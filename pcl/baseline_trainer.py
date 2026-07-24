import logging

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch.cuda import amp
from tqdm import tqdm

from utils.faiss_rerank import compute_jaccard_distance
from utils.meter import AverageMeter
from .loss import ClusterMemoryAMP


@torch.no_grad()
def extract_baseline_features(model, data_loader):
    """Extract descriptors in the loader's deterministic sorted order."""
    model.eval()
    features = []
    filenames = []
    for images, _, batch_filenames in tqdm(
        data_loader, total=len(data_loader), desc="Baseline feature extraction"
    ):
        outputs = model(images.cuda())
        features.append(outputs.cpu())
        filenames.extend(batch_filenames)
    if not features:
        raise RuntimeError("Cannot cluster an empty domain dataset")
    return torch.cat(features, dim=0), filenames


def cluster_domain(features, eps, min_samples, k1, k2):
    normalized = torch.nn.functional.normalize(features.cuda(), p=2, dim=1)
    distance = compute_jaccard_distance(
        normalized,
        k1=k1,
        k2=k2,
        search_option=3,
    )
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
        n_jobs=8,
    ).fit_predict(distance)
    del distance
    return labels


def build_domain_memory(dataset, features, labels, modality, cfg):
    """Drop DBSCAN outliers and initialize one cluster-level PCL memory."""
    if len(dataset) != len(labels) or features.shape[0] != len(labels):
        raise RuntimeError(
            f"{modality} feature/label/data lengths do not match: "
            f"{features.shape[0]}/{len(labels)}/{len(dataset)}"
        )

    pseudo_dataset = []
    cluster_features = {}
    outliers = 0
    for (image_path, _), feature, label in zip(sorted(dataset), features, labels):
        label = int(label)
        if label == -1:
            outliers += 1
            continue
        pseudo_dataset.append((image_path, label))
        cluster_features.setdefault(label, []).append(feature)

    if not cluster_features:
        raise RuntimeError(f"DBSCAN produced no valid {modality} clusters")

    expected_labels = list(range(len(cluster_features)))
    if sorted(cluster_features) != expected_labels:
        raise RuntimeError(
            f"{modality} DBSCAN labels must be contiguous from zero, got "
            f"{sorted(cluster_features)}"
        )

    centers = torch.stack(
        [torch.stack(cluster_features[label]).mean(0) for label in expected_labels]
    )
    centers = torch.nn.functional.normalize(centers.float(), p=2, dim=1).cuda()
    memory = ClusterMemoryAMP(
        temp=cfg.BASELINE.PCL_TEMP,
        momentum=cfg.MODEL.MEMORY_MOMENTUM,
        use_hard=cfg.BASELINE.PCL_HARD_MEMORY,
    ).cuda()
    memory.features = centers

    logging.getLogger("PCL.baseline").info(
        "%s clustering: %d clusters, %d clustered images, %d outliers",
        modality,
        len(expected_labels),
        len(pseudo_dataset),
        outliers,
    )
    return pseudo_dataset, memory


class TwoDomainPCLTrainer:
    """Single-stage trainer with exactly two independent PCL losses."""

    def __init__(self, model):
        self.model = model
        self.scaler = amp.GradScaler()

    def train_epoch(
        self,
        cfg,
        visible_loader,
        infrared_loader,
        visible_memory,
        infrared_memory,
        optimizer,
        epoch,
    ):
        logger = logging.getLogger("PCL.baseline")
        total_meter = AverageMeter()
        visible_meter = AverageMeter()
        infrared_meter = AverageMeter()
        self.model.train()

        visible_loader.new_epoch()
        infrared_loader.new_epoch()
        for iteration in range(cfg.DATALOADER.NUM_ITERS):
            visible_images, visible_pids, _ = visible_loader.next()
            infrared_images, infrared_pids, _ = infrared_loader.next()
            visible_images = visible_images.cuda(non_blocking=True)
            infrared_images = infrared_images.cuda(non_blocking=True)
            visible_pids = visible_pids.cuda(non_blocking=True)
            infrared_pids = infrared_pids.cuda(non_blocking=True)

            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                visible_features = self.model(visible_images, modal=1)
                infrared_features = self.model(infrared_images, modal=2)
                loss_visible = visible_memory(visible_features, visible_pids)
                loss_infrared = infrared_memory(infrared_features, infrared_pids)
                loss = loss_visible + loss_infrared

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            batch_size = visible_images.shape[0] + infrared_images.shape[0]
            total_meter.update(loss.item(), batch_size)
            visible_meter.update(loss_visible.item(), visible_images.shape[0])
            infrared_meter.update(loss_infrared.item(), infrared_images.shape[0])

            if (iteration + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info(
                    "Epoch[%d] Iteration[%d/%d] Loss: %.3f, "
                    "PCL-visible: %.3f, PCL-infrared: %.3f",
                    epoch,
                    iteration + 1,
                    cfg.DATALOADER.NUM_ITERS,
                    total_meter.avg,
                    visible_meter.avg,
                    infrared_meter.avg,
                )

        return {
            "loss": total_meter.avg,
            "pcl_visible": visible_meter.avg,
            "pcl_infrared": infrared_meter.avg,
        }
