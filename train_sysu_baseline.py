import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from datasets.make_dataloader import (
    make_clusterloader,
    make_dataloader_usl_dnreid,
    make_sysu_dataset_manager,
    make_testloader_sysu,
)
from pcl.baseline_model import make_baseline_model
from pcl.baseline_trainer import (
    TwoDomainPCLTrainer,
    build_domain_memory,
    cluster_domain,
    extract_baseline_features,
)
from pcl.processor_pcl import do_inference_sysu
from solver.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(cfg, model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if cfg.SOLVER.OPTIMIZER_NAME == "SGD":
        return torch.optim.SGD(
            parameters,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    if cfg.SOLVER.OPTIMIZER_NAME == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    optimizer_class = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)
    return optimizer_class(parameters, lr=cfg.SOLVER.BASE_LR)


def evaluate_sysu_baseline(cfg, model, dataset, epoch):
    logger = logging.getLogger("PCL.baseline")
    metric_keys = ("mAP", "mINP", "rank1", "rank5", "rank10", "rank20")
    all_results = {}
    for mode in ("all", "indoor"):
        trial_results = []
        for trial in range(cfg.BASELINE.EVAL_TRIALS):
            loader, num_query = make_testloader_sysu(
                cfg, dataset, mode=mode, trial=trial
            )
            result = do_inference_sysu(cfg, model, loader, num_query)
            trial_results.append(result)
            logger.info(
                "Epoch %d SYSU %s trial %d/%d: R1 %.1f%%, mAP %.1f%%, mINP %.1f%%",
                epoch,
                mode,
                trial + 1,
                cfg.BASELINE.EVAL_TRIALS,
                result["rank1"] * 100.0,
                result["mAP"] * 100.0,
                result["mINP"] * 100.0,
            )

        mean_result = {
            key: float(np.mean([result[key] for result in trial_results]))
            for key in metric_keys
        }
        all_results[mode] = mean_result
        logger.info(
            "Epoch %d SYSU %s-search mean: R1 %.1f%%, R5 %.1f%%, "
            "R10 %.1f%%, R20 %.1f%%, mAP %.1f%%, mINP %.1f%%",
            epoch,
            mode,
            mean_result["rank1"] * 100.0,
            mean_result["rank5"] * 100.0,
            mean_result["rank10"] * 100.0,
            mean_result["rank20"] * 100.0,
            mean_result["mAP"] * 100.0,
            mean_result["mINP"] * 100.0,
        )
    return all_results


def checkpoint_state(model, optimizer, scheduler, epoch, best_rank1):
    model_without_parallel = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "model": model_without_parallel.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_rank1": best_rank1,
        "descriptor": "clip_cls_l2_768",
        "objective": "pcl_visible_plus_pcl_infrared",
    }


def validate_feature_order(dataset, filenames, modality):
    expected = ["/".join(str(path).replace("\\", "/").split("/")[-3:]) for path, _ in sorted(dataset)]
    if filenames != expected:
        raise RuntimeError(f"{modality} cluster-loader order does not match the dataset")


def main():
    parser = argparse.ArgumentParser(description="SYSU-MM01 two-domain PCL baseline")
    parser.add_argument("--config_file", default="config/sysu-baseline.yml")
    parser.add_argument("--weights", default="", help="baseline checkpoint to load")
    parser.add_argument("--evaluate", action="store_true", help="evaluate weights and exit")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    dataset_name = str(cfg.DATASETS.NAMES).strip().lower().replace("_", "-")
    if dataset_name not in {"sysu", "sysu-mm01"}:
        raise ValueError("The baseline entry point only supports SYSU-MM01")
    if cfg.MODEL.DIST_TRAIN:
        raise ValueError("The minimal baseline supports DataParallel, not distributed training")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("PCL.baseline", cfg.OUTPUT_DIR, if_train=not args.evaluate)
    logger.info("Single-stage baseline objective: PCL-visible + PCL-infrared")
    logger.info("Descriptor: raw CLIP CLS (768-D) + L2 normalization; no BN neck")
    logger.info("Running with config:\n%s", cfg)
    set_seed(cfg.SOLVER.SEED)

    dataset = make_sysu_dataset_manager(cfg)
    train_visible = dataset.train_visible
    train_infrared = dataset.train_infrared
    cluster_loader_visible = make_clusterloader(cfg, train_visible)
    cluster_loader_infrared = make_clusterloader(cfg, train_infrared)

    model = make_baseline_model(cfg)
    if args.weights:
        model.load_param(args.weights)
        logger.info("Loaded baseline weights from %s", args.weights)
    model.cuda()
    model = nn.DataParallel(model)

    if args.evaluate:
        evaluate_sysu_baseline(cfg, model, dataset, epoch=0)
        return

    optimizer = build_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        cfg.SOLVER.WARMUP_FACTOR,
        cfg.SOLVER.WARMUP_ITERS,
        cfg.SOLVER.WARMUP_METHOD,
    )
    trainer = TwoDomainPCLTrainer(model)

    logger.info("Running epoch-0 evaluation before PCL training")
    initial_results = evaluate_sysu_baseline(cfg, model, dataset, epoch=0)
    best_rank1 = initial_results["all"]["rank1"]
    torch.save(
        checkpoint_state(model, optimizer, scheduler, 0, best_rank1),
        os.path.join(cfg.OUTPUT_DIR, "baseline_best.pth"),
    )

    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        visible_features, visible_filenames = extract_baseline_features(
            model, cluster_loader_visible
        )
        validate_feature_order(train_visible, visible_filenames, "visible")
        visible_labels = cluster_domain(
            visible_features,
            cfg.BASELINE.CLUSTER_EPS,
            cfg.BASELINE.CLUSTER_MIN_SAMPLES,
            cfg.SOLVER.K1,
            cfg.BASELINE.JACCARD_K2,
        )
        pseudo_visible, memory_visible = build_domain_memory(
            train_visible, visible_features, visible_labels, "visible", cfg
        )

        infrared_features, infrared_filenames = extract_baseline_features(
            model, cluster_loader_infrared
        )
        validate_feature_order(train_infrared, infrared_filenames, "infrared")
        infrared_labels = cluster_domain(
            infrared_features,
            cfg.BASELINE.CLUSTER_EPS,
            cfg.BASELINE.CLUSTER_MIN_SAMPLES,
            cfg.SOLVER.K1,
            cfg.BASELINE.JACCARD_K2,
        )
        pseudo_infrared, memory_infrared = build_domain_memory(
            train_infrared, infrared_features, infrared_labels, "infrared", cfg
        )

        train_loader_visible = make_dataloader_usl_dnreid(cfg, pseudo_visible)
        train_loader_infrared = make_dataloader_usl_dnreid(cfg, pseudo_infrared)
        trainer.train_epoch(
            cfg,
            train_loader_visible,
            train_loader_infrared,
            memory_visible,
            memory_infrared,
            optimizer,
            epoch,
        )
        scheduler.step()

        should_evaluate = (
            epoch % cfg.SOLVER.EVAL_PERIOD == 0
            or epoch == cfg.SOLVER.MAX_EPOCHS
        )
        if should_evaluate:
            results = evaluate_sysu_baseline(cfg, model, dataset, epoch)
            rank1 = results["all"]["rank1"]
            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(
                    checkpoint_state(model, optimizer, scheduler, epoch, best_rank1),
                    os.path.join(cfg.OUTPUT_DIR, "baseline_best.pth"),
                )
                logger.info("New best all-search Rank-1: %.1f%%", best_rank1 * 100.0)

        if (
            epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0
            or epoch == cfg.SOLVER.MAX_EPOCHS
        ):
            torch.save(
                checkpoint_state(model, optimizer, scheduler, epoch, best_rank1),
                os.path.join(cfg.OUTPUT_DIR, f"baseline_epoch_{epoch:03d}.pth"),
            )

    logger.info("Training finished. Best all-search Rank-1: %.1f%%", best_rank1 * 100.0)


if __name__ == "__main__":
    main()
