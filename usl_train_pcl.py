from utils.logger import setup_logger
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from solver.lr_scheduler import WarmupMultiStepLR
from datasets.make_dataloader import make_dataloader_usl_dnreid, make_dataset, make_clusterloader, make_testloader_usl_dnreid, make_dataset_dnwild
from pcl.processor_pcl import ClusterContrastTrainer, do_inference, do_train_stage1
from pcl.optimizer import make_optimizer_2stage, make_optimizer_1stage
from pcl.model import make_model
from utils.meter import AverageMeter, to_torch
from collections import OrderedDict
import time
import torch.nn as nn
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from utils.faiss_rerank import compute_jaccard_distance
from tqdm import tqdm
from pcl.loss import ClusterMemoryAMP, CrossEntropyLabelSmooth
from pcl.utils import compute_cluster_centroids
from solver.scheduler_factory import create_scheduler
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_all_features(model, data_loader, print_freq=50):
    model.eval()
    features_g = OrderedDict()
    features_proj = OrderedDict()
    img_labels = OrderedDict()

    with torch.no_grad():
        for i, (imgs, pids, fnames) in tqdm(enumerate(data_loader), total=len(data_loader),
                                                  desc='Vision Feature Extracting...'):
            # data_time.update(time.time() - end)
            inputs = to_torch(imgs).cuda()
            if isinstance(model, nn.DataParallel):
                outputs_g, outputs_proj = model.module.extract_all_features(inputs)
            else:
                outputs_g, outputs_proj = model.extract_all_features(inputs)
            outputs_g, outputs_proj = outputs_g.data.cpu(), outputs_proj.data.cpu()
            for fname, output_g, output_proj, pid in zip(fnames, outputs_g, outputs_proj, pids):
                features_g[fname] = output_g
                features_proj[fname] = output_proj
                img_labels[fname] = pid

        return features_g, features_proj, img_labels

def compute_pseudo_labels(features, cluster, k1):
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6,
                                              search_option=3)  # rerank_dist_all_jacard[features_day.size(0):,features_day.size(0):]#
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids



from typing import Tuple
@torch.no_grad()
def mutual_topk_partition(
    features_a: torch.Tensor,  # (a, d)
    features_b: torch.Tensor,  # (b, d)
    k: int = 5,
) -> Tuple[
    torch.Tensor,  # mutual_counts_a: (a,)
    torch.Tensor,  # mutual_counts_b: (b,)
    torch.Tensor,  # pos_pairs: (m, 2) unique mutual pairs (i,j)
    torch.Tensor,  # pos_pairs_from_A: (mA, 2) mutual restricted to A's top-k
    torch.Tensor,  # neg_pairs_a2b: (a*k - mA, 2) per A anchor, non-mutual in its top-k
    torch.Tensor,  # pos_pairs_from_B: (mB, 2) mutual restricted to B's top-k
    torch.Tensor,  # neg_pairs_b2a: (b*k - mB, 2) per B anchor, non-mutual in its top-k
]:
    """
    将每个锚的 top-k 精确划分为 正/负 对，从而满足：
      对 A 侧：pos_from_A 个数 + neg_a2b 个数 == a * k
      对 B 侧：pos_from_B 个数 + neg_b2a 个数 == b * k
    同时返回去重后的 mutual 正对集合 pos_pairs（与方向无关）。
    """
    device = features_a.device
    a, d = features_a.shape
    b, _ = features_b.shape
    kA = min(k, b)
    kB = min(k, a)

    # 1) 归一化 + 相似度
    fa = torch.nn.functional.normalize(features_a, dim=1)
    fb = torch.nn.functional.normalize(features_b, dim=1)
    sim = fa @ fb.T  # (a, b)

    # 2) A->B 与 B->A 的 top-k
    topk_b_for_a = torch.topk(sim, k=kA, dim=1).indices        # (a, kA)
    topk_a_for_b = torch.topk(sim.T, k=kB, dim=1).indices      # (b, kB)

    # 3) 互为 top-k 掩码 M(i,j) = True 当且仅当 j∈Topk_B(i) 且 i∈Topk_A(j)
    A_top = torch.zeros((a, b), dtype=torch.bool, device=device)
    A_top.scatter_(1, topk_b_for_a, True)
    B_top = torch.zeros((b, a), dtype=torch.bool, device=device)
    B_top.scatter_(1, topk_a_for_b, True)
    M = A_top & B_top.T  # (a, b)

    # 4) 去重后的 mutual 正对（与方向无关）
    pos_pairs = M.nonzero(as_tuple=False)  # (m, 2) [i, j]

    # 5) 针对 A 侧：把每个 i 的 top-k 精确划分为 正/负
    #    mask_A(i, t) 表示 topk_b_for_a[i, t] 是否与 i 互为 top-k
    mask_A = M[torch.arange(a, device=device).unsqueeze(1), topk_b_for_a]  # (a, kA)

    # 正对（来源于 A 的 top-k 视角）
    rows_A = torch.arange(a, device=device).unsqueeze(1).expand_as(topk_b_for_a)  # (a, kA)
    pos_A_flat_mask = mask_A.reshape(-1)                      # (a*kA,)
    pos_pairs_from_A = torch.stack([
        rows_A.reshape(-1)[pos_A_flat_mask],
        topk_b_for_a.reshape(-1)[pos_A_flat_mask]
    ], dim=1)                                                # (mA, 2)

    # 负对（每个 i 的 top-k 中非互配项）
    neg_A_flat_mask = (~mask_A).reshape(-1)
    neg_pairs_a2b = torch.stack([
        rows_A.reshape(-1)[neg_A_flat_mask],
        topk_b_for_a.reshape(-1)[neg_A_flat_mask]
    ], dim=1)                                                # (a*kA - mA, 2)

    # 6) 针对 B 侧：同理划分
    mask_B = M.T[torch.arange(b, device=device).unsqueeze(1), topk_a_for_b]  # (b, kB)
    rows_B = torch.arange(b, device=device).unsqueeze(1).expand_as(topk_a_for_b)

    pos_B_flat_mask = mask_B.reshape(-1)
    pos_pairs_from_B = torch.stack([
        topk_a_for_b.reshape(-1)[pos_B_flat_mask],  # i in A
        rows_B.reshape(-1)[pos_B_flat_mask]         # j in B
    ], dim=1)                                       # (mB, 2) 注意顺序统一为 (i, j)

    neg_B_flat_mask = (~mask_B).reshape(-1)
    neg_pairs_b2a = torch.stack([
        topk_a_for_b.reshape(-1)[neg_B_flat_mask],  # i in A
        rows_B.reshape(-1)[neg_B_flat_mask]         # j in B
    ], dim=1)                                       # (b*kB - mB, 2)

    # 7) 统计（可用于断言/调试）
    mutual_counts_a = M.sum(dim=1)  # (a,)
    mutual_counts_b = M.sum(dim=0)  # (b,)

    # 可选：一致性检查（训练时可注释掉避免开销）
    # assert pos_pairs_from_A.size(0) + neg_pairs_a2b.size(0) == a * kA
    # assert pos_pairs_from_B.size(0) + neg_pairs_b2a.size(0) == b * kB

    return (mutual_counts_a, mutual_counts_b,
            pos_pairs, pos_pairs_from_A, neg_pairs_a2b,
            pos_pairs_from_B, neg_pairs_b2a)

# 统一的键构造方式（和你原来一致）
def key3(path):
    return "/".join(str(path).split("/")[-3:])
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="config/dn348-vit.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(cfg.SOLVER.LOCAL_RANK)

    logger = setup_logger("PCL", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    device = "cuda"
    if cfg.DATASETS.NAMES == "dn348":
        train_day, train_night, Test_day, Test_night = make_dataset(cfg)
        day_eps, night_eps = 0.7, 0.7
    else:
        train_day, train_night, Query_day, Query_night, Test_day, Test_night = make_dataset_dnwild(cfg)
        day_eps, night_eps = 0.8, 0.8
    cluster_loader_day = make_clusterloader(cfg, train_day)
    cluster_loader_night = make_clusterloader(cfg, train_night)

    # 1) 生成与拼接顺序严格一致的有序键列表
    ordered_keys_day = [key3(f) for (f, _) in sorted(train_day)]
    ordered_keys_night = [key3(f) for (f, _) in sorted(train_night)]
    # 4) 建立双向映射：索引 <-> 文件键
    fname2index_day = {k: i for i, k in enumerate(ordered_keys_day)}
    fname2index_night = {k: i for i, k in enumerate(ordered_keys_night)}



    model = make_model(cfg, num_classes=3000, num_img_day=len(train_day), num_img_night=len(train_night))
    # model.load_param(cfg.TEST.WEIGHT)
    model.cuda()
    model = nn.DataParallel(model)

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
                                        lr_min=cfg.SOLVER.STAGE1.LR_MIN, \
                                        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
                                        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range=None)

    do_train_stage1(
        cfg,
        model,
        cluster_loader_day,
        cluster_loader_night,
        ordered_keys_day,
        ordered_keys_night,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    optimizer = make_optimizer_2stage(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                         cfg.SOLVER.WARMUP_FACTOR,
                                         cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    epochs = cfg.SOLVER.MAX_EPOCHS
    pseudo_precisions = []
    BEST = [0.0, 0.0, 0.0, 0.0]
    score = 0
    eval_period = cfg.SOLVER.EVAL_PERIOD
    trainer = ClusterContrastTrainer(model)
    for epoch in range(1, epochs + 1):

        with torch.no_grad():
            if epoch == 1:
                # DBSCAN cluster precomputed euclidean
                cluster_day = DBSCAN(eps=day_eps, min_samples=4, metric='precomputed', n_jobs=8)
                cluster_night = DBSCAN(eps=night_eps, min_samples=4, metric='precomputed', n_jobs=8)

            print(f'==> Create pseudo labels for unlabeled day data, epse:{day_eps}')
            features_day, features_day_proj, img_labels_day = extract_all_features(model, cluster_loader_day, logger)
            features_day = torch.cat(
                [features_day["/".join(str(f).split("/")[-3:])].unsqueeze(0) for f, _, in sorted(train_day)],
                0)
            features_day_proj = torch.cat(
                [features_day_proj["/".join(str(f).split("/")[-3:])].unsqueeze(0) for f, _, in sorted(train_day)], 0)

            features_total_day = torch.cat([features_day, features_day_proj],
                                           dim=1)  # Similar to evaluation, it is used to cluster
            # features_total_day = features_day_proj
            features_total_day = F.normalize(features_total_day, p=2, dim=1)
            features_total_day = features_total_day.to('cuda', dtype=torch.float32)
            # # assign pseudo-labels
            pseudo_labels_day, num_class_day = compute_pseudo_labels(features_total_day, cluster_day, cfg.SOLVER.K1)

            del features_day, features_day_proj, img_labels_day
            torch.cuda.empty_cache()

            # generate new dataset with pseudo-labels
            num_outliers_day = 0
            new_dataset_day = []
            pids_day, all_pids_day = [], []
            for i, ((fname, _), label) in enumerate(zip(sorted(train_day), pseudo_labels_day)):
                pid = label.item()
                if pid >= num_class_day:  # append data except outliers
                    num_outliers_day += 1
                else:
                    new_dataset_day.append((fname, pid))
                    pids_day.append(pid)
                all_pids_day.append(pid)

            # statistics of clusters and un-clustered instances
            print('==> Statistics for epoch {} day train_set: {} clusters, {} un-clustered instances'.format(epoch,
                                                                                                             num_class_day,
                                                                                                             num_outliers_day))

            print(f'==> Create pseudo labels for unlabeled night data, epse:{night_eps}')
            features_night, features_night_proj, img_labels_night = extract_all_features(model, cluster_loader_night,
                                                                                         logger)
            features_night = torch.cat(
                [features_night["/".join(str(f).split("/")[-3:])].unsqueeze(0) for f, _, in sorted(train_night)],
                0)
            features_night_proj = torch.cat(
                [features_night_proj["/".join(str(f).split("/")[-3:])].unsqueeze(0) for f, _, in sorted(train_night)],
                0)

            features_total_night = torch.cat([features_night, features_night_proj],
                                             dim=1)  # Similar to evaluation, it is used to cluster
            # features_total_night = features_night_proj
            features_total_night = F.normalize(features_total_night, p=2, dim=1)

            # assign pseudo-labels
            features_total_night = features_total_night.to('cuda', dtype=torch.float32)
            pseudo_labels_night, num_class_night = compute_pseudo_labels(features_total_night, cluster_night,
                                                                         cfg.SOLVER.K1)

            # === 释放夜间特征 ===
            del features_night, features_night_proj, img_labels_night
            torch.cuda.empty_cache()

            # generate new dataset with pseudo-labels
            num_outliers_night = 0
            new_dataset_night = []
            pids_night, all_pids_night = [], []
            for i, ((fname, _), label) in enumerate(zip(sorted(train_night), pseudo_labels_night)):
                pid = label.item()
                if pid >= num_class_night:  # append data except outliers
                    num_outliers_day += 1
                else:
                    new_dataset_night.append((fname, pid))
                    pids_night.append(pid)
                all_pids_night.append(pid)

            # statistics of clusters and un-clustered instances
            print('==> Statistics for epoch {} night train_set: {} clusters, {} un-clustered instances'.format(epoch,
                                                                                                               num_class_night,
                                                                                                               num_outliers_night))

            train_loader_day = make_dataloader_usl_dnreid(cfg, new_dataset_day)

            train_loader_night = make_dataloader_usl_dnreid(cfg, new_dataset_night)

            # CAP memory
            memory_day = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
            memory_day.features = compute_cluster_centroids(features_total_day, pseudo_labels_day).to(device)

            # CAP memory
            memory_night = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
            memory_night.features = compute_cluster_centroids(features_total_night, pseudo_labels_night).to(device)

            # compute cluster centroids
            centroids_total_day, centroids_total_night = [], []
            for pid in sorted(np.unique(pids_day)):  # loop all pids
                idxs_p = np.where(all_pids_day == pid)[0]
                centroids_total_day.append(features_total_day[idxs_p].mean(0))
            model.module.classifier_day.weight.data[:num_class_day].copy_(memory_day.features[:num_class_day])

            for pid in sorted(np.unique(pids_night)):  # loop all pids
                idxs_p = np.where(all_pids_night == pid)[0]
                centroids_total_night.append(features_total_night[idxs_p].mean(0))
            model.module.classifier_night.weight.data[:num_class_night].copy_(memory_night.features[:num_class_night])

            mutual_counts_a, mutual_counts_b, pos_pairs, posA, neg_day2night, posB, neg_night2day = mutual_topk_partition(memory_day.features[:num_class_day].contiguous(), memory_night.features[:num_class_night].contiguous(), k=15)
            print("每个 a_i 的互为 top-k 个数:", mutual_counts_a.tolist())  # 范围 0..k
            print("每个 b_j 的互为 top-k 个数:", mutual_counts_b.tolist())  # 范围 0..k
            print("互为 top-k 的 (i, j) 对:\n", len(pos_pairs))  # (i in A, j in B)
            # del mutual_counts_a, mutual_counts_b, pos_pairs

        trainer.memory_day = memory_day
        trainer.memory_night = memory_night

        trainer.train(cfg, train_loader_day, train_loader_night, epoch, optimizer, scheduler, num_class_day,
                      num_class_night, fname2index_day, fname2index_night, pos_pairs=pos_pairs, neg_day2night=neg_day2night,
                        neg_night2day=neg_night2day)

        if epoch % eval_period == 0:
            if cfg.DATASETS.NAMES == "dn348":
                print("--------Day to Night Test-----------")
                val_loader = make_testloader_usl_dnreid(cfg, Test_day, Test_night)
                map_d2n, r1_d2n, r5d2n = do_inference(cfg,
                            model,
                            val_loader,
                            len(Test_day))

                print("--------Night to Day Test-----------")
                val_loader = make_testloader_usl_dnreid(cfg, Test_night, Test_day)
                map_d2n, r1_n2d, r5n2d = do_inference(cfg,
                            model,
                            val_loader,
                            len(Test_night))
            else:
                print("--------Day to Night Test-----------")
                val_loader = make_testloader_usl_dnreid(cfg, Query_day, Test_night)
                map_n2d, r1_d2n, r5d2n = do_inference(cfg,
                            model,
                            val_loader,
                            len(Query_day))

                print("--------Night to Day Test-----------")
                val_loader = make_testloader_usl_dnreid(cfg, Query_night, Test_day)
                map_n2d, r1_n2d, r5n2d = do_inference(cfg,
                            model,
                            val_loader,
                            len(Query_night))
            
            if r1_d2n+r1_n2d >score:
                score = r1_n2d + r1_d2n
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f'{cfg.DATASETS.NAMES}_reid_best.pth'))