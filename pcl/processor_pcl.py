import logging
import os
import torch
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from .utils import *
from .loss import ClusterMemoryAMP, CrossEntropyLabelSmooth
from .supcontrast import SupConLoss
import torch.distributed as dist
import torch.nn as nn
# from .utils.meter import AverageMeter, to_torch
from collections import OrderedDict
from tqdm import tqdm
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def extract_all_features(model, data_loader):
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

def do_train_stage1(cfg,
                    model,
                    cluster_loader_day,
                    cluster_loader_night,
                    ordered_keys_day,
                    ordered_keys_night,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("PCL")
    logger.info('start training stage1')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():

        features_day, features_day_proj, img_labels_day = extract_all_features(model, cluster_loader_day)
        features_day = torch.cat([features_day[k].unsqueeze(0) for k in ordered_keys_day], 0)
        features_day_proj = torch.cat([features_day_proj[k].unsqueeze(0) for k in ordered_keys_day], 0)
        features_total_day = torch.cat([features_day, features_day_proj],
                                         dim=1)  # Similar to evaluation, it is used to cluster
        # features_total_day = features_day_proj
        features_total_day = F.normalize(features_total_day, p=2, dim=1).cuda()

        idx_day = torch.arange(0, features_day_proj.shape[0], dtype=torch.long, device='cuda')

        fname2index_day = {k: i for i, k in enumerate(ordered_keys_day)}

        features_night, features_night_proj, img_labels_night = extract_all_features(model, cluster_loader_night)
        features_night = torch.cat([features_night[k].unsqueeze(0) for k in ordered_keys_night], 0)
        features_night_proj = torch.cat([features_night_proj[k].unsqueeze(0) for k in ordered_keys_night], 0)
        
        features_total_night = torch.cat([features_night, features_night_proj],
                                         dim=1)  # Similar to evaluation, it is used to cluster
        # features_total_night = features_night_proj
        features_total_night = F.normalize(features_total_night, p=2, dim=1).cuda()

        idx_night = torch.arange(0, features_night_proj.shape[0], dtype=torch.long, device='cuda')
        # fname2index_night = {k: i for i, k in enumerate(ordered_keys_night)}


        nums_day, nums_night = features_day_proj.shape[0], features_night_proj.shape[0]
    del labels, image_features

    features_day_proj = features_day_proj.cuda()
    features_night_proj = features_night_proj.cuda()
    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        if nums_day > nums_night:
            iter_list_day = torch.randperm(nums_day).to(device)
            iter_list_night = torch.cat([torch.randperm(nums_night),torch.randint(0,nums_night,(nums_day-nums_night,))],dim=0).to(device)
        elif nums_day == nums_night:
            iter_list_day = torch.randperm(nums_day).to(device)
            iter_list_night = torch.randperm(nums_night).to(device)
        else:
            iter_list_night = torch.randperm(nums_night).to(device)
            iter_list_day = torch.cat([torch.randperm(nums_day), torch.randint(0, nums_day, (nums_night - nums_day,))], dim=0).to(device)

        i_ter = len(iter_list_day) // batch

        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list_day = iter_list_day[i * batch:(i + 1) * batch]
                b_list_night = iter_list_night[i * batch:(i + 1) * batch]
            else:
                b_list_day = iter_list_day[i * batch:len(iter_list_day)]
                b_list_night = iter_list_night[i * batch:len(iter_list_day)]


            target_day = idx_day[b_list_day]
            target_night = idx_night[b_list_night]

            image_features_day = features_total_day[b_list_day]
            image_features_night = features_total_night[b_list_night]
            image_features_day_proj = features_day_proj[b_list_day]
            image_features_night_proj = features_night_proj[b_list_night]
            
            with amp.autocast(enabled=True):
                text_features_day = model(idx=target_day, modal=1, vis_feat=image_features_day, get_text=True)
                text_features_night = model(idx=target_night, modal=2, vis_feat=image_features_night, get_text=True)


            loss_i2t_day = xent(image_features_day_proj, text_features_day, target_day, target_day)
            loss_t2i_day = xent(text_features_day, image_features_day_proj, target_day, target_day)

            loss_i2t_night = xent(image_features_night_proj, text_features_night, target_night, target_night)
            loss_t2i_night = xent(text_features_night, image_features_night_proj, target_night, target_night)

            loss = loss_i2t_day + loss_t2i_day + loss_t2i_night + loss_i2t_night

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), features_day_proj.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), i_ter+1,
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch == epochs:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))



def train_pcl(cfg,
              model,
              train_loader,
              val_loader,
              cluster_loader,
              optimizer,
              scheduler,
              num_query,
              num_classes):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PCL")
    logger.info('start training')
    
    model.to(device)
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    xent = CrossEntropyLabelSmooth(num_classes)
    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    # training epochs
    for epoch in range(1, epochs+1):
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        # create memory bank
        image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
            
        num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
        logger.info(f'Memory has {num_classes} classes.')
        
        train_loader.new_epoch()
        
        # memory
        memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
        logger.info('Create memory bank with shape = {}'.format(memory.features.shape))
        
        # train one iteration
        model.train()
        num_iters = len(train_loader)
        for n_iter in range(num_iters):
            img, target, target_cam, _ = train_loader.next()
            
            optimizer.zero_grad()
            
            img = img.to(device)
            target = target.to(device)
            target_cam = target_cam.to(device)
            
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            with amp.autocast(enabled=True):
                feat, logit = model(img, cam_label=target_cam, view_label=target_view)
                loss = memory(feat, target) * cfg.MODEL.PCL_LOSS_WEIGHT
                if cfg.MODEL.ID_LOSS_WEIGHT > 0:
                    loss_id = xent(logit, target) * cfg.MODEL.ID_LOSS_WEIGHT
                else:
                    loss_id = 0
                loss = loss + loss_id

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_lr()[0]))
        
        scheduler.step()
        logger.info("Epoch {} done.".format(epoch))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else: 
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else: 
                        target_view = None
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)

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

from collections import defaultdict
@torch.no_grad()
def _group_neg_by_rowid(pairs_rowid, num_rows):
    """
    pairs_rowid: (m, 2) -> [row_id_in_pos_pairs, neg_index_on_other_side]
    返回长度 num_rows 的 list，每个元素是 Tensor(neg_idx) 或 None
    """
    buckets = [[] for _ in range(num_rows)]
    if pairs_rowid is not None and pairs_rowid.numel() > 0:
        r_idx, n_idx = pairs_rowid[:, 0].tolist(), pairs_rowid[:, 1].tolist()
        for r, n in zip(r_idx, n_idx):
            buckets[r].append(n)
    out = []
    for lst in buckets:
        out.append(torch.tensor(lst, dtype=torch.long) if lst else None)
    return out

def sym_infoc_nce_all_ij(
    fa, fb,
    pos_pairs_ij,             # (m,2)  (i,j) 正对
    neg_pairs_ab_ij=None,     # (p,2)  (i,j-)  A->B 的负对
    neg_pairs_ba_ij=None,     # (q,2)  (i-,j)  B->A 的负对（注意仍是 (i,j) 顺序，这里 i 属于 A）
    tau: float = 0.07
):
    """
    假设所有 pair 都是 (i_in_A, j_in_B) 顺序。
    """
    fa = F.normalize(fa, dim=1)
    fb = F.normalize(fb, dim=1)

    with torch.cuda.amp.autocast(enabled=False):
        sim = (fa @ fb.t()).float()
        if pos_pairs_ij is None or pos_pairs_ij.numel() == 0:
            return sim.new_zeros(())

        i_pos = pos_pairs_ij[:,0].long().to(sim.device)
        j_pos = pos_pairs_ij[:,1].long().to(sim.device)
        tau = float(tau)

        # ===== A -> B =====
        pos_ab = sim[i_pos, j_pos] / tau  # (m,)
        if neg_pairs_ab_ij is None or neg_pairs_ab_ij.numel() == 0:
            # in-batch negatives
            row = (sim[i_pos] / tau)                       # (m,B)
            row[torch.arange(i_pos.size(0), device=sim.device), j_pos] = float('-inf')
            loss_ab = -(pos_ab - torch.logsumexp(row, dim=1)).mean()
        else:
            # 按 (i,jneg) 汇总
            # 为每个 (i_pos[k]) 收集它的 jneg 列表
            m = i_pos.size(0)
            # 建立 i -> 所有对应的行号 rid（因为同一个 i 可能出现在 pos_pairs_ij 的多个 rid）
            from collections import defaultdict
            i_to_rows = defaultdict(list)
            for rid, i in enumerate(i_pos.tolist()):
                i_to_rows[i].append(rid)

            # 将 (i,jneg) 映射到每个 rid
            rows, negs = [], []
            for (ii, jn) in neg_pairs_ab_ij.long().tolist():
                if ii in i_to_rows:
                    for rid in i_to_rows[ii]:
                        rows.append(rid); negs.append(jn)
            if len(rows) == 0:
                row = (sim[i_pos] / tau)
                row[torch.arange(m, device=sim.device), j_pos] = float('-inf')
                loss_ab = -(pos_ab - torch.logsumexp(row, dim=1)).mean()
            else:
                rows = torch.tensor(rows, device=sim.device, dtype=torch.long)
                negs = torch.tensor(negs, device=sim.device, dtype=torch.long)
                # 对每个 rid 构造 [pos | negs] 并计算 -log softmax
                loss_list = []
                for rid in range(m):
                    pos = (sim[i_pos[rid], j_pos[rid]] / tau).view(1)
                    sel = (rows == rid).nonzero(as_tuple=False).flatten()
                    if sel.numel() == 0:
                        row = (sim[i_pos[rid]] / tau).clone()
                        row[j_pos[rid]] = float('-inf')
                        loss_list.append( - (pos - torch.logsumexp(row, dim=0)) )
                    else:
                        neg = (sim[i_pos[rid], negs[sel]] / tau)   # (k,)
                        logits = torch.cat([pos, neg], dim=0)
                        loss_list.append( - (pos - torch.logsumexp(logits, dim=0)) )
                loss_ab = torch.stack(loss_list).mean()

        # ===== B -> A =====
        # 方向相反：以 B 为锚等价于在 A 中找 negatives，但我们仍保持 (i,j) 顺序
        pos_ba = pos_ab  # 同一个 (i,j)
        if neg_pairs_ba_ij is None or neg_pairs_ba_ij.numel() == 0:
            col = (sim.t()[j_pos] / tau)                    # (m,A)
            col[torch.arange(j_pos.size(0), device=sim.device), i_pos] = float('-inf')
            loss_ba = -(pos_ba - torch.logsumexp(col, dim=1)).mean()
        else:
            # neg_pairs_ba_ij 是 (i_neg, j) 的集合（仍是 (i,j) 顺序）
            # 为每个 (j_pos[k]) 收集它的 i_neg 列表
            from collections import defaultdict
            j_to_rows = defaultdict(list)
            for rid, j in enumerate(j_pos.tolist()):
                j_to_rows[j].append(rid)
            rows, negs = [], []
            for (ineg, jj) in neg_pairs_ba_ij.long().tolist():
                if jj in j_to_rows:
                    for rid in j_to_rows[jj]:
                        rows.append(rid); negs.append(ineg)
            if len(rows) == 0:
                col = (sim.t()[j_pos] / tau)
                col[torch.arange(j_pos.size(0), device=sim.device), i_pos] = float('-inf')
                loss_ba = -(pos_ba - torch.logsumexp(col, dim=1)).mean()
            else:
                rows = torch.tensor(rows, device=sim.device, dtype=torch.long)
                negs = torch.tensor(negs, device=sim.device, dtype=torch.long)
                loss_list = []
                for rid in range(i_pos.size(0)):
                    pos = (sim[i_pos[rid], j_pos[rid]] / tau).view(1)
                    sel = (rows == rid).nonzero(as_tuple=False).flatten()
                    if sel.numel() == 0:
                        col = (sim.t()[j_pos[rid]] / tau).clone()
                        col[i_pos[rid]] = float('-inf')
                        loss_list.append( - (pos - torch.logsumexp(col, dim=0)) )
                    else:
                        neg = (sim[negs[sel], j_pos[rid]] / tau)   # (k,)
                        logits = torch.cat([pos, neg], dim=0)
                        loss_list.append( - (pos - torch.logsumexp(logits, dim=0)) )
                loss_ba = torch.stack(loss_list).mean()

        return 0.5 * (loss_ab + loss_ba)
def _check_pairs_range(fa, fb, pos_pairs, neg_pairs_a2b, neg_pairs_b2a, tag=""):
    A, B = fa.size(0), fb.size(0)
    dev = fa.device
    def _to(x): 
        return x.to(dev) if x is not None else None

    pos_pairs    = _to(pos_pairs)
    neg_pairs_a2b = _to(neg_pairs_a2b)
    neg_pairs_b2a = _to(neg_pairs_b2a)

    def oob(mask): 
        return mask.any().item()

    msgs = []
    if pos_pairs is not None and pos_pairs.numel()>0:
        if oob(pos_pairs[:,0] >= A) or oob(pos_pairs[:,1] >= B):
            msgs.append(f"[{tag}] pos_pairs out of range: max_i={pos_pairs[:,0].max().item()} vs A={A-1}, "
                        f"max_j={pos_pairs[:,1].max().item()} vs B={B-1}")

    if neg_pairs_a2b is not None and neg_pairs_a2b.numel()>0:
        if oob(neg_pairs_a2b[:,0] >= A) or oob(neg_pairs_a2b[:,1] >= B):
            msgs.append(f"[{tag}] neg_pairs_a2b out of range")

    # 注意：如果你坚持“全部都是 (i_in_A, j_in_B) 顺序”，这里也按 (i,j) 检查：
    if neg_pairs_b2a is not None and neg_pairs_b2a.numel()>0:
        if oob(neg_pairs_b2a[:,0] >= A) or oob(neg_pairs_b2a[:,1] >= B):
            msgs.append(f"[{tag}] neg_pairs_b2a out of range (as (i,j) ordering)")

    if msgs:
        raise RuntimeError("\n".join(msgs))


def compute_centroids(text_features, labels):
    # 计算文本质心
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)

    # 创建标签映射
    label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_to_idx[label.item()] for label in labels],
                                 device=labels.device)

    # 计算每个类别的文本质心
    centroids = torch.zeros(num_classes, text_features.shape[1], device=text_features.device)
    for i in range(num_classes):
        mask = (mapped_labels == i)
        if mask.sum() > 0:
            centroids[i] = text_features[mask].mean(dim=0)

    centroids = F.normalize(centroids, dim=1)
    return mapped_labels, centroids

class ClusterContrastTrainer(object):
    def __init__(self, model, memory=None, device="cuda"):
        super(ClusterContrastTrainer, self).__init__()
        self.model = model
        self.memory_day = memory
        self.memory_night = memory
        self.device = device

    def train(self, cfg, train_loader_day, train_loader_night, epoch, optimizer, scheduler, num_class_day, num_class_night, fname2index_day, fname2index_night, pos_pairs=None, neg_day2night=None, neg_night2day=None):
        log_period = cfg.SOLVER.LOG_PERIOD
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        eval_period = cfg.SOLVER.EVAL_PERIOD

        logger = logging.getLogger("PCL")
        logger.info('start training')


        loss_meter = AverageMeter()
        loss_i2tce_meter = AverageMeter()
        loss_cm_meter = AverageMeter()
        loss_id_meter = AverageMeter()
        acc_meter = AverageMeter()
        xent_day = CrossEntropyLabelSmooth(num_class_day)
        xent_night = CrossEntropyLabelSmooth(num_class_night)


        # evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        scaler = amp.GradScaler()
        self.model.train()
        for n_iter in range(cfg.DATALOADER.NUM_ITERS):
            img_day, pid_day, fnames_day = train_loader_day.next()
            img_night, pid_night, fnames_night = train_loader_night.next()

            optimizer.zero_grad()
            img_day, pid_day = img_day.to(self.device), pid_day.to(self.device)
            img_night, pid_night = img_night.to(self.device), pid_night.to(self.device)

            with amp.autocast(enabled=True):
                feat_day, logit_day, feat_day_proj = self.model(x=img_day, modal=1)
                feat_night, logit_night, feat_night_proj = self.model(x=img_night, modal=2)

                #get text
                target_day = torch.tensor([fname2index_day[key] for key in fnames_day])
                target_night = torch.tensor([fname2index_night[key] for key in fnames_night])
                text_features_day = self.model(idx=target_day, modal=1, vis_feat=feat_day, get_text=True)
                text_features_night = self.model(idx=target_night, modal=2, vis_feat=feat_night, get_text=True)


                mapped_labels_day, text_centroids_day = compute_centroids(text_features_day, pid_day)
                
                mapped_labels_night, text_centroids_night = compute_centroids(text_features_night, pid_night)

                logits_day = feat_day_proj @ text_centroids_day.t()
                logits_night = feat_night_proj @ text_centroids_night.t()
                loss_i2tce_day = xent_day(logits_day, mapped_labels_day, text_centroids_day.shape[0])
                loss_i2tce_night = xent_night(logits_night, mapped_labels_night, text_centroids_night.shape[0])
                loss_i2tce = loss_i2tce_day + loss_i2tce_night

                loss_day = self.memory_day(feat_day, pid_day)
                loss_night = self.memory_night(feat_night, pid_night)

                # loss_id_day =  xent_day(F.softmax(logit_day[:, :num_class_day], dim=1), pid_day, num_class_day) 
                # loss_id_night = xent_night(F.softmax(logit_night[:, :num_class_night], dim=1), pid_night, num_class_night) 
                loss_id_day =  xent_day(logit_day[:, :num_class_day], pid_day, num_class_day) 
                loss_id_night = xent_night(logit_night[:, :num_class_night], pid_night, num_class_night) 
                
                loss_id = loss_id_day + loss_id_night


                fa = self.memory_day.features[:num_class_day].contiguous()
                fb = self.memory_night.features[:num_class_night].contiguous()
                # mutual_counts_a, mutual_counts_b, pos_pairs, posA, neg_day2night, posB, neg_night2day = mutual_topk_partition(fa, fb, k=50)
                # _check_pairs_range(fa, fb, pos_pairs, neg_ab, neg_ba, tag="pre-loss")
                loss_cm = sym_infoc_nce_all_ij(fa, fb, pos_pairs, neg_day2night, neg_night2day)
                loss = loss_day + loss_night + loss_i2tce + loss_cm + loss_id

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img_day.shape[0])
            loss_i2tce_meter.update(loss_i2tce.item(), img_day.shape[0])
            loss_cm_meter.update(loss_cm.item(), img_day.shape[0])
            loss_id_meter.update(loss_id.item(), img_day.shape[0])
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_i2tce: {:.3f}, Loss_cm: {:.3f}, Loss_id: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_day),
                                    loss_meter.avg, loss_i2tce_meter.avg, loss_cm_meter.avg, loss_id_meter.avg, scheduler.get_lr()[0]))

        scheduler.step()
        logger.info("Epoch {} done.".format(epoch))




def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("PCL")
    logger.info("Enter inferencing")
    model.to(device)

    evaluator= R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()


    model.eval()
    for n_iter, (img, pid, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img)
            evaluator.update((feat, pid))

    cmc, mAP, minp, = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}, minp:{:.1%}".format(mAP, minp))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return mAP, cmc[0], cmc[4]