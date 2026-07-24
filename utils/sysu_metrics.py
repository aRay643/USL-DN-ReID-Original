import numpy as np


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluate the official SYSU-MM01 IR-to-visible protocol.

    CMC follows the released SYSU protocol and counts each predicted identity
    only once. AP and INP retain all gallery images. For a cam3 infrared query,
    cam2 visible gallery images are excluded before scoring.
    """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    predicted_pids = g_pids[indices]
    matches = (predicted_pids == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []

    for query_index in range(num_q):
        query_pid = q_pids[query_index]
        query_camid = q_camids[query_index]
        order = indices[query_index]

        remove = (query_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        raw_cmc = matches[query_index][keep]
        if not np.any(raw_cmc):
            continue

        ranked_pids = predicted_pids[query_index][keep]
        first_occurrences = np.unique(ranked_pids, return_index=True)[1]
        ranked_pids = ranked_pids[np.sort(first_occurrences)]
        cmc = (ranked_pids == query_pid).astype(np.int32).cumsum()
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        if cmc.shape[0] < max_rank:
            cmc = np.pad(cmc, (0, max_rank - cmc.shape[0]), mode="edge")
        all_cmc.append(cmc)

        raw_cumsum = raw_cmc.cumsum()
        positive_indices = np.where(raw_cmc == 1)[0]
        last_positive = positive_indices[-1]
        all_INP.append(raw_cumsum[last_positive] / (last_positive + 1.0))

        num_relevant = raw_cmc.sum()
        precisions = raw_cumsum / (np.arange(raw_cumsum.shape[0]) + 1.0)
        all_AP.append((precisions * raw_cmc).sum() / num_relevant)

    if not all_cmc:
        raise AssertionError("Error: all query identities do not appear in gallery")

    return (
        np.asarray(all_cmc, dtype=np.float32).mean(axis=0),
        float(np.mean(all_AP)),
        float(np.mean(all_INP)),
    )
