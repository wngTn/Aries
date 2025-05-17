import numpy as np


def batch_triangulate(keypoints_, Pall, min_view=2):
    """triangulate the keypoints of whole body

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4) | (nViews, nJoints, 3, 4): projection matrix of each view
        min_view (int, optional): min view for visible points. Defaults to 2.

    Returns:
        keypoints3d: (nJoints, 4)
    """
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    if len(Pall.shape) == 3:
        P0 = Pall[None, :, 0, :]
        P1 = Pall[None, :, 1, :]
        P2 = Pall[None, :, 2, :]
    else:
        P0 = Pall[:, :, 0, :].swapaxes(0, 1)
        P1 = Pall[:, :, 1, :].swapaxes(0, 1)
        P2 = Pall[:, :, 2, :].swapaxes(0, 1)
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d  # * (conf[..., 0].sum(axis=-1)>min_view)
    return result


def project_points(keypoints, RT, einsum=None):
    homo = np.concatenate([keypoints[..., :3], np.ones_like(keypoints[..., :1])], axis=-1)
    if einsum is None:
        if len(homo.shape) == 2 and len(RT.shape) == 3:
            kpts2d = np.einsum("vab,kb->vka", RT, homo)
        elif len(homo.shape) == 2 and len(RT.shape) == 4:
            kpts2d = np.einsum("vkab,kb->vka", RT, homo)
        else:
            import ipdb

            ipdb.set_trace()
    else:
        kpts2d = np.einsum(einsum, RT, homo)
    kpts2d[..., :2] /= kpts2d[..., 2:]
    return kpts2d


def make_Cnk(n, k):
    import itertools

    res = {}
    for n_ in range(3, n + 1):
        n_0 = [i for i in range(n_)]
        for k_ in range(2, k + 1):
            res[(n_, k_)] = list(map(list, itertools.combinations(n_0, k_)))
    return res


MAX_VIEWS = 30
Cnk = make_Cnk(MAX_VIEWS, 3)


def robust_triangulate_point(kpts2d, Pall, dist_max, min_v=3):
    nV = kpts2d.shape[0]
    if len(kpts2d) < min_v:  # 重建失败
        return [], None
    # min_v = max(2, nV//2)
    # 1. choose the combination of min_v
    index_ = Cnk[(len(kpts2d), min(min_v, len(kpts2d)))]
    # 2. proposals: store the reconstruction points of each proposal
    proposals = np.zeros((len(index_), 4))
    weight_self = np.zeros((nV, len(index_)))
    for i, index in enumerate(index_):
        weight_self[index, i] = 100.0
        point = batch_triangulate(kpts2d[index, :], Pall[index], min_view=min_v)
        proposals[i] = point
    # 3. project the proposals to each view
    #    and calculate the reprojection error
    # (nViews, nProposals, 4)
    kpts_repro = project_points(proposals, Pall)
    conf = (proposals[None, :, -1] > 0) * (kpts2d[..., -1] > 0)
    # err: (nViews, nProposals)
    err = np.linalg.norm(kpts_repro[..., :2] - kpts2d[..., :2], axis=-1) * conf
    valid = 1.0 - err / dist_max
    valid[valid < 0] = 0
    # consider the weight of different view
    # TODO:naive weight:
    conf = kpts2d[..., -1]
    weight = conf
    # (valid > 0)*weight_self 一项用于强制要求使用到的两个视角都需要被用到
    # 增加一项使用的视角数的加成
    weight_sum = (weight * valid).sum(axis=0) + ((valid > 0) * weight_self).sum(axis=0) - min_v * 100
    if weight_sum.max() < 0:  # 重建失败
        return [], None
    best = weight_sum.argmax()
    if (err[index_[best], best] > dist_max).any():
        return [], None
    # 对于选出来的proposal，寻找其大于0的其他视角
    point = proposals[best]
    best_add = np.where(valid[:, best])[0].tolist()
    index = list(index_[best])
    best_add.sort(key=lambda x: -weight[x])
    for add in best_add:
        if add in index:
            continue
        index.append(add)
        point = batch_triangulate(kpts2d[index, :], Pall[index], min_view=min_v)
        kpts_repro = project_points(point, Pall[index])
        err = np.linalg.norm(kpts_repro[..., :2] - kpts2d[index, ..., :2], axis=-1)
        if (err > dist_max).any():
            index.remove(add)
            break
    return index, point


def remove_outview(kpts2d, out_view, debug):
    if len(out_view) == 0:
        return False
    outv = out_view[0]
    kpts2d[outv] = 0.0
    return True


def remove_outjoint(kpts2d, Pall, out_joint, dist_max, min_view=3, debug=False):
    if len(out_joint) == 0:
        return False
    for nj in out_joint:
        valid = np.where(kpts2d[:, nj, -1] > 0)[0]
        if len(valid) < min_view:
            # if less than 3 visible view, set these unvisible
            kpts2d[:, nj, -1] = 0
            continue
        if len(valid) > MAX_VIEWS:
            # only select max points
            conf = -kpts2d[:, nj, -1]
            valid = conf.argsort()[:MAX_VIEWS]
        index_j, point = robust_triangulate_point(kpts2d[valid, nj : nj + 1], Pall[valid], dist_max=dist_max, min_v=3)
        index_j = valid[index_j]
        # print('select {} for joint {}'.format(index_j, nj))
        set0 = np.zeros(kpts2d.shape[0])
        set0[index_j] = 1.0
        kpts2d[:, nj, -1] *= set0
    return True


def project_and_distance(kpts3d, RT, kpts2d):
    kpts_proj = project_points(kpts3d, RT)
    # 1. distance between input and projection
    conf = (kpts3d[None, :, -1] > 0) * (kpts2d[:, :, -1] > 0)
    dist = np.linalg.norm(kpts_proj[..., :2] - kpts2d[..., :2], axis=-1) * conf
    return dist, conf


def iterative_triangulate(
    kpts2d,
    RT,
    previous=None,
    min_conf=0.1,
    min_view=3,
    min_joints=3,
    dist_max=0.05,
    dist_vel=0.05,
    thres_outlier_view=0.4,
    thres_outlier_joint=0.4,
    debug=False,
):
    kpts2d = kpts2d.copy()
    conf = kpts2d[..., -1]
    kpts2d[conf < min_conf] = 0.0
    # TODO: consider large motion
    if previous is not None:
        dist, conf = project_and_distance(previous, RT, kpts2d)
        nottrack = (dist > dist_vel) & conf
        if nottrack.sum() > 0:
            kpts2d[nottrack] = 0.0
    while True:
        # 0. triangulate and project
        kpts3d = batch_triangulate(kpts2d, RT, min_view=min_view)
        dist, conf = project_and_distance(kpts3d, RT, kpts2d)
        # 2. find the outlier
        vv, jj = np.where(dist > dist_max)
        if vv.shape[0] < 1:
            break
        ratio_outlier_view = (dist > dist_max).sum(axis=1) / (1e-5 + conf.sum(axis=1))
        ratio_outlier_joint = (dist > dist_max).sum(axis=0) / (1e-5 + conf.sum(axis=0))
        # 3. find the totally wrong detections
        out_view = np.where(ratio_outlier_view > thres_outlier_view)[0]
        out_joint = np.where(ratio_outlier_joint > thres_outlier_joint)[0]
        if len(out_view) > 1:
            dist_view = dist.sum(axis=1) / (1e-5 + conf.sum(axis=1))
            out_view = out_view.tolist()
            out_view.sort(key=lambda x: -dist_view[x])
        if remove_outview(kpts2d, out_view, debug):
            continue
        if remove_outjoint(kpts2d, RT, out_joint, dist_max, debug=debug):
            continue
        kpts2d[vv, jj, -1] = 0.0
    if (kpts3d[..., -1] > 0).sum() < min_joints:
        kpts3d[..., -1] = 0.0
        kpts2d[..., -1] = 0.0
        return kpts3d, kpts2d
    return kpts3d, kpts2d
