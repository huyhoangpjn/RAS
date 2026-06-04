import torch
import torch.nn as nn
import numpy as np
from mmcv.ops.box_iou_rotated import box_iou_rotated
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

def convert_tensor_bbox_to_3D_box(bboxes, box_type_3d, normalized=True):
    '''
    in:
        bboxes: tensor of shape (N, 10) if normalized or (N,9)
    out:
        bboxes_3d: mmdet3D box
    '''
    if normalized:
        bboxes = denormalize_bbox(bboxes, None)
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5]*0.5
    bboxes = box_type_3d(bboxes, bboxes.size(-1))
    return bboxes

def convert_3Dbox_to_2D(bboxes_3d, lidar2img, h, w):
    '''
    Project all bboxes_3D on views: first project to all views and then filter valid views only
    Currently, implement for bboxes_3d shape (n_box, ...), after extending this to batch manner
    Also, doing simple case: take bbox not polygon for fast computation -> If want more precise -> Refer: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py
    in:
        bboxes_3d: (N, ...)
        lidar2img: (n_cam, 4, 4)
    out:
        bboxes_2d: projected bboxes (n_cam, N, 4)
        depths: (n_cam, N,)
    '''
    corners = bboxes_3d.corners.reshape(-1,8,3)
    centers = bboxes_3d.gravity_center.reshape(-1,3)

    n_boxes = corners.shape[0]
    n_cam = lidar2img.shape[0]

    corners = torch.cat((corners, corners.new_ones((n_boxes, 8, 1))), dim=-1)
    centers = torch.cat((centers, corners.new_ones((n_boxes, 1))), dim=-1)
    corners = corners.view(-1, 4).unsqueeze(0)
    centers = centers.view(-1, 4).unsqueeze(0)

    corners = corners@torch.transpose(lidar2img, -2, -1) # (cam, n_boxes*8, 4)
    corners[..., 2] = torch.clamp(corners[..., 2], min=-1e-5, max=1e5)
    corners[..., 0] = torch.clamp(corners[..., 0]/corners[..., 2], min=0, max=w)
    corners[..., 1] = torch.clamp(corners[..., 1]/corners[..., 2], min=0, max=h)
    corners = corners.view(n_cam, n_boxes, 8, 4)

    centers = centers@torch.transpose(lidar2img, -2, -1)
    centers[..., 2] = torch.clamp(centers[..., 2], min=-1e-5, max=1e5)
    centers = centers.view(n_cam, n_boxes, 4)

    top_corner = corners.min(dim=-2).values[..., :2]
    bot_corner = corners.max(dim=-2).values[..., :2]
    
    bboxes_2d = torch.cat((top_corner, bot_corner), dim=-1) # (cam, n_boxes, 4)
    invalid_mask = ((bboxes_2d[..., 0] == bboxes_2d[..., 2]) |
                    (bboxes_2d[..., 1] == bboxes_2d[..., 3]) |
                    ((bboxes_2d[..., 0] == 0)&(bboxes_2d[..., 1] == 0)&(bboxes_2d[..., 2] == w)&(bboxes_2d[..., 3] == h)))
    bboxes_2d[invalid_mask] = -1
    return bboxes_2d.detach(), centers[..., 2].detach()

def rel_depth_err(depths1, depths2):
    '''
    in:
        depths1: Tensor of shape (N,)
        depths2: Tensor of shape (M,)
        depths2: prior as anchor
    out:
        rel_depth_err: (N,M)
    '''
    depths1 = depths1.unsqueeze(-1)
    depths2 = depths2.unsqueeze(0)
    res = 1 - torch.abs(depths2-depths1)/depths2
    return torch.relu(res)

def rel_depth_err_batched(depths1, depths2):
    '''
    in:
        depths1: Tensor of shape (B, N)
        depths2: Tensor of shape (B, M)
        depths2: prior as anchor (denominator)
    out:
        rel_depth_err: Tensor of shape (B, N, M)
    '''
    depths1 = depths1.unsqueeze(2)  # (B, N, 1)
    depths2 = depths2.unsqueeze(1)  # (B, 1, M)
    res = 1 - torch.abs(depths2 - depths1) / depths2.clamp(min=1e-6)
    return torch.relu(res)

def compute_iou(boxes1, boxes2):
    '''
    in:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)
    out:
        iou: (N,M)
    '''
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2 - inter_area
    return inter_area / union_area

def compute_iou_batched(boxes1, boxes2):
    '''
    in:
        boxes1: Tensor of shape (B, N, 4)
        boxes2: Tensor of shape (B, M, 4)
    out:
        iou: Tensor of shape (B, N, M)
    '''

    inter_x1 = torch.max(boxes1[:, :, None, 0], boxes2[:, None, :, 0])  # (B, N, M)
    inter_y1 = torch.max(boxes1[:, :, None, 1], boxes2[:, None, :, 1])
    inter_x2 = torch.min(boxes1[:, :, None, 2], boxes2[:, None, :, 2])
    inter_y2 = torch.min(boxes1[:, :, None, 3], boxes2[:, None, :, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])  # (B, N)
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])  # (B, M)

    union_area = area1[:, :, None] + area2[:, None, :] - inter_area  # (B, N, M)

    return inter_area / union_area.clamp(min=1e-6)

def compute_giou(boxes1, boxes2):
    '''
    in:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)
    out:
        giou: Tensor of shape (N, M)
    '''
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)
    return giou

def compute_giou_batched(boxes1, boxes2, rescale=True):
    '''
    in:
        boxes1: Tensor of shape (B, N, 4)
        boxes2: Tensor of shape (B, M, 4)
    out:
        giou: Tensor of shape (B, N, M)
    '''
    # Intersection
    inter_x1 = torch.max(boxes1[:, :, None, 0], boxes2[:, None, :, 0])
    inter_y1 = torch.max(boxes1[:, :, None, 1], boxes2[:, None, :, 1])
    inter_x2 = torch.min(boxes1[:, :, None, 2], boxes2[:, None, :, 2])
    inter_y2 = torch.min(boxes1[:, :, None, 3], boxes2[:, None, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Area
    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])  # (B, N)
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])  # (B, M)
    union_area = area1[:, :, None] + area2[:, None, :] - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, :, None, 0], boxes2[:, None, :, 0])
    enc_y1 = torch.min(boxes1[:, :, None, 1], boxes2[:, None, :, 1])
    enc_x2 = torch.max(boxes1[:, :, None, 2], boxes2[:, None, :, 2])
    enc_y2 = torch.max(boxes1[:, :, None, 3], boxes2[:, None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)
    if rescale:
        # Rescale giou [-1, 1] -> [0, 1]
        giou = (giou + 1) / 2
        giou = torch.clamp(giou, min=0, max=1)
    return giou


'''
Notes:
In the priorhead, make a dict to allow to switch between iou and giou
'''

def depth_aware_iou(boxes1, depths1, boxes2, depths2, ioufunc=compute_iou, depth_err_func=rel_depth_err):
    # Assume boxes1 is propagated and boxes2 is prior
    iou = ioufunc(boxes1, boxes2)
    rel_err = depth_err_func(depths1, depths2)
    return iou*rel_err

def loc_exp_similarity(boxes1, boxes2, std=2):
    '''
    Compute location similarity between 2 set of points
    '''
    assert boxes1.size(0) == boxes2.size(0), 'Pred and GT num are not equal'
    distance = torch.norm(boxes1[..., :2] - boxes2[..., :2], dim=-1) # 2 - get only x,y. Could be changed to 3 (also get z)
    return torch.exp(-distance**2/(std**2)).detach()

def get_corresponding_scores(mres, concat_scores_2d, min_conf=0.2):
    '''
    mres: result from P2PAssigner \\
        here prop can have multiple prior matching (on different views) -> greedy: get max matching score and conf
    Return 2 tensors shape of (num_prior+num_propagated, ): 
        mscore: matching score with its assigned prior
        conf2d: corresponding 2D confidence score
    '''
    inds_2d = mres.gt_inds.clone()
    if len(inds_2d.shape) > 1:
        mscore = mres.max_overlaps[:, 0] # 1st index contains max matching score
        valid_mask = inds_2d > 0
        conf2d = mscore.new_full((mscore.size(0),), min_conf)
        if len(concat_scores_2d):        
            conf2d_ = concat_scores_2d[inds_2d-1]
            conf2d_[~valid_mask] = 0
            pos_inds = (inds_2d[:, 0] > 0)
            conf2d = mscore.new_full((mscore.size(0),), min_conf)
            conf2d[pos_inds] = conf2d_[pos_inds].max(dim=-1).values
    else:
        pos_inds = inds_2d > 0
        inds_2d[pos_inds] -= 1 # Assign result gt indexes being +1
        mscore = mres.max_overlaps.clone()
        conf2d = mscore.new_full((mscore.size(0),), min_conf)
        if len(concat_scores_2d):
            conf2d[pos_inds] = concat_scores_2d[inds_2d[pos_inds]]
        
    # Also extend score of prior queries, with mscore = 0
    mscore = torch.cat((torch.zeros_like(concat_scores_2d), mscore))
    conf2d = torch.cat((concat_scores_2d, conf2d))
    return mscore.detach(), conf2d.detach()

def generate_soft_label(cor_scores, hsampling):
    '''
        cor_scores: corresponding scores between propagated and prior queries (mscore, conf2d),
                    this is concatenated version, (mscore=0 for prior)
        hsampling: result of one-to-one sampling
    out:
        sflabels: if temporal and match score > 0 -> 1/2*(delta_s + match_score)*conf2d
                  if prior -> delta_s*conf2d
                  else delta_s*min_conf2d
    '''
    num_bboxes = hsampling.pos_inds.size(0) + hsampling.neg_inds.size(0)
    sflabels = hsampling.pos_bboxes.new_zeros((num_bboxes, ))
    mscore, conf2d = cor_scores # prior+propagated

    pos_inds = hsampling.pos_inds
    delta_s = loc_exp_similarity(hsampling.pos_bboxes, hsampling.pos_gt_bboxes)
    # print(delta_s[37])
    sflabels[pos_inds] = (delta_s + mscore[pos_inds])*conf2d[pos_inds]

    mask_inds = torch.zeros_like(sflabels, dtype=torch.bool)
    mask_inds[pos_inds] = 1
    mask_inds = mask_inds & (mscore > 0)
    sflabels[mask_inds] /= 2

    return sflabels

def depth_aware_score(pred_bboxes_3d, gt_bboxes_3d, gamma=2, rescale_strategy=0, **kwargs):
    '''
    Compute depth aware iou score between predicted and ground truth 3D boxes, based on 2D IoU and depth error
    Args:
        pred_bboxes_3d: denormalized boxes (N, 9)
        gt_bboxes_3d: (M, 9)
        *args: should contain (lidar2img, h, w) in order
        gamma: power to apply to the score
    Returns:
        score_matrix: (N, M)
    '''
    # Unpack required args
 
    lidar2img = kwargs.get('lidar2img', None)
    h = kwargs.get('h', None)
    w = kwargs.get('w', None)
    depth_err_func = kwargs.get('depth_err_func', rel_depth_err_batched)
    ioufunc = kwargs.get('ioufunc', compute_iou_batched)
    assert lidar2img is not None and h is not None and w is not None, "lidar2img, h, w must be provided"

    pred_bboxes_2d, pred_depths = convert_3Dbox_to_2D(pred_bboxes_3d, lidar2img, h, w)  # (n_cam, N, 4), (n_cam, N)
    gt_bboxes_2d, gt_depths = convert_3Dbox_to_2D(gt_bboxes_3d, lidar2img, h, w)        # (n_cam, M, 4), (n_cam, M)

    # Compute invalid masks: (n_cam, N) and (n_cam, M)
    pred_invalid = (pred_bboxes_2d == -1).all(dim=-1)  # (n_cam, N)
    gt_invalid = (gt_bboxes_2d == -1).all(dim=-1)      # (n_cam, M)

    # Compute depth-aware IoU in batch: (n_cam, N, M)
    scores = depth_aware_iou(pred_bboxes_2d, pred_depths, gt_bboxes_2d, gt_depths, ioufunc=ioufunc, depth_err_func=depth_err_func)**gamma

    # Mask out invalid boxes: set score to 0 if either pred or gt is invalid in that view
    pred_invalid = pred_invalid.unsqueeze(2)  # (n_cam, N, 1)
    gt_invalid = gt_invalid.unsqueeze(1)      # (n_cam, 1, M)
    mask = (~pred_invalid) & (~gt_invalid)    # (n_cam, N, M)
    scores = scores * mask.float()

    # Take max over views (n_cam, N, M) -> (N, M)
    scores, _ = scores.max(dim=0)
    # print(scores.shape, scores.flatten().topk(3).values)
    if rescale_strategy == 0:
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
    elif rescale_strategy == 1:
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score * (max_score ** (-gamma))
    return scores.detach()

def compute_ro_iou(pred_bboxes, gt_bboxes):
    '''
    Compute Ro_IoU between predicted and ground truth 3D boxes
    in:
        pred_bboxes: (N, 9) - x,y,z,w,l,h,theta,vx,vy
        gt_bboxes: (M, 9)
    out:
        ro_iou: (N, M) - Ro_IoU score
    '''
    b1 = torch.stack([pred_bboxes[:, 0], pred_bboxes[:, 1], pred_bboxes[:, 3], pred_bboxes[:, 4], pred_bboxes[:, 6]], dim=-1)
    b2 = torch.stack([gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 3], gt_bboxes[:, 4], gt_bboxes[:, 6]], dim=-1)
    return box_iou_rotated(b1, b2)

def compute_ro_diou(pred_bboxes, gt_bboxes, rescale=True):
    '''
    in:
        pred_bboxes and gt_bboxes are denormalized boxes
        pred_bboxes: (N, 9) - x,y,z,w,l,h,theta,vx,vy
        gt_bboxes: (M, 9)
    out:
        ro_gd_iou: (N, M) - Ro_DIOU score
    '''
    b1 = torch.stack([pred_bboxes[:, 0], pred_bboxes[:, 1], pred_bboxes[:, 3], pred_bboxes[:, 4], pred_bboxes[:, 6]], dim=-1)
    b2 = torch.stack([gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 3], gt_bboxes[:, 4], gt_bboxes[:, 6]], dim=-1)
    ro_iou = compute_ro_iou(pred_bboxes, gt_bboxes,)

    c = (torch.cdist(pred_bboxes[:, :2], gt_bboxes[:, :2], p=2)**2).to(pred_bboxes.device)
    N, M = b1.size(0), b2.size(0)
    x1_min = b1[:, 0].unsqueeze(1).expand(N, M).clone()
    y1_min = b1[:, 1].unsqueeze(1).expand(N, M).clone()
    x2_min = b2[:, 0].unsqueeze(0).expand(N, M).clone()
    y2_min = b2[:, 1].unsqueeze(0).expand(N, M).clone()

    w1 = b1[:, 2].unsqueeze(1)
    l1 = b1[:, 3].unsqueeze(1)
    w2 = b2[:, 2].unsqueeze(0)
    l2 = b2[:, 3].unsqueeze(0)

    x1_left = x1_min - w1 / 2
    x1_right = x1_min + w1 / 2
    y1_top = y1_min - l1 / 2
    y1_bottom = y1_min + l1 / 2

    x2_left = x2_min - w2 / 2
    x2_right = x2_min + w2 / 2
    y2_top = y2_min - l2 / 2
    y2_bottom = y2_min + l2 / 2

    min_x = torch.min(x1_left, x2_left)
    max_x = torch.max(x1_right, x2_right)
    min_y = torch.min(y1_top, y2_top)
    max_y = torch.max(y1_bottom, y2_bottom)

    d = (max_x - min_x) ** 2 + (max_y - min_y) ** 2

    ro_diou = ro_iou - (c / d.clamp(min=1e-6))
    if rescale:
        ro_diou = (ro_diou + 1) / 2
        ro_diou = ro_diou.clamp(min=0, max=1)
    return ro_diou 

def compute_ro_diou_pairwise(pred_bboxes, gt_bboxes, rescale=True):
    '''
    Compute Ro_DIOU for pairwise boxes (i.e., pred_bboxes[i] vs gt_bboxes[i])
    in:
        pred_bboxes: (N, 9)
        gt_bboxes: (N, 9)
    out:
        ro_diou: (N,)
    '''
    assert pred_bboxes.shape[0] == gt_bboxes.shape[0], "Number of boxes must match for pairwise computation"
    b1 = torch.stack([pred_bboxes[:, 0], pred_bboxes[:, 1], pred_bboxes[:, 3], pred_bboxes[:, 4], pred_bboxes[:, 6]], dim=-1)
    b2 = torch.stack([gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 3], gt_bboxes[:, 4], gt_bboxes[:, 6]], dim=-1)
    ro_iou = box_iou_rotated(b1, b2, aligned=True)

    c = torch.sum((pred_bboxes[:, :2] - gt_bboxes[:, :2]) ** 2, dim=-1)

    x1_min = b1[:, 0]
    y1_min = b1[:, 1]
    x2_min = b2[:, 0]
    y2_min = b2[:, 1]

    w1 = b1[:, 2]
    l1 = b1[:, 3]
    w2 = b2[:, 2]
    l2 = b2[:, 3]

    x1_left = x1_min - w1 / 2
    x1_right = x1_min + w1 / 2
    y1_top = y1_min - l1 / 2
    y1_bottom = y1_min + l1 / 2

    x2_left = x2_min - w2 / 2
    x2_right = x2_min + w2 / 2
    y2_top = y2_min - l2 / 2
    y2_bottom = y2_min + l2 / 2

    min_x = torch.min(x1_left, x2_left)
    max_x = torch.max(x1_right, x2_right)
    min_y = torch.min(y1_top, y2_top)
    max_y = torch.max(y1_bottom, y2_bottom)

    d = (max_x - min_x) ** 2 + (max_y - min_y) ** 2

    ro_diou = ro_iou - (c / d.clamp(min=1e-6))
    if rescale:
        ro_diou = (ro_diou + 1) / 2
        ro_diou = ro_diou.clamp(min=0, max=1)
    return ro_diou

def inbev_score(pred_bboxes, gt_bboxes, gamma=2, rescale_strategy=0, **kwargs):
    '''
    A wrapper for compute iou scores in BEV
    Compute in BEV score between predicted and ground truth 3D boxes
    in:
        prev_boxes: denormalized boxes (N, 9)
        gt_boxes: (M, 9)
    out:
        score_matrix: (N,M)
    '''
    ioufunc = kwargs.get('ioufunc', compute_ro_diou)
    scores = ioufunc(pred_bboxes, gt_bboxes)**gamma
    if rescale_strategy == 0:
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
    elif rescale_strategy == 1:
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score * (max_score ** (-gamma))
    return scores.detach()

'''
# Pseudo test case for validation
if __name__ == "__main__":
    class DummyBox3D:
        def __init__(self, corners, gravity_center):
            self.corners = corners
            self.gravity_center = gravity_center

    # 2 cameras, 1 GT, 2 predictions (one close, one far)
    n_cam, N, M = 2, 2, 1
    gt_corners = torch.rand(M, 8, 3)
    gt_centers = torch.rand(M, 3)
    # Positive prediction: small noise
    pos_pred_corners = gt_corners + torch.randn_like(gt_corners) * 0.01
    pos_pred_centers = gt_centers + torch.randn_like(gt_centers) * 0.01
    # Negative prediction: larger noise
    neg_pred_corners = gt_corners + torch.randn_like(gt_corners) * 0.2
    neg_pred_centers = gt_centers + torch.randn_like(gt_centers) * 0.2
    pred_corners = torch.cat([pos_pred_corners, neg_pred_corners], dim=0)
    pred_centers = torch.cat([pos_pred_centers, neg_pred_centers], dim=0)

    pred_bboxes_3d = DummyBox3D(pred_corners, pred_centers)
    gt_bboxes_3d = DummyBox3D(gt_corners, gt_centers)
    lidar2img = torch.eye(4).unsqueeze(0).repeat(n_cam, 1, 1)
    h, w = 100, 100
    scores = depth_aware_score(pred_bboxes_3d, gt_bboxes_3d, lidar2img, h, w)
    print("Score matrix shape:", scores.shape)
    print(scores)
'''