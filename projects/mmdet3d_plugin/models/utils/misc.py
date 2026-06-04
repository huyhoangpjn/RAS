import torch
import torch.nn as nn
import numpy as np
from mmdet.core import bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid

def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def apply_ltrb(locations, pred_ltrb): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)


        return pred_boxes    

def apply_center_offset(locations, center_offset): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Get the center (or close) of receptive field in the image of the location (x,y) in feature map i.
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations



def gaussian_2d(shape, sigma=1.0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
        

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta # element wise

        return out


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

def get_yaw_from_egopose(egopose):
    return torch.atan2(egopose[..., 1, 0], egopose[..., 0, 0])

def fix_2D_label_inconsistency(labels_2d):
    '''
    cat_2d = ['car', 
            'truck', 
            'trailer', 
            'bus', 
            'construction_vehicle',
            'bicycle', 
            'motorcycle', 
            'pedestrian', 
            'traffic_cone',
            'barrier']
    cat_3d = ['car',
              'truck',                
              'construction_vehicle', 
              'bus',                   
              'trailer',              
              'barrier',               
              'motorcycle',          
              'bicycle',               
              'pedestrian',            
              'traffic_cone'     
        ]
    '''
    index_map = labels_2d.new_tensor([0, 1, 4, 3, 2, 7, 6, 8, 9, 5, 10]) # 10 -> bg
    return index_map[labels_2d]

def transform_2d_to_3d_pts(centers2d, depths, img2lidars, bbox_nums, eps=1e-5):
    '''
        centers2d, depths: already concatenated (M,2), (M,)
        img2lidars: (n_cam,4,4)
        bbox_nums: number of bboxes in each single image
    '''
    coords = torch.cat([centers2d, depths.unsqueeze(1)], dim=1).to(depths.device)
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1) 
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)
    coords = coords.unsqueeze(-1) 

    # img2lidar array build
    img2lidars = img2lidars.unsqueeze(1) # (n_cam, 1, 4, 4)
    img2lidars_ = torch.cat([img2lidars[kth].repeat(num, 1, 1) for kth, num in enumerate(bbox_nums)], dim=0) 

    coords3d = torch.matmul(img2lidars_, coords).squeeze(-1)[..., :3]
    return coords3d

def ray_eq_generator(gt_bboxes_list, gt_labels_list, k=3, num_pseudo=10, inv_range_scale=False, img_metas=None, depth_scaling=False, depth_sat_min=10.0, depth_sat_max=30.0, **data):
    """
    Generate pseudo ground truth 3D points along rays for equidistant sampling.
    This function takes 3D bounding boxes and generates pseudo ground truth points
    by sampling along rays from camera centers through the projected 2D centers of
    valid 3D boxes. The sampling is done equidistantly around the original depth
    with a range proportional to the box dimensions.
    Args:
        gt_bboxes_list (list): List of ground truth 3D bounding boxes for each batch.
            Each element has shape (M, 9) containing [x, y, z, w, l, h, yaw, vx, vy]
            in denormalized coordinates.
        gt_labels_list (list): List of ground truth labels for each batch.
            Each element has shape (M,) containing class labels.
        k (int, optional): Scaling factor for sampling range based on box dimensions.
            Range = k * (width + length + height)/6. Defaults to 3.
        num_pseudo (int, optional): Number of pseudo points to generate per valid box.
            Defaults to 10.
    Returns:
        tuple: A tuple containing:
            - pseudo_gt_bboxes_list (list): List of pseudo 3D points for each batch.
              Each element has shape (N_valid, num_pseudo, 3) where N_valid is the
              number of valid boxes that project within image boundaries.
            - pseudo_gt_labels_list (list): List of pseudo labels for each batch.
              Each element has shape (N_valid, num_pseudo) with repeated labels.
            - pseudo_corr_idx_list (list): List of correspondence indices for each batch.
              Each element has shape (N_valid, num_pseudo) indicating which original
              box each pseudo point corresponds to.
    Note:
        - Only boxes that project within image boundaries with positive depth are considered valid
        - Pseudo points are generated by sampling depths along the ray from camera center
        - The sampling range is proportional to the sum of box dimensions (w + l + h)
        - If no valid boxes exist for a batch, empty tensors are returned
    """
    pseudo_gt_bboxes_list = []
    pseudo_gt_labels_list = []
    pseudo_corr_idx_list = []
    num_pseudo = num_pseudo if num_pseudo % 2 else num_pseudo + 1  # Ensure odd number for symmetric sampling
    for b in range(len(gt_bboxes_list)):
        gt_bboxes = gt_bboxes_list[b]
        gt_labels = gt_labels_list[b]
        img_metas_b = img_metas[b]
        
        # get lidar2img, padhw, hw
        lidar2img = data['lidar2img'][b] # data # data['lidar2img'][b], data for testing
        pad_h, pad_w, _ = img_metas_b['pad_shape'][0]
        n_cam = lidar2img.shape[0]
        n_boxes = gt_bboxes.shape[0]
        
        centers = torch.cat((gt_bboxes[:, :3], gt_bboxes.new_ones(gt_bboxes.shape[0], 1)), dim=1)
        centers = centers.view(-1, 4).unsqueeze(0)
        centers = centers@torch.transpose(lidar2img, -2, -1)
        z_mask = centers[..., 2:3] > 0
        centers[..., 2] = torch.clamp(centers[..., 2], min=1e-5, max=1e5) # (ud, vd, d, 1)
        centers[..., :2] = centers[..., :2] / centers[..., 2:3]

        centers = centers.view(n_cam, n_boxes, 4)[..., :3]
        z_mask = z_mask.view(n_cam, n_boxes, )
        # Mask indicated truncated centers
        valid_mask = ((centers[...,0]>=0)&(centers[...,0]<pad_w) &
                       (centers[...,1]>=0)&(centers[...,1]<pad_h))

        valid_mask = valid_mask & z_mask
        # Generate pseudo centers for valid boxes in batch manner
        valid_indices = torch.where(valid_mask)  # (cam_indices, box_indices)
        
        if len(valid_indices[0]) > 0:
            # Get valid centers, depths, and boxes
            valid_centers_2d = centers[valid_indices][:, :2]  # (N_valid, 2)
            valid_depths = centers[valid_indices][:, 2]  # (N_valid,)
            valid_box_indices = valid_indices[1]  # (N_valid,)
            valid_cam_indices = valid_indices[0]  # (N_valid,)
            
            # Get dimensions for valid boxes
            valid_boxes_3d = gt_bboxes[valid_box_indices]  # (N_valid, 9)
            w, l, h = valid_boxes_3d[:, 3], valid_boxes_3d[:, 4], valid_boxes_3d[:, 5]
            # Change so that smaller objects have larger range
            if inv_range_scale:
                range_vals = k * 6 / (w + l + h)
            else:
                range_vals = k * (w + l + h)/6  # (N_valid,)

            if depth_scaling:
                # Apply depth scaling based on saturation limits
                range_vals = torch.where(
                    valid_depths < depth_sat_min,
                    0.0,
                    torch.where(
                        valid_depths > depth_sat_max,
                        range_vals,
                        range_vals * (valid_depths - depth_sat_min) / (depth_sat_max - depth_sat_min)
                    )
                )

            # Clamp range values to reasonable bounds
            range_vals = torch.clamp(range_vals, min=1.0, max=6.0)

            # Generate offset grid
            offsets = torch.linspace(-1, 1, num_pseudo, device=gt_bboxes.device)
            
            # Create pseudo centers: (N_valid, num_pseudo, 2)
            pseudo_depths = valid_depths.unsqueeze(1) + offsets.view(1, -1) * range_vals.unsqueeze(1)  # (N_valid, num_pseudo)
            pseudo_centers_2d = valid_centers_2d.unsqueeze(1) + valid_centers_2d.new_zeros(valid_centers_2d.shape[0], num_pseudo, 2)  # (N_valid, num_pseudo, 2)

            # Concatenate pseudo centers and depths
            pseudo_coords = torch.cat([pseudo_centers_2d, pseudo_depths.unsqueeze(-1)], dim=-1)  # (N_valid, num_pseudo, 3)
            pseudo_coords = torch.cat([pseudo_coords, torch.ones_like(pseudo_coords[..., :1])], dim=-1)  # (N_valid, num_pseudo, 4)

            # Convert to homogeneous coordinates and apply depth scaling
            pseudo_coords[..., :2] = pseudo_coords[..., :2] * torch.maximum(pseudo_coords[..., 2:3], torch.ones_like(pseudo_coords[..., 2:3]) * 1e-5)

            # Get corresponding img2lidar matrices for each valid camera
            img2lidars = torch.inverse(lidar2img)  # (n_cam, 4, 4)
            valid_img2lidars = img2lidars[valid_cam_indices]  # (N_valid, 4, 4)

            # Project pseudo centers back to 3D
            pseudo_coords_3d = torch.matmul(valid_img2lidars.unsqueeze(1), pseudo_coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # (N_valid, num_pseudo, 3)
            
            # Create pseudo bboxes with original dimensions and attributes
            valid_boxes_attrs = valid_boxes_3d[:, 3:]  # (N_valid, 6) - w, l, h, yaw, vx, vy
            pseudo_boxes_attrs = valid_boxes_attrs.unsqueeze(1).repeat(1, num_pseudo, 1)  # (N_valid, num_pseudo, 6)
            pseudo_coords_full = torch.cat([pseudo_coords_3d, pseudo_boxes_attrs], dim=-1)  # (N_valid, num_pseudo, 9)
            
            pseudo_gt_bboxes_list.append(pseudo_coords_full)
            # Create pseudo labels
            pseudo_gt_labels = gt_labels[valid_box_indices].unsqueeze(1).repeat(1, num_pseudo)
            pseudo_gt_labels_list.append(pseudo_gt_labels)
            # Get correspondence of original boxes to pseudo boxes
            # pseudo_corr_idx = valid_box_indices.unsqueeze(1).repeat(1, num_pseudo) # (N_valid, num_pseudo)
            pseudo_corr_idx_list.append(valid_box_indices)
        else:
            # If no valid boxes, append empty tensor
            pseudo_gt_bboxes_list.append(gt_bboxes.new_zeros((0, num_pseudo, 9)))
            pseudo_gt_labels_list.append(gt_bboxes.new_zeros((0, num_pseudo), dtype=torch.long))
            pseudo_corr_idx_list.append(gt_bboxes.new_zeros((0, ), dtype=torch.long))
    
    return pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list

@torch.no_grad()
def ray_adaptive_generator(gt_bboxes, gt_labels, o2o_pred_cls=None, num_pseudo=10, k=1, gen_cls_thr=0.3, pad_shape=None, lidar2img=None, depth_scaling=False, depth_sat_min=10.0, depth_sat_max=30.0):
    # For single batch
    # Adaptive range sampling along rays

    num_pseudo = num_pseudo if num_pseudo % 2 else num_pseudo + 1
    n_cam = lidar2img.shape[0]
    pad_h, pad_w, _ = pad_shape[0]
    # Filter to keep only boxes with low confidence
    low_cls_mask = o2o_pred_cls < gen_cls_thr
    n_boxes = low_cls_mask.sum()

    low_cls_gt_bboxes = gt_bboxes[low_cls_mask]
    low_cls_gt_labels = gt_labels[low_cls_mask]

    centers = torch.cat((low_cls_gt_bboxes[:, :3], low_cls_gt_bboxes.new_ones(low_cls_gt_bboxes.shape[0], 1)), dim=1)
    centers = centers.view(-1, 4).unsqueeze(0)
    centers = centers@torch.transpose(lidar2img, -2, -1)
    z_mask = centers[..., 2:3] > 0
    centers[..., 2] = torch.clamp(centers[..., 2], min=1e-5, max=1e5) # (ud, vd, d, 1)
    centers[..., :2] = centers[..., :2] / centers[..., 2:3]

    centers = centers.view(n_cam, n_boxes, 4)[..., :3]
    z_mask = z_mask.view(n_cam, n_boxes, )
    # Mask indicated truncated centers
    valid_mask = ((centers[...,0]>=0)&(centers[...,0]<pad_w) &
                    (centers[...,1]>=0)&(centers[...,1]<pad_h))

    valid_mask = valid_mask & z_mask
    # Generate pseudo centers for valid boxes in batch manner
    valid_indices = torch.where(valid_mask)  # (cam_indices, box_indices)

    pseudo_gt_bboxes = gt_bboxes.new_zeros((0, num_pseudo, 9))
    pseudo_gt_labels = gt_bboxes.new_zeros((0, num_pseudo), dtype=torch.long)
    pseudo_corr_idx = gt_bboxes.new_zeros((0, ), dtype=torch.long)

    if len(valid_indices[0]) > 0:
        # Get valid centers, depths, and boxes
        valid_centers_2d = centers[valid_indices][:, :2]  # (N_valid, 2)
        valid_depths = centers[valid_indices][:, 2]  # (N_valid,)
        valid_box_indices = valid_indices[1]  # (N_valid,)
        valid_cam_indices = valid_indices[0]  # (N_valid,)
        
        # Get dimensions for valid boxes
        valid_boxes_3d = low_cls_gt_bboxes[valid_box_indices]  # (N_valid, 9)

        # Get dimensions for valid boxes
        valid_boxes_3d = gt_bboxes[valid_box_indices]  # (N_valid, 9)
        w, l, h = valid_boxes_3d[:, 3], valid_boxes_3d[:, 4], valid_boxes_3d[:, 5]
        range_vals = k * (w + l + h)/6  # (N_valid,)

        if depth_scaling:
            # Apply depth scaling based on saturation limits
            range_vals = torch.where(
                valid_depths < depth_sat_min,
                0.0,
                torch.where(
                    valid_depths > depth_sat_max,
                    range_vals,
                    range_vals * (valid_depths - depth_sat_min) / (depth_sat_max - depth_sat_min)
                )
            )

        # Apply minimum range based on average object size - size trick
        # size_avg = (w + l + h) / 6
        # min_range = torch.where(size_avg < 0.5, torch.ones_like(range_vals), torch.ones_like(range_vals) * 2)
        # range_vals = torch.maximum(range_vals, min_range)

        # Clamp range values to reasonable bounds
        range_vals = torch.clamp(range_vals, min=1.0, max=6.0)

        # Generate offset grid
        offsets = torch.linspace(-1, 1, num_pseudo, device=gt_bboxes.device)

        
        # Create pseudo centers: (N_valid, num_pseudo, 2)
        pseudo_depths = valid_depths.unsqueeze(1) + offsets.view(1, -1) * range_vals.unsqueeze(1)  # (N_valid, num_pseudo)
        pseudo_centers_2d = valid_centers_2d.unsqueeze(1) + valid_centers_2d.new_zeros(valid_centers_2d.shape[0], num_pseudo, 2)  # (N_valid, num_pseudo, 2)

        # Concatenate pseudo centers and depths
        pseudo_coords = torch.cat([pseudo_centers_2d, pseudo_depths.unsqueeze(-1)], dim=-1)  # (N_valid, num_pseudo, 3)
        pseudo_coords = torch.cat([pseudo_coords, torch.ones_like(pseudo_coords[..., :1])], dim=-1)  # (N_valid, num_pseudo, 4)

        # Convert to homogeneous coordinates and apply depth scaling
        pseudo_coords[..., :2] = pseudo_coords[..., :2] * torch.maximum(pseudo_coords[..., 2:3], torch.ones_like(pseudo_coords[..., 2:3]) * 1e-5)

        # Get corresponding img2lidar matrices for each valid camera
        img2lidars = torch.inverse(lidar2img)  # (n_cam, 4, 4)
        valid_img2lidars = img2lidars[valid_cam_indices]  # (N_valid, 4, 4)

        # Project pseudo centers back to 3D
        pseudo_coords_3d = torch.matmul(valid_img2lidars.unsqueeze(1), pseudo_coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # (N_valid, num_pseudo, 3)
        
        # Create pseudo bboxes with original dimensions and attributes
        valid_boxes_attrs = valid_boxes_3d[:, 3:]  # (N_valid, 6) - w, l, h, yaw, vx, vy
        pseudo_boxes_attrs = valid_boxes_attrs.unsqueeze(1).repeat(1, num_pseudo, 1)  # (N_valid, num_pseudo, 6)
        pseudo_gt_bboxes = torch.cat([pseudo_coords_3d, pseudo_boxes_attrs], dim=-1)  # (N_valid, num_pseudo, 9)
        
        # Create pseudo labels
        pseudo_gt_labels = low_cls_gt_labels[valid_box_indices].unsqueeze(1).repeat(1, num_pseudo)

        # Get correspondence of original boxes to pseudo boxes
        pseudo_corr_idx = torch.where(low_cls_mask)[0][valid_box_indices]

    return pseudo_gt_bboxes, pseudo_gt_labels, pseudo_corr_idx