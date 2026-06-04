# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
#  Modified by Huy Hoang Duong
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmcv.ops import RoIAlign
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, \
                                                    SELayer_Linear, get_yaw_from_egopose, fix_2D_label_inconsistency, transform_2d_to_3d_pts
from projects.mmdet3d_plugin.models.utils.score_functions import get_corresponding_scores, generate_soft_label, \
                                                    convert_3Dbox_to_2D, convert_tensor_bbox_to_3D_box, \
                                                    depth_aware_iou, loc_exp_similarity, \
                                                    depth_aware_score, compute_iou_batched, \
                                                    compute_giou_batched, compute_ro_diou, inbev_score
@HEADS.register_module()
class PriorHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stride=16,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 noise_label = 0.2,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 use_2d_prior=False,
                 depthnet_config={},
                 is_obj_depth=False,
                 use_prior_class_emb=False,
                 use_prior_gt=False, # in this method, using prior gt in EARLY STAGE acts as denoising -> faster convergence
                 add_prior_noise=False,
                 # gt_prior_trans=0.0,
                 use_soft_label=False,
                 p2p_assigner=dict(type='P2PAssigner', alpha=0.5, beta=0.1),
                 use_ori_comp=False,
                 depth_gamma=2,
                 min_conf=0.2,
                 use_soft_label_v1=False,
                 scoring_method='depth_aware_iou',
                 cost_ioufunc='compute_giou_batched',
                 loss_ioufunc='compute_iou_batched',
                 pos_score_for_cost_power=1,
                 pos_score_for_loss_power=1,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is PriorHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        # Config for soft label training
        self.use_soft_label_v1 = use_soft_label_v1
        self.pos_score_for_cost_power = pos_score_for_cost_power
        self.pos_score_for_loss_power = pos_score_for_loss_power
        if self.use_soft_label_v1:
            scoring_methods = {
                'depth_aware_score': depth_aware_score,
                'inbev_score': inbev_score,
            }
            self.scoring_method = scoring_methods.get(scoring_method, depth_aware_score)

            ioufunc = {
                'compute_giou_batched': compute_giou_batched,
                'compute_iou_batched': compute_iou_batched,
                'compute_ro_diou': compute_ro_diou
            }

            self.cost_ioufunc = ioufunc.get(cost_ioufunc, compute_giou_batched)
            self.loss_ioufunc = ioufunc.get(loss_ioufunc, compute_iou_batched)

        self.use_soft_label = use_soft_label
        self.use_ori_comp = use_ori_comp
        self.depth_gamma = depth_gamma
        self.min_conf = min_conf
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            if self.use_soft_label:
                # assert self.use_soft_label and loss_cls.get('type', None) != 'QualityFocalLoss', 'Currently not supported'
                self.p2p_assigner = build_assigner(p2p_assigner)


        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals # Keep topk_proposal each frame, topk_proposal != topk 2d feat
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.LID = LID
        self.depth_start = depth_start
        self.stride=stride

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.label_noise_scale = noise_label
        self.dn_weight = dn_weight
        self.split = split 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear

        self.use_2d_prior = use_2d_prior
        self.depthnet_config = depthnet_config
        self.is_obj_depth = is_obj_depth
        self.use_prior_class_emb = use_prior_class_emb
        self.use_prior_gt = use_prior_gt
        self.add_prior_noise = add_prior_noise
        #self.gt_prior_trans = gt_prior_trans

        super(PriorHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # when construct depth, they already set the max depth to be position_range(3) -> reason why they dont apply filter to filter points that outside the config range 
        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        assert not (not self.use_2d_prior and self.use_prior_class_emb), 'Can not use class prior when not using 2D detector'
        assert not (not self.use_2d_prior and self.use_prior_gt), 'Currently not supported'

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )

        self.memory_embed = nn.Sequential(
                nn.Linear(self.in_channels, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        # can be replaced with MLN
        # self.featurized_pe = SELayer_Linear(self.embed_dims)
        
        if not self.use_2d_prior: 
            self.reference_points = nn.Embedding(self.num_query, 3) # Initialize anchors 
        else:
            if not self.is_obj_depth:
                self.depth_roi = RoIAlign(
                    output_size=(3,3),
                    spatial_scale=1/self.depthnet_config['stride']
                )
            if self.use_prior_class_emb:
                # Class embedding for prior queries
                self.label_enc = nn.Embedding(self.num_classes+1, self.embed_dims, padding_idx=self.num_classes) # last is for the label of padded queries

        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        # 128*3 (emb of x,y,z, each 128) -> 256
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(14, use_ln=False)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        if not self.use_2d_prior:
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        # To store bbox of propagated queries in previous frame
        if self.training and self.use_soft_label:
            self.memory_bbox = None
            self.memory_label = None

    def pre_update_memory(self, data):
        '''
            Currently, only apply position compensation for ego pose,
            not taking into account ego motion (velocity*t of moving objects)
        '''
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
            if self.training and self.use_soft_label:
                self.memory_bbox = x.new_zeros(B, self.num_propagated, 9) # Denormalize version of bbox for hard matching on perspective views
                self.memory_label = x.new_zeros((B, self.num_propagated,))
        else:
            if self.training and self.use_soft_label:
                #  Compensate velo -> To global -> To ego at current frame -> Orientation compensation (optional)
                self.memory_bbox[..., :2] += self.memory_bbox[..., -2:]*((self.memory_timestamp[:, 0]+data['timestamp'].unsqueeze(-1))).unsqueeze(-1)
                self.memory_bbox[..., :3] = transform_reference_points(self.memory_bbox[..., :3], data['ego_pose_inv']@self.memory_egopose[:, 0, :, :])
                if self.use_ori_comp:
                    prev_yaw_ego = get_yaw_from_egopose(self.memory_egopose[:, 0, :, :]) # B,4,4 -> B,
                    curr_yaw_ego = get_yaw_from_egopose(data['ego_pose'])
                    self.memory_bbox[..., 6] -= (curr_yaw_ego - prev_yaw_ego).unsqueeze(-1)
                self.memory_bbox = memory_refresh(self.memory_bbox, x)
                self.memory_label = memory_refresh(self.memory_label, x)
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)

        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)
            if self.training and self.use_soft_label:
                self.memory_bbox[..., :3] += (1 - x).view(B, 1, 1) * pseudo_reference_points

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, valid_idx=None):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1] # actually, the box has 3d pos, 3 dim, 2 ori, 2 velo, so this one is good
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            topk = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1)
            rec_score = topk.values[..., 0:1]
            rec_labels = topk.indices[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            rec_bboxes = all_bbox_preds[:, :, mask_dict['pad_size']:, :][-1]
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            topk = all_cls_scores[-1].sigmoid().topk(1, dim=-1)
            rec_score = topk.values[..., 0:1]
            rec_labels = topk.indices[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            rec_bboxes = all_bbox_preds[-1]
        # Padded query score to 0
        rec_score_valid = torch.zeros_like(rec_score)
        rec_score_valid[valid_idx] = rec_score[valid_idx]
        rec_score_valid[:, -self.num_propagated:] = rec_score[:, -self.num_propagated:]

        # topk proposals
        _, topk_indexes = torch.topk(rec_score_valid, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

        if self.training and self.use_soft_label:
            # Pick topk pred in current frame -> denormalize
            self.memory_bbox = topk_gather(rec_bboxes, topk_indexes).detach()
            self.memory_bbox = denormalize_bbox(self.memory_bbox, pc_range=None)
            self.memory_label = topk_gather(rec_labels, topk_indexes).squeeze(-1)

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        # lidar2img = intrinsic@extrinsic  
        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # normalize the 3d coordinates 
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        # We choose two points on the camera ray and use it as the representation of the intrinsic and extrinsic. (Focal PETR)
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords_position_embeding, cone

    def temporal_alignment(self, query_pos, tgt, reference_points):
        # tgt: context of queries
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1) # v=0, t=0
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            # like the functions mentioned in the paper 
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float() 
            # Ego pose: 3x4, 3x3 rotation + 3 translation, augment 1 dim to be square matrix
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion) #nerf_pos_encoding: for higher dim, in this case (2+1+3*4)=15 --> 12*15 = 180 (6 func)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            # Basically, self attention: Current + Propagated attention to Past queries + itself
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose
    
    def prepare_for_dn(self, batch_size, reference_points, img_metas, valid_idx=None, gt_conf3d=None):
        # Denoising
        # In pure streampetr, those with highest pos deviation consider as negative, default 25% is neg
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_labels_expand = known_labels.clone()
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.use_soft_label:
                gt_conf3d = torch.cat([t for t in gt_conf3d])
                known_soft_scores = gt_conf3d.repeat(self.scalar, 1).view(-1)

            if self.label_noise_scale > 0:
                p = torch.rand_like(known_labels_expand.float())
                chosen_indice = torch.nonzero(p < (self.label_noise_scale)).view(-1)
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
                known_labels_expand.scatter_(0, chosen_indice, new_label)

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0 
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale

                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes # Contrastive: 25% is negative (cls to background, all other logits -> 0) 

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device) # (denoise_pad_size,3)

            if not self.use_2d_prior:
                padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
                dn_labels = None
            else:
                padding_bbox = padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1)
                padded_reference_points = torch.cat((padding_bbox,reference_points), dim=1) # (B, noise_w_pad + valid + pad, 3)
                # update valid index
                valid_idx = (valid_idx[0], (valid_idx[1]+pad_size).long())
                dn_labels = torch.full((reference_points.size(0), pad_size+reference_points.size(1)), self.num_classes).to(reference_points.device)
                
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)
                if self.use_2d_prior:
                    dn_labels[(known_bid.long(), map_known_indice)] = known_labels_expand.to(reference_points.device)
            tgt_size = padded_reference_points.size(1)
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling
            query_size = tgt_size + self.num_propagated
            tgt_size = tgt_size + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            if self.use_soft_label:
                known_labels = (known_labels, known_soft_scores.to(known_labels.device))

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            if self.use_2d_prior:
                padded_reference_points = reference_points
            else:
                padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None
            dn_labels = None
        return padded_reference_points, attn_mask, mask_dict, valid_idx, dn_labels

    def get_hard_score(self, propagated_queries, propagated_labels, bboxes_2d, labels_2d, depths_2d, lidar2img, h, w):
        '''
        Matching propagated and prior queries on each view based on IoU (not class-agnostic)
        in:
            propagated_queries: denormalized 3D boxes (N, 9)
            propagated_labels: (N, )
            bboxes_2d: bboxes on each view, list (n_cam, Mi, 4)
            labels_2d, depths_2d: list on each view (n_cam, Mi, )
        out:
            hard_scores: (N, M), with M = sum(Mi) over n_cam views
            class_masks: boolean mask indicating if the boxes belong to same class (N, M)
            box_ids: list of length n_cam, each entry is a range of indices of 2D boxes when concatenating them (will be re-used in soft matching)
        '''
        num_propagated = len(propagated_queries)
        num_bboxes = [len(bbox) for bbox in bboxes_2d]
        n_cam = len(num_bboxes)
        prefix_ids = [0]*n_cam
        for i in range(1, n_cam): prefix_ids[i] = prefix_ids[i-1]+num_bboxes[i-1]
        box_inds = [range(prefix_ids[i], prefix_ids[i]+num_bboxes[i]) for i in range(n_cam)]
        total_bboxes = sum(num_bboxes)
        proj_bboxes, proj_depths = convert_3Dbox_to_2D(propagated_queries, lidar2img, h=h, w=w) # (n_cam, N, 4), (n_cam, N)
        proj_bboxes = proj_bboxes.to(bboxes_2d[0].device)
        proj_depths = proj_depths.to(bboxes_2d[0].device)
        hard_scores = proj_bboxes.new_zeros((num_propagated, total_bboxes))

        for i in range(n_cam):
            # print(proj_bboxes[i].shape)
            # print(bboxes_2d[i].shape)
            if bboxes_2d[i].numel():
                hard_scores[:, box_inds[i]] = depth_aware_iou(
                    proj_bboxes[i],
                    proj_depths[i],
                    bboxes_2d[i],
                    self._convert_bin_to_float_depth(depths_2d[i]) \
                        if not (self.use_prior_gt and self.training) else depths_2d[i],
                    self.depth_gamma
                )
        
        # Keep score of same class boxes
        labels_2d = fix_2D_label_inconsistency(torch.cat(labels_2d)) # (sum(Mi),)
        labels_2d = labels_2d[None, :] # (1, sum(Mi))
        propagated_labels = propagated_labels[:, None] # (N, 1)
        class_masks = (propagated_labels.to(labels_2d)==labels_2d)
        # print(hard_scores.shape)
        # print(class_masks.shape)
        hard_scores[~class_masks] = 0

        return hard_scores, class_masks, box_inds

    def get_highest_conf_for_gt_3d_single(self, unproj_centers2d, conf2d, centers3d, eps=1e-4):
        '''
        Simple trick to get correspondence between 2d and 3d gt is by distance computation
        in:
            unproj_centers2d: concatenated center gt in views of a frame (M, 3)
            centers3d: 
            conf2d: (M, )
        out:
            cor_matrix: (N, M), N: 3D labels, M = sum(Mi) 2D labels over views
        '''
        n_3d = centers3d.size(0)
        dist = (centers3d.unsqueeze(1) - unproj_centers2d.unsqueeze(0))**2
        dist = dist.sum(dim=-1).to(conf2d)
        row_idx, col_idx = (dist<eps).nonzero(as_tuple=True)
        conf3d = conf2d.new_zeros((n_3d, ))
        conf3d = conf3d.scatter_reduce(0, row_idx, conf2d[col_idx], reduce='amax', include_self=False)
        conf3d[conf3d<self.min_conf] = self.min_conf
        return conf3d

    def forward(self, img_metas, out_roi=None, **data):
        """Forward function.
        Self attention + Cross attention with multi level features
        """
        # zero init the memory bank
        self.pre_update_memory(data)

        mlvl_feats = data['multi_lv_feats']

        B, N, _, _, _ = mlvl_feats[0].shape

        valid_idx=None
        device = mlvl_feats[0].device
        if self.use_2d_prior:
            if self.use_prior_gt and self.training:
                # Use gt in early stage to encourage stable training
                if self.is_obj_depth:
                    depth_preds = [depths.to(device, non_blocking=True) for img_meta in img_metas for depths in img_meta['depths']._data]
                else:
                    gt_ins_depth = [img_meta['ins_depthmap'] for img_meta in img_metas]
                    gt_depths = gt_ins_depth[0].unsqueeze(0).to(device, non_blocking=True) if len(gt_ins_depth) == 1 else torch.stack(gt_ins_depth).to(device, non_blocking=True)
                    depth_preds = gt_depths.flatten(0, 1).unsqueeze(1) # (BN, 1, H, W)   
                bbox_list = [bboxes.to(device, non_blocking=True) for img_meta in img_metas for bboxes in img_meta['gt_bboxes']._data]
                center2d_list = [centers2d.to(device, non_blocking=True) for img_meta in img_metas for centers2d in img_meta['centers2d']._data]
                label_list = [labels.to(device, non_blocking=True) for img_meta in img_metas for labels in img_meta['gt_labels']._data]
            else:
                depth_preds = out_roi['depth_list']
                bbox_list = out_roi['bbox_list'] # list of B*N
                center2d_list = out_roi['center2d_list']
                label_list = out_roi['label_list']

            if self.training and self.use_soft_label:
                score_list = out_roi['gt_conf2d'] if self.use_prior_gt else out_roi['bbox2d_scores']
                self.memory_bbox[..., 2] = self.memory_bbox[..., 2] - self.memory_bbox[..., 5] * 0.5 # Prepare to convert to mmdet Lidar3D box
                cor_scores = []

                gt_conf3d = []
                gt_conf2d = out_roi['gt_conf2d']
                gt_centers2d = [centers2d.to(device) for img_meta in img_metas for centers2d in img_meta['centers2d']._data]
                gt_depths = [depths.to(device) for img_meta in img_metas for depths in img_meta['depths']._data]
                gt_centers3d = [img_meta['gt_bboxes_3d']._data.gravity_center.to(device) for img_meta in img_metas]
                for b in range(B):
                    prop_bbox = img_metas[b]['box_type_3d'](self.memory_bbox[b], self.memory_bbox[b].size(-1))
                    prop_label = self.memory_label[b]
                    lidar2img = data['lidar2img'][b].to(device)
                    h, w, _ = img_metas[b]['img_shape'][0] # All imgs have same shape
                    bboxes_2d = bbox_list[b*N:(b+1)*N]
                    labels_2d = label_list[b*N:(b+1)*N]
                    if not self.is_obj_depth: raise NotImplementedError
                    depths_2d = depth_preds[b*N:(b+1)*N]
                    scores_2d = score_list[b*N:(b+1)*N]
                    score_matrix, class_masks, box_inds = self.get_hard_score(prop_bbox, prop_label, bboxes_2d, labels_2d, depths_2d, lidar2img, h, w)
                    concat_labels_2d = torch.cat(labels_2d)
                    mres = self.p2p_assigner.assign(score_matrix, concat_labels_2d, box_inds)
                    concat_scores_2d = torch.cat(scores_2d)
                    cor_scores.append(get_corresponding_scores(mres, concat_scores_2d)) # Eventually, cor_scores: list of B, each shape (num_propagated, num_prior)
                    # Get 2d confidence scores to make soft label for denoising process
                    bbox_nums = [len(centers) for centers in gt_centers2d[b*N:(b+1)*N]]
                    gt_centers2d_b = torch.cat(gt_centers2d[b*N:(b+1)*N])
                    gt_depths_b = torch.cat(gt_depths[b*N:(b+1)*N])
                    gt_conf2d_b = torch.cat(gt_conf2d[b*N:(b+1)*N])
                    gt_centers3d_b = gt_centers3d[b]
                    if len(gt_centers2d_b):
                        unproj_centers2d = transform_2d_to_3d_pts(gt_centers2d_b, gt_depths_b, lidar2img.inverse(), bbox_nums)
                        gt_conf3d.append(self.get_highest_conf_for_gt_3d_single(unproj_centers2d, gt_conf2d_b, gt_centers3d_b))
                    else: gt_conf3d.append(gt_conf2d_b.new_full((gt_centers3d_b.size(0),), self.min_conf))

            reference_points, labels, valid_idx = self.build_prior_queries(bbox_list, center2d_list, data, (B,N), depth_preds, label_list)
            labels = fix_2D_label_inconsistency(labels)
            if self.use_prior_gt and self.add_prior_noise:
                raise NotImplementedError
        else:
            reference_points = self.reference_points.weight

        dtype = reference_points.dtype
        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]
        # focal length + extrinsic -> a way to encode view (no idea why no use of principal points)
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1) # BxN, 1, 14
        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            _, _, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten.append(mlvl_feat.to(dtype))
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=mlvl_feats[0].device)
        # Since the features are flatten, this will tell which is the first index of each level
        level_start_index = torch.cat((spatial_flatten.new_zeros((1, )), spatial_flatten.prod(1).cumsum(0)[:-1]))

        if not self.use_soft_label or not self.training:
            gt_conf3d=None
        reference_points, attn_mask, mask_dict, valid_idx_with_dn, dn_labels = self.prepare_for_dn(B, reference_points, img_metas, valid_idx, gt_conf3d)      
        if dn_labels is not None:
            dn_labels[:, mask_dict['pad_size']:] = labels
            labels = dn_labels # (B, dn+prior+pad)
        # Positional encoding (sin,cos PE)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        # StreamPetr not leverage 2d context feats -> init by zeros
        tgt = torch.zeros_like(query_pos)

        if self.use_prior_class_emb:
            tgt = self.label_enc(labels)
            # print(tgt.requires_grad)
        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)

        # reference_points in case training: (B, dn+prior+propagated, 3) with prior = valid + padded queries
        # similarly for query_pos, tgt
        # for temporal: fix shape (B, num_queries_in_memory - num_propagated) -> stack with queries of current frame in self attention

        # Get original query_positions
        ori_query_pos = reference_points.clone()

        key_padding_mask = torch.ones((B, query_pos.size(1)+temp_pos.size(1)), device=device, dtype=torch.bool) # For self attention
        if self.training and self.with_dn and self.use_2d_prior:
            key_padding_mask[:, :mask_dict['pad_size']] = 0
            key_padding_mask[valid_idx_with_dn] = 0
            key_padding_mask[:, query_pos.size(1)-self.num_propagated:] = 0
        else:
            if valid_idx is not None:
                # In case not use dn in training
                key_padding_mask[valid_idx] = 0
                key_padding_mask[:, query_pos.size(1)-self.num_propagated:] = 0
            else:    
                key_padding_mask = None
        outs_dec = self.transformer(tgt, query_pos, feat_flatten, spatial_flatten, level_start_index, temp_memory, 
                                     temp_pos, attn_mask, reference_points, self.pc_range, data, img_metas, key_padding_mask)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid() # sigmoid helps the pos stay in range 

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes) # pad queries -> score = 0
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, valid_idx)
    
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict':mask_dict,
                'valid_idx': valid_idx,
            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict':None,
                'valid_idx': valid_idx,
                'ori_query_pos': ori_query_pos,
            }
        if self.training and self.use_soft_label:
            outs.update(cor_scores=cor_scores)
        if self.training and self.use_soft_label_v1:
            soft_label_args = []
            for b in range(B):
                args = {}
                args['box_type_3d'] = img_metas[b]['box_type_3d']
                args['lidar2img'] = data['lidar2img'][b]
                args['h'] = img_metas[b]['img_shape'][0][0]
                args['w'] = img_metas[b]['img_shape'][0][1]
                soft_label_args.append(args)
            outs.update(soft_label_args=soft_label_args)
        return outs
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None,
                           cor_score=None,
                           soft_label_args=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)

        if self.use_soft_label_v1:
            if gt_bboxes.numel():
                # box tensor to box type
                if self.scoring_method.__name__ == 'depth_aware_score':
                    gt_bboxes_3d = convert_tensor_bbox_to_3D_box(gt_bboxes.clone(), soft_label_args['box_type_3d'], normalized=False)
                    bbox_pred_3d = convert_tensor_bbox_to_3D_box(bbox_pred.detach(), soft_label_args['box_type_3d']) # Detach to avoid GPU memory leak
                else:
                    gt_bboxes_3d = gt_bboxes.clone()
                    bbox_pred_3d = denormalize_bbox(bbox_pred.detach(), None)
                # calculate positional scores
                pos_scores_cost = self.scoring_method(
                    bbox_pred_3d, gt_bboxes_3d, lidar2img=soft_label_args['lidar2img'],
                    h=soft_label_args['h'], w=soft_label_args['w'],
                    gamma=self.pos_score_for_cost_power, ioufunc=self.cost_ioufunc, rescale_strategy=2,)
                
                pos_scores_loss = self.scoring_method(
                    bbox_pred_3d, gt_bboxes_3d, lidar2img=soft_label_args['lidar2img'],
                    h=soft_label_args['h'], w=soft_label_args['w'],
                    gamma=self.pos_score_for_loss_power, ioufunc=self.loss_ioufunc, rescale_strategy=0)
                
                # del gt_bboxes_3d, bbox_pred_3d
                assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                    gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo, pos_scores_cost)
                del gt_bboxes_3d, bbox_pred_3d
            else:
                pos_scores_cost = gt_bboxes.new_empty((num_bboxes, 0))
                # pos_scores_loss = bbox_pred.new_empty((num_bboxes, 0))
                assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                    gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo, pos_scores_cost)
        else:
            assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                    gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
    
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # soft label
        if self.use_soft_label:
            soft_scores = generate_soft_label(cor_score, sampling_result).to(gt_bboxes.device)

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long) # set to last index (bg), the purpose is when converting to 1-hot, these negave have target zeros(size=num_class)
                                                      # refer: https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/losses/focal_loss.html - target = target[:, :num_classes]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size] # Negative candidates will be forced to 0
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0 # Can also make this adaptive? For e.g, soft label low -> hard -> more weight?
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        if self.use_soft_label_v1:
            soft_scores = gt_bboxes.new_zeros((num_bboxes, ))
            if sampling_result.num_gts > 0:
                soft_scores[pos_inds] = pos_scores_loss[pos_inds, sampling_result.pos_assigned_gt_inds]
            # del pos_scores_cost, pos_scores_loss

        if self.use_soft_label or self.use_soft_label_v1:
            assert labels.shape == soft_scores.shape, "Inconsistent number of labels and soft scores"
            labels = (labels, soft_scores) # Follow QFL convention
        
        del assign_result, sampling_result

        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    cor_scores=None,
                    soft_label_args=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        if cor_scores is None:
            cor_scores = [None for _ in range(num_imgs)]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, cor_scores, soft_label_args)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    valid_idx=None,
                    cor_scores=None,
                    soft_label_args=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0) # here is batch_size
        num_prior = cls_scores.size(1) - self.num_propagated
        if valid_idx is not None:
            valid_idx_tensor = torch.stack(valid_idx, dim=-1)
            cls_scores_list = []
            bbox_preds_list = []
            for batch_idx in range(num_imgs):
                valid_idx = valid_idx_tensor[valid_idx_tensor[:,0]==batch_idx]
                # prior + propagated queries
                valid_idx = (torch.cat((valid_idx[:,0], torch.full((self.num_propagated, ), batch_idx, device=cls_scores.device))), 
                             torch.cat((valid_idx[:,1], torch.tensor(range(num_prior, cls_scores.size(1)), device=cls_scores.device))))
                cls_scores_list.append(cls_scores[valid_idx])
                bbox_preds_list.append(bbox_preds[valid_idx])
        else:
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        if cor_scores is None:
            cor_scores = [None for _ in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list, cor_scores, soft_label_args)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
    
        # labels = torch.cat(labels_list, 0)
        # labels_list: [(labels, soft_scores)] -> QFL
        if self.use_soft_label or self.use_soft_label_v1:
            labels = (torch.cat([l[0] for l in labels_list], 0), torch.cat([l[1] for l in labels_list], 0))
        else:
            labels = torch.cat(labels_list, 0)

        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = torch.cat(cls_scores_list, 0) #cls_scores.reshape(-1, self.cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor) # Instead of average with equal weight, this will make pos more important while neg one has less impact on the overall loss

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = torch.cat(bbox_preds_list, 0) # bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None,
                    soft_label_args=None
                    ):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        if self.use_soft_label:
            known_labels, known_soft_scores = known_labels
            # update soft score with distance
            known_soft_scores = loc_exp_similarity(bbox_preds, known_bboxs) * known_soft_scores
            known_labels = (known_labels.long(), known_soft_scores)

        if self.use_soft_label_v1:

            # Convert bbox_preds to 3D box
            # Conver known bboxs to 3D box
            # Compute depth aware score - however, this is pairwise, no need to expand -> create another function for this
            # Store it in known_soft_scores

            # First, consider denoising is consistent process -> set soft label = 1
            known_labels = (known_labels.long(), known_labels.new_ones(known_labels.shape, dtype=torch.float32))

        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels) if not self.use_soft_label and not self.use_soft_label_v1 else torch.ones_like(known_labels[0])
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        valid_idx = preds_dicts['valid_idx']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        valid_idx_list = [valid_idx for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        if self.use_soft_label:
            cor_scores = preds_dicts['cor_scores'] # list of B
            all_cor_scores = [cor_scores for _ in range(num_dec_layers)]
        else: 
            all_cor_scores = [None for _ in range(num_dec_layers)]
        
        if self.use_soft_label_v1:
            all_soft_label_args = [preds_dicts['soft_label_args'] for _ in range(num_dec_layers)]
        else: 
            all_soft_label_args = [None for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list,valid_idx_list, all_cor_scores, all_soft_label_args)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list, all_soft_label_args)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:
            # dn also go through Hungarian matching
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions. 
        Output from the model has gravity center. This func convert gravity center to ground center to comply with mmdet LidarBbox standard
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5 # Here
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

    @torch.no_grad()
    def build_prior_queries(self, bbox_list, center2d_list, data, bn, depth_preds=None, label_list=None):
        '''Build prior reference points (also constraints but later), support multi-batch setting
        Args:
            bbox_list: have to be detached from grad
            center2d_list
            data: metadata
            depth_preds: depth map, only used if is_obj_depth=False
            depth_list: corresponding depth for each bbox, used if is_obj_depth
        Out:
            reference_points: (B, max_num_queries_in_that_batch, 3)
            labels: (B, max_num_queries_in_that_batch)
            valid_idx: tuple of (batch_id, valid_query_id)
        '''
        B,N = bn
        if not self.training and B > 1:
            raise NotImplementedError('Inference with multi batch setting not supported')

        # Collect number of bboxes each image for convenience to reconstruct the batch
        bbox_nums = [len(bbox) for bbox in bbox_list] # B*N entries
        num_box_per_frame = [sum(bbox_nums[N*i:N*(i+1)]) for i in range(B)] # per B
        max_box_per_batch = max(num_box_per_frame)
        if max_box_per_batch:
            reference_points = torch.zeros((B, max_box_per_batch, 3), device=bbox_list[0].device)
            labels = torch.full((B, max_box_per_batch), self.num_classes, device=bbox_list[0].device)
            batch_idx = torch.cat([torch.full((t, ), i) for i, t in enumerate(num_box_per_frame)]).to(bbox_list[0].device)
            valid_idx = torch.cat([torch.tensor(range(num)) for num in num_box_per_frame]).to(bbox_list[0].device)
            valid_idx = (batch_idx.long(), valid_idx.long())
        else:
            # No box in batch -> Fake pts
            reference_points = torch.zeros((B, 1, 3), device=self.pc_range.device)
            labels = torch.full((B, 1), self.num_classes, device=self.pc_range.device)
            num_box_per_frame = [1 for _ in range(B)]
            batch_idx = torch.cat([torch.full((t, ), i) for i, t in enumerate(num_box_per_frame)]).to(self.pc_range.device)
            valid_idx = torch.cat([torch.tensor(range(num)) for num in num_box_per_frame]).to(self.pc_range.device)
            valid_idx = (batch_idx.long(), valid_idx.long())

        # Group all bboxes into a tensor
        concat_reference_points, concat_labels = self._single_batch_build_prior_query(bbox_list, center2d_list, data, (1, B*N), depth_preds, label_list, N)
        # Split back to batch
        start_idx = 0
        for ith, bbox_num in enumerate(num_box_per_frame):
            end_idx = start_idx + bbox_num
            reference_points[ith, :bbox_num] = concat_reference_points[0, start_idx:end_idx]
            labels[ith, :bbox_num] = concat_labels[0, start_idx:end_idx]
            start_idx = end_idx
        
        return reference_points, labels, valid_idx

    @torch.no_grad()
    def _single_batch_build_prior_query(self, bbox_list, center2d_list, data, bn, depth_preds=None, label_list=None, n=6):
        '''Build prior reference points in single batch setting
        Args:
            bbox_list: have to be detached from grad
            center2d_list
            data: metadata
            depth_preds: depth map tensor if is_obj_depth, otherwises a depth object list
        Out:
            reference_points: (1, sum(Mi), 3), Mi is number of bbox in each image
            labels: (1, sum(Mi),)
        '''
        B,N = bn
        assert B == 1, "B must be 1 in single batch mode"
        eps = 1e-5

        bbox_nums = [len(bbox) for bbox in bbox_list] # number each bboxes in each image
        bboxes = torch.cat(bbox_list, 0).detach().float()
        centers2d = torch.cat(center2d_list, 0).detach().float()
        labels = torch.cat(label_list, 0).detach().unsqueeze(0)

        if not sum(bbox_nums): 
            # process when no bbox in a frame, could happen in inference -> give a fake points to avoid error
            B_ = N//n
            labels = torch.tensor(self.num_classes).repeat(B_).reshape(1, B_)
            return torch.rand((1, B_, 3), device=bboxes.device), labels
        
        if not self.is_obj_depth and not self.use_prior_gt:
            # this case we have predicted depth map (per pixel) -> Average the depth values within bbox as object depth
            depth_preds = depth_preds.detach().float()
            depth_preds = torch.argmax(depth_preds, dim=1, keepdim=True).float() # (BN, 1, H, W) -> single batch: BN=6
            rois = []
            for ith, pred_bbox in enumerate(bbox_list):    
                if pred_bbox.numel() == 0:
                    continue
                batch_inds = torch.full((pred_bbox.size(0), 1), ith, device=pred_bbox.device)
                rois.append(torch.cat([batch_inds, pred_bbox], dim=1)) # (Mi, 5)
            
            rois = torch.cat(rois, dim=0)
            pooled = self.depth_roi(depth_preds, rois) # (sum(Mi), 1, h, w), roi align output size (h,w), default (3,3)
            depths = pooled.round()
        else:
            depths = torch.cat(depth_preds, 0).detach()
        
        if not (self.use_prior_gt and self.training): depths = self._convert_bin_to_float_depth(depths)
        if not self.is_obj_depth: depths = depths.mean(dim=[1,2,3]) # (sum(Mi), )
        coords = torch.cat([centers2d, depths.unsqueeze(1)], dim=1).to(depths.device)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1) 
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)
        coords = coords.unsqueeze(-1) 

        # img2lidar array build
        img2lidars = data['lidar2img'].inverse()  # (B, N, 4, 4)
        img2lidars = img2lidars.view(B*N, 1, 4, 4) # (BN, 1, 4, 4)
        img2lidars_ = torch.cat([img2lidars[kth].repeat(num, 1, 1) for kth, num in enumerate(bbox_nums)], dim=0) 

        coords3d = torch.matmul(img2lidars_, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        coords3d = coords3d.clamp(min=0.0, max=1.0)
        return coords3d.unsqueeze(0), labels # (1, sum(Mi), 3), (1, sum(Mi))

    def _convert_bin_to_float_depth(self, depths):
        '''
            Currently, support LID mode only
        '''
        depth_max = self.depthnet_config['depth_max']
        depth_min = self.depthnet_config['depth_min']
        num_bins = self.depthnet_config['num_depth_bins']

        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depths = depth_min + bin_size / 8 * (torch.square(depths / 0.5 + 1) - 1)

        return depths