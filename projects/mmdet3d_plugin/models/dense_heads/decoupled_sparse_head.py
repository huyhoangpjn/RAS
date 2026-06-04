import torch
import torch.nn as nn 
from mmcv.cnn import Linear, bias_init_with_prob, Scale

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from projects.mmdet3d_plugin.models.utils.o2m_assigner import O2MAssigner
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear, ray_adaptive_generator, ray_eq_generator
import copy
from mmdet.models.utils import NormedLinear

@HEADS.register_module()
class DecoupledSparseHead(AnchorFreeHead):
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
                 stride=[16],
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
                 loss_cls_o2m=dict(
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
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 use_ray_o2m=False,
                 num_pseudo=10,
                 threshold_o2m=0.4,
                 max_pred_per_gt_o2m=6,
                 coef_box_o2m=0.7,
                 coef_cls_o2m=0.3,
                 coef_decay_o2m=1,
                 use_decay_o2m=True,
                 k_range_scaling=3,
                 inv_range_scale=False,
                 weight_gap_scaling=1.0,
                 depth_scaling=False,
                 depth_sat_min=20.0,
                 depth_sat_max=40.0,
                 gen_cls_thr=0.5,
                 gt_balancing=True,
                 temporal_masking=False,
                 distance_temp=2.0,
                 include_ori=False,
                 shared_o2m_head=False,
                 post_soft_nms=False,
                 post_bbox_coder=None,
                 ray_o2m_loss_weight=1.0,
                 self_attn_enhance=False,
                 self_attn_reduce_mode='average',  # 'average' or 'mlp'
                 self_attn_num_heads=8,
                 self_attn_super_mode='o2o',  # 'o2o' or 'closest' - which query doesn't need supervision
                 self_attn_loss_weight=1.0,
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
        if class_weight is not None and (self.__class__ is DecoupledSparseHead):
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

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.with_dn = with_dn
        self.stride=stride

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 

        self.use_ray_o2m = use_ray_o2m
        self.num_pseudo = num_pseudo
        self.k_range_scaling = k_range_scaling
        self.inv_range_scale = inv_range_scale
        self.weight_gap_scaling = weight_gap_scaling
        self.depth_scaling = depth_scaling
        self.depth_sat_min = depth_sat_min
        self.depth_sat_max = depth_sat_max
        self.gen_cls_thr = gen_cls_thr
        self.gt_balancing = gt_balancing
        self.temporal_masking = temporal_masking
        self.distance_temp = distance_temp
        self.include_ori = include_ori
        self.shared_o2m_head = shared_o2m_head
        self.post_soft_nms = post_soft_nms
        self.self_attn_enhance = self_attn_enhance
        self.self_attn_reduce_mode = self_attn_reduce_mode
        self.self_attn_num_heads = self_attn_num_heads
        self.self_attn_super_mode = self_attn_super_mode
        self.self_attn_loss_weight = self_attn_loss_weight
        self.ray_o2m_loss_weight = ray_o2m_loss_weight

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(DecoupledSparseHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        if self.use_ray_o2m:
            self.loss_cls_o2m = build_loss(loss_cls_o2m)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)
        
        # Set return_self_attn_output flag based on self_attn_enhance
        if self.self_attn_enhance:
            self.transformer.return_self_attn_output = True
            # Build MLP for head reduction if using 'mlp' mode
            if self.self_attn_reduce_mode == 'mlp':
                self.attn_head_reduce = nn.Sequential(
                    nn.Linear(self.self_attn_num_heads, self.self_attn_num_heads),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.self_attn_num_heads, 1),
                )

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)


        self._init_layers()
        self.reset_memory()

        if self.use_ray_o2m:
            self.o2m_assigner = O2MAssigner(
                threshold=threshold_o2m, k=max_pred_per_gt_o2m, coef_box=coef_box_o2m, coef_cls=coef_cls_o2m, coef_decay=coef_decay_o2m, use_decay=use_decay_o2m)
            self.num_pos_o2o_per_batch = 0
            self.num_pos_o2m_per_batch = 0
            if self.post_soft_nms:
                self.post_bbox_coder = build_bbox_coder(post_bbox_coder)
        
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

        if self.use_ray_o2m:
            if self.shared_o2m_head:
                self.cls_branches_o2m = self.cls_branches
            else:
                cls_branch_o2m = []
                for _ in range(self.num_reg_fcs):
                    cls_branch_o2m.append(Linear(self.embed_dims, self.embed_dims))
                    cls_branch_o2m.append(nn.LayerNorm(self.embed_dims))
                    cls_branch_o2m.append(nn.ReLU(inplace=True))
                if self.normedlinear:
                    cls_branch_o2m.append(NormedLinear(self.embed_dims, self.cls_out_channels))
                else:
                    cls_branch_o2m.append(Linear(self.embed_dims, self.cls_out_channels))
                fc_cls_o2m = nn.Sequential(*cls_branch_o2m)
                self.cls_branches_o2m = nn.ModuleList(
                    [fc_cls_o2m for _ in range(self.num_pred)])

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)


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

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
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
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
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
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict


    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
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

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
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

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # Post soft NMS
        if self.post_soft_nms:
            # Only pass non-DN queries to the NMS coder (exclude pad_size queries used for denoising)
            if self.training and mask_dict and mask_dict['pad_size'] > 0:
                nms_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
                nms_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            else:
                nms_bbox_preds = all_bbox_preds
                nms_cls_scores = all_cls_scores
            nms_preds = self.post_bbox_coder.decode({
                'all_bbox_preds': nms_bbox_preds,
                'all_cls_scores': nms_cls_scores
            })
            # Rescore so that less duplicate boxes are propagated
            rec_score = torch.stack([nms_pred['scores'] for nms_pred in nms_preds]).to(rec_score.device).unsqueeze(-1)

        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
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
    
    def forward(self, img_metas, **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        self.pre_update_memory(data)
        mlvl_feats = data['img_feats']
        B = mlvl_feats[0].size(0)

        reference_points = self.reference_points.weight
        dtype = reference_points.dtype
        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1)
        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten.append(mlvl_feat.to(dtype))
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1, )), spatial_flatten.prod(1).cumsum(0)[:-1]))
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)

        # Call transformer - return format depends on self_attn_enhance flag
        transformer_outputs = self.transformer(tgt, query_pos, feat_flatten, spatial_flatten, level_start_index, temp_memory, 
                                    temp_pos, attn_mask, reference_points, self.pc_range, data, img_metas)
        
        # Parse transformer outputs based on flags
        if self.self_attn_enhance:
            # Returns: (outs_dec, outs_cross_attn, outs_self_attn), self_self_attn: attn map
            outs_dec, outs_cross_attn, outs_self_attn = transformer_outputs
        else:
            # Returns: (outs_dec, outs_cross_attn)
            outs_dec, outs_cross_attn = transformer_outputs
            outs_self_attn = None

        outs_dec = torch.nan_to_num(outs_dec)
        outs_cross_attn = torch.nan_to_num(outs_cross_attn)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

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

            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict':None,
            }

        if self.use_ray_o2m and self.training:
            outputs_classes_o2m = []
            for lvl in range(outs_dec.shape[0]):
                outputs_class_o2m = self.cls_branches_o2m[lvl](outs_cross_attn[lvl])
                outputs_classes_o2m.append(outputs_class_o2m)
            all_cls_scores_o2m = torch.stack(outputs_classes_o2m)
            outs['all_cls_scores_o2m'] = all_cls_scores_o2m[:, :, mask_dict['pad_size']:, :] if mask_dict and mask_dict['pad_size'] > 0 else all_cls_scores_o2m
        
        # Add self-attention maps for supervision if enabled
        if self.self_attn_enhance and outs_self_attn is not None:
            # outs_self_attn shape: [num_layers, bs, num_heads, num_query, num_query]
            
            # If DN padding is used, we need to exclude the padded queries from attention supervision
            if mask_dict and mask_dict['pad_size'] > 0:
                pad_size = mask_dict['pad_size']
                # Extract only the attention for non-padded queries
                # Shape: [num_layers, bs, num_heads, num_query-pad_size, num_query-pad_size]
                outs_self_attn = outs_self_attn[:, :, :, pad_size:, pad_size:]

            # Reduce heads to get: [num_layers, bs, num_query, num_query]
            if self.self_attn_reduce_mode == 'average':
                outs['out_self_attn'] = outs_self_attn.mean(dim=2)
            elif self.self_attn_reduce_mode == 'mlp':
                # outs_self_attn: [num_layers, bs, num_heads, num_query, num_query]
                # Permute to [num_layers, bs, num_query, num_query, num_heads] for MLP
                outs_self_attn = outs_self_attn.permute(0, 1, 3, 4, 2)
                # Apply MLP: [..., num_heads] -> [..., 1]
                outs_self_attn = self.attn_head_reduce(outs_self_attn).squeeze(-1)
                # Output shape: [num_layers, bs, num_query, num_query]
                outs['out_self_attn'] = outs_self_attn
        outs['ori_query_pos'] = reference_points
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
                           cls_scores_o2m=None,
                           pseudo_gt_bboxes=None,
                           pseudo_gt_labels=None,
                           pseudo_corr_idx=None,
                           ):
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
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if self.use_ray_o2m:
            num_gt = sampling_result.num_gts
            # pad_shape = img_metas['pad_shape']

            o2o_pred_cls = cls_score.new_ones((num_gt, ))
            if num_gt > 0:
                o2o_pred_cls[sampling_result.pos_assigned_gt_inds] = cls_score[sampling_result.pos_inds, gt_labels[sampling_result.pos_assigned_gt_inds]].sigmoid().detach()
            
            # pseudo_gt_bboxes, pseudo_gt_labels, pseudo_corr_idx = ray_adaptive_generator(gt_bboxes, 
            #                                                                              gt_labels, 
            #                                                                              o2o_pred_cls=o2o_pred_cls, 
            #                                                                              num_pseudo=self.num_pseudo, 
            #                                                                              k=self.k_range_scaling, 
            #                                                                              gen_cls_thr=self.gen_cls_thr, 
            #                                                                              pad_shape=pad_shape, 
            #                                                                              lidar2img=lidar2img, 
            #                                                                              depth_scaling=self.depth_scaling,
            #                                                                              depth_sat_min=self.depth_sat_min,
            #                                                                              depth_sat_max=self.depth_sat_max,)

            (pos_inds_o2m, pos_gt_inds_o2m), _ = self.o2m_assigner(cls_scores_o2m.detach(), denormalize_bbox(bbox_pred.detach(), None), pseudo_gt_labels, pseudo_gt_bboxes, pseudo_corr_idx, num_gt, gt_labels)

            # Create mapping from pos_inds to their corresponding gt indices
            pos_inds_to_gt_mapping = torch.full((num_bboxes,), -1, dtype=torch.long, device=pos_inds.device)
            pos_inds_to_gt_mapping[pos_inds] = sampling_result.pos_assigned_gt_inds
            
            # Update pos_gt_inds_o2m for overlapping indices (o2o has priority)
            overlap_mask = torch.isin(pos_inds_o2m, pos_inds)
            pos_gt_inds_o2m[overlap_mask] = pos_inds_to_gt_mapping[pos_inds_o2m[overlap_mask]]

            # Apply temporal masking if enabled
            if self.temporal_masking:
                # Identify temporal queries: they come after the main queries
                # Query order: [DN queries (if any), main queries, temporal queries]
                # We want to mask half of the temporal queries from O2M matching
                start_temporal_idx = num_bboxes - self.num_propagated
                end_temporal_idx = num_bboxes - self.num_propagated // 2
                
                if self.num_propagated > 0 and start_temporal_idx < end_temporal_idx:
                    # Create mask for temporal queries that should be excluded from O2M
                    temporal_mask_indices = torch.arange(start_temporal_idx, end_temporal_idx, 
                                                       device=pos_inds_o2m.device)
                    
                    # Remove temporal queries from O2M positive indices
                    # Keep only pos_inds_o2m that are not in the masked temporal range
                    mask = ~torch.isin(pos_inds_o2m, temporal_mask_indices)
                    pos_inds_o2m = pos_inds_o2m[mask]
                    pos_gt_inds_o2m = pos_gt_inds_o2m[mask]

            if self.gt_balancing:
                # Calculate distance-based softmax annealing weights
                
                # Combine all positive indices and their corresponding GT indices
                all_pos_inds = torch.cat([pos_inds_o2m, pos_inds])
                all_pos_gt_inds = torch.cat([pos_gt_inds_o2m, sampling_result.pos_assigned_gt_inds])
                
                if len(all_pos_inds) > 0:
                    # Get predictions and targets for distance calculation
                    all_pos_bbox_preds = bbox_pred[all_pos_inds]  # [N_pos, bbox_dim]
                    all_pos_gt_bboxes = gt_bboxes[all_pos_gt_inds]  # [N_pos, bbox_dim]
                    
                    # Calculate weighted distance between predictions and ground truth
                    if self.include_ori:
                        # Denormalize predictions to get orientation in proper format
                        # Format after denormalization: [x, y, z, w, l, h, theta, vx, vy]
                        denorm_pred_bboxes = denormalize_bbox(all_pos_bbox_preds, self.pc_range)
                        denorm_gt_bboxes = all_pos_gt_bboxes  # GT is already denormalized
                        
                        # Extract spatial coordinates (x, y) and orientation (theta)
                        pred_centers = denorm_pred_bboxes[:, :2]  # [N_pos, 2]
                        gt_centers = denorm_gt_bboxes[:, :2]      # [N_pos, 2]
                        pred_theta = denorm_pred_bboxes[:, 6]     # [N_pos]
                        gt_theta = denorm_gt_bboxes[:, 6]         # [N_pos]
                        
                        # Calculate spatial distance (x, y)
                        spatial_distances = torch.norm(pred_centers - gt_centers, p=2, dim=1)  # [N_pos]
                        
                        # Calculate orientation distance (accounting for circular nature of angles)
                        # Use minimum angular distance considering 2π periodicity
                        theta_diff = torch.abs(pred_theta - gt_theta)
                        theta_diff = torch.min(theta_diff, 2 * torch.pi - theta_diff)  # [N_pos]
                        
                        # Weighted combination: localization (weight=2) + orientation (weight=1)
                        distances = 2.0 * spatial_distances + 1.0 * theta_diff
                    else:
                        # Original behavior: only spatial dimensions (x, y)
                        pred_centers = all_pos_bbox_preds[:, :2]  # [N_pos, 2]
                        gt_centers = all_pos_gt_bboxes[:, :2]     # [N_pos, 2]
                        distances = torch.norm(pred_centers - gt_centers, p=2, dim=1)  # [N_pos]
                    
                    # Group distances by GT index for softmax computation
                    unique_gt_inds, inverse_indices = torch.unique(all_pos_gt_inds, return_inverse=True)
                    
                    # Create softmax weights using temperature annealing (lower distance = higher weight)
                    # Use negative distances for softmax so lower distance gets higher weight
                    neg_distances = -distances / self.distance_temp
                    
                    # Compute softmax weights for each GT group in a batch manner
                    softmax_weights = torch.zeros_like(distances)
                    
                    # For self-attention supervision: store group info
                    # attn_targets: [num_bboxes, num_bboxes] - sparse target for attention map
                    # attn_masks: [num_bboxes] - which queries need supervision (1=need, 0=skip)
                    attn_targets = torch.zeros(num_bboxes, num_bboxes, device=cls_score.device)
                    attn_masks = torch.zeros(num_bboxes, device=cls_score.device)
                    
                    # Count elements per group
                    num_groups = len(unique_gt_inds)
                    group_counts = torch.bincount(inverse_indices, minlength=num_groups)
                    max_group_size = group_counts.max().item()
                    
                    # Create padded tensors for batched softmax computation
                    # Sort by group for efficient gathering
                    sorted_indices = torch.argsort(inverse_indices)
                    sorted_neg_distances = neg_distances[sorted_indices]
                    sorted_query_inds = all_pos_inds[sorted_indices]
                    
                    # Compute group start positions using cumsum
                    group_offsets = torch.zeros(num_groups + 1, dtype=torch.long, device=cls_score.device)
                    group_offsets[1:] = group_counts.cumsum(0)
                    
                    # Create position-within-group indices for scatter
                    # E.g., if groups are [3, 2, 4], positions are [0,1,2, 0,1, 0,1,2,3]
                    positions_in_group = torch.arange(len(sorted_indices), device=cls_score.device)
                    positions_in_group = positions_in_group - group_offsets[inverse_indices[sorted_indices]]
                    
                    # Pad each group to max_group_size for batched operations using scatter
                    padded_neg_distances = torch.full((num_groups, max_group_size), float('-inf'), device=cls_score.device)
                    padded_query_inds = torch.zeros((num_groups, max_group_size), dtype=torch.long, device=cls_score.device)
                    
                    # Scatter values into padded tensors (fully vectorized)
                    group_indices_sorted = inverse_indices[sorted_indices]
                    padded_neg_distances[group_indices_sorted, positions_in_group] = sorted_neg_distances
                    padded_query_inds[group_indices_sorted, positions_in_group] = sorted_query_inds
                    
                    # Batched softmax across all groups: [num_groups, max_group_size]
                    padded_softmax = torch.softmax(padded_neg_distances, dim=1)
                    
                    # Create valid mask (non-padded positions)
                    position_indices = torch.arange(max_group_size, device=cls_score.device).unsqueeze(0)
                    valid_mask = position_indices < group_counts.unsqueeze(1)  # [num_groups, max_group_size]
                    
                    # Groups with size 1 get weight 1.0 (use non-inplace operation to avoid autograd issues)
                    single_group_mask = (group_counts == 1)
                    single_group_correction = torch.zeros_like(padded_softmax)
                    single_group_correction[single_group_mask, 0] = 1.0 - padded_softmax[single_group_mask, 0].detach()
                    padded_softmax = padded_softmax + single_group_correction
                    
                    # Gather softmax weights back to original order (fully vectorized)
                    softmax_weights[sorted_indices] = padded_softmax[group_indices_sorted, positions_in_group]
                    
                    # Self-attention supervision (batched)
                    if self.self_attn_enhance:
                        # Find super query index for each group
                        # Mask out padded positions for argmax (use torch.where to avoid inplace ops)
                        masked_softmax_for_argmax = torch.where(
                            valid_mask, 
                            padded_softmax, 
                            torch.tensor(float('-inf'), device=cls_score.device)
                        )
                        
                        if self.self_attn_super_mode == 'closest':
                            super_idx_per_group = torch.argmax(masked_softmax_for_argmax, dim=1)  # [num_groups]
                        elif self.self_attn_super_mode == 'o2o':
                            # Check which positions in each group are o2o matched
                            is_o2o = torch.isin(padded_query_inds, pos_inds)  # [num_groups, max_group_size]
                            is_o2o = is_o2o & valid_mask  # Only consider valid positions
                            
                            # For groups with o2o match, use first o2o position; otherwise use closest
                            has_o2o = is_o2o.any(dim=1)  # [num_groups]
                            
                            # Get first o2o index per group (use large value for non-o2o positions)
                            o2o_positions = torch.where(is_o2o, 
                                                       position_indices.expand(num_groups, -1),
                                                       torch.tensor(max_group_size, device=cls_score.device))
                            first_o2o_idx = o2o_positions.argmin(dim=1)  # [num_groups]
                            
                            # Fallback to closest for groups without o2o
                            closest_idx = torch.argmax(masked_softmax_for_argmax, dim=1)
                            super_idx_per_group = torch.where(has_o2o, first_o2o_idx, closest_idx)
                        else:
                            super_idx_per_group = torch.argmax(masked_softmax_for_argmax, dim=1)
                        
                        # Only process groups with size > 1 (single-element groups don't need supervision)
                        multi_element_groups = (group_counts > 1)
                        
                        if multi_element_groups.any():
                            # Create super query mask: [num_groups, max_group_size]
                            # True for the super query position in each group
                            super_mask = position_indices == super_idx_per_group.unsqueeze(1)
                            
                            # Non-super mask: valid positions that are not super queries
                            non_super_valid_mask = valid_mask & ~super_mask & multi_element_groups.unsqueeze(1)
                            
                            # Get indices of all non-super queries across all groups
                            non_super_group_indices, non_super_positions = non_super_valid_mask.nonzero(as_tuple=True)
                            non_super_query_inds = padded_query_inds[non_super_group_indices, non_super_positions]
                            
                            # Set attention masks for all non-super queries at once
                            attn_masks[non_super_query_inds] = 1.0
                            
                            # Build attention targets for all non-super queries
                            # For each non-super query, we need to set attention to all members in its group
                            # non_super_group_indices tells us which group each non-super query belongs to
                            
                            # Get the softmax values and query indices for each group
                            # We need to map: non_super_query -> all group members with their softmax weights
                            num_non_super = len(non_super_query_inds)
                            if num_non_super > 0:
                                # For each non-super query, get its group's members and softmax weights
                                # group_softmax_all: [num_non_super, max_group_size]
                                group_softmax_all = padded_softmax[non_super_group_indices]
                                group_query_inds_all = padded_query_inds[non_super_group_indices]
                                group_valid_mask = valid_mask[non_super_group_indices]
                                
                                # Use advanced indexing to set attention targets
                                # Create row indices for each valid (non_super_query, group_member) pair
                                for j in range(max_group_size):
                                    valid_col = group_valid_mask[:, j]
                                    if valid_col.any():
                                        row_inds = non_super_query_inds[valid_col]
                                        col_inds = group_query_inds_all[valid_col, j]
                                        vals = group_softmax_all[valid_col, j]
                                        attn_targets[row_inds, col_inds] = vals
                    
                    # Map softmax weights back to specific prediction groups
                    n_o2m = len(pos_inds_o2m)
                    
                    # Weights for o2m predictions
                    if n_o2m > 0:
                        box_weights_o2m = softmax_weights[:n_o2m]
                    else:
                        box_weights_o2m = torch.tensor([], dtype=torch.float, device=pos_inds_o2m.device)
                    
                    # Weights for o2o predictions  
                    if len(pos_inds) > 0:
                        box_weights_o2o = softmax_weights[n_o2m:]
                    else:
                        box_weights_o2o = torch.tensor([], dtype=torch.float, device=pos_inds_o2m.device)
                else:
                    # No positive samples, use empty tensors
                    box_weights_o2m = torch.tensor([], dtype=torch.float, device=pos_inds_o2m.device)
                    box_weights_o2o = torch.tensor([], dtype=torch.float, device=pos_inds_o2m.device)
                    # Empty attention supervision targets (self_attn_enhance assumed with gt_balancing)
                    if self.self_attn_enhance:
                        attn_targets = torch.zeros(num_bboxes, num_bboxes, device=cls_score.device)
                        attn_masks = torch.zeros(num_bboxes, device=cls_score.device)
                    
            labels_o2m = cls_score.new_full((num_bboxes, ),
                                            self.num_classes,
                                            dtype=torch.long)
            label_weights_o2m = cls_score.new_ones(num_bboxes) # weighted: cls_score.new_zeros(num_bboxes) 
            neg_inds_o2m = torch.ones(num_bboxes, dtype=torch.bool, device=pos_inds_o2m.device)
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.num_gts > 0:
            if self.use_ray_o2m:
                # Currently using shared weights for o2m
                bbox_targets[pos_inds_o2m] = gt_bboxes[pos_gt_inds_o2m]
                bbox_weights[pos_inds_o2m] = 1.0 if not self.gt_balancing else box_weights_o2m.unsqueeze(-1).repeat(1, 10)
                labels_o2m[pos_inds_o2m] = gt_labels[pos_gt_inds_o2m]
                # label_weights_o2m[pos_inds_o2m] = 1.0 if not self.gt_balancing else box_weights_o2m
                # weighted: label_weights_o2m[pos_inds_o2m] = box_weights_o2m
            # Priority for o2o matching
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0 if not self.gt_balancing else box_weights_o2o.unsqueeze(-1).repeat(1, 10)

            # weighted: bbox_weights = bbox_weights ** self.weight_gap_scaling
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.use_ray_o2m:
                # Update pos_inds_o2m to include elements from pos_inds that are not already in pos_inds_o2m
                pos_inds_o2m = torch.cat([pos_inds_o2m, pos_inds[~torch.isin(pos_inds, pos_inds_o2m)]])
                neg_inds_o2m[pos_inds_o2m] = False
                neg_inds_o2m = neg_inds_o2m.nonzero(as_tuple=False).squeeze(1)
        
        # Build return tuple based on mode
        if self.self_attn_enhance:
            # self_attn_enhance assumes use_ray_o2m and gt_balancing are on
            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds,
                    labels_o2m, label_weights_o2m, pos_inds_o2m, neg_inds_o2m,
                    attn_targets, attn_masks)
        elif self.use_ray_o2m:
            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds,
                    labels_o2m, label_weights_o2m, pos_inds_o2m, neg_inds_o2m)
        else:
            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    cls_scores_o2m=None,
                    pseudo_gt_bboxes_list=None,
                    pseudo_gt_labels_list=None,
                    pseudo_corr_idx_list=None,):
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
        if self.self_attn_enhance:
            # self_attn_enhance assumes use_ray_o2m and gt_balancing are both on
            (labels_list, label_weights_list, bbox_targets_list,
             bbox_weights_list, pos_inds_list, neg_inds_list,
             labels_o2m_list, label_weights_o2m_list, pos_inds_o2m_list, neg_inds_o2m_list,
             self_attn_targets_list, self_attn_masks_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list,
                cls_scores_o2m, pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list)
            num_total_o2m_pos = sum((inds.numel() for inds in pos_inds_o2m_list))
            num_total_o2m_neg = sum((inds.numel() for inds in neg_inds_o2m_list))
        elif self.use_ray_o2m:
            (labels_list, label_weights_list, bbox_targets_list,
             bbox_weights_list, pos_inds_list, neg_inds_list,
             labels_o2m_list, label_weights_o2m_list, pos_inds_o2m_list, neg_inds_o2m_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list,
                cls_scores_o2m, pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list)
            num_total_o2m_pos = sum((inds.numel() for inds in pos_inds_o2m_list))
            num_total_o2m_neg = sum((inds.numel() for inds in neg_inds_o2m_list))
        else:
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
                gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        if self.use_ray_o2m:
            self.num_pos_o2m_per_batch = num_total_o2m_pos
            self.num_pos_o2o_per_batch = num_total_pos
            
            # For visualize the matching results
            self.vis_cls_scores_list = cls_scores_list
            self.vis_bbox_preds_list = [denormalize_bbox(preds, None) for preds in bbox_preds_list]
            self.vis_labels_list = labels_list
            self.vis_bbox_target_list = bbox_targets_list
        
        # Return based on mode
        if self.self_attn_enhance:
            # Stack attention targets across batch
            self_attn_targets = torch.stack(self_attn_targets_list, dim=0)  # [bs, num_query, num_query]
            self_attn_masks = torch.stack(self_attn_masks_list, dim=0)  # [bs, num_query]
            return (labels_list, label_weights_list, bbox_targets_list,
                    bbox_weights_list, num_total_pos, num_total_neg,
                    labels_o2m_list, label_weights_o2m_list,
                    num_total_o2m_pos, num_total_o2m_neg,
                    self_attn_targets, self_attn_masks)
        elif self.use_ray_o2m:
            return (labels_list, label_weights_list, bbox_targets_list,
                    bbox_weights_list, num_total_pos, num_total_neg,
                    labels_o2m_list, label_weights_o2m_list,
                    num_total_o2m_pos, num_total_o2m_neg)
        else:
            return (labels_list, label_weights_list, bbox_targets_list,
                    bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    cls_scores_o2m=None,
                    pseudo_gt_bboxes_list=None,
                    pseudo_gt_labels_list=None,
                    pseudo_corr_idx_list=None,
                    self_attn=None,
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
            self_attn (Tensor, optional): Self-attention map for this layer.
                Shape [bs, num_query, num_query]. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        if self.self_attn_enhance:
            # self_attn_enhance assumes use_ray_o2m is on
            cls_scores_o2m_list = [cls_scores_o2m[i] for i in range(num_imgs)]
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                            gt_bboxes_list, gt_labels_list,
                                            gt_bboxes_ignore_list, cls_scores_o2m_list,
                                            pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list)
                        
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg, labels_o2m_list, label_weights_o2m_list,
            num_total_o2m_pos, num_total_o2m_neg,
            self_attn_targets, self_attn_masks) = cls_reg_targets

            labels_o2m = torch.cat(labels_o2m_list, 0)
            label_weights_o2m = torch.cat(label_weights_o2m_list, 0)
            cls_scores_o2m = cls_scores_o2m.reshape(-1, self.cls_out_channels)
            cls_avg_factor_o2m = num_total_o2m_pos * 1.0 + num_total_o2m_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor_o2m = reduce_mean(cls_scores_o2m.new_tensor([cls_avg_factor_o2m]))
            cls_avg_factor_o2m = max(cls_avg_factor_o2m, 1)
            loss_cls_o2m = self.loss_cls_o2m(
                cls_scores_o2m, labels_o2m, label_weights_o2m, avg_factor=cls_avg_factor_o2m)
            loss_cls_o2m = torch.nan_to_num(loss_cls_o2m)
        elif self.use_ray_o2m:
            cls_scores_o2m_list = [cls_scores_o2m[i] for i in range(num_imgs)]
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                            gt_bboxes_list, gt_labels_list,
                                            gt_bboxes_ignore_list, cls_scores_o2m_list,
                                            pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list)
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg, labels_o2m_list, label_weights_o2m_list,
            num_total_o2m_pos, num_total_o2m_neg) = cls_reg_targets

            labels_o2m = torch.cat(labels_o2m_list, 0)
            label_weights_o2m = torch.cat(label_weights_o2m_list, 0)
            cls_scores_o2m = cls_scores_o2m.reshape(-1, self.cls_out_channels)
            cls_avg_factor_o2m = num_total_o2m_pos * 1.0 + num_total_o2m_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor_o2m = reduce_mean(cls_scores_o2m.new_tensor([cls_avg_factor_o2m]))
            cls_avg_factor_o2m = max(cls_avg_factor_o2m, 1)
            loss_cls_o2m = self.loss_cls_o2m(
                cls_scores_o2m, labels_o2m, label_weights_o2m, avg_factor=cls_avg_factor_o2m)
            loss_cls_o2m = torch.nan_to_num(loss_cls_o2m)
        else:
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                            gt_bboxes_list, gt_labels_list, 
                                            gt_bboxes_ignore_list)
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)                        
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes - gt_balancing will treat count o2m as one by average the box weight to avoid imbalance
        num_total_pos = loss_cls.new_tensor([num_total_pos]) if (not self.use_ray_o2m or self.gt_balancing) else loss_cls.new_tensor([num_total_o2m_pos]) # weighted: loss_cls.new_tensor([bbox_weights[:, 0].sum().item()])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        # Self-attention supervision loss
        loss_self_attn = None
        if self.self_attn_enhance and self_attn is not None and self_attn_targets is not None:
            # self_attn: [bs, num_query, num_query]
            # self_attn_targets: [bs, num_query, num_query]
            # self_attn_masks: [bs, num_query] - True for queries that need supervision
            bs, num_query, _ = self_attn.shape
            
            # Flatten for loss computation
            # Only compute loss for masked queries (non-leader queries in O2M groups)
            self_attn_flat = self_attn.reshape(-1, num_query)  # [bs*num_query, num_query]
            self_attn_targets_flat = self_attn_targets.reshape(-1, num_query)  # [bs*num_query, num_query]
            self_attn_masks_flat = self_attn_masks.reshape(-1).bool()  # [bs*num_query]
            
            # Only supervise queries that are marked (non-leader O2M queries)
            if self_attn_masks_flat.any():
                # Get masked predictions and targets
                masked_attn = self_attn_flat[self_attn_masks_flat]  # [num_masked, num_query]
                masked_targets = self_attn_targets_flat[self_attn_masks_flat]  # [num_masked, num_query]
                
                # Use KL divergence since both attention and target are already in softmax format
                # masked_attn: predicted attention distribution [num_masked, num_query]
                # masked_targets: target attention distribution [num_masked, num_query]
                
                # Add small epsilon for numerical stability and renormalize
                eps = 1e-8
                masked_attn = masked_attn.clamp(min=eps)
                masked_attn = masked_attn / masked_attn.sum(dim=-1, keepdim=True)  # Renormalize
                masked_targets = masked_targets.clamp(min=eps)
                masked_targets = masked_targets / masked_targets.sum(dim=-1, keepdim=True)  # Renormalize
                
                # KL divergence: sum(target * log(target / pred)) = sum(target * (log(target) - log(pred)))
                kl_div = masked_targets * (torch.log(masked_targets) - torch.log(masked_attn))
                kl_div = kl_div.sum(dim=-1)  # Sum over query dimension [num_masked]
                
                loss_self_attn = kl_div.mean()  # Mean over masked queries
                loss_self_attn = torch.nan_to_num(loss_self_attn)
            else:
                loss_self_attn = self_attn.new_tensor(0.0)
            
            return (loss_cls, loss_bbox, loss_cls_o2m, loss_self_attn)
        elif self.use_ray_o2m:
            return (loss_cls, loss_bbox, loss_cls_o2m)
        else:
            return (loss_cls, loss_bbox)

   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
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
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

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
             gt_bboxes_ignore=None,
             img_metas=None,
             **data):
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

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        loss_dict = dict()

        if self.self_attn_enhance:
            # self_attn_enhance assumes use_ray_o2m is on
            all_cls_scores_o2m = preds_dicts['all_cls_scores_o2m']
            all_self_attn = preds_dicts['out_self_attn']  # list of [bs, num_query, num_query] for each layer
            pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list = ray_eq_generator(
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                k=self.k_range_scaling,
                num_pseudo=self.num_pseudo,
                img_metas=img_metas,
                inv_range_scale=self.inv_range_scale,
                depth_scaling=self.depth_scaling,
                depth_sat_min=self.depth_sat_min,
                depth_sat_max=self.depth_sat_max,
                **data
            )
            all_pseudo_gt_bboxes_list = [pseudo_gt_bboxes_list for _ in range(num_dec_layers)]
            all_pseudo_gt_labels_list = [pseudo_gt_labels_list for _ in range(num_dec_layers)]
            all_pseudo_corr_idx_list = [pseudo_corr_idx_list for _ in range(num_dec_layers)]
            
            (losses_cls, losses_bbox, losses_cls_o2m, losses_self_attn) = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list, all_cls_scores_o2m,
                all_pseudo_gt_bboxes_list, all_pseudo_gt_labels_list,
                all_pseudo_corr_idx_list, all_self_attn)
            
            # O2M cls losses
            loss_dict['loss_cls_o2m'] = losses_cls_o2m[-1]
            for i, loss_cls_o2m_i in enumerate(losses_cls_o2m[:-1]):
                loss_dict[f'd{i}.loss_cls_o2m'] = loss_cls_o2m_i
            
            # Self-attention losses
            loss_dict['loss_self_attn'] = losses_self_attn[-1] * self.self_attn_loss_weight
            for i, loss_self_attn_i in enumerate(losses_self_attn[:-1]):
                loss_dict[f'd{i}.loss_self_attn'] = loss_self_attn_i * self.self_attn_loss_weight
                
        elif self.use_ray_o2m:
            all_cls_scores_o2m = preds_dicts['all_cls_scores_o2m']
            pseudo_gt_bboxes_list, pseudo_gt_labels_list, pseudo_corr_idx_list = ray_eq_generator(
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                k=self.k_range_scaling,
                num_pseudo=self.num_pseudo,
                img_metas=img_metas,
                inv_range_scale=self.inv_range_scale,
                depth_scaling=self.depth_scaling,
                depth_sat_min=self.depth_sat_min,
                depth_sat_max=self.depth_sat_max,
                **data
            )
            all_pseudo_gt_bboxes_list = [pseudo_gt_bboxes_list for _ in range(num_dec_layers)]
            all_pseudo_gt_labels_list = [pseudo_gt_labels_list for _ in range(num_dec_layers)]
            all_pseudo_corr_idx_list = [pseudo_corr_idx_list for _ in range(num_dec_layers)]
            
            (losses_cls, losses_bbox, losses_cls_o2m) = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list, all_cls_scores_o2m,
                all_pseudo_gt_bboxes_list, all_pseudo_gt_labels_list,
                all_pseudo_corr_idx_list)
            
            loss_dict['loss_cls_o2m'] = losses_cls_o2m[-1]*self.ray_o2m_loss_weight
            for i, loss_cls_o2m_i in enumerate(losses_cls_o2m[:-1]):
                loss_dict[f'd{i}.loss_cls_o2m'] = loss_cls_o2m_i*self.ray_o2m_loss_weight
        else:
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)

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
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:
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
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

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
        out = gamma * x + beta

        return out