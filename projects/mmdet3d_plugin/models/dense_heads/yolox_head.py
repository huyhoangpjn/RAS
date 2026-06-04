import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

from ..depthnet import SimpleDepthNet
from ..depthnet.ddn_loss import DDNLoss

@HEADS.register_module()
class YOLOXHeadCustom(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False, # Depthwise conv, not depth estimation
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 loss_centers2d=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                 sample_with_score=False,
                 threshold_score=0.05,
                 topk_proposal=None, # Get top k proposal
                 # return_context_feat=False, # Get 2D prior feats, initially get it directly from FPN
                 with_depth = False,
                 is_obj_depth = False,
                 depthnet_config={},
                 reg_depth_level=0, # 1/8, 1/16, 1/32
                 loss_depth_weight=1.0,
                 ):

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)
        self.loss_centers2d = build_loss(loss_centers2d)

        self.use_l1 = True  # This flag will be modified by hooks.
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.sampler_ = build_sampler(sampler_cfg, context=self)
            self.sampler_depth = build_sampler(sampler_cfg, context=self)

        self.threshold_score = threshold_score
        # self.return_context_feat = return_context_feat

        self.with_depth = with_depth
        self.is_obj_depth = is_obj_depth
        self.depthnet_config = depthnet_config
        self.reg_depth_level = reg_depth_level

        if self.with_depth:
            self.ddn_loss = DDNLoss(depthnet_config) # Should have different ddn losses
            self.loss_depth_weight = loss_depth_weight
            if self.is_obj_depth:
                self.ddn_losses = []
                for stride in self.strides:
                    new_depthconfig = depthnet_config.copy()
                    new_depthconfig.update(stride=stride)
                    self.ddn_losses.append(DDNLoss(new_depthconfig, fg_weight=15)) # increase fg weight since the map now sparser

        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_centers2d = nn.ModuleList()
        for _ in self.strides:
            # Cls and Reg conv branches
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj, conv_centers2d = self._build_predictor()
            # 4 heads
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            self.multi_level_conv_centers2d.append(conv_centers2d)
        
        if self.with_depth:
            self.depthnet = SimpleDepthNet(self.depthnet_config, self.is_obj_depth, len(self.strides))


    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        conv_centers2d = nn.Conv2d(self.feat_channels, 2, 1)
        return conv_cls, conv_reg, conv_obj, conv_centers2d

    def init_weights(self):
        super(YOLOXHeadCustom, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj, conv_centers2d):
        """Forward feature of a single scale level."""
        if x.dim() == 5:
            bs, n, c, h, w= x.shape
            x = x.reshape(bs*n, c, h, w)

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)
        centers2d_offset = conv_centers2d(reg_feat)

        return cls_score, bbox_pred, objectness, centers2d_offset

    @force_fp32(apply_to=('img',)) # 'multi_lv_feats'
    def forward(self, **data):
        """Forward features from the upstream network.
        """
        # feats = data['multi_lv_feats']
        feats = data['img_feats']
        cls_scores, bbox_preds, objectnesses, centers2d_offsets=multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj,
                           self.multi_level_conv_centers2d,
                           )
        out = {
            'enc_cls_scores': cls_scores,
            'enc_bbox_preds': bbox_preds,
            'pred_centers2d_offset': centers2d_offsets,
            'objectnesses':objectnesses,
            'topk_indexes':None
        }

        if self.with_depth: # Can supervise with all level depth
            depth_feat = feats[self.reg_depth_level] if not self.is_obj_depth else feats # Can add camera aware here (instrinsic SE/MLN)
            depth_logits = self.depthnet(depth_feat)
            if self.is_obj_depth:
                depth_preds = [depth_logit.softmax(dim=1) for depth_logit in depth_logits] # depth_logit: B*N, bins, H, W
            else: 
                depth_preds = depth_logits.softmax(dim=1)
            out.update(depth_logits = depth_logits, depth_preds = depth_preds)
        return out

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_bboxes(self, preds_dicts,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   threshold_score=0.1,
                   **data
                   ):
        cls_scores = preds_dicts['enc_cls_scores'] # By default, logits, it's a list with the output of 3 levels: 3x(BN num_cls Hi Wi) - cls of each pixel on feat maps
        bbox_preds = preds_dicts['enc_bbox_preds'] # 3x(BN 4 Hi Wi)
        objectnesses = preds_dicts['objectnesses'] # 3x(BN 1 Hi Wi)
        centers_2d = preds_dicts['pred_centers2d_offset'] # 3x(BN 2 Hi Wi)
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True) # 3x(Hi*Wi, 4)
        
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        # Bring channels (box-4params, cls-10classes,...) to the last for easily flatten (to filter high quality predictions)
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centers_2d = [
            center_2d.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for center_2d in centers_2d
        ]
        num_cls = cls_scores[0].shape[1]
        flatten_scores = [
            score.permute(0,2,3,1).reshape(num_imgs, -1, num_cls).topk(1, dim=2)
            for score in cls_scores
        ]
 
        valid_indices_list = []
        # First filter by objectness score * cls (since during training using multiple positives in 3x3 area so need to use nms for the score in 3x3 area)
        for i in range(len(objectnesses)):
            sample_weight = objectnesses[i].sigmoid() * flatten_scores[i].values.sigmoid().reshape(objectnesses[i].shape)#cls_scores[i].topk(1, dim=1).values.sigmoid()
            sample_weight_nms = nn.functional.max_pool2d(sample_weight, (3, 3), stride=1, padding=1)
            sample_weight_nms = sample_weight_nms.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1) # (BN, Hi*Wi, 1)
            sample_weight_ = sample_weight.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            sample_weight = sample_weight_ * (sample_weight_ == sample_weight_nms).float() # (BN, Hi*Wi, 1)
            valid_indices_list.append(sample_weight)
        valid_indices = torch.cat(valid_indices_list, dim=1)
        flatten_sample_weight = valid_indices.clone() # (BN,sum(Hi*Wi), 1)

        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1) # (BN,sum(Hi*Wi),4) 
        flatten_centers_2d = torch.cat(flatten_centers_2d, dim=1) # (BN,sum(Hi*Wi),2)
        flatten_priors = torch.cat(mlvl_priors) # (sum(Hi*Wi),4)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds) # including converting cx,cy,w,h to x1,y1,x2,y2
        flatten_centers_2d = self._centers2d_decode(flatten_priors, flatten_centers_2d)
        flatten_labels = torch.cat([flatten_score.indices for flatten_score in flatten_scores], dim=1) # (BN,sum(Hi*Wi),1)

        valid_indices = valid_indices > threshold_score

        bbox_list = []
        center2d_list = []
        score_list = []
        label_list = []
        for i in range(num_imgs):
            pred_bbox = flatten_bboxes[i][valid_indices[i].repeat(1, 4)].reshape(-1, 4)
            center_2d = flatten_centers_2d[i][valid_indices[i].repeat(1, 2)].reshape(-1,2)
            label = flatten_labels[i][valid_indices[i]]
            score = flatten_sample_weight[i][valid_indices[i]]
            
            # since in nms has sort operation -> should run this per img
            if with_nms and len(pred_bbox):
                _, keep = batched_nms(pred_bbox, score, label, cfg.nms)
                keep = keep.sort().values
                keep_mask = torch.zeros_like(label, dtype=torch.bool)
                keep_mask[keep] = True
                pred_bbox = pred_bbox[keep]
                center_2d = center_2d[keep]
                label = label[keep]
                score = score[keep]
                # update valid index mask
                prev_valid = (valid_indices[i]==True)
                valid_indices[i][prev_valid] = keep_mask
                
            # if i == 0:
            #     print(pred_bbox)
            #     print(flatten_bboxes[i][valid_indices[i].repeat(1, 4)].reshape(-1, 4))
            # bbox = bbox_xyxy_to_cxcywh(pred_bbox)
            bbox_list.append(pred_bbox) # bbox: [x1, y1, x2, y2]
            center2d_list.append(center_2d)
            score_list.append(score)
            label_list.append(label)
        
        valid_depth_list = []
        if self.is_obj_depth and self.with_depth:
            # return only depth list in case object depth, otherwises, get depthmap from forward function is enough
            # this return the bin index of each predicted object
            depth_preds = preds_dicts['depth_preds']
            flatten_depth_preds = [
                depth_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.depthnet_config['num_depth_bins']+1)
                for depth_pred in depth_preds
            ]
            flatten_depth_preds = torch.cat(flatten_depth_preds, dim=1)
            flatten_depth_preds = torch.argmax(flatten_depth_preds, dim=-1, keepdim=True)
            for ith in range(num_imgs):
                valid_depth = flatten_depth_preds[ith][valid_indices[ith]]
                valid_depth_list.append(valid_depth)   # BN x (Mi, 1)
        else:
            valid_depth_list = preds_dicts['depth_preds']
        return {
            'bbox_list': bbox_list,
            'label_list': label_list,
            'center2d_list': center2d_list,
            'depth_list': valid_depth_list,
            'bbox2d_scores': score_list,
            'valid_indices': valid_indices
        }
    
    def _get_highest_conf_for_gt_2d_single(self, cls_preds, objectness, priors, decoded_bboxes,
                    gt_bboxes, gt_labels):
        '''
        Find highest confident pred for a gt in a single image - compatible with pred format
        If no gt, return empty tensor
        '''
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes)
        gt_labels = gt_labels.to(decoded_bboxes)
        conf2d = priors.new_full((num_gts, ), self.threshold_score)

        if num_gts == 0: return conf2d,
        offset_priors = torch.cat(
                [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)
        sampling_result = self.sampler.sample(assign_result, decoded_bboxes, gt_bboxes)

        scores = cls_preds[sampling_result.pos_inds].sigmoid() * objectness[sampling_result.pos_inds].unsqueeze(1).sigmoid()
        pos_gt_labels = gt_labels[sampling_result.pos_assigned_gt_inds]
        scores = scores[range(len(scores)), pos_gt_labels.long()]
        # scores = torch.cat((sampling_result.pos_assigned_gt_inds.unsqueeze(-1), scores.unsqueeze(-1)), dim=-1)
        # Reduce to get max for each gt_index
        unique_pos_gt_inds, inverse = torch.unique(sampling_result.pos_assigned_gt_inds, return_inverse=True)
        max_score_gt = torch.zeros_like(unique_pos_gt_inds, dtype=scores.dtype)
        max_score_gt = max_score_gt.scatter_reduce(0, inverse, scores, reduce='amax', include_self=False)
        conf2d[unique_pos_gt_inds] = max_score_gt
        conf2d[conf2d < self.threshold_score] = self.threshold_score
        return conf2d,

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_highest_conf_for_gt_2d(self, preds_dicts, gt_bboxes2d_list, gt_labels2d_list,):

        cls_scores = preds_dicts['enc_cls_scores']
        bbox_preds = preds_dicts['enc_bbox_preds']
        objectnesses = preds_dicts['objectnesses']
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
            
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)

        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        gt_bboxes = [bboxes2d for i in gt_bboxes2d_list for bboxes2d in i]
        gt_labels = [labels2d for i in gt_labels2d_list for labels2d in i]

        conf2d, = multi_apply(self._get_highest_conf_for_gt_2d_single, flatten_cls_preds.detach(),
                    flatten_objectness.detach(),
                    flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
                    flatten_bboxes.detach(), gt_bboxes, gt_labels)
        
        return conf2d

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2] # Scale and translate (cx,cy,strides) a location on feat map back to input image (near or center of conv kernel)
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        # Also convert cx,cy,w,h to x1,y1,x2,y2
        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes
    
    def _centers2d_decode(self, priors, centers2d):
        centers2d = (centers2d[..., :2] * priors[:, 2:]) + priors[:, :2]
        return centers2d

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses', 'centers2d'))
    def loss(self,
             gt_bboxes2d_list,
             gt_labels2d_list,
             centers2d,
             preds_dicts,
             depths,
             img_metas, #len=B
             gt_bboxes_ignore=None):
        """Compute loss of the head.`
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        cls_scores = preds_dicts['enc_cls_scores']
        bbox_preds = preds_dicts['enc_bbox_preds']
        objectnesses = preds_dicts['objectnesses']
        centers2d_offset = preds_dicts['pred_centers2d_offset']
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
            
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_centers2d_offset = [
            center2d_offset.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for center2d_offset in centers2d_offset
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_centers2d_offset = torch.cat(flatten_centers2d_offset, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        gt_bboxes = [bboxes2d for i in gt_bboxes2d_list for bboxes2d in i]
        gt_labels = [labels2d for i in gt_labels2d_list for labels2d in i]
        centers2d = [center2d for i in centers2d for center2d in i]
        gt_center_depths = [depth for i in depths for depth in i]

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, centers2d_target,
         num_fg_imgs, depth_targets) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels, centers2d, gt_center_depths)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0) # BN*sum(Hi*Wi)
        cls_targets = torch.cat(cls_targets, 0) # gather all labels (meaning that sample points to bg is not optimize instead of trying map it to 0)
        obj_targets = torch.cat(obj_targets, 0) # same size as mask
        bbox_targets = torch.cat(bbox_targets, 0) # same as cls
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        centers2d_target = torch.cat(centers2d_target, 0)

        # There are 5 losses in total: 
        # - loss_bbox: IOU loss between positive sample and GT
        # - loss_obj: objectness - 1 indicate that sample pixel is a object and vice versa
        # - loss_cls
        # - center loss
        # - loss_l1: L1 regression of bbox

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples
        loss_centers2d = self.loss_centers2d(
            flatten_centers2d_offset.view(-1, 2)[pos_masks],
            centers2d_target) / num_total_samples

        loss_dict = dict(
            enc_loss_cls=loss_cls, enc_loss_iou=loss_bbox, enc_loss_obj=loss_obj, enc_loss_centers2d=loss_centers2d)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(enc_loss_bbox=loss_l1)

        if self.with_depth:
            if not self.is_obj_depth:
                device = preds_dicts['enc_cls_scores'][0].device
                if 'ins_depthmap' in img_metas[0].keys(): # instance depth map
                    gt_ins_depth = [img_meta['ins_depthmap'] for img_meta in img_metas] # load gt: batch is a list
                    gt_ins_depth_mask = [img_meta['ins_depthmap_mask'] for img_meta in img_metas]
                    if len(gt_ins_depth) == 1: 
                        gt_depths = gt_ins_depth[0].to(device)
                        gt_depths_mask = gt_ins_depth_mask[0].to(device)
                    else:
                        gt_depths = torch.cat(gt_ins_depth).to(device)
                        gt_depths_mask = torch.cat(gt_ins_depth_mask).to(device)
                else:
                    gt_depths, gt_depths_mask = None, None

                depth_map_logits = preds_dicts['depth_logits']  # (BN, D, H, W)
                loss_depth_map = self.ddn_loss(depth_map_logits, None, None, type='argo', gt_depths=gt_depths,
                                                   gt_depths_mask=gt_depths_mask) * self.loss_depth_weight
                loss_dict.update(loss_depth=loss_depth_map)
            else:
                depth_map_logits = preds_dicts['depth_logits'] # 3*(BN,D,Hi,Wi)
                depth_targets = torch.cat(depth_targets, 0).view(num_imgs, -1) # BN,sum(Hi*Wi)
                pos_masks = pos_masks.view(num_imgs, -1)
                start_lv = 0
                for lv in range(len(depth_map_logits)):
                    _, _, Hi, Wi = depth_map_logits[lv].shape
                    depth_target = depth_targets[:, start_lv:start_lv+Hi*Wi].view(num_imgs, Hi, Wi)
                    depth_mask = pos_masks[:, start_lv:start_lv+Hi*Wi].view(num_imgs, Hi, Wi)
                    loss_depth_lv = self.ddn_losses[lv](depth_map_logits[lv], None, None, type='argo', gt_depths=depth_target,
                                                   gt_depths_mask=depth_mask) * self.loss_depth_weight
                    loss_dict[f'loss_depth_lv_{lv}'] = loss_depth_lv
                    start_lv += Hi*Wi

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                    gt_bboxes, gt_labels, centers2d, gt_center_depths=None):
        """Compute classification, regression, and objectness targets for
        priors in a SINGLE IMAGE.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0) # sum(Hi,Wi)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        centers2d = centers2d.to(decoded_bboxes.dtype)
        depth_min, depth_max, num_bins = [self.depthnet_config.get(key) for key in
                                                ['depth_min', 'depth_max', 'num_depth_bins']]
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            centers2d_target = cls_preds.new_zeros((0, 2))

            if self.is_obj_depth:
                depth_target = cls_preds.new_zeros(num_priors)
                depth_target = self.bin_depths(depth_target, num_bins=num_bins, target=True)
            else: depth_target = None
            return (foreground_mask, cls_target, obj_target, bbox_target,
                        l1_target, centers2d_target, 0, depth_target)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        # need different sampler since the outputs has different shape
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        sampling_result_centers2d = self.sampler_.sample(assign_result, priors, centers2d)

        pos_inds = sampling_result.pos_inds # 1 GT has at most k matched prediction according to Optimal Transportation paper
                                            # Based on cost matrix + center radius (this can pose additional penalty if center is far GT center) 
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds] # IOU between predicted and its assigned gt
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool) # sum(Hi,Wi)
        foreground_mask[pos_inds] = 1

        #centers2d target

        centers2d_labels = sampling_result_centers2d.pos_gt_bboxes
        centers2d_target = cls_preds.new_zeros((num_pos_per_img, 2))
        centers2d_target = self._get_centers2d_target(centers2d_target, centers2d_labels, priors[pos_inds])

        if self.is_obj_depth:
            gt_center_depths = gt_center_depths.to(decoded_bboxes.dtype).view(-1, 1) # a simple trick to reuse sampling method with minimal effort
            sampling_result_depth = self.sampler_depth.sample(assign_result, priors, gt_center_depths) # prior here just for the shape
            depth_labels = sampling_result_depth.pos_gt_bboxes.view(-1,)
            depth_target = torch.zeros_like(objectness)
            # convert to bin
            depth_target[pos_inds] = depth_labels
            depth_target = self.bin_depths(depth_target, depth_min=depth_min, depth_max=depth_max, num_bins=num_bins, target=True)
            
        else: depth_target = None
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, centers2d_target, num_pos_per_img, depth_target)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
    
    def _get_centers2d_target(self, centers2d_target, centers2d_labels, priors):
        centers2d_target = (centers2d_labels - priors[:, :2]) / priors[:, 2:]
        return centers2d_target

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        return indices