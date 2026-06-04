import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from mmcv.ops import nms_rotated, batched_nms
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSCoder(BaseBBoxCoder):
    """Bbox coder with traditional NMS using rotated IoU on BEV.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        nms_threshold (float): IoU threshold for NMS. Default: 0.5.
        num_classes (int): Number of classes. Default: 10.
        soft_nms (bool): Whether to use soft-NMS instead of standard NMS.
            Default: False.
        cls_spec_nms (dict[int, float], optional): Class-specific NMS thresholds.
            Maps class index to NMS threshold. For example, {2: 0.3} sets
            NMS threshold to 0.3 for class 2 (bus). Default: None.
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 nms_threshold=0.5,
                 num_classes=10,
                 soft_nms=False,
                 soft_nms_rescore_only=False,
                 cls_spec_nms=None):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.num_classes = num_classes
        self.soft_nms = soft_nms
        self.soft_nms_rescore_only = soft_nms_rescore_only
        # Be permissive about config typos like `cls_spec_nms = {...},` (a 1-tuple)
        if isinstance(cls_spec_nms, tuple) and len(cls_spec_nms) == 1 and isinstance(cls_spec_nms[0], dict):
            cls_spec_nms = cls_spec_nms[0]
        if cls_spec_nms is None:
            self.cls_spec_nms = {}
        elif not isinstance(cls_spec_nms, dict):
            raise TypeError(f'cls_spec_nms must be a dict[int, float] or None, got {type(cls_spec_nms)}')
        else:
            # Normalize keys/values to plain int/float
            self.cls_spec_nms = {int(k): float(v) for k, v in cls_spec_nms.items()}

    def encode(self):
        pass

    def _bev_boxes_from_3d(self, boxes_3d):
        """Extract BEV boxes (x, y, w, l, theta) from 3D boxes.
        Args:
            boxes_3d (Tensor): 3D boxes with format (x, y, z, w, l, h, theta, vx, vy).
                Shape [N, 9].
        Returns:
            Tensor: BEV boxes with format (x, y, w, l, theta). Shape [N, 5].
        """
        return torch.stack([
            boxes_3d[:, 0],  # x
            boxes_3d[:, 1],  # y  
            boxes_3d[:, 3],  # w
            boxes_3d[:, 4],  # l
            boxes_3d[:, 6],  # theta
        ], dim=-1)

    def _soft_nms_rotated(self, boxes, scores, labels, nms_threshold, sigma=0.5, score_threshold=0.001, rescore_only=False):
        """Perform soft-NMS on rotated BEV boxes in a batch manner.
        Args:
            boxes (Tensor): BEV boxes with format (x, y, w, l, theta). Shape [N, 5].
            scores (Tensor): Classification scores. Shape [N].
            labels (Tensor): Classification labels. Shape [N].
            nms_threshold (float): IoU threshold for soft-NMS decay (not used in Gaussian variant).
            sigma (float): Gaussian function parameter for soft-NMS. Default: 0.5.
            score_threshold (float): Score threshold to filter boxes after decay. Default: 0.001.
            rescore_only (bool): If True, only rescore boxes without removing any, keeping same dimension. Default: False.
        Returns:
            Tuple[Tensor, Tensor]: Indices of boxes to keep and updated scores.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device), scores

        from mmcv.ops import box_iou_rotated
        
        # If rescore_only, we'll update scores in place to preserve order
        if rescore_only:
            updated_scores = scores.clone()
        
        all_keep_indices = []
        all_keep_scores = []
        
        # Process each class separately
        unique_labels = labels.unique()
        
        for label in unique_labels:
            # Get boxes and scores for current class
            class_mask = labels == label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask].clone()
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            
            if len(class_boxes) == 0:
                continue
            
            # Get class-specific NMS threshold if provided
            class_label = int(label.item())
            
            # Sort by score in descending order
            sorted_scores, sorted_idx = class_scores.sort(descending=True)
            sorted_boxes = class_boxes[sorted_idx]
            sorted_indices = class_indices[sorted_idx]
            
            N = len(sorted_boxes)
            
            # Batch compute IoU matrix for all boxes
            # Shape: [N, N]
            iou_matrix = box_iou_rotated(sorted_boxes, sorted_boxes)
            
            # Create upper triangular mask to only consider j > i pairs
            # This ensures we only apply decay from higher-scored boxes to lower-scored ones
            triu_mask = torch.triu(torch.ones((N, N), device=boxes.device, dtype=torch.bool), diagonal=1)
            
            # Apply Gaussian decay in batch
            # For each box i, compute decay weights for all boxes j > i
            # weight = exp(-(iou^2) / sigma)
            decay_weights = torch.exp(-(iou_matrix ** 2) / sigma)
            
            # Mask out the lower triangle and diagonal (only keep upper triangle)
            decay_weights = torch.where(triu_mask, decay_weights, torch.ones_like(decay_weights))
            
            # For each box j, multiply all decay weights from boxes i where i < j
            # This is equivalent to the iterative soft-NMS but computed in batch
            # Shape: [N] - product along dimension 0 gives cumulative decay for each box
            cumulative_decay = torch.prod(decay_weights, dim=0)
            
            # Apply decay to scores
            sorted_scores = sorted_scores * cumulative_decay
            
            if rescore_only:
                # Update scores in place to preserve original order
                updated_scores[class_indices] = sorted_scores[torch.argsort(sorted_idx)]
            else:
                # Keep boxes with scores above threshold
                final_mask = sorted_scores >= score_threshold
                final_indices = sorted_indices[final_mask]
                final_scores = sorted_scores[final_mask]
                all_keep_indices.append(final_indices)
                all_keep_scores.append(final_scores)
        
        if rescore_only:
            # Return all indices in original order with updated scores
            all_indices = torch.arange(len(boxes), device=boxes.device)
            return all_indices, updated_scores
        
        if len(all_keep_indices) == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device), scores.clone()
        
        # Concatenate all kept indices and scores
        keep_indices = torch.cat(all_keep_indices)
        keep_scores_list = torch.cat(all_keep_scores)
        
        # Create updated scores tensor with original shape
        updated_scores = scores.clone()
        updated_scores[keep_indices] = keep_scores_list
        
        return keep_indices, updated_scores

    def _rotated_nms(self, boxes, scores, labels, nms_threshold):
        """Perform rotated NMS on BEV boxes.
        Args:
            boxes (Tensor): BEV boxes with format (x, y, w, l, theta). Shape [N, 5].
            scores (Tensor): Classification scores. Shape [N].
            labels (Tensor): Classification labels. Shape [N].
            nms_threshold (float): IoU threshold for NMS.
        Returns:
            Tensor: Indices of boxes to keep after NMS.
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        keep_indices = []
        
        # Process each class separately
        unique_labels = labels.unique()
        
        for label in unique_labels:
            # Get boxes and scores for current class
            class_mask = labels == label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            
            if len(class_boxes) == 0:
                continue
            
            # Get class-specific NMS threshold if provided
            class_label = int(label.item())
            class_nms_thr = float(self.cls_spec_nms.get(class_label, nms_threshold))

            # mmcv.ops.nms_rotated can be sensitive to dtype/NaNs; sanitize per-class
            class_boxes = class_boxes.to(dtype=torch.float32)
            class_scores = class_scores.to(dtype=torch.float32)
            finite_mask = torch.isfinite(class_boxes).all(dim=1) & torch.isfinite(class_scores)
            if not bool(finite_mask.all()):
                class_boxes = class_boxes[finite_mask]
                class_scores = class_scores[finite_mask]
                class_indices = class_indices[finite_mask]
                if len(class_boxes) == 0:
                    continue
                
            # Use mmcv's nms_rotated function
            # nms_rotated returns (dets, keep_indices) tuple
            try:
                result = nms_rotated(class_boxes, class_scores, class_nms_thr)
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) == 2:
                    # Result is (dets, indices) tuple
                    _, class_keep_indices = result
                else:
                    # Result is just indices
                    class_keep_indices = result
                
                # Ensure it's 1D tensor
                if class_keep_indices.dim() == 0:
                    class_keep_indices = class_keep_indices.unsqueeze(0)
                elif class_keep_indices.dim() > 1:
                    class_keep_indices = class_keep_indices.flatten()
                
                # Map back to original indices
                if len(class_keep_indices) > 0:
                    original_keep_indices = class_indices[class_keep_indices]
                    keep_indices.append(original_keep_indices)
                    
            except Exception as e:
                print(f"Warning: NMS failed for class {label}: {e}")
                print(f"class_boxes shape: {class_boxes.shape}")
                print(f"class_scores shape: {class_scores.shape}")
                # Fallback: axis-aligned BEV NMS (ignores theta) to avoid keeping all duplicates.
                x = class_boxes[:, 0]
                y = class_boxes[:, 1]
                w = class_boxes[:, 2].clamp(min=1e-6)
                l = class_boxes[:, 3].clamp(min=1e-6)
                x1 = x - w * 0.5
                y1 = y - l * 0.5
                x2 = x + w * 0.5
                y2 = y + l * 0.5
                aa_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                _, aa_keep = batched_nms(aa_boxes, class_scores, torch.zeros_like(class_scores, dtype=torch.long), class_nms_thr)
                if aa_keep.numel() > 0:
                    keep_indices.append(class_indices[aa_keep])
                continue
        
        if len(keep_indices) == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
            
        # Concatenate all kept indices
        keep_indices = torch.cat(keep_indices)
        
        return keep_indices

    def decode_single(self, cls_scores, bbox_preds):
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        
        # Get top-k scores across all classes
        if not self.soft_nms_rescore_only:
            scores, indices = cls_scores.view(-1).topk(max_num)
            labels = indices % self.num_classes
            bbox_index = torch.div(indices, self.num_classes, rounding_mode='floor')
            bbox_preds = bbox_preds[bbox_index]
        else:
            # Take top-1 score and label per query to keep structure
            # cls_scores: [num_queries, num_classes]
            scores, labels = cls_scores.max(dim=-1)

        # Denormalize bboxes
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_labels = labels 

        # Apply score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores >= self.score_threshold
            final_box_preds = final_box_preds[thresh_mask]
            final_scores = final_scores[thresh_mask]
            final_labels = final_labels[thresh_mask]

        # Apply post center range filter
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=final_scores.device)
            
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            final_box_preds = final_box_preds[mask]
            final_scores = final_scores[mask]
            final_labels = final_labels[mask]

        # Perform rotated NMS on BEV projection
        if len(final_box_preds) > 0:
            # Project to BEV (x, y, w, l, theta)
            bev_boxes = self._bev_boxes_from_3d(final_box_preds)
            
            # Validate BEV boxes (ensure positive dimensions)
            valid_mask = (bev_boxes[:, 2] > 0) & (bev_boxes[:, 3] > 0)  # w > 0 and l > 0
            if not valid_mask.all():
                final_box_preds = final_box_preds[valid_mask]
                final_scores = final_scores[valid_mask]
                final_labels = final_labels[valid_mask]
                bev_boxes = bev_boxes[valid_mask]
            
            # Apply NMS only if we have boxes left
            if len(bev_boxes) > 0:
                if self.soft_nms:
                    # Use soft-NMS
                    keep_indices, updated_scores = self._soft_nms_rotated(
                        bev_boxes, final_scores, final_labels, self.nms_threshold,
                        rescore_only=self.soft_nms_rescore_only
                    )
                    # Filter results based on soft-NMS
                    boxes3d = final_box_preds[keep_indices]
                    scores = updated_scores[keep_indices]
                    labels = final_labels[keep_indices]
                else:
                    # Use standard NMS
                    keep_indices = self._rotated_nms(bev_boxes, final_scores, final_labels, self.nms_threshold)
                    # Filter results based on NMS
                    boxes3d = final_box_preds[keep_indices]
                    scores = final_scores[keep_indices]
                    labels = final_labels[keep_indices]
            else:
                boxes3d = final_box_preds
                scores = final_scores
                labels = final_labels
        else:
            boxes3d = final_box_preds
            scores = final_scores
            labels = final_labels

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            preds_dicts (dict): Prediction dictionary containing:
                all_cls_scores (Tensor): Classification scores from last decoder layer.
                    Shape [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Bbox predictions from last decoder layer.
                    Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes for each sample in the batch.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list
