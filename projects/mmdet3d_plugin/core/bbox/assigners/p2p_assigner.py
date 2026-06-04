import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
import warnings

@BBOX_ASSIGNERS.register_module()
class P2PAssigner(BaseAssigner): 
    '''Propagated to Prior Assigner, inspired from SimOTAAssigner \\
    Match n propagated queries to m prior queries \\ 
    assign method return AssignResult in which each propagated query correspond to a single prior query and its matching score \\
    Valid matching score > alpha \\
    There are k prop queries being matched to a prior query \\
    k is decided by: top-1 score > 0.5, top-k score > 0.5 and top-1 - top-k <= 0.1 \\
    Only 1 prior is assigned to 1 prop

    Additional notes: \\
    Prop can be matched to 2 priors on different views -> greedy: max match score + max confidence score
    '''
    def __init__(self,
                 alpha=0.5,
                 beta=0.1,
                 ):
        self.alpha = alpha
        self.beta = beta
    
    def assign(self, score_matrix, labels_2d, box_inds):
        '''
        Follow convention
        '''
        try:
            assign_result = self._assign(score_matrix, labels_2d, box_inds)
            return assign_result
        except:
            device = score_matrix.device
            warnings.warn('OOM RuntimeError is raised due to the huge memory '
                          'cost during label assignment. CPU mode is applied '
                          'in this batch. If you want to avoid this issue, '
                          'try to reduce the batch size or image size.')
            torch.cuda.empty_cache()

            score_matrix = score_matrix.cpu()
            labels_2d = labels_2d.cpu()
            box_inds = box_inds.cpu()

            assign_result = self._assign(score_matrix, labels_2d, box_inds)
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            assign_result.labels = assign_result.labels.to(device)

            return assign_result

    def _assign(self, score_matrix, labels_2d, box_inds):
        num_prior = score_matrix.size(1)
        num_prop = score_matrix.size(0)

        # Assign 0 by default: mean not match with any prior
        # Score 0 by default
        # Label -1 by default
        assigned_prior_inds = score_matrix.new_zeros((num_prop, 1), dtype=torch.long)
        assigned_labels = assigned_prior_inds.new_full((num_prop, 1), -1)

        if num_prior == 0 or num_prop == 0:
            assigned_scores = torch.zeros_like(assigned_prior_inds)
            return AssignResult(num_prior, assigned_prior_inds, assigned_scores, assigned_labels)

        matching_matrix = torch.zeros_like(score_matrix)
        valid_mask_1 = score_matrix >= self.alpha
        top_1_prop_match = score_matrix.topk(1, dim=0).values
        valid_mask_2 = (top_1_prop_match - score_matrix) < self.beta

        matching_matrix[(valid_mask_1 & valid_mask_2)] = 1.0

        del valid_mask_1, valid_mask_2, top_1_prop_match

        '''
        # Filter prop belonging to more than 2 priors
        prop_has_multiple_matching = matching_matrix.sum(-1) > 1
        if prop_has_multiple_matching.sum() > 0:
            score_max, score_argmax = torch.max(
                score_matrix[prop_has_multiple_matching, :], dim=1
            )

            matching_matrix[prop_has_multiple_matching, :] = 0.0
            matching_matrix[prop_has_multiple_matching, score_argmax] = 1.0
        '''

        cam_of_prior = score_matrix.new_empty(num_prior, dtype=torch.long)
        for cam_idx, idx_range in enumerate(box_inds):
            cam_of_prior[idx_range, ...] = cam_idx
        
        # For each prop, finding the best corresponding prior on EACH VIEW
        for i in range(num_prop):
            matched_js = (matching_matrix[i]==1).nonzero(as_tuple=False).view(-1) # Return prior indexes that have matching score = 1
            if matched_js.numel() <= 1:
                continue
            
            cam_idx = cam_of_prior[matched_js]
            scores = score_matrix[i, matched_js]
            # For each cam within this row, pick the best:
            keep_mask = torch.zeros_like(matched_js, dtype=torch.bool)
            for c in cam_idx.unique():
                in_c = (cam_idx==c).nonzero(as_tuple=False).view(-1)
                best_idx = in_c[scores[in_c].argmax()]
                keep_mask[best_idx] = True
            
            matching_matrix[i].zero_()
            matching_matrix[i, matched_js[keep_mask]] = 1
        
        num_matches_per_prop = matching_matrix.sum(dim=1).long()
        max_matched = max(int(num_matches_per_prop.max().item()), 1)

        assigned_prior_inds   = score_matrix.new_zeros((num_prop, max_matched), dtype=torch.long)
        assigned_scores = score_matrix.new_zeros((num_prop, max_matched), dtype=score_matrix.dtype)
        assigned_labels = score_matrix.new_full((num_prop, max_matched), -1, dtype=torch.long)

        for i in range(num_prop):
            js = (matching_matrix[i] == 1).nonzero(as_tuple=False).view(-1)
            if js.numel() == 0:
                continue
            # sort by score descending (optional, but stable ordering helps)
            scs, order = score_matrix[i, js].sort(descending=True)
            js = js[order]

            assigned_prior_inds[i, : len(js)]   = js + 1
            assigned_scores[i, : len(js)] = scs
            assigned_labels[i, : len(js)] = labels_2d[js].long()

        '''
        matched_mask = matching_matrix.sum(1) > 0.0
        matched_prior_inds = matching_matrix[matched_mask, :].argmax(1)
        assigned_scores = (matching_matrix*score_matrix).sum(1)
        # Follow AssignResult format
        assigned_prior_inds[matched_mask] = matched_prior_inds + 1
        assigned_labels[matched_mask] = labels_2d[matched_prior_inds].long()
        '''

        return AssignResult(
            num_prior, assigned_prior_inds, assigned_scores, assigned_labels
        )