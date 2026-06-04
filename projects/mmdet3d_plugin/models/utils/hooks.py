from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

import torch

import matplotlib.pyplot as plt

from .misc import ray_eq_generator
import os
import math

# Hook for getting GT early stage
@HOOKS.register_module()
class UseGTHook(Hook):
    def __init__(
        self,
        stop_gt_iter=0,
    ):
        self.stop_gt_iter = stop_gt_iter

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if cur_iter >= self.stop_gt_iter:
            model.pts_bbox_head.use_prior_gt = False

@HOOKS.register_module()
class GetNumPosHook(Hook):
    def __init__(self, log_interval=100, plot_interval=1000, save_dir='logs'):
        self.num_pos_o2m_per_batch = 0
        self.num_pos_o2o_per_batch = 0
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.save_dir = save_dir
        self.iterations = []
        self.difference_values = []
        self.plot_iterations = []
        self.plot_avg_differences = []
        self._resolved_save_dir = None
        
    def _get_save_dir(self, runner):
        """Get the resolved save directory, creating it if necessary."""
        if self._resolved_save_dir is None:
            # If save_dir is relative, resolve it relative to work_dir
            if not os.path.isabs(self.save_dir):
                self._resolved_save_dir = os.path.join(runner.work_dir, self.save_dir)
            else:
                self._resolved_save_dir = self.save_dir
            
            # Create save directory if it doesn't exist
            os.makedirs(self._resolved_save_dir, exist_ok=True)
            
        return self._resolved_save_dir

    def after_train_iter(self, runner):
        cur_iter = runner.iter
        
        if cur_iter % self.log_interval == 0:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
                self.num_pos_o2m_per_batch = model.pts_bbox_head.num_pos_o2m_per_batch
                self.num_pos_o2o_per_batch = model.pts_bbox_head.num_pos_o2o_per_batch
            
            # Calculate difference
            difference = self.num_pos_o2m_per_batch - self.num_pos_o2o_per_batch
            
            # Store values
            self.iterations.append(cur_iter)
            self.difference_values.append(difference)
            
            # Get resolved save directory
            save_dir = self._get_save_dir(runner)
            
            # Save to file
            log_file = f"{save_dir}/pos_values_log.txt"
            with open(log_file, 'a') as f:
                f.write(f"Iter: {cur_iter}, O2M: {self.num_pos_o2m_per_batch}, O2O: {self.num_pos_o2o_per_batch}, Diff: {difference}\n")
            
            # Plot and save
            if cur_iter % self.plot_interval == 0 and len(self.difference_values) > 0:
                # Calculate average of unplotted points since last plot
                points_per_plot = self.plot_interval // self.log_interval
                if len(self.difference_values) >= points_per_plot:
                    # Get the last N unplotted points to average
                    recent_values = self.difference_values[-points_per_plot:]
                    avg_difference = sum(recent_values) / len(recent_values)
                
                    self.plot_iterations.append(cur_iter)
                    self.plot_avg_differences.append(avg_difference)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(self.plot_iterations, self.plot_avg_differences, 'g-', label='O2M - O2O')
                    plt.xlabel('Iteration')
                    plt.ylabel('Difference (O2M - O2O)')
                    batch_size = 16
                    plt.title(f'Difference Between O2M and O2O Positive Samples Over Training (Batch: {batch_size})')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"{save_dir}/pos_values_plot.png")
                    plt.close()

@HOOKS.register_module()
class IterAdaptiveRange(Hook):
    def __init__(self, k_range_scaling=3, min_k=0):
        self.k_range_scaling = k_range_scaling
        self.min_k = min_k

    def before_train_iter(self, runner, decrease_k=True):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        
        # Get current iteration and total iterations
        cur_iter = runner.iter
        max_iters = runner.max_iters
        
        cosine = math.cos(math.pi * cur_iter / max_iters)
        factor = 0.5 * (1 + cosine) if decrease_k else 0.5 * (1 - cosine)
        k_value = self.k_range_scaling * factor
        k_value = max(k_value, self.min_k)
        model.pts_bbox_head.k_range_scaling = k_value

@HOOKS.register_module()
class VisualizeKRangeHook(Hook):
    def __init__(self,plot_interval=1000, save_dir='logs'):
        self.plot_interval = plot_interval
        self.save_dir = save_dir
        self.iterations = []
        self.k_values = []
        self._resolved_save_dir = None
        
    def _get_save_dir(self, runner):
        """Get the resolved save directory, creating it if necessary."""
        if self._resolved_save_dir is None:
            if not os.path.isabs(self.save_dir):
                self._resolved_save_dir = os.path.join(runner.work_dir, self.save_dir)
            else:
                self._resolved_save_dir = self.save_dir
            
            os.makedirs(self._resolved_save_dir, exist_ok=True)
            
        return self._resolved_save_dir

    def after_train_iter(self, runner):
        cur_iter = runner.iter
        
        if cur_iter % self.plot_interval == 0:
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            
            k_value = getattr(model.pts_bbox_head, 'k_range_scaling', 0)
            
            self.iterations.append(cur_iter)
            self.k_values.append(k_value)
            
            # Plot and save
            if cur_iter % self.plot_interval == 0 and len(self.k_values) > 0:
                save_dir = self._get_save_dir(runner)
                
                plt.figure(figsize=(10, 6))
                plt.plot(self.iterations, self.k_values, 'b-', label='k_range_scaling')
                plt.xlabel('Iteration')
                plt.ylabel('k_range_scaling Value')
                plt.title('k_range_scaling Value Over Training')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{save_dir}/k_range_scaling_plot.png")
                plt.close()

@HOOKS.register_module()
class TemperatureAnnealingHook(Hook):
    def __init__(self, switch_iter=60000, switch_temp=0.1):
        self.switch_iter = switch_iter  # Iteration to switch temperature
        self.switch_temp = switch_temp  # Temperature value after switch

    def before_train_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        
        cur_iter = runner.iter
        
        if cur_iter == self.switch_iter:
            model.pts_bbox_head.distance_temp = self.switch_temp

@HOOKS.register_module()
class VisualizeMatchingHook(Hook):
    def __init__(self,plot_interval=1000, save_dir='logs'):
        self.plot_interval = plot_interval
        self.save_dir = save_dir
        self._resolved_save_dir = None
    
    def _get_save_dir(self, runner):
        """Get the resolved save directory, creating it if necessary."""
        if self._resolved_save_dir is None:
            if not os.path.isabs(self.save_dir):
                self._resolved_save_dir = os.path.join(runner.work_dir, self.save_dir)
            else:
                self._resolved_save_dir = self.save_dir
            
            os.makedirs(self._resolved_save_dir, exist_ok=True)
            
        return self._resolved_save_dir

    def after_train_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter % self.plot_interval == 0: # and cur_iter > 0
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            
            box_preds = model.pts_bbox_head.vis_bbox_preds_list
            box_targets = model.pts_bbox_head.vis_bbox_target_list

            if not box_preds or not box_targets:
                return

            save_dir = self._get_save_dir(runner)

            def boxes_to_bev_corners(boxes: torch.Tensor):
                """Convert boxes (cx, cy, cz, w, l, h, yaw[, ...]) to BEV 4-corner polygons (N,4,2).
                Uses a stable base template to avoid stacking shape pitfalls."""
                if boxes.numel() == 0:
                    return boxes.new_zeros((0,4,2))
                cx = boxes[:,0]; cy = boxes[:,1]; w = boxes[:,3]; l = boxes[:,4]; yaw = boxes[:,6]
                base = boxes.new_tensor([
                    [-0.5, -0.5],
                    [-0.5,  0.5],
                    [ 0.5,  0.5],
                    [ 0.5, -0.5],
                ])  # (4,2)
                scale = torch.stack([w, l], -1).unsqueeze(1)  # (N,1,2)
                pts = base.unsqueeze(0) * scale  # (N,4,2)
                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)
                rot = torch.stack([
                    torch.stack([cos_yaw, -sin_yaw], -1),
                    torch.stack([sin_yaw,  cos_yaw], -1)
                ], -2)  # (N,2,2)
                rotated = torch.matmul(pts, rot.transpose(-1,-2))  # (N,4,2)
                centers = torch.stack([cx, cy], -1).unsqueeze(1)
                return rotated + centers

            def deduplicate_boxes(gt_boxes: torch.Tensor, eps: float = 1e-3):
                """Return indices of unique gt boxes (first occurrence) based on (cx,cy,w,l,yaw)."""
                if gt_boxes.numel() == 0:
                    return []
                key = gt_boxes[:, [0,1,3,4,6]]  # select key dims
                rounded = torch.round(key / eps)  # scale then round
                seen = {}
                unique_idx = []
                for i, k in enumerate(rounded.tolist()):
                    t = tuple(k)
                    if t not in seen:
                        seen[t] = i
                        unique_idx.append(i)
                return unique_idx

            # Plot at most 4 samples
            num_samples = min(4, len(box_preds))
            cmap = plt.get_cmap('tab20')

            for b in range(num_samples):
                preds_b = box_preds[b].detach().cpu()
                targets_b = box_targets[b].detach().cpu()
                if targets_b.ndim == 1:
                    targets_b = targets_b.unsqueeze(0)
                # Ensure equal length (they should be); keep unmatched preds by not truncating preds_b
                if preds_b.shape[0] != targets_b.shape[0]:
                    m = min(preds_b.shape[0], targets_b.shape[0])
                    preds_b = preds_b[:m]
                    targets_b = targets_b[:m]

                # Identify matched (non-zero) targets and unmatched predictions
                pos_mask = (targets_b.abs().sum(dim=-1) > 0)
                preds_pos = preds_b[pos_mask]
                targets_pos = targets_b[pos_mask]
                preds_unmatched = preds_b[~pos_mask]

                # Deduplicate GT (due to one-to-many). Keep first occurrence.
                if targets_pos.numel() > 0:
                    unique_gt_indices = deduplicate_boxes(targets_pos)
                    unique_gt_boxes = targets_pos[unique_gt_indices]
                    # Group mapping for predictions tied to GT
                    gt_keys = unique_gt_boxes[:, [0,1,3,4,6]]
                    pred_keys = targets_pos[:, [0,1,3,4,6]]
                    group_map = []
                    for pk in pred_keys:
                        d = (gt_keys - pk).abs().sum(dim=-1)
                        group_map.append(int(d.argmin().item()))
                    gt_corners = boxes_to_bev_corners(unique_gt_boxes)
                    pred_corners = boxes_to_bev_corners(preds_pos)
                else:
                    unique_gt_boxes = targets_pos.new_zeros((0, targets_b.shape[-1]))
                    gt_corners = boxes_to_bev_corners(unique_gt_boxes)
                    pred_corners = boxes_to_bev_corners(preds_pos)
                    group_map = []

                unmatched_corners = boxes_to_bev_corners(preds_unmatched)

                # Skip entirely empty sample
                if preds_pos.numel() == 0 and preds_unmatched.numel() == 0:
                    continue

                def _plot(ax, include_unmatched: bool):
                    handles, labels = [], []
                    added_gt_legend = False
                    added_matched_legend = False
                    # GT & matched
                    for gi in range(gt_corners.shape[0]):
                        gt_poly = gt_corners[gi].detach().cpu().numpy()
                        color = cmap(gi % cmap.N)
                        x_seq = gt_poly[:,0].tolist() + [gt_poly[0,0]]
                        y_seq = gt_poly[:,1].tolist() + [gt_poly[0,1]]
                        ax.plot(x_seq, y_seq, color=color, linewidth=1.0, alpha=1.0)
                        ax.fill(x_seq, y_seq, color=color, alpha=0.15)
                        if not added_gt_legend:
                            gt_handle, = ax.plot([], [], color='k', linewidth=1.0, alpha=1.0, label='GT Box')
                            handles.append(gt_handle); labels.append('GT Box'); added_gt_legend = True
                        for pi, grp in enumerate(group_map):
                            if grp != gi: continue
                            pc = pred_corners[pi].detach().cpu().numpy()
                            px = pc[:,0].tolist() + [pc[0,0]]
                            py = pc[:,1].tolist() + [pc[0,1]]
                            ax.plot(px, py, color=color, linewidth=1.0, alpha=0.7, linestyle='--')
                            if not added_matched_legend:
                                matched_handle, = ax.plot([], [], color='k', linewidth=1.0, alpha=0.7, linestyle='--', label='Matched Pred')
                                handles.append(matched_handle); labels.append('Matched Pred'); added_matched_legend = True
                    if include_unmatched and unmatched_corners.numel() > 0:
                        um_color = 'black'
                        for ui in range(unmatched_corners.shape[0]):
                            uc = unmatched_corners[ui].detach().cpu().numpy()
                            ux = uc[:,0].tolist() + [uc[0,0]]
                            uy = uc[:,1].tolist() + [uc[0,1]]
                            ax.plot(ux, uy, color=um_color, linewidth=0.5, alpha=0.5, linestyle=':')
                        dummy_um, = ax.plot([], [], color='black', linewidth=0.5, alpha=0.5, linestyle=':', label='Unmatched Pred')
                        handles.append(dummy_um); labels.append('Unmatched Pred')
                    # limits
                    all_sets = []
                    all_sets.append(gt_corners.reshape(-1,2)) if gt_corners.numel() else None
                    all_sets.append(pred_corners.reshape(-1,2)) if pred_corners.numel() else None
                    if include_unmatched and unmatched_corners.numel():
                        all_sets.append(unmatched_corners.reshape(-1,2))
                    # Fixed global limits for consistent BEV framing
                    ax.set_xlim([-60, 60])
                    ax.set_ylim([-60, 60])
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
                    suf = 'with unmatched' if include_unmatched else 'matched only'
                    ax.set_title(f'Iter {cur_iter} Sample {b}')
                    if handles:
                        ax.legend(handles, labels, loc='upper right', fontsize=8)
                    ax.grid(True, linestyle='--', alpha=0.3)

                # Plot 1: matched only
                fig1, ax1 = plt.subplots(figsize=(6,6))
                _plot(ax1, include_unmatched=False)
                out_path1 = os.path.join(save_dir, f'iter{cur_iter:06d}_sample{b}.png')
                fig1.tight_layout(); fig1.savefig(out_path1, dpi=150); plt.close(fig1)

                # Plot 2: with unmatched
                fig2, ax2 = plt.subplots(figsize=(6,6))
                _plot(ax2, include_unmatched=True)
                out_path2 = os.path.join(save_dir, f'iter{cur_iter:06d}_sample{b}_with_unmatched.png')
                fig2.tight_layout(); fig2.savefig(out_path2, dpi=150); plt.close(fig2)

