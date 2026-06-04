#!/usr/bin/env python
"""Quick test for VisualizeMatchingHook.after_train_iter logic.
Generates synthetic GT + prediction boxes including:
 - Duplicated GT boxes (one-to-many)
 - Matched predictions (noisy versions of GT)
 - Unmatched predictions (with zero targets)
Outputs BEV visualization PNGs in ./_vis_match_test
Run: python StreamPETR/tools/test_visualize_matching.py
"""
import os
import math
try:
    import torch
except ImportError:
    torch = None
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# -----------------------------
# Helpers reproduced from hook
# -----------------------------

if torch is not None:
    def boxes_to_bev_corners(boxes):
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
        corners = base.unsqueeze(0) * scale  # (N,4,2)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat = torch.stack([torch.stack([cos_yaw, -sin_yaw], -1),
                               torch.stack([sin_yaw,  cos_yaw], -1)], -2)  # (N,2,2)
        rotated = torch.matmul(corners, rot_mat.transpose(-1,-2))
        centers = torch.stack([cx, cy], -1).unsqueeze(1)
        return rotated + centers
    def deduplicate_boxes(gt_boxes, eps: float = 1e-3):
        if gt_boxes.numel() == 0:
            return []
        key = gt_boxes[:, [0,1,3,4,6]]
        rounded = torch.round(key / eps)
        seen = {}
        unique_idx = []
        for i, k in enumerate(rounded.tolist()):
            t = tuple(k)
            if t not in seen:
                seen[t] = i
                unique_idx.append(i)
        return unique_idx
else:
    print('Torch not installed; exiting test.')
    exit(0)

# -----------------------------
# Synthetic data generator
# -----------------------------

def make_box(cx, cy, yaw_deg, w=2.0, l=4.0, cz=0.0, h=1.5):
    yaw = math.radians(yaw_deg)
    return torch.tensor([cx, cy, cz, w, l, h, yaw], dtype=torch.float32)

def make_sample(seed=0):
    torch.manual_seed(seed)
    # unique GTs
    g1 = make_box(0.0, 0.0, 0)
    g2 = make_box(10.0, 5.0, 30)
    g3 = make_box(-6.0, 12.0, -45)
    # duplicates of g2 (one-to-many)
    dup1 = g2.clone()
    dup2 = g2.clone()
    # zeros to represent unmatched target slots
    zero = torch.zeros_like(g1)
    zero2 = torch.zeros_like(g1)
    targets = torch.stack([g1, g2, g3, dup1, dup2, zero, zero2])  # (7,7)
    # predictions: noisy matches for first five, random unmatched for last two
    noise = lambda b: b + torch.randn_like(b) * torch.tensor([0.2,0.2,0.0,0.05,0.05,0.0,0.05])
    p1 = noise(g1)
    p2 = noise(g2)
    p3 = noise(g3)
    p4 = noise(g2)
    p5 = noise(g2)
    unmatched1 = make_box(20.0, -10.0, 75, 3.0, 6.0)
    unmatched2 = make_box(-25.0, 18.0, -10, 2.5, 5.5)
    preds = torch.stack([p1, p2, p3, p4, p5, unmatched1, unmatched2])
    return preds, targets

# Create a batch of 2 samples
box_preds_list = []
box_targets_list = []
for s in range(2):
    p, t = make_sample(seed=s)
    box_preds_list.append(p)
    box_targets_list.append(t)

# -----------------------------
# Visualization logic (mirrors hook.after_train_iter)
# -----------------------------

def visualize_matching(cur_iter, box_preds, box_targets, out_dir='./_vis_match_test', max_samples=4):
    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.get_cmap('tab20')
    num_samples = min(max_samples, len(box_preds))
    for b in range(num_samples):
        preds_b = box_preds[b].clone()
        targets_b = box_targets[b].clone()
        if targets_b.ndim == 1:
            targets_b = targets_b.unsqueeze(0)
        if preds_b.shape[0] != targets_b.shape[0]:
            m = min(preds_b.shape[0], targets_b.shape[0])
            preds_b = preds_b[:m]
            targets_b = targets_b[:m]
        pos_mask = (targets_b.abs().sum(dim=-1) > 0)
        preds_pos = preds_b[pos_mask]
        targets_pos = targets_b[pos_mask]
        preds_unmatched = preds_b[~pos_mask]
        if targets_pos.numel() > 0:
            unique_gt_indices = deduplicate_boxes(targets_pos)
            unique_gt_boxes = targets_pos[unique_gt_indices]
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
        if preds_pos.numel() == 0 and preds_unmatched.numel() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6,6))
        handles = []
        labels = []
        for gi in range(gt_corners.shape[0]):
            gt_poly = gt_corners[gi]
            color = cmap(gi % cmap.N)
            x_pts = gt_poly[:,0].detach().cpu().tolist() + [gt_poly[0,0].item()]
            y_pts = gt_poly[:,1].detach().cpu().tolist() + [gt_poly[0,1].item()]
            gt_line, = ax.plot(x_pts, y_pts, color=color, linewidth=2.0, alpha=1.0)
            ax.fill(x_pts, y_pts, color=color, alpha=0.15)
            handles.append(gt_line); labels.append(f'GT {gi}')
            for pi, grp in enumerate(group_map):
                if grp != gi: continue
                pc = pred_corners[pi]
                px = pc[:,0].detach().cpu().tolist() + [pc[0,0].item()]
                py = pc[:,1].detach().cpu().tolist() + [pc[0,1].item()]
                ax.plot(px, py, color=color, linewidth=1.5, alpha=0.75, linestyle='--')
        if unmatched_corners.numel() > 0:
            um_color = 'black'
            for ui in range(unmatched_corners.shape[0]):
                uc = unmatched_corners[ui]
                ux = uc[:,0].detach().cpu().tolist() + [uc[0,0].item()]
                uy = uc[:,1].detach().cpu().tolist() + [uc[0,1].item()]
                ax.plot(ux, uy, color=um_color, linewidth=1.0, alpha=0.5, linestyle=':')
            dummy_um, = ax.plot([], [], color=um_color, linewidth=1.0, alpha=0.5, linestyle=':', label='Unmatched Pred')
            handles.append(dummy_um); labels.append('Unmatched Pred')
        all_sets = []
        if gt_corners.numel(): all_sets.append(gt_corners.reshape(-1,2))
        if pred_corners.numel(): all_sets.append(pred_corners.reshape(-1,2))
        if unmatched_corners.numel(): all_sets.append(unmatched_corners.reshape(-1,2))
        if all_sets:
            all_pts = torch.cat(all_sets, dim=0)
            min_xy, _ = all_pts.min(dim=0)
            max_xy, _ = all_pts.max(dim=0)
            pad = 2.0
            ax.set_xlim([min_xy[0]-pad, max_xy[0]+pad])
            ax.set_ylim([min_xy[1]-pad, max_xy[1]+pad])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'Test Iter {cur_iter} - Sample {b} (Unique GT: {len(gt_corners)})')
        if handles: ax.legend(handles, labels, loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        out_path = os.path.join(out_dir, f'test_match_iter{cur_iter:06d}_sample{b}.png')
        fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f'Saved {out_path}')

if __name__ == '__main__':
    if torch is None:
        print('Torch not available; skip.')
    else:
        visualize_matching(cur_iter=12345, box_preds=box_preds_list, box_targets=box_targets_list)
        print('Done.')
