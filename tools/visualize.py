import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = True
out_dir = './result_vis/r50_o2m_thr_03/'
result_json = './test/repdetr3d_r50_428q_nui_60e_train_decoupled/Mon_Oct_27_16_52_05_2025/pts_bbox/results_nusc'
# result_json = './val/notebooks/pretrained/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_retrain/Wed_Jul_30_22_20_05_2025/pts_bbox/results_nusc'
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.35, version='v1.0-trainval' ) # version='v1.0-trainval' 
else:
    nusc = NuScenes(dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.3, version='v1.0-trainval' )

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens[:100]):
    if use_gt:
        nusc.render_sample(token, out_path = out_dir+token+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = out_dir+token+"_pred.png", verbose=False)

