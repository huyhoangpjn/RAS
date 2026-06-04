
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.mmdet3d_plugin.core.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
import torch
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv import Config
import copy

# Non-distributed eval hook
@HOOKS.register_module()
class EvalLossHook(Hook):
    '''
        Get losses over the entire val set, non-distributed and sequence training mode only (Config in data pipeline)
    '''
    def __init__(self, data, interval=1):
        self.interval = interval
        self.data = Config(data)
        self.dataloader = self.build_dataloader()

    def build_dataloader(self):
        dataset = custom_build_dataset(self.data.val)
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1, # self.data.samples_per_gpu, since it just val -> batch = 1 is ok
            workers_per_gpu=self.data.workers_per_gpu,
            shuffle=False, #skip frame during training
            dist=False,
            shuffler_sampler=self.data.shuffler_sampler,
            nonshuffler_sampler=self.data.nonshuffler_sampler,
            runner_type=dict(type='EpochBasedRunner', max_epochs=1)
            )
        return dataloader

    def after_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        model = copy.deepcopy(runner.model).eval()
        # check mem
        # print(model.module.pts_bbox_head.memory_embedding.shape)
        model.module.pts_bbox_head.reset_memory()
        # print(runner.model.module.pts_bbox_head.memory_embedding.shape)
        losses = dict()
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                loss = model(return_loss=True, **data)
                for key, value in loss.items():
                    if key not in losses: losses[key] = value
                    else: losses[key] += value

        runner.log_buffer.clear_output()
        for key, value in losses.items():
            #runner.log_buffer.update({f'val/{key}': (losses[key]/(i+1)).cpu()}, runner.iter)
            runner.log_buffer.output[f'val/{key}'] = float((losses[key]/(i+1)).cpu())
        runner.log_buffer.clear_output()