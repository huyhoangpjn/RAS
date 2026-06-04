import torch.nn as nn
from mmdet.core import multi_apply

class SimpleDepthNet(nn.Module):

    def __init__(self, model_cfg, is_obj_depth=False, levels = 3):
        """
        Simple depth estimation module using 1 single scale feature
        Single feature from FPN (H,W,fpn_out_dim) -> some conv -> (H,W,D) with D is the number of depth bins
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        self.is_obj_depth = is_obj_depth
        depth_num_bins = int(model_cfg["num_depth_bins"])

        # Create modules
        input_dim = model_cfg["hidden_dim"]
        d_model = model_cfg["depthnet_dim"]
        
        _build_conv_layer = lambda dim0, dim1: nn.Sequential(
            nn.Conv2d(dim0, dim1, kernel_size=(3, 3), padding=1), nn.GroupNorm(32, dim1), nn.ReLU())
        
        self.conv_layer_num = 2-1     # default value
        if 'conv_layer_num' in model_cfg:
            self.conv_layer_num = model_cfg['conv_layer_num'] - 1
        if not is_obj_depth:
            conv_layers = []
            conv_layers.append(_build_conv_layer(input_dim, d_model))
            conv_layers += [_build_conv_layer(d_model, d_model) for _ in range(self.conv_layer_num)]
            self.depth_head = nn.Sequential(*conv_layers)
            self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))
        else:
            self.depth_head = nn.ModuleList()
            self.depth_classifier = nn.ModuleList()
            for _ in range(levels):
                conv_layers = []
                conv_layers.append(_build_conv_layer(input_dim, d_model))
                conv_layers += [_build_conv_layer(d_model, d_model) for _ in range(self.conv_layer_num)]
                self.depth_head.append(nn.Sequential(*conv_layers))
                self.depth_classifier.append(nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1)))
    
    def forward_single(self, feature, depth_head, depth_classifier):
        if feature.dim() == 5:
            b,n,c,h,w = feature.shape
            feature = feature.view(b*n, c, h, w)
        src = depth_head(feature)
        depth_logits = depth_classifier(src)
        return depth_logits

    def forward(self, feature):
        if not self.is_obj_depth:
            if feature.dim() == 5:
                b,n,c,h,w = feature.shape
                feature = feature.view(b*n, c, h, w)
            src = self.depth_head(feature)
            depth_logits = self.depth_classifier(src)
        else:
            depth_logits = [self.forward_single(feat, depth_head, depth_classifier) for feat, depth_head, depth_classifier in zip(feature, self.depth_head, self.depth_classifier)]
        return depth_logits