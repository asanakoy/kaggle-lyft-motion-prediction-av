from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_common import build_backbone


class LyftMultiModelSimple(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()

        model_params = cfg["model_params"]

        self.backbone, backbone_out_features = build_backbone(
            model_params["model_architecture"], model_params["nb_raster_channels"]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.future_len = model_params["future_num_frames"]
        num_targets = 2 * self.future_len

        self.extend_agent_state = model_params.get("extend_agent_state", False)
        self.add_agent_state = model_params["add_agent_state"]
        self.add_agent_state_size = model_params["add_agent_state_size"]
        self.exclude_agents = model_params["exclude_agents"]
        self.use_center_point_only = model_params.get("use_center_point_only", False)
        self.all_points = model_params.get("use_all_points", False)
        self.num_modes = model_params.get("num_modes", 3)
        self.use_dyn_model = model_params.get("use_dyn_model", False)

        print('use dyn model:', self.use_dyn_model)

        if self.add_agent_state:
            self.agent_state_fc1 = nn.Linear(
                in_features=self.add_agent_state_size + backbone_out_features,
                out_features=1024,
            )
            self.agent_state_norm = nn.LayerNorm(1024)
            self.agent_state_fc2 = nn.Linear(in_features=1024, out_features=backbone_out_features)

        self.num_preds = num_targets * self.num_modes

        self.pred_pos = nn.Linear(backbone_out_features, out_features=self.num_preds)
        self.pred_mode_conf = nn.Linear(backbone_out_features, out_features=self.num_modes)

    def forward(self, x, x_state=None):
        if self.exclude_agents:
            x = x[:, 5:, :, :]
        x = self.backbone.forward_features(x)
        # print(x.shape)
        if self.use_center_point_only:
            h, w = x.shape[-2:]
            x = x[:, :, h // 2, w // 2]
        else:
            x = x[:, :, 1:-1, 1:-1]
            x = self.avgpool(x)

        img_out = torch.flatten(x, 1)

        if self.add_agent_state:
            x = torch.cat([img_out, x_state], dim=1)
            x1 = self.agent_state_fc1(x)
            x1 = F.relu(self.agent_state_norm(x1))
            x1 = self.agent_state_fc2(x1)

            x = img_out + x1
        else:
            x = img_out

        pred_pos = self.pred_pos(x).view(-1, self.num_modes, self.future_len, 2)

        confidences = torch.log_softmax(self.pred_mode_conf(x), dim=1)
        return pred_pos, confidences


class LyftMultiModelWeightedPool(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()

        model_params = cfg["model_params"]
        self.backbone, backbone_out_features = build_backbone(
            model_params["model_architecture"], model_params["nb_raster_channels"]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.future_len = model_params["future_num_frames"]
        num_targets = 2 * self.future_len

        features_w = model_params.get("features_w", 7)
        features_h = model_params.get("features_h", 7)

        self.block_weight = nn.Parameter(
            torch.ones(1, backbone_out_features, features_h, features_w, requires_grad=True),
            requires_grad=True
        )
        self.num_modes = model_params.get("num_modes", 3)
        self.num_preds = num_targets * self.num_modes
        self.pred_pos = nn.Linear(backbone_out_features, out_features=self.num_preds)
        self.pred_mode_conf = nn.Linear(backbone_out_features, out_features=self.num_modes)

    def forward(self, x, x_state=None):
        x = self.backbone.forward_features(x)
        x = x * self.block_weight
        x = self.avgpool(x)
        img_out = torch.flatten(x, 1)
        x = img_out

        pred_pos = self.pred_pos(x).view(-1, self.num_modes, self.future_len, 2)
        confidences = torch.log_softmax(self.pred_mode_conf(x), dim=1)
        return pred_pos, confidences

