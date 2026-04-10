"""Custom Sample Factory encoder for MaCA."""

from __future__ import annotations

import torch
from torch import nn

from sample_factory.algorithms.appo.model_utils import EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements


class MaCAEncoder(EncoderBase):
    """Compact CNN + measurements MLP encoder sized for a single 8 GB GPU."""

    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        activation = nonlinearity(cfg)

        self.conv_head = nn.Sequential(
            nn.Conv2d(obs_shape.obs[0], 16, kernel_size=5, stride=2, padding=2),
            activation,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            activation,
        )
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        measurement_input_size = 0
        if "measurements" in obs_shape:
            measurement_input_size += obs_shape.measurements[0]
        if "is_alive" in obs_shape:
            measurement_input_size += obs_shape.is_alive[0]

        self.measurement_head = None
        measurement_out_size = 0
        if measurement_input_size > 0:
            self.measurement_head = nn.Sequential(
                nn.Linear(measurement_input_size, 64),
                nonlinearity(cfg),
                nn.Linear(64, 64),
                nonlinearity(cfg),
            )
            measurement_out_size = calc_num_elements(self.measurement_head, (measurement_input_size,))

        self.init_fc_blocks(self.conv_head_out_size + measurement_out_size)

    def forward(self, obs_dict):
        x = self.conv_head(obs_dict["obs"].float())
        x = x.contiguous().view(-1, self.conv_head_out_size)

        if self.measurement_head is not None:
            features = [obs_dict["measurements"].float()]
            if "is_alive" in obs_dict:
                features.append(obs_dict["is_alive"].float())
            measurements = torch.cat(features, dim=1)
            measurements = self.measurement_head(measurements)
            x = torch.cat((x, measurements), dim=1)

        x = self.forward_fc_blocks(x)
        return x
