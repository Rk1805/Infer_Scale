"""Fixed input scaling stored inside an SB3 checkpoint for portable inference."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ScaledObservations(BaseFeaturesExtractor):
    """Scale all 12 raw features without external VecNormalize state.

    Scales are engineering constants, not fitted using held-out observations.
    The registered buffer travels with the checkpoint and PPOPolicy loads it.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=12)
        self.register_buffer("scales", torch.tensor(
            [20, 20000, 20, 600, 350, 8, 8, 1, 5, 1, 10, 8],
            dtype=torch.float32,
        ))

    def forward(self, observations):
        return torch.clamp(observations / self.scales, 0, 10)
