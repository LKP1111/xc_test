from torch import nn
import numpy as np
from argparse import Namespace
from . import dotdict

# Step 1: Create a policy.
class DreamerV3Policy(nn.Module):
    def __init__(self, model: nn.Module, config: Namespace):
        super(DreamerV3Policy, self).__init__()
        self.config = dotdict(vars(config))

        self.model = model
        self.world_model = self.model.world_model
        self.actor = self.model.actor
        self.critic = self.model.critic
        self.target_critic = self.model.target_critic

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        from obs to action

        Args:
            obs (np.ndarray)

        Returns:
            actions (np.ndarray)
        """

        pass

    def model_forward(self, x):
        pass

    def actor_critic_forward(self, x):
        pass
