from xuance.torch.agents import OffPolicyAgent

# REGISTRY
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import REGISTRY_Learners

from . import DreamerV3WorldModel  # '.': import from __init__
from . import DreamerV3Policy
from . import DreamerV3Learner

from tqdm import tqdm


# Step 3: Create the agent.
class DreamerV3Agent(OffPolicyAgent):
    def __init__(self, config, envs):
        super(DreamerV3Agent, self).__init__(config, envs)
        REGISTRY_Representation['DreamerV3WorldModel'] = DreamerV3WorldModel
        self.player = self._build_representation("DreamerV3WorldModel", self.observation_space, self.config)
        REGISTRY_Policy["DreamerV3Policy"] = DreamerV3Policy
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        REGISTRY_Learners['DreamerV3Agent'] = DreamerV3Learner  # Registry your pre-defined learner.
        self.learner = self._build_learner(self.config, self.policy)  # Build the learner.

    def _build_policy(self):
        policy = REGISTRY_Policy["DreamerV3Policy"](self.player, 64, self.action_space.n, self.config)
        return policy

    def train(self, train_steps):
        # self.player.init_states()  # TODO
        # for steps in range(train_steps):
        pass

    def test(self, env_fn, test_episodes: int) -> list:
        pass

