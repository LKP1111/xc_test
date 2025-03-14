from xuance.torch.learners import Learner
import torch
from torch import nn

class DreamerV3Learner(Learner):
    def __init__(self, config, policy):
        super(DreamerV3Learner, self).__init__(config, policy)
        pass

    def update(self, **samples):

        pass
