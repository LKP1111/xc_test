# Test the policy-based algorithms with PyTorch.

from argparse import Namespace
from xuance import get_runner
import unittest

n_steps = 10000
device = 'cuda:0'
# device = 'cpu'
test_mode = False


class TestValueBaseAlgo(unittest.TestCase):
    def test_dreamerv2_discrete(self):
        args = Namespace(dl_toolbox='torch', device=device, running_steps=n_steps, test_mode=test_mode)
        runner = get_runner(method="dreamerv2", env='classic_control', env_id='CartPole-v1', parser_args=args)
        runner.run()


if __name__ == "__main__":
    unittest.main()
