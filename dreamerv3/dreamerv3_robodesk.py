import argparse
from typing import Union

import numpy as np
from copy import deepcopy
from gym.spaces import Box
from xuance.torch.utils.operations import set_seed
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, RawEnvironment, REGISTRY_ENV
from common import DreamerV3Agent
import robodesk

import warnings
# 忽略所有弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MyNewEnv(RawEnvironment):
    def __init__(self, env_config):
        super(MyNewEnv, self).__init__()
        self.env_id = env_config.env_id
        self.is_render = env_config.render
        self.render_mode = env_config.render_mode
        self.env = robodesk.RoboDesk(
            task=self.env_id,
            reward='dense',
            action_repeat=2,
            episode_length=500,
            image_size=64)

        self.observation_space = self.env.observation_space['image']  # Box(0, 255, (64, 64, 3), uint8)
        self.action_space = self.env.action_space
        self.max_episode_steps = self.env.episode_length
        self._current_step = 0


    def reset(self, **kwargs):
        self._current_step = 0
        obs = self.env.reset()
        assert isinstance(obs, dict)
        return obs['image'], {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._current_step += 1
        # truncated = False if self._current_step < self.max_episode_steps else True
        return obs['image'], reward, done, done, info

    # def render(self, *args, **kwargs):
    #     return self.env.render(self.render_mode)
    def render(self, *args, **kwargs):
        return self.env.render(self.render_mode)

    def close(self):
        return


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for Robodesk.")
    parser.add_argument("--env-id", type=str, default="open_slide")
    parser.add_argument("--log-dir", type=str, default="./logs/open_slide/")
    parser.add_argument("--model-dir", type=str, default="./models/open_slide/")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--render_mode", type=str, default='rgb_array')

    # atari5M, ratio=0.25, gradient_step=1250k
    # parser.add_argument("--running-steps", type=int, default=5_000_000)  # 5M
    # parser.add_argument("--eval-interval", type=int, default=100_000)
    # parser.add_argument("--replay-ratio", type=int, default=0.25)

    # atari1M, ratio=1/8=0.125, gradient_step=125k
    parser.add_argument("--running-steps", type=int, default=1_000_000)  # 1M
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--replay-ratio", type=int, default=0.125)

    # atari100k, ratio=1, gradient_step=100k
    # parser.add_argument("--running-steps", type=int, default=100_000)  # 100k
    # parser.add_argument("--eval-interval", type=int, default=2_000)  # 50 logs
    # parser.add_argument("--replay-ratio", type=int, default=1)

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=4)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    # render test
    # parser.add_argument("--env_seed", type=int, default=1)
    # parser.add_argument("--render", type=bool, default=True)
    # parser.add_argument("--render_mode", type=str, default='human')
    # parser.add_argument("--parallels", type=int, default=1)
    # parser.add_argument("--test-episode", type=int, default=1)
    # parser.add_argument("--test", type=int, default=1)
    # parser.add_argument("--benchmark", type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    # print(sys.path)  # python path
    parser = parse_args()
    configs_dict = get_configs(file_dir="config/atari.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_ENV[configs.env_name] = MyNewEnv
    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DreamerV3Agent(config=configs, envs=envs)
    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)


        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            can_save = np.mean(test_scores) > best_scores_info["mean"]
            can_save |= (abs(np.mean(test_scores) - best_scores_info["mean"]) < 1e-6
                         and np.std(test_scores) < best_scores_info["std"])
            if can_save:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            model = None
            # model = 'seed_1_2025_0324_100206'
            Agent.load_model(path=Agent.model_dir_load, model=model)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f'scores: {scores}')
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()


"""
50m_sheeprl_time
    atari5M, ratio=0.03125, gradient_step=156.25k, 14h
    atari1M, ratio=1/8=0.125, gradient_step=125k, 11h
    atari100k, ratio=1, gradient_step=100k, 5.5h
"""