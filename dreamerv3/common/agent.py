from copy import deepcopy
from typing import List

import torch.nn

from xuance.torch.agents import OffPolicyAgent

# REGISTRY
from xuance.torch import REGISTRY_Representation, REGISTRY_Policy, REGISTRY_Learners, Module
from xuance.torch.utils import ActivationFunctions

# '.': import from __init__
from . import DreamerV3WorldModel
from . import DreamerV3Policy
from . import DreamerV3Learner
from . import SequentialReplayBuffer

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from argparse import Namespace
from xuance.common import space2shape, Optional

# Step 3: Create the agent.
class DreamerV3Agent(OffPolicyAgent):
    def __init__(self, config, envs):
        super(DreamerV3Agent, self).__init__(config, envs)

        # continuous or not
        self.is_continuous = isinstance(self.envs.action_space, gym.spaces.Box)
        self.is_multidiscrete = isinstance(self.envs.action_space, gym.spaces.MultiDiscrete)

        # ratio
        self.replay_ratio = self.config.replay_ratio
        self.current_step, self.gradient_step = 0, 0

        # REGISTRY & create: representation, policy, learner
        ActivationFunctions['silu'] = torch.nn.SiLU
        REGISTRY_Representation['DreamerV3WorldModel'] = DreamerV3WorldModel
        self.model = self._build_representation(representation_key="DreamerV3WorldModel",
                                                config=self.config, input_space=None)
        self.player = self.model.player
        REGISTRY_Policy["DreamerV3Policy"] = DreamerV3Policy
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        REGISTRY_Learners['DreamerV3Learner'] = DreamerV3Learner
        self.learner: DreamerV3Learner = self._build_learner(self.config, self.policy)

    def _build_representation(self, representation_key: str,
                              input_space: Optional[gym.spaces.Space],
                              config: Namespace) -> DreamerV3WorldModel:
        # specify the type in order to use code completion
        actions_dim = tuple(
            self.envs.action_space.shape
            if self.is_continuous else (
                self.envs.action_space.nvec.tolist() if self.is_multidiscrete else [self.envs.action_space.n]
            )
        )
        input_representations = dict(
            actions_dim=actions_dim,
            is_continuous=self.is_continuous,
            config=self.config,
            obs_space=self.envs.observation_space
        )
        representation = REGISTRY_Representation[representation_key](**input_representations)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_memory(self, auxiliary_info_shape=None) -> SequentialReplayBuffer:
        self.atari = True if self.config.env_name == "Atari" else False  # TODO atari
        # Buffer = DummyOffPolicyBuffer_Atari if self.atari else DummyOffPolicyBuffer
        Buffer = SequentialReplayBuffer
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size)
        return Buffer(**input_buffer)

    def _build_policy(self) -> DreamerV3Policy:
        return REGISTRY_Policy["DreamerV3Policy"](self.model, self.config)

    def action(self, observations: np.ndarray,
               test_mode: Optional[bool] = False) -> np.ndarray:
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        # actions_output = self.policy(observations)
        # [envs, *obs_shape] -> [1: batch, envs, *obs_shape]
        obs = torch.as_tensor(observations, device=self.device).unsqueeze(0)
        actions = self.player.get_actions(obs, greedy=test_mode, mask=None)[0][0]
        # ont-hot -> real_actions
        if not self.is_continuous:
            actions = actions.argmax(dim=1).detach().cpu().numpy()
        else:
            actions = actions.detach().cpu().numpy()  # TODO continuous action
        # TODO not test_mode exploration
        """
        for env_interaction: actions.shape, (envs, )
        """
        return actions

    def train_epochs(self, n_epochs: int = 1):
        train_info = {}
        for _ in range(n_epochs):
            samples = self.memory.sample(self.config.seq_len)
            train_info = self.learner.update(**samples)
        # train_info["epsilon-greedy"] = self.e_greedy
        # train_info["noise_scale"] = self.noise_scale
        return train_info

    def squeeze_and_store(self, x: List[np.ndarray]):
        # deal with xuance memory store
        self.memory.store(*(lambda t: [np.squeeze(_) for _ in t])(x))

    def train(self, train_steps):
        self.player.init_states()
        return_info = {}
        obs = self.envs.buf_obs  # (envs, *obs_shape)
        rews = np.zeros((self.envs.num_envs, 1))
        terms = np.zeros((self.envs.num_envs, 1))
        truncs = np.zeros((self.envs.num_envs, 1))
        is_first = np.ones_like(terms)
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)  # ?
            obs = self._process_observation(obs)  # obs_norm
            acts = self.action(obs, test_mode=False)
            """(o1, a1, r0, term0, trunc0, is_first1), act: not one-hot"""
            self.squeeze_and_store([obs, acts, self._process_reward(rews), terms, truncs, is_first])
            next_obs, rews, terms, truncs, infos = self.envs.step(acts)
            """
            set to zeros after the first step
            (o2, a1, r1, term1, trunc1, is_first2)
            """
            is_first = np.zeros_like(terms)
            obs = next_obs
            self.returns = self.gamma * self.returns + rews

            done_idxes = []
            for i in range(self.n_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):  # do not term until trunc
                        pass
                    else:
                        # carry the reset procedure to the outside
                        done_idxes.append(i)
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info[f"Episode-Steps/rank_{self.rank}/env-{i}"] = infos[i]["episode_step"]
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}/env-{i}"] = infos[i]["episode_score"]
                        else:
                            step_info[f"Episode-Steps/rank_{self.rank}"] = {f"env-{i}": infos[i]["episode_step"]}
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}"] = {
                                f"env-{i}": infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)
                        return_info.update(step_info)
            self.current_step += self.n_envs
            # self._update_explore_factor()

            # TODO when an env is done, one more frame need to be stored, which may cause problem to other envs
            if len(done_idxes) > 0:
                """
                store the last data and reset all
                (o_t, a_t = 0 for dones, r_{t-1}, term_{t-1}, trunc_{t-1}, is_first_t)
                """
                acts[done_idxes] = np.zeros((len(done_idxes), 1))
                self.squeeze_and_store([obs, acts, self._process_reward(rews), terms, truncs, is_first])
                obs[done_idxes] = infos[done_idxes]["reset_obs"]  # reset obs
                self.envs.buf_obs[done_idxes] = obs[done_idxes]
                rews[done_idxes] = np.zeros((len(done_idxes), 1))
                terms[done_idxes] = np.zeros((len(done_idxes), 1))
                truncs[done_idxes] = np.zeros((len(done_idxes), 1))
                is_first[done_idxes] = np.ones_like(terms)
                """reset DreamerV3 Player's states"""
                self.player.init_states(done_idxes)

            """
            start training 
            replay_ratio = self.gradient_step / self.current_step
            """
            if self.current_step > self.start_training:
                n_epochs = max(int(self.current_step * self.replay_ratio - self.gradient_step), 0)
                train_info = self.train_epochs(n_epochs=n_epochs)
                self.log_infos(train_info, self.current_step)
                return_info.update(train_info)

            return return_info

    def test(self, env_fn, test_episodes: int) -> list:
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        self.player.init_states(num_envs=num_envs)
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts = self.action(obs, test_mode=True)
            next_obs, rews, terms, truncs, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            done_idxes = []
            for i in range(num_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        scores.append(infos[i]["episode_score"])
                        current_episode += 1
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))

            self.player.init_states(reset_envs=done_idxes, num_envs=num_envs)

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores

