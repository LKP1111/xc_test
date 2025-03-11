from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union, DummyOnPolicyBuffer
from xuance.common.memory_tools import TransitionBuffer, DummyOnPolicyBuffer_Atari, TransitionBuffer_Atari
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.representations.wm_dreamer.rssm_utils import RSSMDiscState
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OnPolicyAgent
from xuance.torch.utils import split_distributions
from gym.spaces import Dict, Space
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import destroy_process_group
from xuance.common import get_time_string, create_directory, RunningMeanStd, space2shape, EPS, Optional, Union
from xuance.torch import REGISTRY_Representation, REGISTRY_Learners, Module


# Copy from PPOCLIP_Agent
class DreamerV2_Agent(OnPolicyAgent):
    """The implementation of DreamerV2 agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(DreamerV2_Agent, self).__init__(config, envs)
        self.memory = self._build_memory()
        """n_envs -> representation"""
        self.config.n_envs = self.envs.num_envs
        self.representation = self._build_representation(self.config.representation, self.observation_space,
                                                         self.config)
        self.policy = self._build_policy()
        self.learner = self._build_learner(self.config, self.policy)

    def _build_memory(self, auxiliary_info_shape=None):
        self.atari = True if self.config.env_name == "Atari" else False
        # TODO
        Buffer = TransitionBuffer_Atari if self.atari else TransitionBuffer
        self.buffer_size = self.n_envs * self.horizon_size
        self.batch_size = self.buffer_size // self.n_minibatch
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            horizon_size=self.horizon_size,
                            use_gae=self.config.use_gae,
                            use_advnorm=self.config.use_advnorm,
                            gamma=self.gamma,
                            gae_lam=self.gae_lam)
        return Buffer(**input_buffer)

    def _build_representation(self, representation_key: str,
                              input_space: Optional[Space],
                              config: Namespace) -> Module:
        """
        Build representation for policies.

        Parameters:
            representation_key (str): The selection of representation, e.g., "Basic_MLP", "Basic_RNN", etc.
            input_space (Optional[Space]): The space of input tensors.
            config: The configurations for creating the representation module.

        Returns:
            representation (Module): The representation Module.
        """
        config.obs_shape = self.envs.obs_shape
        config.action_size = self.action_space.n
        representation = REGISTRY_Representation[representation_key](config)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        # build policy.
        if self.config.policy == "Categorical_DreamerV2":
            policy = REGISTRY_Policy["Categorical_DreamerV2"](
                action_space=self.action_space, representation=self.representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training)
        elif self.config.policy == "Gaussian_DreamerV2":
            policy = REGISTRY_Policy["Gaussian_DreamerV2"](
                action_space=self.action_space, representation=self.representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                activation_action=ActivationFunctions[self.config.activation_action])
        else:
            raise AttributeError(f"DreamerV2 currently does not support the policy named {self.config.policy}.")
        return policy

    def dreamer_action(self, observations: np.ndarray,
                       prev_actions: np.ndarray,
                       prev_nonterm: np.ndarray,
                       prev_rssm_states: List[RSSMDiscState],
                       num_envs: int) -> dict:
        """Returns actions and rssm_state.
            prev_rssm_states(h_0) change to rssm_states(h_1) through prev_actions(a_0)
            rssm_states(h_1) contain information of current observations(x_1).
        Parameters:
            observations (np.ndarray): The observation.
            prev_actions (np.ndarray): The previous actions.
            prev_nonterm (np.ndarray): The previous nonterm.
            prev_rssm_states (List[RSSMDiscState]): The previous rssm states.
            num_envs (int): The number of environments.
        Returns:
            actions: The actions to be executed.
            rssm_state: The current posterior_rssm_state after prev_actions.
        """
        actions = []
        rssm_states = []
        for i in range(num_envs):
            act_dist, posterior_rssm_state = self.policy(observations[i], prev_actions[i], prev_nonterm[i],
                                                         prev_rssm_states[i])
            action = act_dist.sample()  # one-hot actions
            # action = action.long().numpy()[0]  # 目前 batch 只有 1
            action = action.long().cpu().numpy()[0]  # 目前 batch 只有 1
            actions.append(action)
            rssm_states.append(posterior_rssm_state)
        return {"actions": actions, "rssm_states": rssm_states}

    def train_epochs(self, n_epochs: int = 1) -> dict:
        train_info = {}
        for _ in range(n_epochs):
            for start in range(0, self.buffer_size, self.batch_size):
                samples = self.memory.sample_traj(self.config.batch_size, self.config.seq_len)
                obs, act, rew, term = samples
                # if _ == 4:  # kl_loss -> inf  # TODO
                #     print()
                train_info = self.learner.update(
                    obs=obs,
                    act=act,
                    rew=rew,
                    noterm=1 - term
                )
        return train_info

    def train(self, train_steps):
        obs = self.envs.buf_obs
        """prev_rssm_states get prev_done through prev_actions"""
        prev_rssm_states = [self.representation.RSSM.init_rssm_state(1, self.device) for _ in range(self.envs.num_envs)]
        prev_actions = np.zeros([self.envs.num_envs, self.envs.action_space.n])
        prev_done = np.ones(self.envs.num_envs, dtype=np.bool8)
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)  # obs: (10, ~)
            with torch.no_grad():
                policy_out = self.dreamer_action(obs, prev_actions, np.logical_not(prev_done), prev_rssm_states, self.envs.num_envs)
            one_hot_acts, rssm_states = policy_out['actions'], policy_out['rssm_states']
            # if not hasattr(self.config, 'action') or not self.config.action == "one-hot":
            acts = np.array(one_hot_acts).argmax(axis=1)
            next_obs, rewards, terminals, truncations, infos = self.envs.step(acts)
            """
            no need to store value and log_p, because the full traj can be imagined by single obs
            (act, rew, next_obs, term)
            normalization of next_obs & rew
            """
            self.memory.store(self._process_observation(next_obs), acts,
                              self._process_reward(rewards),
                              None, np.logical_or(terminals, truncations), None)
            # 至少存了 seq_len 的数据
            if _ >= self.config.seq_len and _ % self.config.training_frequency == 0:
                # if _ == 100:  # kl_loss -> inf  # TODO
                #     print()
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)

            if _ > 0 and _ % self.config.soft_update_frequency == 0:
                self.policy.soft_update(tau=self.config.tau)

            self.returns = self.gamma * self.returns + rewards  # discounted return
            obs = deepcopy(next_obs)
            """update prev variables!!!!!!"""
            prev_rssm_states = deepcopy(rssm_states)
            prev_actions = deepcopy(one_hot_acts)
            prev_done = np.logical_or(terminals, truncations)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    """reset rssm_state and prev_action when env terminate"""
                    prev_rssm_states[i] = self.representation.RSSM.init_rssm_state(1, self.device)
                    prev_actions[i] = np.zeros(self.envs.action_space.n)
                    prev_done[i] = True
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info["Episode-Steps/env-%d" % i] = infos[i]["episode_step"]
                            step_info["Train-Episode-Rewards/env-%d" % i] = infos[i]["episode_score"]
                        else:
                            step_info["Episode-Steps"] = {"env-%d" % i: infos[i]["episode_step"]}
                            step_info["Train-Episode-Rewards"] = {"env-%d" % i: infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)
            self.current_step += self.n_envs


    def test(self, env_fn, test_episodes: int) -> list:
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        """prev_rssm_states get prev_done through prev_actions"""
        prev_rssm_states = [self.representation.RSSM.init_rssm_state(1, self.device) for _ in range(num_envs)]
        prev_actions = np.zeros([num_envs, test_envs.action_space.n])
        prev_done = np.ones(num_envs, dtype=np.bool8)
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            with torch.no_grad():
                policy_out = self.dreamer_action(obs, prev_actions, np.logical_not(prev_done), prev_rssm_states, num_envs)
            one_hot_acts, rssm_states = policy_out['actions'], policy_out['rssm_states']
            acts = np.array(one_hot_acts).argmax(axis=1)
            next_obs, rewards, terminals, truncations, infos = test_envs.step(acts)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
            obs = deepcopy(next_obs)
            """update prev variables!!!!!!"""
            prev_rssm_states = deepcopy(rssm_states)
            prev_actions = deepcopy(one_hot_acts)
            prev_done = np.logical_or(terminals, truncations)
            for i in range(num_envs):
                if terminals[i] or truncations[i]:
                    """reset rssm_state and prev_action when env terminate"""
                    prev_rssm_states[i] = self.representation.RSSM.init_rssm_state(1, self.device)
                    prev_actions[i] = np.zeros(test_envs.action_space.n)
                    prev_done[i] = True
                    if self.atari and (~truncations[i]):
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

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)
        test_envs.close()
        return scores
