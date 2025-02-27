from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Union, DummyOnPolicyBuffer
from xuance.common.memory_tools import TransitionBuffer, DummyOnPolicyBuffer_Atari
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
        Buffer = DummyOnPolicyBuffer_Atari if self.atari else TransitionBuffer
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
                       prev_rssm_states: List[RSSMDiscState]) -> dict:
        """Returns actions and dists.
        Parameters:
            observations (np.ndarray): The observation.
            prev_actions (np.ndarray): The previous actions.
            prev_nonterm (np.ndarray): The previous nonterm.
            prev_rssm_states (List[RSSMDiscState]): The previous rssm states.
        Returns:
            actions: The actions to be executed.
            dists: The policy distributions.
        """
        actions = []
        act_dists = []
        for i in range(self.envs.num_envs):
            act_dist, posterior_rssm_state = self.policy(observations[i], prev_actions[i], prev_nonterm[i],
                                                         prev_rssm_states[i])
            action = act_dist.sample()  # one-hot actions
            # action = action.long().numpy()[0]  # 目前 batch 只有 1
            action = action.long().cpu().numpy()[0]  # 目前 batch 只有 1
            actions.append(action)
            act_dists.append(act_dist)
        return {"actions": actions, "dists": act_dists}

    def train_epochs(self, n_epochs: int = 1) -> dict:
        train_info = {}
        for _ in range(n_epochs):
            for start in range(0, self.buffer_size, self.batch_size):
                samples = self.memory.sample_traj(self.config.batch_size, self.config.seq_len)
                obs, act, rew, term = samples
                train_info = self.learner.update(
                    obs=obs,
                    act=act,
                    rew=rew,
                    noterm=1 - term
                )
        return train_info

    def train(self, train_steps):
        obs = self.envs.buf_obs
        """create rssm_states and prev_actions"""
        prev_rssm_states = [self.representation.RSSM.init_rssm_state(1, self.device) for _ in range(self.envs.num_envs)]
        prev_actions = np.zeros([self.envs.num_envs, self.envs.action_space.n])
        prev_done = np.zeros(self.envs.num_envs, dtype=np.bool8)
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)  # obs: (10, ~)
            with torch.no_grad():
                policy_out = self.dreamer_action(obs, prev_actions, ~prev_done, prev_rssm_states)
            one_hot_acts, dists = policy_out['actions'], policy_out['dists']
            # if not hasattr(self.config, 'action') or not self.config.action == "one-hot":
            acts = np.array(one_hot_acts).argmax(axis=1)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            """
                不需要存 value 和 log_p, 因为 actor_critic update 时只需一个 obs 即可生成完整路径
                model_loss = obs_loss + reward_loss + term_loss + kl_loss; (obs, act, rew, term)
                actor_critic_loss = 
                    actor_loss(rho * reinforce_loss + (1 - rho) * dynamics_loss + entropy_loss) + 
                    critic_loss(value_loss); (posterior)
            """
            """注意这里存的是 (act, rew, next_obs, term)"""
            self.memory.store(next_obs, acts, self._process_reward(rewards), None, (terminals.any() or trunctions.any()), None)
            # 至少存了 seq_len 的数据
            if _ >= self.config.seq_len and _ % self.config.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)

            if self.learner.iterations % self.config.soft_update_frequency == 0:
                self.policy.soft_update(tau=self.config.tau)


            # 为什么满了要清空??
            # if self.memory.full:
            #     train_info = self.train_epochs(n_epochs=self.n_epochs)
            #     self.log_infos(train_info, self.current_step)
            #     self.memory.clear()

            self.returns = self.gamma * self.returns + rewards  # discounted return
            obs = deepcopy(next_obs)
            prev_actions = deepcopy(one_hot_acts)

            # TODO
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    """added after noterm_loss checked"""
                    prev_done[i] = terminals[i] or trunctions[i]
                    """reset rssm_state and prev_action when env terminate"""
                    prev_rssm_states[i] = self.representation.RSSM.init_rssm_state(1, self.device)
                    prev_actions[i] = np.zeros(self.envs.action_space.n)
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~trunctions[i]):
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
