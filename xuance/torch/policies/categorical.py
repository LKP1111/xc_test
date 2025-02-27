import os
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete
from sympy import gamma

from xuance.common import Sequence, Optional, Callable, Union
from xuance.torch import Module, Tensor, DistributedDataParallel
from xuance.torch.utils import ModuleType
from .core import CategoricalActorNet as ActorNet
from .core import CategoricalActorNet_SAC as Actor_SAC
from .core import OneHotCategoricalActorNet, GaussianCriticNet
from .core import BasicQhead, CriticNet


def _init_layer(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorPolicy(Module):
    """
    Actor for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(ActorPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.actor = DistributedDataParallel(module=self.actor, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the hidden states, action distribution.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_dist: The distribution of actions output by actor.
        """
        outputs = self.representation(observation)
        a_dist = self.actor(outputs['state'])
        return outputs, a_dist, None


class ActorCriticPolicy(Module):
    """
    Actor-Critic for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(ActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.actor = DistributedDataParallel(module=self.actor, device_ids=[self.rank])
            self.critic = DistributedDataParallel(module=self.critic, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the hidden states, action distribution, and values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            outputs: The outputs of representation.
            a_dist: The distribution of actions output by actor.
            value: The state values output by critic.
        """
        outputs = self.representation(observation)
        a_dist = self.actor(outputs['state'])
        value = self.critic(outputs['state'])
        return outputs, a_dist, value[:, 0]


class PPGActorCritic(Module):
    """
    Actor-Critic for PPG with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(PPGActorCritic, self).__init__()
        self.action_dim = action_space.n
        self.actor_representation = representation
        self.critic_representation = deepcopy(representation)
        self.aux_critic_representation = deepcopy(representation)
        self.representation_info_shape = self.actor_representation.output_shapes

        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                normalize, initialize, activation, device)
        self.aux_critic = CriticNet(representation.output_shapes['state'][0], critic_hidden_size,
                                    normalize, initialize, activation, device)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.actor_representation._get_name() != "Basic_Identical":
                self.actor_representation = DistributedDataParallel(self.representation, device_ids=[self.rank])
            if self.critic_representation._get_name() != "Basic_Identical":
                self.critic_representation = DistributedDataParallel(self.representation, device_ids=[self.rank])
            if self.aux_critic_representation._get_name() != "Basic_Identical":
                self.aux_critic_representation = DistributedDataParallel(self.representation, device_ids=[self.rank])
            self.actor = DistributedDataParallel(self.actor, device_ids=[self.rank])
            self.critic = DistributedDataParallel(self.critic, device_ids=[self.rank])
            self.aux_critic = DistributedDataParallel(self.aux_critic, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the actors representation output, action distribution, values, and auxiliary values.

        Parameters:
            observation: The original observation of agent.

        Returns:
            policy_outputs: The outputs of actor representation.
            a_dist: The distribution of actions output by actor.
            value: The state values output by critic.
            aux_value: The auxiliary values output by aux_critic.
        """
        policy_outputs = self.actor_representation(observation)
        critic_outputs = self.critic_representation(observation)
        aux_critic_outputs = self.aux_critic_representation(observation)
        a_dist = self.actor(policy_outputs['state'])
        value = self.critic(critic_outputs['state'])
        aux_value = self.aux_critic(aux_critic_outputs['state'])
        return policy_outputs, a_dist, value[:, 0], aux_value[:, 0]


class SACDISPolicy(Module):
    """
    Actor-Critic for SAC with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(SACDISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation_info_shape = representation.output_shapes

        self.actor_representation = representation
        self.actor = Actor_SAC(representation.output_shapes['state'][0], self.action_dim, actor_hidden_size,
                               normalize, initialize, activation, device)

        self.critic_1_representation = deepcopy(representation)
        self.critic_1 = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation, device)
        self.critic_2_representation = deepcopy(representation)
        self.critic_2 = BasicQhead(representation.output_shapes['state'][0], self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation, device)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.actor_parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.critic_1_representation.parameters()) + list(
            self.critic_1.parameters()) + list(self.critic_2_representation.parameters()) + list(
            self.critic_2.parameters())

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.actor_representation._get_name() != "Basic_Identical":
                self.actor_representation = DistributedDataParallel(self.actor_representation, device_ids=[self.rank])
            if self.critic_1_representation._get_name() != "Basic_Identical":
                self.critic_1_representation = DistributedDataParallel(self.critic_1_representation,
                                                                       device_ids=[self.rank])
            if self.critic_2_representation._get_name() != "Basic_Identical":
                self.critic_2_representation = DistributedDataParallel(self.critic_2_representation,
                                                                       device_ids=[self.rank])
            self.actor = DistributedDataParallel(module=self.actor, device_ids=[self.rank])
            self.critic_1 = DistributedDataParallel(module=self.critic_1, device_ids=[self.rank])
            self.critic_2 = DistributedDataParallel(module=self.critic_2, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of actor representation and samples of actions.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            outputs: The outputs of the actor representation.
            act_sample: The sampled actions from the distribution output by the actor.
        """
        outputs = self.actor_representation(observation)
        act_dist = self.actor(outputs['state'])
        act_samples = act_dist.stochastic_sample()
        return outputs, act_samples

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """
        Feedforward and calculate the action probabilities, log of action probabilities, and Q-values.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            act_prob: The probabilities of actions.
            log_action_prob: The log of action probabilities.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)

        act_dist = self.actor(outputs_actor['state'])
        act_prob = act_dist.probs
        z = act_prob == 0.0
        z = z.float() * 1e-8
        log_action_prob = torch.log(act_prob + z)

        q_1 = self.critic_1(outputs_critic_1['state'])
        q_2 = self.critic_2(outputs_critic_2['state'])
        return act_prob, log_action_prob, q_1, q_2

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """
        Calculate the action probabilities, log of action probabilities, and Q-values with target networks.

        Parameters:
            observation: The original observation of an agent.

        Returns:
            new_act_prob: The probabilities of actions.
            log_action_prob: The log of action probabilities.
            target_q: The minimum of Q-values calculated by the target critic networks.
        """
        outputs_actor = self.actor_representation(observation)
        outputs_critic_1 = self.target_critic_1_representation(observation)
        outputs_critic_2 = self.target_critic_2_representation(observation)

        new_act_dist = self.actor(outputs_actor['state'])
        new_act_prob = new_act_dist.probs
        z = new_act_prob == 0.0
        z = z.float() * 1e-8  # avoid log(0)
        log_action_prob = torch.log(new_act_prob + z)

        target_q_1 = self.target_critic_1(outputs_critic_1['state'])
        target_q_2 = self.target_critic_2(outputs_critic_2['state'])
        target_q = torch.min(target_q_1, target_q_2)
        return new_act_prob, log_action_prob, target_q

    def Qaction(self, observation: Union[np.ndarray, dict]):
        """
        Returns the evaluated Q-values for current observations.

        Parameters:
            observation: The original observation.

        Returns:
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        outputs_critic_1 = self.critic_1_representation(observation)
        outputs_critic_2 = self.critic_2_representation(observation)
        q_1 = self.critic_1(outputs_critic_1['state'])
        q_2 = self.critic_2(outputs_critic_2['state'])
        return q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class DreamerV2DISPolicy(Module):
    """
    Actor-Critic for DreamerV2 with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(DreamerV2DISPolicy, self).__init__()
        self.action_dim = action_space.n
        self.device = device
        # TODO 这里就直接用 xuance 的网络结构了
        """actor Net: model_state -> onehot_action"""
        self.representation = representation
        self.actor = OneHotCategoricalActorNet(self.representation.modelstate_size, self.action_dim, actor_hidden_size,
                                               normalize, initialize, activation, device)
        """critic Net & Target critic Net: model_state -> value_normal_dist"""
        self.critic_1 = GaussianCriticNet(self.representation.modelstate_size, critic_hidden_size,
                                          normalize, initialize, activation, device)
        self.target_critic_1 = deepcopy(self.critic_1)

        # representation 参数共享
        self.model_parameters = list(self.representation.parameters())
        self.actor_parameters = list(self.representation.parameters()) + list(self.actor.parameters())
        self.critic_parameters = list(self.representation.parameters()) + list(self.critic_1.parameters())

    """(n_env = 1, ~)"""

    def forward(self,
                observation,
                prev_action,
                prev_nonterm,
                prev_rssm_state):
        """
        Returns the action distribution and posterior_rssm_state for agent action

        Parameters:
            observation: The observation after previous action of an agent.  (1, 4)
            prev_action: The previous action of an agent.  (2, )
            prev_nonterm: The previous nonterm of an agent.  (1, )
            prev_rssm_state: The previous rssm state of an agent.  (1, 400)

        Returns:
            act_dist: The action distribution.
            posterior_rssm_state: The posterior rssm state.
        """
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_nonterm = torch.tensor(prev_nonterm, dtype=torch.float32, device=self.device).unsqueeze(0)
        prev_action = torch.tensor(prev_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        rssm_model_state, posterior_rssm_state = self.representation.rssm_observe(
            observation,
            prev_action,
            prev_nonterm,
            prev_rssm_state
        )
        act_dist = self.actor(rssm_model_state)  # OneHot_dist
        # value_dist = self.critic(rssm_model_state)  # Normal_dist
        return act_dist, posterior_rssm_state

    def model_forward(self,
                      obs_seq_batch,
                      act_seq_batch,
                      noterm_seq_batch):
        """ TODO docstring
        Returns the action distribution and posterior_rssm_state for agent action

        Parameters:
            obs_seq_batch: observation
            act_seq_batch:
            noterm_seq_batch:

        Returns:
            obs_dist:
            rew_dist:
            noterm_dist:
            prior, post
        """
        prior, post = self.representation.rollout_observation(obs_seq_batch, act_seq_batch, noterm_seq_batch)
        obs_dist, rew_dist, noterm_dist = self.representation.return_dists(post)
        return obs_dist, rew_dist, noterm_dist, prior, post

    def actor_critic_forward(self, prev_rssm_state):
        seq_len, batch_size = prev_rssm_state.stoch.shape[0:2]
        """(n_envs * seq * batch, ~)"""
        with torch.no_grad():
            prev_rssm_state = self.representation.RSSM.rssm_seq_to_batch(prev_rssm_state, batch_size, seq_len)
        out = self.representation.rollout_imagination(self.actor, self.target_critic_1, prev_rssm_state)
        imag_reward, imag_value, discount_arr, act_log_probs, act_ent = out['for_actor']
        """(imagine_horizon, n_envs * seq * batch, ~)"""
        lambda_ = self.representation.config.lambda_
        V_lambda = DreamerV2DISPolicy.compute_lambda_return(imag_reward, imag_value, discount_arr, lambda_)

        imag_modelstate = out['for_critic']
        value_dist = self.critic_1.to(self.device)(imag_modelstate)
        value_dist.loc = value_dist.loc.to(self.device)
        # value_dist.scale.to(self.device)  # 不赋值就不行
        value_dist.scale = value_dist.scale.to(self.device)
        # for actor_loss: imag_reward, imag_value, discount_arr, act_log_probs, act_ent
        # for critic_loss: imag_modelstate, discount, lambda_returns
        return act_log_probs, act_ent, imag_value, V_lambda, value_dist

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)

    @staticmethod
    def compute_lambda_return(r, v, gamma, lambda_):
        """checked!!!"""
        """
            v[i]: expectation of discounted return, learn from world model
            V_lambda[i] = r[i] + gamma[i] * (lambda * v[i + 1] + (1 - lambda) * V_lambda[i + 1]) 
            V_lambda[H - 1] = v[H - 1]  
        """
        """(imagine_horizon, n_envs * seq * batch, ~)"""
        H = v.shape[0]
        # V_lambda = r[H - 1] + gamma[H - 1] * v[H - 1]  # the paper's approach
        V_lambda = v[H - 1]
        V_lambda_list = [V_lambda]
        for i in reversed(range(0, H - 1)):
            V_lambda = r[i] + gamma[i] * ((1 - lambda_) * v[i + 1] + lambda_ * V_lambda)
            V_lambda_list.append(V_lambda)
        return torch.flip(torch.stack(V_lambda_list), [0])





