import numpy as np
from xuance.common import Sequence, Optional, Union, Callable
from xuance.torch import Module, Tensor
from xuance.torch.utils import torch, nn, mlp_block, ModuleType
from .rssm import RSSM
from .pixel import ObsDecoder, ObsEncoder

import numpy as np
from .dense import DenseModel


class WorldModel_DreamerV2(Module):
    def __init__(self, config):
        super(WorldModel_DreamerV2, self).__init__()
        """WM"""
        # TODO unique and delete unnecessary var & comments
        # 定义 _model_initialize 方法，初始化模型组件 (initialize model components)
        self.config = config  # 将配置对象赋值给 self.config 属性
        self.action_size = config.action_size  # 从配置中获取动作空间大小 (action space size)
        self.pixel = config.pixel  # 从配置中获取是否使用像素观测的标志 (flag indicating pixel observation)
        self.kl_info = config.kl  # 从配置中获取 KL 散度相关信息 (KL divergence related information)
        self.seq_len = config.seq_len  # 从配置中获取序列长度 (sequence length)
        self.batch_size = config.batch_size  # 从配置中获取批大小 (batch size)
        # self.collect_intervals = config.collect_intervals  # 从配置中获取数据收集间隔 (data collection interval)
        # self.seed_steps = config.seed_steps  # 从配置中获取 seed steps 数量 (seed steps count)
        self.discount = config.discount_  # 从配置中获取折扣因子 (discount factor)
        self.lambda_ = config.lambda_  # 从配置中获取 lambda 参数，用于 GAE 或 lambda-return (lambda parameter for GAE or lambda-return)
        self.horizon = config.horizon  # 从配置中获取想象力 horizon (imagination horizon)
        self.loss_scale = config.loss_scale  # 从配置中获取损失缩放比例 (loss scaling)
        # self.actor_entropy_scale = config.actor_entropy_scale  # 从配置中获取 actor entropy 的缩放比例系数 (actor entropy scaling coefficient)
        self.grad_clip_norm = config.grad_clip  # 从配置中获取梯度裁剪范数 (gradient clip norm)
        self.device = config.device

        # self._model_initialize(config)  # 调用 _model_initialize 方法初始化模型 (initialize models)
        # self._optim_initialize(config)  # 调用 _optim_initialize 方法初始化优化器 (initialize optimizers)
        obs_shape = config.obs_shape  # 从配置中获取观测形状 (get observation shape from config)
        action_size = config.action_size  # 从配置中获取动作空间大小 (get action space size from config)
        deter_size = config.rssm_info['deter_size']  # 从配置中获取确定性状态大小 (get deterministic state size from config)
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']  # 32
            class_size = config.rssm_info['class_size']  # 32
            stoch_size = category_size * class_size  # calculate stochastic state size

        embedding_size = config.embedding_size  # 200
        rssm_node_size = config.rssm_node_size  # 200
        self.modelstate_size = stoch_size + deter_size  # 400

        self.RSSM = RSSM(
            action_size,
            rssm_node_size,
            embedding_size,
            config.rssm_type,
            config.rssm_info
        ).to(self.device)  # 创建 RSSM 实例并放到指定设备上 (create RSSM instance and move to device)

        """RewardDecoder: h + z -> r"""
        self.RewardDecoder = DenseModel((1,), self.modelstate_size, config.reward).to(
            self.device)

        # if config.discount['use']:
        """DiscountModel: h + z -> gamma"""
        self.DiscountModel = DenseModel((1,), self.modelstate_size, config.discount).to(
            self.device)  # 创建 DiscountModel 实例并放到指定设备上 (create DiscountModel instance and move to device)

        if config.pixel:  # if using pixel observations
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, self.modelstate_size, config.obs_decoder).to(self.device)
        else:  # if not using pixel observations
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, self.modelstate_size, config.obs_decoder).to(self.device)

    """obs(x2) -> rssm_model_state[deter(h2), stoch(z2)]"""
    def rssm_observe(self, obs, prev_action, prev_nonterm, prev_rssm_state):
        obs_embed = self.ObsEncoder(obs)
        # 返回先验 RSSM 状态和后验 RSSM 状态
        prior_rssm_state, posterior_rssm_state = self.RSSM.rssm_observe(obs_embed, prev_action, prev_nonterm, prev_rssm_state)
        rssm_model_state = self.RSSM.get_model_state(posterior_rssm_state)
        return rssm_model_state, posterior_rssm_state

    """(seq, n_envs * batch, ~)"""
    def return_dists(self, posterior):
        model_state = self.RSSM.get_model_state(posterior)
        """seq_shift: model_state[:-1]"""
        obs_dist = self.ObsDecoder(model_state[:-1])
        rew_dist = self.RewardDecoder(model_state[:-1])
        noterm_dist = self.DiscountModel(model_state[:-1])
        """(seq * n_envs * batch, ~)"""
        return obs_dist, rew_dist, noterm_dist

    """(seq, n_envs * batch, ~)"""
    def rollout_observation(self,
                            obs: torch.Tensor,
                            actions: torch.Tensor,
                            nonterms: torch.Tensor):
        """actions -> obs, nonterms"""
        embed = self.ObsEncoder(obs)
        prev_rssm_state = self.RSSM.init_rssm_state(obs.shape[1], self.device)
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        """(seq, n_envs * batch, ~)"""
        return prior, posterior

    """(seq * n_envs * batch, ~)"""
    def rollout_imagination(self,
                            actor: nn.Module,
                            target_critic: nn.Module,
                            prev_rssm_state):
        # with torch.no_grad(): # 在无梯度计算的环境下 (in no gradient calculation context)
        #     """转化为 batch * seq 大小的单个批次的数据, 对每个后验都进行 imagine 再更新"""
        #     batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1)) # 将序列后验状态转换为批次形式并分离计算图 (convert sequence posterior state to batch form and detach from computation graph)
        #
        # with FreezeParameters(self.world_list): # 在冻结世界模型参数的情况下 (in frozen world model parameters context)
        #     imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior) # 使用想象力 (imagination) rollout 得到想象轨迹 (imagination trajectory)
        # rollout_imagination(obs_seq_batch, act_seq_batch, noterm_seq_batch)
        """(horizon, n_envs * seq * batch, ~)"""
        prior, imag_act_log_probs, act_ent = self.RSSM.rollout_imagination(self.horizon, actor, prev_rssm_state)

        with torch.no_grad():
            imag_modelstate = self.RSSM.get_model_state(prior)
        """reward: directly use the mean of the Normal dist"""
        imag_reward_dist = self.RewardDecoder(imag_modelstate)
        imag_reward = imag_reward_dist.mean

        """target_value: directly use the mean of the Normal dist"""
        imag_value_dist = target_critic.to(self.device)(imag_modelstate)
        imag_value = imag_value_dist.mean

        """discount: not to sample Bernoulli dist, but to round"""
        imag_discount_dist = self.DiscountModel(imag_modelstate)
        discount_arr = self.discount * torch.round(imag_discount_dist.base_dist.probs)
        return {
            "for_actor": (imag_reward, imag_value.detach(), discount_arr.detach(), imag_act_log_probs, act_ent),
            "for_critic": imag_modelstate
        }

