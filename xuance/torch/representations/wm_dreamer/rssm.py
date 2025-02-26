import torch # 导入 torch 库，PyTorch 的核心库
import torch.nn as nn # 导入 torch.nn 模块并简写为 nn，用于构建神经网络
from .rssm_utils import RSSMUtils, RSSMContState, RSSMDiscState # 从 dreamerv2.utils.rssm 模块导入 RSSMUtils 工具类, RSSMContState (连续型 RSSM 状态) 和 RSSMDiscState (离散型 RSSM 状态)

class RSSM(nn.Module, RSSMUtils): # 定义 RSSM 类，继承自 nn.Module (PyTorch 神经网络模块基类) 和 RSSMUtils (RSSM 工具类)
    def __init__( # 定义构造函数 __init__，在创建 RSSM 对象时调用
        self, # self 代表类的实例
        action_size, # 动作空间大小 (Action space size)，即动作的维度
        rssm_node_size, # RSSM 节点的隐藏层大小 (Hidden layer size of RSSM nodes)，用于定义网络层的大小
        embedding_size, # 嵌入向量大小 (Embedding size)，通常是观测 (Observation) 经过编码器后的维度
        rssm_type, # RSSM 类型 (RSSM type)，可以是 'continuous' (连续型) 或 'discrete' (离散型)
        info, # 包含 RSSM 相关信息的字典 (Information dictionary for RSSM)，例如状态维度等
        act_fn=nn.ELU, # 激活函数 (Activation function)，默认为 ELU (指数线性单元)
    ):
        nn.Module.__init__(self) # 调用 nn.Module 的构造函数，完成 nn.Module 的初始化
        RSSMUtils.__init__(self, rssm_type=rssm_type, info=info) # 调用 RSSMUtils 的构造函数，初始化 RSSMUtils 的功能
        self.action_size = action_size # 将传入的 action_size (动作空间大小) 赋值给对象属性 self.action_size
        self.node_size = rssm_node_size # 将传入的 rssm_node_size (RSSM 节点大小) 赋值给对象属性 self.node_size
        self.embedding_size = embedding_size # 将传入的 embedding_size (嵌入向量大小) 赋值给对象属性 self.embedding_size
        self.act_fn = act_fn # 将传入的 act_fn (激活函数) 赋值给对象属性 self.act_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size) # 定义 GRUCell (门控循环单元)，作为 RSSM 的循环核 (Recurrent kernel)，输入和输出维度均为确定性状态大小 (deter_size)

        self.fc_embed_state_action = self._build_embed_state_action()

        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()

    """state_action pair embed"""
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action
        and embed it to deter size for rnn input
        """
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.action_size, self.deter_size)]
        fc_embed_state_action += [self.act_fn()]
        return nn.Sequential(*fc_embed_state_action)

    """Transition predictor"""
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state
        and output prior over stochastic state
        """
        temporal_prior = [nn.Linear(self.deter_size, self.node_size)]
        temporal_prior += [self.act_fn()]
        if self.rssm_type == 'discrete':
            temporal_prior += [nn.Linear(self.node_size, self.stoch_size)]
        elif self.rssm_type == 'continuous':
             temporal_prior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_prior)

    """Representation model"""
    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state
        and output posterior over stochastic states
        """
        temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, self.node_size)]
        temporal_posterior += [self.act_fn()]
        if self.rssm_type == 'discrete':
            temporal_posterior += [nn.Linear(self.node_size, self.stoch_size)]
        elif self.rssm_type == 'continuous':
            temporal_posterior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_posterior)

    """a_1, rssm_state[h_1, z_1] -> rssm_state[h_2, z_hat_2]"""
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        """prev_stoch(z_1) & prev_action(a_1) + prev_deter(h_1, rnn_hidden) -> deter(h_2); net: rnn(Recurrent model)"""
        # if torch.any(torch.isnan(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))):
        #     print()
        # if torch.sum(torch.isnan(self.fc_embed_state_action[0].weight)).item() > 0:
        #     print()
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter*nonterms)
        if self.rssm_type == 'discrete':
            """deter(h_2) -> prior_stoch(z_hat_2); net: fc_prior(Transition predictor)"""
            prior_logit = self.fc_prior(deter_state)
            stats = {'logit':prior_logit}
            prior_stoch_state = self.get_stoch_state(stats)
            """(z2_hat_logit, z2_hat_sample, h2); 打包先验状态为 rssm_discrete_state"""
            prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)
        elif self.rssm_type == 'continuous':
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            stats = {'mean':prior_mean, 'std':prior_std}
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state

    """一系列想象"""
    def rollout_imagination(self, horizon: int, actor: nn.Module, prev_rssm_state): # 定义方法 rollout_imagination，进行想象轨迹 (Imagination trajectory rollout) 的展开
        rssm_state = prev_rssm_state # 初始化 RSSM 状态为给定的 prev_rssm_state
        next_rssm_states = [] # 初始化列表 (list) 用于存储未来的 RSSM 状态
        action_entropy = [] # 初始化列表用于存储动作的熵 (entropy of actions)
        imag_log_probs = [] # 初始化列表用于存储想象轨迹中动作的对数概率 (log probabilities of actions in imagination trajectory)
        for t in range(horizon): # 循环 horizon (想象步数) 次
            """modified"""
            # action, action_dist = actor((self.get_model_state(rssm_state)).detach()) # 使用 actor (策略网络) 根据当前模型状态 (get_model_state) 生成动作 (action) 和动作分布 (action_dist)，使用 detach() 分离梯度
            action_dist = actor((self.get_model_state(rssm_state)).detach()) # 使用 actor (策略网络) 根据当前模型状态 (get_model_state) 生成动作 (action) 和动作分布 (action_dist)，使用 detach() 分离梯度
            action = action_dist.rsample()
            rssm_state = self.rssm_imagine(action, rssm_state) # 使用 rssm_imagine 方法根据生成的动作更新 RSSM 状态
            next_rssm_states.append(rssm_state) # 将新的 RSSM 状态添加到列表中
            action_entropy.append(action_dist.entropy()) # 计算动作分布的熵并添加到列表中
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach()))) # 计算动作分布下，对四舍五入后的动作 (torch.round(action.detach())) 取对数概率，并添加到列表中
        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0) # 将列表中所有 RSSM 状态堆叠 (stack) 成批次形式 (batch form)，沿时间维度 (dim=0)
        action_entropy = torch.stack(action_entropy, dim=0) # 将列表中所有动作熵堆叠成张量 (tensor)，沿时间维度
        imag_log_probs = torch.stack(imag_log_probs, dim=0) # 将列表中所有对数概率堆叠成张量，沿时间维度
        return next_rssm_states, imag_log_probs, action_entropy # 返回未来 RSSM 状态序列，动作对数概率序列和动作熵序列

    """想象+观测一次(多了个后验), a_1, rssm_state[h_1, z_1] -> rssm_state[h_2, z_hat_2], rssm_state[h_2, z_2]"""
    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        if self.rssm_type == 'discrete':
            """deter(h2) & obs_embed(x2) -> stoch_posterior(z2); net: fc_posterior(Rrepresentation model)"""
            posterior_logit = self.fc_posterior(x)
            stats = {'logit':posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            """(z2_logit, z2_sample, h2); 打包后验状态为 rssm_discrete_state"""
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state) # 构建离散型后验 RSSM 状态 (posterior_rssm_state)
        elif self.rssm_type == 'continuous': # 如果 RSSM 类型是 'continuous' (连续型)
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1) # 通过后验网络预测均值和标准差
            stats = {'mean':posterior_mean, 'std':posterior_std} # 将均值和标准差存入字典 stats
            posterior_stoch_state, std = self.get_stoch_state(stats) # 从 stats 中采样得到后验随机状态和标准差
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state) # 构建连续型后验 RSSM 状态 (posterior_rssm_state)
        """prior: (z2_hat_logit, z2_hat_sample, h2)"""
        """posterior: (z2_logit, z2_sample, h2)"""
        return prior_rssm_state, posterior_rssm_state  # 返回先验 RSSM 状态和后验 RSSM 状态

    """一系列观测"""
    def rollout_observation(self, seq_len: int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state): # 定义方法 rollout_observation，进行观测序列 (Observation sequence) 的展开
        priors = [] # 初始化列表用于存储先验 RSSM 状态
        posteriors = [] # 初始化列表用于存储后验 RSSM 状态
        for t in range(seq_len): # 循环 seq_len (序列长度) 次
            prev_action = action[t]*nonterms[t] # 获取当前时间步的动作，并与 nonterms 相乘处理 episode 终止
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], prev_action, nonterms[t], prev_rssm_state) # 使用 rssm_observe 方法根据当前观测嵌入 (obs_embed), 上一步动作 (prev_action), nonterms 和上一步状态 (prev_rssm_state) 更新状态，得到先验和后验状态
            priors.append(prior_rssm_state) # 将先验状态添加到列表
            posteriors.append(posterior_rssm_state) # 将后验状态添加到列表
            prev_rssm_state = posterior_rssm_state # 将后验状态更新为下一步的 prev_rssm_state
        prior = self.rssm_stack_states(priors, dim=0) # 将列表中所有先验 RSSM 状态堆叠成批次形式，沿时间维度
        post = self.rssm_stack_states(posteriors, dim=0) # 将列表中所有后验 RSSM 状态堆叠成批次形式，沿时间维度
        return prior, post # 返回先验 RSSM 状态序列和后验 RSSM 状态序列
