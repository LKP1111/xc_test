from collections import namedtuple # 导入 namedtuple，用于创建具名元组，这是一种轻量级的对象类型，类似于只有属性的类
import torch.distributions as td # 导入 torch.distributions 模块并简写为 td，用于处理概率分布
import torch # 导入 torch 库，PyTorch 的核心库
import torch.nn.functional as F # 导入 torch.nn.functional 模块并简写为 F，提供了一些函数式神经网络操作，如激活函数等
from typing import Union # 导入 Union 类型，用于类型提示，表示可以是多种类型之一

RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter']) # 定义具名元组 RSSMDiscState，用于表示离散型 RSSM (Recurrent State Space Model, 循环状态空间模型) 的状态，包含 logit (对数几率，用于生成离散分布), stoch (stochastic state, 随机状态), deter (deterministic state, 确定性状态)
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])  # 定义具名元组 RSSMContState，用于表示连续型 RSSM 的状态，包含 mean (均值，用于生成正态分布), std (标准差，用于生成正态分布), stoch (随机状态), deter (确定性状态)

RSSMState = Union[RSSMDiscState, RSSMContState] # 定义 RSSMState 类型为 RSSMDiscState 或 RSSMContState 的联合类型，表示 RSSM 的状态可以是离散型或连续型

class RSSMUtils(object): # 定义 RSSMUtils 类，用于提供处理 RSSM 状态的工具函数
    '''utility functions for dealing with rssm states''' # 类的文档字符串，说明该类是用于处理 rssm 状态的工具函数集合
    def __init__(self, rssm_type, info): # 定义构造函数 __init__，在创建 RSSMUtils 对象时调用
        self.rssm_type = rssm_type # 将传入的 rssm_type (rssm 类型，'continuous' 或 'discrete') 赋值给对象属性 self.rssm_type
        if rssm_type == 'continuous': # 如果 rssm_type 是 'continuous' (连续型)
            self.deter_size = info['deter_size'] # 从 info 字典中获取 'deter_size' (确定性状态的大小) 并赋值给对象属性 self.deter_size
            self.stoch_size = info['stoch_size'] # 从 info 字典中获取 'stoch_size' (随机状态的大小) 并赋值给对象属性 self.stoch_size
            self.min_std = info['min_std'] # 从 info 字典中获取 'min_std' (最小标准差) 并赋值给对象属性 self.min_std，用于保证标准差为正值
        elif rssm_type == 'discrete': # 如果 rssm_type 是 'discrete' (离散型)
            self.deter_size = info['deter_size'] # 从 info 字典中获取 'deter_size' 并赋值给对象属性 self.deter_size
            self.class_size = info['class_size'] # 从 info 字典中获取 'class_size' (类别数量) 并赋值给对象属性 self.class_size，用于离散分布
            self.category_size = info['category_size'] # 从 info 字典中获取 'category_size' (类别组数) 并赋值给对象属性 self.category_size，用于离散分布
            self.stoch_size  = self.class_size*self.category_size # 计算离散状态的总大小，等于类别数量乘以类别组数，并赋值给对象属性 self.stoch_size
        else: # 如果 rssm_type 既不是 'continuous' 也不是 'discrete'
            raise NotImplementedError # 抛出 NotImplementedError 异常，表示不支持该 rssm_type

    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len): # 定义方法 rssm_seq_to_batch，将序列形式的 RSSM 状态转换为批次形式
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            return RSSMDiscState( # 返回一个新的 RSSMDiscState 对象
                seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len), # 将 logit 序列转换为批次形式
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len), # 将 stoch 序列转换为批次形式
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)  # 将 deter 序列转换为批次形式
            )
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return RSSMContState( # 返回一个新的 RSSMContState 对象
                seq_to_batch(rssm_state.mean[:seq_len], batch_size, seq_len), # 将 mean 序列转换为批次形式
                seq_to_batch(rssm_state.std[:seq_len], batch_size, seq_len), # 将 std 序列转换为批次形式
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len), # 将 stoch 序列转换为批次形式
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)  # 将 deter 序列转换为批次形式
            )

    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len): # 定义方法 rssm_batch_to_seq，将批次形式的 RSSM 状态转换为序列形式
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            return RSSMDiscState( # 返回一个新的 RSSMDiscState 对象
                batch_to_seq(rssm_state.logit, batch_size, seq_len), # 将 logit 批次转换为序列形式
                batch_to_seq(rssm_state.stoch, batch_size, seq_len), # 将 stoch 批次转换为序列形式
                batch_to_seq(rssm_state.deter, batch_size, seq_len)  # 将 deter 批次转换为序列形式
            )
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return RSSMContState( # 返回一个新的 RSSMContState 对象
                batch_to_seq(rssm_state.mean, batch_size, seq_len), # 将 mean 批次转换为序列形式
                batch_to_seq(rssm_state.std, batch_size, seq_len), # 将 std 批次转换为序列形式
                batch_to_seq(rssm_state.stoch, batch_size, seq_len), # 将 stoch 批次转换为序列形式
                batch_to_seq(rssm_state.deter, batch_size, seq_len)  # 将 deter 批次转换为序列形式
            )

    """离散分布"""
    def get_dist(self, rssm_state): # 定义方法 get_dist，根据 RSSM 状态获取概率分布 (Distribution) 对象
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            shape = rssm_state.logit.shape # 获取 logit 的形状
            logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size)) # 将 logit 的形状重塑为 (*shape[:-1], category_size, class_size)，以便创建 OneHotCategoricalStraightThrough 分布
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1) # 返回 Independent (独立分布)，包裹 OneHotCategoricalStraightThrough (One-Hot 直通分类分布)，用于离散状态的采样和梯度传播，参数 logits (对数几率) 为重塑后的 logit，事件维度为 1
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1) # 返回 Independent (独立分布)，包裹 Normal (正态分布)，用于连续状态的采样，参数 mean (均值) 和 std (标准差) 来自 RSSM 状态，事件维度为 1

    def get_stoch_state(self, stats): # 定义方法 get_stoch_state，从模型的输出统计量 (stats) 中获取随机状态 (stochastic state)
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            logit = stats['logit'] # 从 stats 字典中获取 'logit'
            shape = logit.shape # 获取 logit 的形状
            logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size)) # 重塑 logit 形状
            dist = torch.distributions.OneHotCategorical(logits=logit) # 创建 OneHotCategorical (One-Hot 分类分布)，用于从 logit 采样离散状态
            stoch = dist.sample() # 从分布中采样一个 one-hot 向量作为随机状态 stoch
            stoch += dist.probs - dist.probs.detach() # Straight-Through (直通) 梯度估计技巧：在 forward pass (前向传播) 中使用采样值，但在 backward pass (反向传播) 中使用概率值，以解决离散采样的不可导问题
            return torch.flatten(stoch, start_dim=-2, end_dim=-1) # 将 stoch 展平，返回展平后的随机状态

        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            mean = stats['mean'] # 从 stats 字典中获取 'mean'
            std = stats['std'] # 从 stats 字典中获取 'std'
            std = F.softplus(std) + self.min_std # 使用 softplus (softplus 函数，平滑的 ReLU 函数) 保证 std 为正值，并加上 min_std (最小标准差) 避免 std 过小
            return mean + std*torch.randn_like(mean), std # 从以 mean 为均值，std 为标准差的正态分布中采样，返回采样值 (随机状态 stoch) 和标准差 std

    def rssm_stack_states(self, rssm_states, dim): # 定义方法 rssm_stack_states，将多个 RSSM 状态沿指定维度 (dim) 堆叠在一起
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            return RSSMDiscState( # 返回一个新的 RSSMDiscState 对象
                torch.stack([state.logit for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 logit 属性，沿维度 dim
                torch.stack([state.stoch for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 stoch 属性，沿维度 dim
                torch.stack([state.deter for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 deter 属性，沿维度 dim
            )
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return RSSMContState( # 返回一个新的 RSSMContState 对象
            torch.stack([state.mean for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 mean 属性，沿维度 dim
            torch.stack([state.std for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 std 属性，沿维度 dim
            torch.stack([state.stoch for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 stoch 属性，沿维度 dim
            torch.stack([state.deter for state in rssm_states], dim=dim), # 堆叠所有 RSSM 状态的 deter 属性，沿维度 dim
        )

    """deter & post_stoch 状态拼接作为模型最终预测的 model_state"""
    def get_model_state(self, rssm_state): # 定义方法 get_model_state，获取模型的状态表示，通常是确定性状态和随机状态的拼接
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1) # 将 deter (确定性状态) 和 stoch (随机状态) 在最后一个维度 (dim=-1) 上拼接
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1) # 将 deter (确定性状态) 和 stoch (随机状态) 在最后一个维度 (dim=-1) 上拼接

    def rssm_detach(self, rssm_state): # 定义方法 rssm_detach，将 RSSM 状态从计算图中分离 (detach)，用于停止梯度传播
        if self.rssm_type == 'discrete': # 如果是离散型 RSSM
            return RSSMDiscState( # 返回一个新的 RSSMDiscState 对象
                rssm_state.logit.detach(),  # 分离 logit
                rssm_state.stoch.detach(), # 分离 stoch
                rssm_state.deter.detach(), # 分离 deter
            )
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return RSSMContState( # 返回一个新的 RSSMContState 对象
                rssm_state.mean.detach(), # 分离 mean
                rssm_state.std.detach(),  # 分离 std
                rssm_state.stoch.detach(), # 分离 stoch
                rssm_state.deter.detach()  # 分离 deter
            )

    def init_rssm_state(self, batch_size, device, **kwargs): # 定义私有方法 _init_rssm_state，初始化 RSSM 状态为零状态
        if self.rssm_type  == 'discrete': # 如果是离散型 RSSM
            return RSSMDiscState( # 返回一个新的 RSSMDiscState 对象，所有状态都初始化为零张量 (zero tensor)
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(device), # 初始化 logit 为零张量，形状为 (batch_size, stoch_size)，并移动到指定设备 (device) (假设 self.device 已定义)
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(device), # 初始化 stoch 为零张量，形状为 (batch_size, stoch_size)，并移动到指定设备
                torch.zeros(batch_size, self.deter_size, **kwargs).to(device), # 初始化 deter 为零张量，形状为 (batch_size, deter_size)，并移动到指定设备
            )
        elif self.rssm_type == 'continuous': # 如果是连续型 RSSM
            return RSSMContState( # 返回一个新的 RSSMContState 对象，所有状态都初始化为零张量
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(device), # 初始化 mean 为零张量，形状为 (batch_size, stoch_size)，并移动到指定设备
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(device), # 初始化 std 为零张量，形状为 (batch_size, stoch_size)，并移动到指定设备
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(device), # 初始化 stoch 为零张量，形状为 (batch_size, stoch_size)，并移动到指定设备
                torch.zeros(batch_size, self.deter_size, **kwargs).to(device), # 初始化 deter 为零张量，形状为 (batch_size, deter_size)，并移动到指定设备
            )

def seq_to_batch(sequence_data, batch_size, seq_len): # 定义函数 seq_to_batch，将序列数据转换为批次数据
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """ # 函数的文档字符串，说明将长度为 L，批大小为 B 的序列数据转换为大小为 L*B 的单个批次数据
    shp = tuple(sequence_data.shape) # 获取输入序列数据 sequence_data 的形状 (shape) 并转换为元组 (tuple)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]]) # 将序列数据 sequence_data 重塑 (reshape) 为批次数据 batch_data，新的形状为 [shp[0]*shp[1], *shp[2:]]，即将前两个维度 (序列长度和批大小) 合并为一个维度
    return batch_data # 返回转换后的批次数据 batch_data

def batch_to_seq(batch_data, batch_size, seq_len): # 定义函数 batch_to_seq，将批次数据转换为序列数据
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """ # 函数的文档字符串，说明将大小为 L*B 的单个批次数据转换为长度为 L，批大小为 B 的序列数据
    shp = tuple(batch_data.shape) # 获取输入批次数据 batch_data 的形状并转换为元组
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]]) # 将批次数据 batch_data 重塑为序列数据 seq_data，新的形状为 [seq_len, batch_size, *shp[1:]]，即将第一个维度 (合并后的维度) 拆分为两个维度 (序列长度和批大小)
    return seq_data # 返回转换后的序列数据 seq_data