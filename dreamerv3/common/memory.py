from xuance.common.memory_tools import Buffer, create_memory, store_element, sample_batch
import numpy as np
from gym import Space
from xuance.common import Optional, Union
from xuance.common import space2shape
from xuance.common import Dict

# modified from DummyOffPolicyBuffer
class SequentialReplayBuffer(Buffer):
    """
    Sequential Replay buffer for Dreamerv3

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        auxiliary_shape: data shape of auxiliary information (if exists).
        n_envs: number of parallel environments.
        buffer_size: the total size of the replay buffer.
        batch_size: size of transition data for a batch of sample.
    """
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 n_envs: int,
                 buffer_size: int,
                 batch_size: int):
        super(SequentialReplayBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
        self.n_envs, self.batch_size = n_envs, batch_size
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = buffer_size // self.n_envs
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)
        self.truncations = create_memory((), self.n_envs, self.n_size)
        self.is_first = create_memory((), self.n_envs, self.n_size)

    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)
        self.truncations = create_memory((), self.n_envs, self.n_size)
        self.is_first = create_memory((), self.n_envs, self.n_size)

    def store(self, obs, acts, rews, terminals, truncations, is_first):
        """

        Args:
            all arguments are numpy arrays, shape: [envs, ~] if ~ != 1 else [envs, ]

        Returns:

        """
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(truncations, self.truncations, self.ptr)
        store_element(is_first, self.is_first, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def sample(self, seq_len: int):
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode boundaries.
        Args:
            seq_len (int)
        Returns:
            Dict[str, np.ndarray]: the sampled dictionary with a shape of
            [envs, sequence_length, batch_size, ...].
        """
        # [self.ptr, (self.ptr + sequence_length) % self.n_size)
        # make sure there are more than seq_len of data stored
        assert self.size - seq_len >= 0
        first_range_end = self.ptr - seq_len + 1
        indices = np.arange(0, first_range_end)
        if self.size == self.n_size:
            second_range_end = self.ptr + seq_len if first_range_end <= 0 else self.n_size
            indices = np.concatenate([
                indices,
                np.arange(self.ptr, second_range_end)
            ])
        li = []
        for _ in range(self.n_envs):
            start = np.random.choice(indices, size=self.batch_size).reshape(-1, 1)  # (batch, 1)
            seq_arange = np.arange(seq_len, dtype=np.intp).reshape(1, -1)  # (1, seq)
            idxes = (start + seq_arange) % self.n_size  # (batch, seq)
            li.append(np.swapaxes(idxes, 0, 1))  # (seq, batch)
        idxes = np.stack(li)  # (envs, seq, batch)
        envs = np.broadcast_to(np.arange(self.n_envs)[:, None, None], idxes.shape)  # (env, seq, batch)
        envs, idxes = envs.ravel(), idxes.ravel()
        samples_dict = {  # (envs, seq, batch, ~)
            'obs': self.observations[envs, idxes].reshape(self.n_envs, seq_len, self.batch_size, *space2shape(self.observation_space)),
            'actions': self.actions[envs, idxes].reshape(self.n_envs, seq_len, self.batch_size, *space2shape(self.action_space)),
            'rewards': self.rewards[envs, idxes].reshape(self.n_envs, seq_len, self.batch_size, 1),
            'terminals': self.terminals[envs, idxes].reshape(self.n_envs, seq_len, self.batch_size, 1),
            'truncations': self.truncations[envs, idxes].reshape(self.n_envs, seq_len, self.batch_size, 1),
            'is_first': self.is_first[envs, idxes].reshape(self.n_envs, seq_len, self.batch_size, 1),
        }
        return samples_dict
