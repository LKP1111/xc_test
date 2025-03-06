from collections import deque

from xuance.environment import RawEnvironment
import gymnasium as gym
from gym.spaces import Box
import numpy as np
try:
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
except ImportError:
    pass

try:
    import cv2
except ImportError:
    print("The module opencv-python might not be installed."
          "Please ensure you have installed opencv-python via `pip install opencv-python==4.5.4.58`.")


class MiniGridEnv(RawEnvironment):
    """
    The wrapper of minigrid environment.

    Args:
        config: the configurations for the environment.
    """
    def __init__(self, config):
        super(MiniGridEnv, self).__init__()
        self.rgb_img_partial_obs_wrapper = config.RGBImgPartialObsWrapper
        img_obs_wrapper = config.ImgObsWrapper
        self.env = gym.make(config.env_id, render_mode=config.render_mode)
        if self.rgb_img_partial_obs_wrapper:
            self.env = RGBImgPartialObsWrapper(self.env)
        if img_obs_wrapper:
            self.env = ImgObsWrapper(self.env)
        self.env.reset(seed=config.env_seed)

        self.env_id = config.env_id
        self.render_mode = config.render_mode

        # for pixel obs
        self.num_stack = config.num_stack
        self.frames = deque([], maxlen=self.num_stack)
        self.image_size = self.env.observation_space['image'].shape[:2]
        if self.rgb_img_partial_obs_wrapper:  # pomdp do not need direction info
            # self.env.observation_space['image'].shape: (56, 56, 3)
            # there is no grayscale setting?
            self.image_size = self.image_size if config.img_size is None else config.img_size
            self.observation_space = Box(low=0, high=255,
                                         shape=(self.image_size + [self.num_stack * 3,]), dtype=np.uint8)
        else:
            flat_vec = np.prod(self.env.observation_space['image'].shape)  # height * width * channels
            dim_obs = flat_vec + 1  # direction
            self.observation_space = Box(low=0, high=255, shape=(dim_obs, ), dtype=np.uint8)
        self.action_space = self.env.action_space
        self.max_episode_steps = self.env.env.env.max_steps

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Return the rendering result"""
        return self.env.render()

    def reset(self):
        """Reset the environment."""
        obs_raw, info = self.env.reset()
        if self.rgb_img_partial_obs_wrapper:
            # stack reset obs
            for _ in range(self.num_stack):
                self.frames.append(self.resize_pixel_obs(obs_raw['image']))
        else:
            self.frames.append(obs_raw)
        return self._get_obs(), info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        obs_raw, reward, terminated, truncated, info = self.env.step(actions)
        reward *= 10
        if self.rgb_img_partial_obs_wrapper:
            self.frames.append(self.resize_pixel_obs(obs_raw['image']))
        else:
            self.frames.append(obs_raw)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        if self.rgb_img_partial_obs_wrapper:
            assert len(self.frames) == self.num_stack
            obs = LazyFrames(list(self.frames))
        else:
            obs = self.flatten_raw_obs(self.frames.pop())
        return obs

    # get pixel obs and resize
    def resize_pixel_obs(self, frame):
        return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)

    def flatten_raw_obs(self, obs_raw):
        """Convert image observation to vectors"""
        image = obs_raw['image']
        direction = obs_raw['direction']
        obs = np.append(image.reshape(-1), direction)
        return obs


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to the model.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]
