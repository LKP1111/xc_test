from xuance.environment import RawEnvironment
import gymnasium as gym
from gym.spaces import Box
import numpy as np
try:
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
except ImportError:
    pass


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
        if self.rgb_img_partial_obs_wrapper:  # pomdp do not need direction info
            self.image_size = self.env.observation_space['image'].shape
            self.observation_space = Box(low=0, high=255, shape=[*self.image_size], dtype=np.uint8)
        else:
            self.image_size = np.prod(self.env.observation_space['image'].shape)  # height * width * channels
            self.dim_obs = self.image_size + 1  # direction
            self.observation_space = Box(low=0, high=255, shape=[self.dim_obs, ], dtype=np.uint8)
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
        return self._get_obs(obs_raw), info

    def step(self, actions):
        """Execute the actions and get next observations, rewards, and other information."""
        obs_raw, reward, terminated, truncated, info = self.env.step(actions)
        reward *= 10
        return self._get_obs(obs_raw), reward, terminated, truncated, info

    def flatten_obs(self, obs_raw):
        """Convert image observation to vectors"""
        image = obs_raw['image']
        direction = obs_raw['direction']
        observations = np.append(image.reshape(-1), direction)
        return observations

    def _get_obs(self, obs_raw):
        if self.rgb_img_partial_obs_wrapper:
            obs = obs_raw['image']
        else:
            obs = self.flatten_obs(obs_raw)
        return obs