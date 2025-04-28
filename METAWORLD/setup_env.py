from gymnasium.core import Env
import numpy as np
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from config import METAWORLD_CFGS

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.65, 0.0]),
    }

DEFAULT_SIZE=112

class VisualWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, seed):
        super().__init__(env)

        self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
        self.unwrapped.mujoco_renderer = MujocoRenderer(env.model, env.data, default_cam_config=DEFAULT_CAMERA_CONFIG, width=DEFAULT_SIZE, height=DEFAULT_SIZE)

        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seeded_rand_vec = True
        
        self.seed = self.unwrapped.seed(seed)

    def reset(self, **kwargs):
        vector_obs, info = super().reset(**kwargs)
        
        return vector_obs, info

    def step(self, action):
        vector_obs, reward, done, truncate, info = self.env.step(action)

        return vector_obs, reward, done, truncate, info
    
def setup_metaworld_env(task_name:str, seed:int, render_mode:str="rgb_array"):
    cfgs = METAWORLD_CFGS[task_name]
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[cfgs["env_name"]]
    
    env = VisualWrapper(env_cls(render_mode=render_mode), seed)
    
    return env