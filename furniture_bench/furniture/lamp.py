import numpy as np

from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.lamp_base import LampBase
from furniture_bench.furniture.parts.lamp_bulb import LampBulb
from furniture_bench.furniture.parts.lamp_hood import LampHood
from furniture_bench.config import config


class Lamp(Furniture):
    def __init__(self, env_idx):
        super().__init__(env_idx)
        self.name = "lamp"
        furniture_conf = config["furniture"]["lamp"]
        self.furniture_conf = furniture_conf

        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            LampBase(furniture_conf["lamp_base"], 0, env_idx),
            LampBulb(furniture_conf["lamp_bulb"], 1, env_idx),
            LampHood(furniture_conf["lamp_hood"], 2, env_idx),
        ]
        self.num_parts = len(self.parts)

        self.should_be_assembled = [(0, 1), (0, 2)]
        self.position_only.add((0, 2))

        self.should_assembled_first[(0, 2)] = (0, 1)
        self.ori_bound = 0.6 # 0.995

        self.assembled_rel_poses[(0, 1)] = [
            get_mat([0, -0.0678, 0], [0, 0, np.pi]),
        ]
        self.assembled_rel_poses[(0, 2)] = [
            get_mat([0, -0.088324, 0], [0, 0, 0]),
        ]
        self.env_idx = env_idx
