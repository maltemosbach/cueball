import collections
from dm_control.suite import base
from dm_control.suite.cueball import CueballPhysics
from dm_control.suite.utils import randomizers
import numpy as np
from typing import *


class CueballSortBoxesPhysics(CueballPhysics):
    def box_pos(self, color: str, id: int) -> np.array:
        return self.named.data.geom_xpos[f'{color}_box_{id}'][0:2]

    def target_zone_range(self, color: str) -> Tuple[np.array, np.array]:
        target_zone_center = self.named.data.geom_xpos[
            f'{color}_target_zone'][0:2]
        target_zone_size = self.model.geom(f'{color}_target_zone').size[0:2]
        lower = target_zone_center - target_zone_size
        upper = target_zone_center + target_zone_size
        return lower, upper

    def box_in_target_zone(self, color: str, id: int) -> bool:
        box_pos = self.box_pos(color, id)
        lower, upper = self.target_zone_range(color)
        return np.all(box_pos > lower) and np.all(box_pos < upper)


class CueballSortBoxes(base.Task):
    def __init__(
            self,
            random=None,
            sparse: bool = False,
            terminate_on_completion: bool = False
    ) -> None:
        self._sparse = sparse
        self._terminate_on_completion = terminate_on_completion
        super().__init__(random=random)

    def initialize_episode(self, physics):
        randomizers.randomize_cueball_position(physics, self.random, x_range=(
            -0.1, 0.1), y_range=(-0.1, 0.1))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['cueball_position'] = physics.cueball_pos()
        for color in ['yellow', 'blue']:
            for id in range(2):
                obs[f'{color}_box_{id}_position'] = physics.box_pos(color, id)
        return obs

    def get_reward(self, physics):
        boxes_in_target_zone = 0
        for color in ['yellow', 'blue']:
            for id in range(2):
                boxes_in_target_zone += physics.box_in_target_zone(
                    color, id).astype(float)
        if self._sparse:
            return -1. + float(boxes_in_target_zone == 4.)
        else:
            return boxes_in_target_zone

    def get_termination(self, physics):
        if self._terminate_on_completion:
            if self.get_completion(physics):
                return 0.

    def get_completion(self, physics):
        boxes_in_target_zone = 0
        for color in ['yellow', 'blue']:
            for id in range(2):
                boxes_in_target_zone += physics.box_in_target_zone(
                    color, id).astype(float)
        return boxes_in_target_zone == 4.
