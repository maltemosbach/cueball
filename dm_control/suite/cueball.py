# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cue-ball domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_reach_model():
    return common.read_model('cueball_reach.xml'), common.ASSETS


def get_sink_single_model():
    return common.read_model('cueball_sink_single.xml'), common.ASSETS


def get_sink_pyramid_model():
    return common.read_model('cueball_sink_pyramid.xml'), common.ASSETS


@SUITE.add()
def reach(time_limit=_DEFAULT_TIME_LIMIT, random=None,
          environment_kwargs=None):
    """Returns the easy point_mass task."""
    physics = Physics.from_xml_string(*get_reach_model())
    task = Cueball(random=random, task='reach')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def sink_single(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                environment_kwargs=None):
    """Returns the easy point_mass task."""
    physics = Physics.from_xml_string(*get_sink_single_model())
    task = Cueball(random=random, task='sink_single')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def sink_pyramid(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                 environment_kwargs=None):
    """Returns the easy point_mass task."""
    physics = Physics.from_xml_string(*get_sink_pyramid_model())
    task = Cueball(random=random, task='sink_pyramid')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the cueball domain."""

    def cueball_pos(self):
        return self.named.data.geom_xpos['cueball'][0:2]

    def target_pos(self):
        return self.named.data.mocap_pos['target'][0:2]

    def cueball_to_target(self):
        return (self.cueball_pos() - self.target_pos())

    def cueball_to_target_dist(self):
        """Returns the distance from cueball to the target."""
        return np.linalg.norm(self.cueball_to_target())

    def ball_pos(self, id: int):
        return self.named.data.geom_xpos[f'ball_{id}'][0:2]

    def pocket_pos(self):
        pocket_pos = []
        for id in range(6):
            pocket_pos.append(self.named.data.geom_xpos[f'pocket_{id}'][0:2])
        return np.stack(pocket_pos)

    def ball_in_pocket(self, id: int):
        expanded_ball_pos = np.repeat(np.expand_dims(self.ball_pos(id), 0),
                                      6, axis=0)
        pocket_pos = self.pocket_pos()

        ball_to_pocket_dist = np.linalg.norm(expanded_ball_pos - pocket_pos,
                                             axis=-1)
        ball_in_pocket = np.any(ball_to_pocket_dist < 0.03)
        return ball_in_pocket

    def ball_to_closest_pocket(self, id: int):
        expanded_ball_pos = np.repeat(np.expand_dims(self.ball_pos(id), 0),
                                      6, axis=0)
        pocket_pos = self.pocket_pos()
        ball_to_pocket_dist = np.linalg.norm(expanded_ball_pos - pocket_pos,
                                             axis=-1)
        ball_to_closest_pocket_dist = np.min(ball_to_pocket_dist)
        return ball_to_closest_pocket_dist


class Cueball(base.Task):
    def __init__(self, random=None, task: str = 'reach',
                 sparse: bool = False,
                 target_threshold: float = 0.03,
                 terminate_when_target_reached: bool = False):
        self._task = task
        self._sparse = sparse
        self._target_threshold = target_threshold
        self._terminate_when_target_reached = terminate_when_target_reached
        super().__init__(random=random)

    def initialize_episode(self, physics):
        randomizers.randomize_cueball_position(physics, self.random, x_range=(
            -0.1, 0.1), y_range=(-0.1, 0.1))
        if self._task == 'reach':
            randomizers.randomize_target_position(
                physics, self.random, x_range=(-0.6, 0.6),
                y_range=(-0.3, 0.3))
        elif self._task == 'sink_single':
            randomizers.randomize_ball_position(
                physics, self.random, id=0, x_range=(0.4, 0.475),
                y_range=(0.1, 0.15))
        elif self._task == 'sink_pyramid':
            self._ball_to_sink_id = self.random.choice(np.arange(0, 2))
            print("ball to sink is:", self._ball_to_sink_id)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['cueball_position'] = physics.cueball_pos()

        if self._task == 'reach':
            obs['target_position'] = physics.target_pos()
        elif self._task == 'sink_single':
            obs['ball_0_position'] = physics.ball_pos(0)
        elif self._task == 'sink_pyramid':
            for id in range(3):
                obs[f'ball_{id}_position'] = physics.ball_pos(id)
            obs['ball_to_sink'] = self._ball_to_sink_id
        return obs

    def get_reward(self, physics):
        if self._task == 'reach':
            return self.get_reach_reward(physics)
        elif self._task == 'sink_single':
            return self.get_sink_reward(physics, id=0)
        elif self._task == 'sink_pyramid':
            return self.get_sink_reward(physics, id=self._ball_to_sink_id)
        else:
            assert False

    def get_reach_reward(self, physics):
        dist = physics.cueball_to_target_dist()
        if self._sparse:
            return -(dist > self._target_threshold).astype(np.float32)
        else:
            return -dist.astype(np.float32)

    def get_sink_reward(self, physics, id: int):
        ball_in_pocket = physics.ball_in_pocket(id)
        if self._sparse:
            return -1. + (ball_in_pocket).astype(np.float32)
        else:
            return -physics.ball_to_closest_pocket(id)

    def get_termination(self, physics):
        if self._task == 'reach':
            return self.get_reach_termination(physics)
        elif self._task == 'sink_single':
            return self.get_sink_termination(physics, id=0)
        elif self._task == 'sink_pyramid':
            return self.get_sink_termination(physics, id=self._ball_to_sink_id)
        else:
            assert False

    def get_reach_termination(self, physics):
        if self._terminate_when_target_reached:
            dist = physics.cueball_to_target_dist()
            if dist <= self._target_threshold:
                return 0.

    def get_sink_termination(self, physics, id: int):
        if self._terminate_when_target_reached:
            if physics.ball_in_pocket(id):
                return 0.

