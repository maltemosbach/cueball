import argparse
from dm_control import suite
import numpy as np
from dm_control.suite.wrappers import pixels
from tqdm import tqdm
from typing import *


DATASET_KEYS = ['is_first', 'image', 'action', 'reward', 'is_last']
POLICY_TYPES = ['random', 'constant']
VIEWPOINT_MAPPING = {'birdseye': 0, 'firstperson': 1}


class Policy:
    def __init__(self, action_spec, policy_type: str = 'random'):
        self.policy_type = policy_type
        self.action_spec = action_spec

    def get_action(self, time_step) -> np.array:
        if self.policy_type == 'random':
            action = self.sample_random_action()
        elif self.policy_type == 'constant':
            if time_step.first():
                self.const_action = self.sample_random_action()
            action = self.const_action
        else:
            assert False
        return action

    def sample_random_action(self) -> np.array:
        return np.random.uniform(
            self.action_spec.minimum, self.action_spec.maximum,
            size=self.action_spec.shape)


def append_to_episode(episode: Dict[str, List], time_step, action=None) -> \
        None:
    episode['is_first'].append(time_step.first())
    episode['image'].append(time_step.observation['pixels'])
    if action is not None:
        episode['action'].append(action)
    episode['reward'].append(time_step.reward)
    episode['is_last'].append(time_step.last())


def create_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='cueball')
    parser.add_argument('--task', type=str, default='sink_single')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--policy', type=str, default='random')
    parser.add_argument('--npz_file', type=str,
                        default='cueball_dataset.npz')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=72)
    parser.add_argument('--viewpoint', type=str, default='birdseye')
    args = parser.parse_args()

    # Find camera index.
    assert args.viewpoint in VIEWPOINT_MAPPING.keys(), \
        f"--viewpoint should be in {VIEWPOINT_MAPPING.keys()}, but found " \
        f"unknown viewpoint '{args.viewpoint}'."

    # Initialize dataset.
    dataset = {k: [] for k in DATASET_KEYS}
    episode = {k: [] for k in DATASET_KEYS}

    # Load and reset environment.
    env = suite.load(domain_name=args.domain, task_name=args.task)
    env = pixels.Wrapper(
        env, pixels_only=True, render_kwargs=dict(
            height=args.height, width=args.width, camera_id=VIEWPOINT_MAPPING[
                args.viewpoint]))
    time_step = env.reset()

    # Initialize policy.
    policy = Policy(env.action_spec(), policy_type=args.policy)

    for episode_count in tqdm(range(args.num_episodes),
                              desc="Creating Cueball-dataset"):
        while True:
            # Query action for the current time-step.
            action = policy.get_action(time_step)
            # Append current time-step (including the action taken) to the
            # episode.
            append_to_episode(episode, time_step, action)

            # Step environment to get the next action.
            time_step = env.step(action)

            if time_step.last():
                append_to_episode(episode, time_step)
                for k in DATASET_KEYS:
                    dataset[k].append(np.array(episode[k]))
                time_step = env.reset()
                episode = {k: [] for k in DATASET_KEYS}
                episode_count += 1
                break

    for k in DATASET_KEYS:
        dataset[k] = np.stack(dataset[k])

    np.savez_compressed(args.npz_file, **dataset)


if __name__ == '__main__':
    create_dataset()
