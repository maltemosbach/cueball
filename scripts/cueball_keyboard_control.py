import argparse
from dm_control import suite
import numpy as np
import pygame
import time


viewpoint_mapping = {'birdseye': 0,
                     'firstperson': 1}

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='cueball')
parser.add_argument('--task', type=str, default='reach')
parser.add_argument('--fps', type=int, default=20)
parser.add_argument('--viewpoint', type=str, default='birdseye')
args = parser.parse_args()

# Find camera index.
assert args.viewpoint in viewpoint_mapping.keys(), \
    f"--viewpoint should be in {viewpoint_mapping.keys()}, but found " \
    f"unknown viewpoint '{args.viewpoint}'."

# Mapping from pygame-keys to actions.
control_mapping = {
    pygame.K_UP: np.array([0., 1.]),
    pygame.K_DOWN: np.array([0., -1.]),
    pygame.K_LEFT: np.array([-1., 0.]),
    pygame.K_RIGHT: np.array([1., 0.])
}
key_is_down = {k: False for k in control_mapping.keys()}

# Load environment.
env = suite.load(domain_name=args.domain, task_name=args.task)
action_spec = env.action_spec()
time_step = env.reset()

# Initialize pygame.
pygame.init()
screen = pygame.display.set_mode((1920, 1080))
clock = pygame.time.Clock()

running = True
while running:
    # Render environment.
    image = env.physics.render(width=1920, height=1080,
                               camera_id=viewpoint_mapping[args.viewpoint])
    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(args.fps)

    # Get keyboard inputs.
    pygame.event.pump()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key in control_mapping.keys():
                key_is_down[event.key] = True
        elif event.type == pygame.KEYUP:
            if event.key in control_mapping.keys():
                key_is_down[event.key] = False

    # Map keyboard inputs to actions.
    action = np.array([0., 0.])
    for control_key, control_action in control_mapping.items():
        if key_is_down[control_key]:
            action += control_action
    obs = env.step(action)

    print("obs:", obs)

    if time_step.last():
        time.sleep(2)
        time_step = env.reset()
