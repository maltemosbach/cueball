import cv2
import numpy as np
import pygame
import torch
from torch.utils.data import Dataset


class CueballDataset(Dataset):
    """Dataset for Cueball tasks."""

    def __init__(self, npz_file: str, fps: int = 20, transform=None) -> None:
        self.npz_file = npz_file
        self.dataset = np.load(npz_file, allow_pickle=True)
        self.fps = fps
        self.transform = transform

    def __len__(self):
        return self.dataset[list(self.dataset.keys())[0]].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {k: v[idx] for k, v in self.dataset.items()}

    def play_episode(self, idx, show_action: bool = True,
                     image_scale: int = 10):
        if idx >= len(self):
            raise IndexError(
                f"Episode index {idx} out of bounds for dataset of size "
                f"{len(self)}.")

        episode = self.__getitem__(idx)
        image_sequence = episode['image']
        height, width = image_sequence.shape[1], image_sequence.shape[2]

        if show_action:

            action_image_sequence = []
            for action in episode['action']:
                action_image = 255 * np.ones((height, height, 3), dtype=np.uint8)
                action_image = cv2.arrowedLine(
                    action_image, (int(height/2), int(height/2)),
                    (int(height/2 + height/2 * action[0]),
                     int(height/2 - height/2 * action[1])),
                    (0, 0, 0), int(height/36))
                action_image_sequence.append(action_image)

            # Append no action for last image.
            action_image = 255 * np.ones((height, height, 3), dtype=np.uint8)
            action_image_sequence.append(action_image)

            action_image_sequence = np.stack(action_image_sequence)
            image_sequence = np.concatenate(
                [image_sequence, action_image_sequence], axis=2)
            width = width + height

        pygame.init()
        screen = pygame.display.set_mode(
            (image_scale * width, image_scale * height))
        clock = pygame.time.Clock()

        for time_step, image in enumerate(image_sequence):
            image = cv2.resize(
                image, (image_scale * width, image_scale * height),
                interpolation=cv2.INTER_LINEAR)
            surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.set_caption(f'Episode {idx}; Time-step {time_step}')
            pygame.display.flip()
            clock.tick(self.fps)
