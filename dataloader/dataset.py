from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch

import os
import json


class CollateFunction:
    def __init__(self, x_lims, y_lims):
        self.x_lims = torch.tensor(x_lims)[None, None, None, :]
        self.y_lims = torch.tensor(y_lims)[None, None, None, :]
        self.num_x_bins = len(x_lims) + 1

    def __call__(self, batch):
        # frames tensor: bs * seq_len * C(1 or 2) * H * W
        # waypoints tensor: bs * seq_len * 8 * 2
        batch = default_collate(batch)
        frames = batch["frames"].permute(1, 0, 2, 3, 4).to(dtype=torch.float32)  # Invert the bs and seq_len

        # Discretize the waypoints
        waypoints = batch["waypoints"]
        waypoints[torch.isnan(waypoints)] = 0  # Replace NaNs

        # Find which bin delta x and y fall into
        x_indices = torch.sum(waypoints[:, :, :, 0:1] > self.x_lims, dim=-1)
        y_indices = torch.sum(waypoints[:, :, :, 1:2] > self.y_lims, dim=-1)
        y_indices = y_indices * self.num_x_bins

        discretized_waypoints = (x_indices + y_indices).permute(1, 0, 2).int()  # Invert the bs and seq_len

        # seq_len * bs * c * h * w
        # seq_len * bs * 8
        batch = {"frames": frames, "actions": discretized_waypoints}

        return batch


def get_collate_fn(x_lims, y_lims):
    x_lims = torch.tensor(x_lims)
    y_lims = torch.tensor(y_lims)

    num_x_bins = len(x_lims) + 1

    x_lims_expanded = x_lims[None, None, None, :]
    y_lims_expanded = y_lims[None, None, None, :]

    def collate_fn(batch):
        # frames tensor: bs * seq_len * C(1 or 2) * H * W
        # waypoints_tensor: bs * seqlen * 8 * 2
        batch = default_collate(batch)
        frames = batch["frames"].permute(1, 0, 2, 3, 4).to(dtype=torch.float32)  # invert the bs and seq_len
        # discretize the waypoints
        waypoints = batch["waypoints"]
        waypoints[torch.isnan(waypoints)] = 0

        # find which bin delta x and y falls into
        x_indices = torch.sum(waypoints[:, :, :, 0:1] > x_lims_expanded, dim=-1)
        y_indices = torch.sum(waypoints[:, :, :, 1:2] > y_lims_expanded, dim=-1)
        y_indices = y_indices * num_x_bins

        discretized_waypoints = (x_indices + y_indices).permute(1, 0, 2)  # invert the bs and seq_len
        batch = {"frames": frames, "actions": discretized_waypoints}

        # seq_len * bs * c * h * w
        # seq_len * bs * 8
        return batch

    return collate_fn


class SequencedDataset(Dataset):
    def __init__(self, data_path, seq_length=12, step=2, road_only=True, samples=None):
        assert seq_length > step

        self.road_only = road_only
        self.data_path = data_path
        self.seq_length = seq_length
        self.step = step
        if samples is None:
            self.samples = self._load_samples()
        else:
            self.samples = samples

    def _load_samples(self):
        samples = []
        for scenario_name in os.listdir(self.data_path):
            scenario_path = os.path.join(self.data_path, scenario_name)
            if os.path.isdir(scenario_path):
                for town_name in os.listdir(scenario_path):
                    town_path = os.path.join(scenario_path, town_name)
                    if os.path.isdir(town_path):
                        for route_name in os.listdir(town_path):
                            route_path = os.path.join(town_path, route_name)
                            if os.path.isdir(route_path):
                                frames_path = os.path.join(route_path, 'topdown')
                                measurements_path = os.path.join(route_path, 'measurements')

                                frame_files = [f for f in os.listdir(frames_path) if f.endswith('.png')]
                                waypoint_files = [f for f in os.listdir(measurements_path) if f.endswith('.json')]
                                assert len(frame_files) == len(waypoint_files)

                                num_frames = len(frame_files)
                                num_sequences = (num_frames - (self.seq_length - self.step)) // self.step

                                if num_sequences > 0:
                                    for seq_num in range(num_sequences):
                                        samples.append((route_path, seq_num))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        route_path, seq_num = self.samples[idx]
        frames_path = os.path.join(route_path, 'topdown')
        measurements_path = os.path.join(route_path, 'measurements')

        frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.png')])
        waypoint_files = sorted([f for f in os.listdir(measurements_path) if f.endswith('.json')])

        # Calculate the start and end frame index for the sequence
        start_idx = seq_num * self.step
        end_idx = start_idx + self.seq_length

        # Load the sequence of frames and waypoints
        frames = [self._load_image(os.path.join(frames_path, frame_files[i])) for i in range(start_idx, end_idx)]
        frames = torch.stack(frames)
        waypoints = [torch.tensor(self._load_json(os.path.join(measurements_path, waypoint_files[i]))['waypoints'],
                                  dtype=torch.float32) for i in range(start_idx, end_idx)]
        waypoints = torch.stack(waypoints)
        # waypoints = self._discretize_waypoints(waypoints)

        if self.road_only:
            frames = frames.unsqueeze(1)  # Resulting shape will be [seq_length, 1, H, W]
        else:
            frames = frames.permute(0, 3, 1, 2)  # Resulting shape will be [seq_length, 2, H, W]

        return {
            'frames': frames,
            'waypoints': waypoints
        }

    def _load_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32)

        # if road_only, return the B channel only, otherwise return G and B channels
        if self.road_only:
            return image_tensor[:, :, 2]
        else:
            return image_tensor[:, :, 1:3]

    def _load_json(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)
