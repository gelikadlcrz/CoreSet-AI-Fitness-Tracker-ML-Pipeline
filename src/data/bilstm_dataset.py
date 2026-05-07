import os
import torch
from torch.utils.data import Dataset


class BiLSTMDataset(Dataset):
    def __init__(self, data_dir, split_list, max_frames=150):
        self.samples = []
        self.data_dir = data_dir
        self.max_frames = max_frames

        self.class_names = ['bench_press', 'pull_up', 'push_up', 'squat']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for item in split_list:
            pt_file = item.replace(".json", ".pt")

            class_name = os.path.dirname(pt_file)
            full_path = os.path.join(data_dir, pt_file)

            if os.path.exists(full_path):
                label = self.class_to_idx[class_name]
                self.samples.append((full_path, label))
            else:
                print(f"Warning: Missing file {full_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        loaded = torch.load(path)

        data = loaded['sequence'].float()

        rep_count = float(loaded.get('rep_count', 0))

        num_frames = data.shape[0]

        if num_frames > self.max_frames:
            data = data[:self.max_frames]

        elif num_frames < self.max_frames:

            padding = torch.zeros(
                (self.max_frames - num_frames, data.shape[1]),
                dtype=data.dtype
            )

            data = torch.cat([data, padding], dim=0)

        return data, label, rep_count