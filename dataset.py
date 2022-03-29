from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image


DATA_MODES = ['train', 'test']


class MNISTDetectionData(Dataset):
    def __init__(self, path: Path, mode: str, transform=None):
        super().__init__()
        self.transform = transform
        self.x_files = sorted(path.joinpath("images").iterdir(), key=lambda f: int(f.name[:-4]))
        self.y_files = sorted(path.joinpath("labels").iterdir(), key=lambda f: int(f.name[:-4]))

        if mode not in DATA_MODES:
            raise ValueError
        self.mode = mode

        self.len_ = len(self.x_files)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()

        return image

    def __getitem__(self, item):
        x = self.load_sample(self.x_files[item])
        if self.transform is not None:
            x = self.transform(x)

        if self.mode == 'test':
            return x
        else:
            y = pd.read_csv(self.y_files[item])

            return x, y
