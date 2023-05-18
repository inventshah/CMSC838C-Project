import os
import numpy as np

import torch
from torch.utils.data import Dataset


def read_merl(filename: str) -> np.ndarray:
    """read a MERL .binary file"""
    with open(filename, "rb") as f:
        shape = np.fromfile(f, np.int32, 3)
        data = np.fromfile(f, np.float64)
    data[data < 0] = 0

    cube = data.reshape((3, *shape))

    a, b, c, d = cube.shape
    # phi x phi x theta*channels
    B = np.zeros([b, c, d * a], np.float32)

    # color scaling
    B[:, :, 0:d] = cube[0, :, :, :] * (1.0 / 1500.0)
    B[:, :, d : d * 2] = cube[1, :, :, :] * (1.15 / 1500.0)
    B[:, :, d * 2 :] = cube[2, :, :, :] * (1.66 / 1500.0)
    B = np.moveaxis(B, -1, 0)
    return B


def read_rgl(filename: str) -> np.ndarray:
    """read a RGL .binary file"""
    with open(filename, "rb") as f:
        data = np.fromfile(f, np.float32)
    data[data < 0] = 0

    cube = data.reshape((180, 90, 90, 3))
    cube = np.swapaxes(cube, 0, -1)

    a, b, c, d = cube.shape
    # phi x phi x theta*channels
    B = np.zeros([b, c, d * a], np.float32)

    # color scaling
    B[:, :, 0:d] = cube[0, :, :, :]
    B[:, :, d : d * 2] = cube[1, :, :, :]
    B[:, :, d * 2 :] = cube[2, :, :, :]
    B = np.moveaxis(B, -1, 0)
    return B


def log_norm(brdf: np.ndarray) -> np.ndarray:
    return (np.log(brdf + 0.01) - np.log(0.01)) / (np.log(1.01) - np.log(0.01))


def undo_log_norm(brdf: torch.Tensor) -> torch.Tensor:
    # return brdf
    t = brdf * (np.log(1.01) - np.log(0.01)) + np.log(0.01)
    return torch.exp(t) - 0.01

def load_merl_as_tensor(filename) -> torch.Tensor:
    return torch.from_numpy(log_norm(read_merl(filename)))


def load_rgl_as_tensor(filename) -> torch.Tensor:
    return torch.from_numpy(log_norm(read_rgl(filename)))


def load_brdf(filename: str) -> torch.Tensor:
    if filename.endswith(".binary"):
        return load_merl_as_tensor(filename)
    elif filename.endswith(".bsdf"):
        return load_rgl_as_tensor(filename)
    else:
        raise ValueError(f"Unknown filetype {filename!r}")

class BRDFDataset(Dataset):
    """MERL and RGL Dataset"""

    def __init__(self, merl_folder: str, rgl_folder: str, preload: bool = False):
        self.files = sorted(
            os.path.join(merl_folder, file)
            for file in os.listdir(merl_folder)
            if file.endswith(".binary") and not file.startswith(".")
        ) + sorted(
            os.path.join(rgl_folder, file)
            for file in os.listdir(rgl_folder)
            if file.endswith(".bsdf") and not file.startswith(".")
        )

        if preload:
            self.data = tuple(map(load_brdf, self.files))

        self.preload = preload

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.preload:
            return self.data[idx]
        mat_name = self.files[idx]
        return load_brdf(mat_name)

