import torch
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


indexes_img = Image.open(f"./indexes.png")
width, height = indexes_img.size
indexes = torch.from_numpy(np.array(indexes_img)).view(-1, 4).long()


def render(brdf: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """render a BRDF from a fixed view/incident direction"""
    ind1 = indexes[:, 0]
    ind2 = indexes[:, 1]
    ind3 = indexes[:, 2]

    red = brdf[ind1 + 180 * 0, ind3, ind2]
    green = brdf[ind1 + 180 * 1, ind3, ind2]
    blue = brdf[ind1 + 180 * 2, ind3, ind2]
    alpha = indexes[:, 3] / 255

    red = torch.pow(red, 1 / gamma)
    green = torch.pow(green, 1 / gamma)
    blue = torch.pow(blue, 1 / gamma)

    colors = torch.stack((red, green, blue, alpha), dim=-1)

    return torch.clamp(colors, min=0, max=1).view(128, 128, 4)


def render_as_img(brdf: torch.Tensor, render=render) -> Image.Image:
    img = render(brdf)
    return Image.fromarray((img * 255).numpy().astype(np.uint8), "RGBA")


def render_slice(brdf: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """render a BRDF from a fixed view/incident direction"""
    ind1 = indexes[:, 0]
    ind2 = indexes[:, 1]
    ind3 = indexes[:, 2]

    t = (ind1 % 9) / 9

    red = (
        brdf[ind1 // 9 + 21 * 0, ind3, ind2] * (1 - t)
        + brdf[torch.clamp(ind1 // 9 + 1, max=21) + 21 * 0, ind3, ind2] * t
    )
    green = (
        brdf[ind1 // 9 + 21 * 1, ind3, ind2] * (1 - t)
        + brdf[torch.clamp(ind1 // 9 + 1, max=21) + 21 * 1, ind3, ind2] * t
    )
    blue = (
        brdf[ind1 // 9 + 21 * 2, ind3, ind2] * (1 - t)
        + brdf[torch.clamp(ind1 // 9 + 1, max=21) + 21 * 2, ind3, ind2] * t
    )
    alpha = indexes[:, 3] / 255

    red = torch.pow(red, 1 / gamma)
    green = torch.pow(green, 1 / gamma)
    blue = torch.pow(blue, 1 / gamma)

    colors = torch.stack((red, green, blue, alpha), dim=-1)

    return torch.clamp(colors, min=0, max=1).view(128, 128, 4)


IDX = np.linspace(1, 20, 20).astype(np.int32) * 9 - 1
IDX = np.array([0, *IDX, 180, *(180 + IDX), 360, *(360 + IDX)])
