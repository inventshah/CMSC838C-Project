from models import DeepBRDFVAE, ColorWiseVAE
from data import BRDFDataset, undo_log_norm
from render import render_as_img, render_slice, IDX

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb

os.environ["WANDB_SILENT"] = "true"
os.environ["WAND_DISABLE_CODE"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--EPOCHS", "-e", type=int, default=2000, required=False)
parser.add_argument("--BATCH_SIZE", "-b", type=int, default=5, required=False)
parser.add_argument("--SAVE_EVERY", "-s", type=int, default=100, required=False)
parser.add_argument(
    "--CHECKPOINT", "-c", type=str, default="checkpoint", required=False
)
parser.add_argument(
    "--MERL", "-m", type=str, default="BRDFDatabase/brdfs", required=False
)
parser.add_argument(
    "--RGL", "-r", type=str, default="BRDFDatabase/rgl-brdfs", required=False
)
parser.add_argument("--LEARN_RATE", "-l", type=float, default=3e-5, required=False)
parser.add_argument("--Z_DIM", "-z", type=int, default=4, required=False)
parser.add_argument("--COLORWISE", "-o", action="store_true", required=False)

args = parser.parse_args(sys.argv[1:])

SAVE_EVERY = args.SAVE_EVERY
CHECKPOINT_DIR = args.CHECKPOINT
MERL_DIR = args.MERL
RGL_DIR = args.RGL
EPOCHS = args.EPOCHS
BATCH_SIZE = args.BATCH_SIZE
LEARN_RATE = args.LEARN_RATE
Z_DIM = args.Z_DIM

seed = 0
# Reproductibilites
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

data = BRDFDataset(MERL_DIR, RGL_DIR, True)
train = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)

model = (
    ColorWiseVAE(z_dim=Z_DIM).to(device)
    if args.COLORWISE
    else DeepBRDFVAE(n_slices=63, z_dim=Z_DIM).to(device)
)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

criterion = torch.nn.MSELoss()

beta = 20 * model.z_dim / (63 * 90 * 90)

wandb.init(name=os.path.basename(CHECKPOINT_DIR))
print(args)

for i in range(1, EPOCHS + 1):
    start = time.time()
    epoch_loss = 0

    recloss = 0
    kld = torch.tensor(0, dtype=torch.float32)
    for j, x in enumerate(train):
        x = x.to(device)[:, IDX]
        mask = (x > 0).to(device)

        optimizer.zero_grad()

        x_recon, mu, logvar = model(x)
        x_recon.where(mask, torch.zeros(x.size()).to(device)).to(device)

        recon_loss = criterion(x, x_recon)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        total_loss = recon_loss + beta * kld_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += total_loss.item()
        recloss += recon_loss.item()
        kld += kld_loss.item()

    if i % 20 == 0:
        example = render_as_img(
            undo_log_norm(
                model(data[4].to(device).unsqueeze(0)[:, IDX])[0]
                .squeeze()
                .detach()
                .cpu()
            ),
            render_slice,
        )
        example2 = render_as_img(
            undo_log_norm(
                model(data[34].to(device).unsqueeze(0)[:, IDX])[0]
                .squeeze()
                .detach()
                .cpu()
            ),
            render_slice,
        )

        wandb.log(
            {
                "img": wandb.Image(example.resize((64, 64))),
                "img2": wandb.Image(example2.resize((64, 64))),
                "total_error": epoch_loss / len(train),
                "recon_error": recloss / len(train),
                "kld_error": kld / len(train),
                **vars(args),
            }
        )

    print(
        f"{i} - tot={epoch_loss/len(train):.4f} (recon={recloss/len(train):.4f} & kld={kld/len(train):.4f})  :: {time.time() - start:.2f}s"
    )

    if i % SAVE_EVERY == 0:
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_{i}.pth"))

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_end.pth"))
