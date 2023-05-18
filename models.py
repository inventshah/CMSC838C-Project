import torch
from torch import nn
import math


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class DeepBRDFVAE(nn.Module):
    def __init__(self, n_slices: int = 540, z_dim: int = 8):
        super(DeepBRDFVAE, self).__init__()
        self.z_dim = z_dim

        self.tanh = nn.Tanh()

        self.conv_layer1 = nn.Conv2d(n_slices, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv_layer2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv_layer_res = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convres_bn = nn.BatchNorm2d(64)

        self.lrelu = nn.LeakyReLU()

        self.fc1 = nn.Linear(64 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.z_dim * 2)
        self.dfc3 = nn.Linear(self.z_dim, 256)
        self.dfc2 = nn.Linear(256, 1024)
        self.dfc1 = nn.Linear(1024, 64 * 12 * 12)
        self.deconv_layer1 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )
        self.deconv_layer2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )
        self.deconv_layer3 = nn.ConvTranspose2d(
            64, n_slices, kernel_size=4, stride=2, padding=1
        )

    def _resBlock(self, x):
        res = x
        out = self.conv_layer_res(x)
        out = self.conv_layer_res(out)
        out = self.lrelu(out)
        out += res

        return out

    def _encode(self, x):
        out = self.conv_layer1(x)
        out = self.conv1_bn(out)
        out = self.lrelu(out)

        out = self.conv_layer2(out)
        out = self.conv2_bn(out)
        out = self.lrelu(out)

        out = self.conv_layer3(out)
        out = self.conv3_bn(out)
        out = self.lrelu(out)

        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self.convres_bn(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.lrelu(out)

        out = self.fc2(out)
        out = self.lrelu(out)

        out = self.fc3(out)
        return out

    def _decode(self, z):
        out = self.dfc3(z)

        out = self.lrelu(out)

        out = self.dfc2(out)
        out = self.lrelu(out)

        out = self.dfc1(out)

        out = out.view(out.size(0), 64, 12, 12)

        out = self.deconv_layer1(out)
        out = self.lrelu(out)

        out = self.deconv_layer2(out)
        out = self.lrelu(out)

        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self.deconv_layer3(out)

        return out

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar


class DeepBRDFTCVAE(nn.Module):
    def __init__(self, n_slices: int = 63, z_dim: int = 8):
        super(DeepBRDFTCVAE, self).__init__()
        self.z_dim = z_dim

        self.tanh = nn.Tanh()

        self.conv_layer1 = nn.Conv2d(n_slices, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv_layer2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv_layer_res = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convres_bn = nn.BatchNorm2d(64)

        self.lrelu = nn.LeakyReLU()

        self.fc1 = nn.Linear(64 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.z_dim * 2)
        self.dfc3 = nn.Linear(self.z_dim, 256)
        self.dfc2 = nn.Linear(256, 1024)
        self.dfc1 = nn.Linear(1024, 64 * 12 * 12)
        self.deconv_layer1 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )
        self.deconv_layer2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )
        self.deconv_layer3 = nn.ConvTranspose2d(
            64, n_slices, kernel_size=4, stride=2, padding=1
        )

        self.criterion = torch.nn.MSELoss()

        self.alpha = 1
        self.beta = 6
        self.gamma = 1
        self.anneal_steps = 1000
        self.num_iter = 0

    def _resBlock(self, x):
        res = x
        out = self.conv_layer_res(x)
        out = self.conv_layer_res(out)
        out = self.lrelu(out)
        out += res

        return out

    def _encode(self, x):
        out = self.conv_layer1(x)
        out = self.conv1_bn(out)
        out = self.lrelu(out)

        out = self.conv_layer2(out)
        out = self.conv2_bn(out)
        out = self.lrelu(out)

        out = self.conv_layer3(out)
        out = self.conv3_bn(out)
        out = self.lrelu(out)

        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self.convres_bn(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.lrelu(out)

        out = self.fc2(out)
        out = self.lrelu(out)

        out = self.fc3(out)
        return out

    def _decode(self, z):
        out = self.dfc3(z)

        out = self.lrelu(out)

        out = self.dfc2(out)
        out = self.lrelu(out)

        out = self.dfc1(out)

        out = out.view(out.size(0), 64, 12, 12)

        out = self.deconv_layer1(out)
        out = self.lrelu(out)

        out = self.deconv_layer2(out)
        out = self.lrelu(out)

        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self.deconv_layer3(out)

        return out

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, x, mu, logvar, z

    def log_density_gaussian(
        self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self, recons, input, mu, log_var, z) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = self.criterion(recons, input)

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(
            z.view(batch_size, 1, latent_dim),
            mu.view(1, batch_size, latent_dim),
            log_var.view(1, batch_size, latent_dim),
        )

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        dataset_size = 151  # dataset size
        strat_weight = (dataset_size - batch_size) / (dataset_size * (batch_size))
        importance_weights = (
            torch.Tensor(batch_size, batch_size)
            .fill_(1 / (batch_size))
            .to(input.device)
        )
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        self.num_iter += 1
        anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)

        loss = (
            recons_loss
            + self.alpha * mi_loss
            + (self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)
        )

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
            "TC_Loss": tc_loss,
            "MI_Loss": mi_loss,
        }


class ColorWiseVAE(nn.Module):
    def __init__(self, n_slices: int = 21, z_dim: int = 4):
        super(ColorWiseVAE, self).__init__()
        self.z_dim = z_dim
        self.n_slices = n_slices

        self.tanh = nn.Tanh()

        self.conv_layer1 = nn.Conv2d(n_slices, 32, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv_layer_res = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.convres_bn = nn.BatchNorm2d(64)

        self.lrelu = nn.LeakyReLU()

        self.fc1 = nn.Linear(64 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.z_dim * 2)
        self.dfc3 = nn.Linear(self.z_dim, 256)
        self.dfc2 = nn.Linear(256, 1024)
        self.dfc1 = nn.Linear(1024, 64 * 12 * 12)
        self.deconv_layer1 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )
        self.deconv_layer2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )
        self.deconv_layer3 = nn.ConvTranspose2d(
            64, n_slices, kernel_size=4, stride=2, padding=1
        )

    def _resBlock(self, x):
        res = x
        out = self.conv_layer_res(x)
        out = self.conv_layer_res(out)
        out = self.lrelu(out)
        out += res

        return out

    def _encode(self, x):
        x = x.reshape((-1, self.n_slices, x.shape[-2], x.shape[-1]))
        out = self.conv_layer1(x)
        out = self.conv1_bn(out)
        out = self.lrelu(out)

        out = self.conv_layer2(out)
        out = self.conv2_bn(out)
        out = self.lrelu(out)

        out = self.conv_layer3(out)
        out = self.conv3_bn(out)
        out = self.lrelu(out)

        out = self._resBlock(out)
        out = self._resBlock(out)
        out = self.convres_bn(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.lrelu(out)

        out = self.fc2(out)
        out = self.lrelu(out)

        out = self.fc3(out)
        return out

    def _decode(self, z) -> torch.Tensor:
        out = self.dfc3(z)

        out = self.lrelu(out)

        out = self.dfc2(out)
        out = self.lrelu(out)

        out = self.dfc1(out)

        out = out.view(out.size(0), 64, 12, 12)

        out = self.deconv_layer1(out)
        out = self.lrelu(out)

        out = self.deconv_layer2(out)
        out = self.lrelu(out)

        out = self._resBlock(out)
        out = self.deconv_layer3(out)

        return out.reshape((-1, self.n_slices * 3, 90, 90))

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        x_recon = x_recon

        return x_recon, mu, logvar
