import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as vision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

z_dim = 60
learning_rate = 5e-4
batch_size = 16
device = torch.device("cuda")
KD_controller = 0.00025
EPOCHS = 51


class q_zx(nn.Module):
    def __init__(self, z_dims, in_channels=1, channels=[32, 64, 64]):
        super(q_zx, self).__init__()
        modules = []
        c1 = self._convblock(1, channels[0], 3, padding='same', bias=True)
        modules.append(*c1)
        c2 = self._convblock(channels[0], channels[1], 3, padding='same', bias=True)
        modules.append(*c2)
        c3 = self._convblock(channels[1], channels[2], 3, padding='same', bias=True)
        modules.append(*c3)
        self.model = nn.Sequential(*modules)
        self.mu_fc = nn.Linear(64 * 64 * 64, z_dims)
        self.logvar_fc = nn.Linear(64 * 64 * 64, z_dims)
        self.flatten = nn.Flatten()

    def _convblock(self, in_channels, out_channels, kernel_size, padding, bias):
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )
        return modules

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        mu_z = self.mu_fc(x)
        logvar_z = self.logvar_fc(x)
        return mu_z, logvar_z


class p_xz(nn.Module):
    def __init__(self, z_dims, out_channels=1, channels=[48, 90, 90], batch_size=1):
        super(p_xz, self).__init__()
        self.batch_size = batch_size
        self.fc = nn.Linear(z_dims, 64 * 64 * 48)
        self.relu_fc = nn.ReLU()

        modules = []
        self.c1 = self._convblock(48, channels[0], kernel_size=3, padding='same', bias=True)
        modules.append(*self.c1)
        self.c2 = self._convblock(channels[0], channels[1], kernel_size=3, padding='same', bias=True)
        modules.append(*self.c2)
        self.c3 = self._convblock(channels[1], channels[2], kernel_size=3, padding='same', bias=True)
        modules.append(*self.c3)
        self.c = nn.Sequential(*modules)

        self.mu_conv = nn.Conv2d(in_channels=channels[2], out_channels=out_channels, kernel_size=3, padding='same',
                                 bias=False)
        self.logvar_conv = nn.Conv2d(in_channels=channels[2], out_channels=out_channels, kernel_size=3, padding='same',
                                     bias=False)

    def _convblock(self, in_channels, out_channels, kernel_size, padding, bias):
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )
        return modules

    def forward(self, x):
        x = self.fc(x)
        x = self.relu_fc(x)
        x = x.view(-1, 48, 64, 64)
        x = self.c(x)
        mu_x = self.mu_conv(x)
        logvar_x = self.logvar_conv(x)
        return mu_x, logvar_x


class VAE(nn.Module):
    def __init__(self, z_dims=60):
        super(VAE, self).__init__()
        self.encoder = q_zx(z_dims=z_dims)
        self.decoder = p_xz(z_dims=z_dims, batch_size=batch_size)

    def encode(self, x):
        mu_z, logvar_z = self.encoder(x)
        # mu_z = (1, z_dims)
        # logvar_z = (1, z_dims)
        return mu_z, logvar_z

    def decode(self, x):
        mu_x, logvar_x = self.decoder(x)
        # mu_x = (batch, 1, 64, 64)
        # logvar_x = (batch, 1, 64, 64)
        return mu_x, logvar_x

    def sample(self, mu_z, logvar_z):
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        return mu_z + eps * std

    def loss_function(self, mu_x, logvar_x, x, mu_z, logvar_z):
        # mu_x = torch.flatten(mu_x, start_dim = 1)
        # logvar_x = torch.flatten(logvar_x, start_dim=1)

        loss_rec = torch.mean(-torch.sum(
            (-0.5 * np.log(2.0 * np.pi))
            + (-0.5 * logvar_x)
            + ((-0.5 / torch.clip(torch.exp(logvar_x), min=1e-5)) * (x - mu_x) ** 2.0),
            dim=1))
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar_z - mu_z ** 2 - logvar_z.exp(), dim=1), dim=0)
        return loss_rec + 0.00025 * KLD

    def loss(self, z_mean, z_logvar, recon_mean, recon_logvarinv, x):
        ####
        # TODO:
        ####
        recon_logvar = torch.clip(torch.exp(recon_logvarinv), min=1e-5)

        return 1

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z_sampled = self.sample(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z_sampled)
        return mu_z, logvar_z, mu_x, logvar_x


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            0.5, 0.5
        ),
    ]
)
# ADD DATA PATH
dataset = vision.datasets.ImageFolder('', transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_data_sample():
    train_loader2 = DataLoader(dataset, batch_size=16, shuffle=True)
    loader = enumerate(train_loader2)
    data = next(loader)
    return data[1][0]


model = VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mu_z, logvar_z, mu_x, logvar_x = model(data)
        loss = model.loss_function(mu_x, logvar_x, data, mu_z, logvar_z)
        loss.backward()
        train_loss += loss.mean().item()
        optimizer.step()
        if batch_idx % 1200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            mu_z, logvar_z, mu_x, logvar_x = model(data)
            recon_batch = mu_x
            test_loss += model.loss_function(mu_x, logvar_x, data, mu_z, logvar_z)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                save_image(comparison.cpu(), 'reconstruction_' + str(epoch) + '.png', nrow=n)


for l in model.modules():
    if isinstance(l, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(l.weight.data, 0.0, 0.05)

for epoch in range(1, EPOCHS):
    train(epoch)
    if epoch % 5 == 0:
        test(epoch)
        filename = 'model-vae-epoch_' + str(epoch) + '.pth.tar'
        checkpoint = {'save_dic': model.state_dict()}
        torch.save(checkpoint, filename)
