import torch
import torch.nn as nn
from model_wgan import Generator, load_checkpoint, Critic, save_checkpoint
from encoder_model import Encoder, Residual_block
import torch.cuda
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from statistics import mean

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 128
NUM_EPOCHS = 5
WEIGHTING_FACTOR = 1

# Load generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
load_checkpoint("generator.pth.tar", gen)
critic = Critic().to(device)
load_checkpoint("critic.pth.tar", critic)

# create encoder
encoder = Encoder(in_channels=3, num_hidden=128, residual_inter=32).to(device)

# TODO: Define dataset and dataloader
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)
dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.Adam(encoder.parameters(), LEARNING_RATE, betas=(0.0, 0.9))
criterion = nn.MSELoss()

gen.eval()
critic.eval()


for epoch in range(NUM_EPOCHS):
    size = len(loader.dataset)
    losses = []
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        encoded = encoder(data)
        generated = gen(encoded)

        f_x = critic(data)
        f_gx = critic(generated)

        loss = criterion(generated, data) + WEIGHTING_FACTOR * criterion(f_x, f_gx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    print(f"Epoch {epoch+1}    loss:{mean(losses)}")

encoder_checkpoint = {'state_dic': encoder.state_dict()}
save_checkpoint(encoder_checkpoint, "encoder.pth.tar")










