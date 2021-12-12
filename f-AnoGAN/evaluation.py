"""
Evaluate performance of trained f-AnoGAN
"""

import torch
import torch.nn as nn
from model_wgan import Critic, Generator, load_checkpoint, save_checkpoint
from encoder_model import Encoder
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(device)
load_checkpoint("generator.pth.tar", gen)

encoder = Encoder(in_channels=3, num_hidden=128, residual_inter=32).to(device)
load_checkpoint("encoder.pth.tar", encoder)

critic = Critic().to(device)
load_checkpoint("critic.pth.tar", critic)


def anomaly_score_izif(x, gen, critic, encoder, weighting_factor):
    gen.eval()
    critic.eval()
    encoder.eval()
    x = x.to(device)

    # A_r(x)
    e_x = encoder(x)
    g_ex = gen(e_x)
    a_r = F.mse_loss(x, g_ex)

    # A_d(x)
    f_x = critic(x)
    f_gex = critic(g_ex)
    a_d = F.mse_loss(f_x, f_gex)

    score = a_r + weighting_factor * a_d
    return score


def anomaly_score_izi(x, gen, critic, encoder):
    gen.eval()
    critic.eval()
    encoder.eval()
    x = x.to(device)

    e_x = encoder(x)
    g_ex = gen(e_x)
    a_r = F.mse_loss(x, g_ex)

    return a_r

# TODO: for score evaluation of a dataset: loop through dataset and accumulate scores