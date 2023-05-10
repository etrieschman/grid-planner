import torch
import torch.nn as nn
import copy
from tqdm import tqdm

def recon_loss(recon_x, x):
    return nn.functional.mse_loss(recon_x, x, reduction='sum')

def vae_loss(recon_x, x, mu, logvar):
    MSE = recon_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

class VAE(nn.Module):
    def __init__(self, latent_dim, dims):
        super(VAE, self).__init__()
        self.C, self.L, self.D = dims
        self.latent_dim = latent_dim
        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=self.C, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.lstm1 = nn.LSTM(input_size=64*self.D//4, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=self.latent_dim*2, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        # decoder layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3,3), stride=(2,2), padding=(0,0), output_padding=(1,1))
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(0,0))
        self.delstm2 = nn.LSTM(input_size=self.latent_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.delstm1 = nn.LSTM(input_size=32, hidden_size=64*self.D//4, num_layers=1, batch_first=True)

    def encode(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.shape[0], self.L//4, 64*self.D//4)
        x, __ = self.lstm1(x)
        __, (x, __) = self.lstm2(x)
        return x.view(x.shape[1], x.shape[-1])

    def decode(self, z):
        # repeat, reshape, and run LSTM
        z = z.repeat(1, self.L//4)
        z = z.view(z.shape[0], self.L//4, self.latent_dim)
        z, __ = self.delstm2(z)
        z = nn.functional.relu(z)
        z, __ = self.delstm1(z)
        z = nn.functional.relu(z)
        # reshape and convolute
        z = z.view(z.shape[0], self.L//4, 64, self.D//4)
        z = z.permute(0, 2, 1, 3)
        z = nn.functional.relu(self.deconv2(z))
        z = self.deconv1(z)
        # z = z[:z.shape, :self.C, :self.L, :self.D]
        return z


    def forward(self, x):
        # Encoder
        mean, logvar = self.encode(x).chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        # Sample from the latent space
        eps = torch.randn(self.latent_dim)
        z = mean + eps * std
        # Decoder
        xhat = self.decode(z)
        return xhat, mean, logvar
    
    