import torch
import torch.nn as nn
import copy
from tqdm import tqdm

def conv_block(inC, outC, strideT, dropout):
    cb = nn.Sequential(
        nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(3,3), 
                  stride=(strideT,1), padding=(0,2)),
        nn.BatchNorm2d(outC),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), padding=(0,1))
    )
    if dropout:
        return nn.Sequential(nn.Dropout(p=0.1), cb)
    return cb

def deconv_block(inC, outC, strideT, padT, activation):
    dcb = nn.ConvTranspose2d(in_channels=inC, out_channels=outC, kernel_size=(3,3), 
                             stride=(strideT,1), padding=(padT,1), output_padding=(1,0))
    if activation:
        return nn.Sequential(dcb, nn.ReLU(), nn.BatchNorm2d(outC))
    return dcb


class Encoder(nn.Module):
    def __init__(self, latent_dim, dims):
        super(Encoder, self).__init__()
        self.C, self.L, self.D = dims
        self.latent_dim = latent_dim
        # encoder layers
        self.lstm = nn.LSTM(input_size=512*3, hidden_size=64, num_layers=1, batch_first=True)
        self.fc_mean = nn.Linear(in_features=64, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=64, out_features=latent_dim)

    def forward(self, x):
        x = conv_block(self.C, 128, 2, False)(x)
        x = conv_block(128, 256, 2, False)(x)
        x = conv_block(256, 512, 1, False)(x)
        x = x.view(x.shape[0], x.shape[2], 512*3)
        __, (x, __) = self.lstm(x)
        x = x.view(x.shape[1], x.shape[2])
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, dims):
        super(Decoder, self).__init__()
        self.C, self.L, self.D = dims
        self.latent_dim = latent_dim
        # decoder layers
        self.fc_time = nn.Linear(in_features=latent_dim, out_features=self.L//(2**3))
        self.delstm1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.delstm2 = nn.LSTM(input_size=32, hidden_size=128*3, num_layers=1, batch_first=True)
        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear')

    def forward(self, z):
        # repeat, reshape, and run LSTM
        z = self.fc_time(z)
        z = z.view(z.shape[0], self.L//(2**3), 1)
        z = nn.functional.relu(z)
        z, __ = self.delstm1(z)
        z, __ = self.delstm2(z)
        z = z.view(z.shape[0], z.shape[1], 128, 3).permute(0, 2, 1, 3)
        z = deconv_block(128, self.C*2, 2, 1, True)(z)
        z = self.upsample(z)
        z = deconv_block(self.C*2, self.C, 2, 1, False)(z)
        return z

class Encoder_n(nn.Module):
    def __init__(self, latent_dim, dims):
        super(Encoder_n, self).__init__()
        self.C, self.L, self.D = dims
        self.latent_dim = latent_dim
        # encoder layers
        self.fc_mean = nn.Linear(in_features=64, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=64, out_features=latent_dim)

    def forward(self, x):
        x = conv_block(self.C, 32, 2, False)(x)
        x = conv_block(32, 64, 2, False)(x)
        x = conv_block(64, 128, 2, False)(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[1]*x.shape[3])

        __, (x, __) = nn.LSTM(input_size=x.shape[1]*x.shape[3], hidden_size=64, 
                       num_layers=1, batch_first=True)(x)
        x = x.view(x.shape[1], x.shape[2])
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    

class Decoder_n(nn.Module):
    def __init__(self, latent_dim, dims):
        super(Decoder_n, self).__init__()
        self.C, self.L, self.D = dims
        self.latent_dim = latent_dim
        # decoder layers
        self.fc_time = nn.Linear(in_features=latent_dim, out_features=self.L//(2**3))
        self.delstm1 = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.delstm2 = nn.LSTM(input_size=32, hidden_size=128*3, num_layers=1, batch_first=True)
        self.upsample = nn.Upsample(scale_factor=(2,2), mode='bilinear')

    def forward(self, z):
        # repeat, reshape, and run LSTM
        z = self.fc_time(z)
        z = z.view(z.shape[0], self.L//(2**3), 1)
        z = nn.functional.relu(z)
        z, __ = self.delstm1(z)
        z, __ = self.delstm2(z)
        z = z.view(z.shape[0], z.shape[1], 128, 3).permute(0, 2, 1, 3)
        z = deconv_block(128, self.C*8, 2, 1, True)(z)
        z = self.upsample(z)
        z = deconv_block(self.C*8, self.C, 2, 1, False)(z)
        return z
    

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder, latent_dim, dims):
        super(VAE, self).__init__()
        self.C, self.L, self.D = dims
        self.latent_dim = latent_dim
        # d/encoder layers
        self.encoder = Encoder(latent_dim, dims)
        self.decoder = Decoder(latent_dim, dims)


    def forward(self, x):
        # Encoder
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        # Sample from the latent space
        eps = torch.randn(self.latent_dim)
        z = mean + eps * std
        # Decoder
        xhat = self.decoder(z)
        return xhat, mean, logvar
    
    