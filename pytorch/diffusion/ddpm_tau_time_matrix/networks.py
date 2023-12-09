import torch
import torch.nn as nn

from utils import GRID_SIZE, INPUT_CHANNELS # rho, vx and vy

# Definitions
VELOCITY_FIELD_SHAPE = (2, GRID_SIZE, GRID_SIZE) # vx and vy -> (2, h, w)
DENSITY_SHAPE = (1, GRID_SIZE, GRID_SIZE)   # rho -> (1, h, w)
INPUT_SHAPE = (INPUT_CHANNELS, GRID_SIZE, GRID_SIZE) # rho (ICs), vx and vy

# Network utils
def sinusoidal_embedding(n, d):
    """Returns the standard positional embedding, d is emb dimension and n is max sequence length"""
    # https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])     # 2i
    embedding[:,1::2] = torch.cos(t * wk[:,::2])    # 2i + 1

    return embedding

# SAGAN
class AttentionBlock(nn.Module):
    # key -> f, query -> g, value -> int(or h)
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0,  bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi * x

class Block(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(Block, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
class UNet(nn.Module):
    def __init__(self, output_channels=2, diffusion_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(diffusion_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(diffusion_steps, time_emb_dim)
        self.time_embed.requires_grad_(False) # Time embedding is not learnt

        # Encoder
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            Block(INPUT_SHAPE, INPUT_CHANNELS, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            Block((10, GRID_SIZE // 2, GRID_SIZE // 2), 10, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            Block((20, GRID_SIZE // 4, GRID_SIZE // 4), 20, 40),
            Block((40, GRID_SIZE // 4, GRID_SIZE // 4), 40, 40),
            Block((40, GRID_SIZE // 4, GRID_SIZE // 4), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=2, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1)
        )

        # Bottleneck
        self.te_mid = self._make_te(dim_in=time_emb_dim, dim_out=40)
        self.b_mid = nn.Sequential(
            Block((40, GRID_SIZE // 8, GRID_SIZE // 8), 40, 20),
            Block((20, GRID_SIZE // 8, GRID_SIZE // 8), 20, 20),
            Block((20, GRID_SIZE // 8, GRID_SIZE // 8), 20, 40)
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2)
        )

        self.te4 = self._make_te(dim_in=time_emb_dim, dim_out=80)
        self.b4 = nn.Sequential(
            Block((80, GRID_SIZE // 4, GRID_SIZE // 4), 80, 40),
            Block((40, GRID_SIZE // 4, GRID_SIZE // 4), 40, 20),
            Block((20, GRID_SIZE // 4, GRID_SIZE // 4), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=1)
        self.te5 = self._make_te(dim_in=time_emb_dim, dim_out=40)
        self.b5 = nn.Sequential(
            Block((40, GRID_SIZE // 2, GRID_SIZE // 2), 40, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 10),
            Block((10, GRID_SIZE // 2, GRID_SIZE // 2), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(dim_in=time_emb_dim, dim_out=20)
        self.b_out = nn.Sequential(
            Block((20, GRID_SIZE, GRID_SIZE), 20, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(in_channels=10, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # x is (N, 2, GRID_SIZE, GRID_SIZE) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)

        # Encoder
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 64, 64)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 32, 32)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 16, 16)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 8, 8)

        # Decoder
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 16, 16)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 16, 16)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 32, 32)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 32, 32)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 64, 64)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 10, 64, 64)

        out = self.conv_out(out)    # (N, output_channels, 64, 64)

        return out

    def _make_te(self, dim_in, dim_out):
        """Temporal embedding with MLP"""
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    
# Note: Trying to approximate https://arxiv.org/pdf/2301.11661.pdf
class AttentionUNet(nn.Module):
    def __init__(self, output_channels=6, diffusion_steps=1000, time_emb_dim=100):
        super(AttentionUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(diffusion_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(diffusion_steps, time_emb_dim)
        self.time_embed.requires_grad_(False) # Time embedding is not learnt

        # Encoder
        self.b1 = nn.Sequential(
            Block(INPUT_SHAPE, INPUT_SHAPE[0], 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.b2 = nn.Sequential(
            Block((10, GRID_SIZE // 2, GRID_SIZE // 2), 10, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te_encoding = self._make_te(time_emb_dim, 20)
        self.att_encoder = AttentionBlock(f_g=20, f_l=20, f_int=20) # Attention
        self.b3 = nn.Sequential(
            Block((20, GRID_SIZE // 4, GRID_SIZE // 4), 20, 40),
            Block((40, GRID_SIZE // 4, GRID_SIZE // 4), 40, 40),
            Block((40, GRID_SIZE // 4, GRID_SIZE // 4), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=2, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1)
        )

        # Bottleneck
        #self.te_mid = self._make_te(dim_in=time_emb_dim, dim_out=40)
        self.att_mid = AttentionBlock(f_g=40, f_l=40, f_int=40) # Attention
        self.b_mid = nn.Sequential(
            Block((40, GRID_SIZE // 8, GRID_SIZE // 8), 40, 20),
            Block((20, GRID_SIZE // 8, GRID_SIZE // 8), 20, 20),
            Block((20, GRID_SIZE // 8, GRID_SIZE // 8), 20, 40)
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2)
        )
        self.b4 = nn.Sequential(
            Block((80, GRID_SIZE // 4, GRID_SIZE // 4), 80, 40),
            Block((40, GRID_SIZE // 4, GRID_SIZE // 4), 40, 20),
            Block((20, GRID_SIZE // 4, GRID_SIZE // 4), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(in_channels=20, out_channels=20, kernel_size=4, stride=2, padding=1)
        self.b5 = nn.Sequential(
            Block((40, GRID_SIZE // 2, GRID_SIZE // 2), 40, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 10),
            Block((10, GRID_SIZE // 2, GRID_SIZE // 2), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(dim_in=time_emb_dim, dim_out=20)
        self.att_decoder = AttentionBlock(f_g=20, f_l=20, f_int=20) # Attention
        self.b_out = nn.Sequential(
            Block((20, GRID_SIZE, GRID_SIZE), 20, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(in_channels=10, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # x is (N, 2, GRID_SIZE, GRID_SIZE) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)

        # Encoder
        out1 = self.b1(x)
        out2 = self.b2(self.down1(out1))
        # [16, 20, 32, 32]
        out_te3 = self.te_encoding(t).reshape(n, -1, 1, 1) # [16, 20, 1, 1]
        att_encoder_in = self.down2(out2) + out_te3
        att_encoder = self.att_encoder(g=att_encoder_in, x=att_encoder_in) # Attention
        out3 = self.b3(att_encoder)

        # Bottleneck
        #out_te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
        #att_mid_in = self.down3(out3) + out_te_mid
        att_mid_in = self.down3(out3)
        att_mid = self.att_mid(g=att_mid_in, x=att_mid_in) # Attention
        out_mid = self.b_mid(att_mid) # [16, 40, 8, 8]

        # Decoder
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5)

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out_te_out = self.te_out(t).reshape(n, -1, 1, 1)
        att_decoder_in = out + out_te_out
        att_decoder = self.att_decoder(g=att_decoder_in, x=att_decoder_in) # Attention
        out = self.b_out(att_decoder) # [16, 10, 64, 64]

        out = self.conv_out(out)    # (N, 1, 64, 64)

        return out

    def _make_te(self, dim_in, dim_out):
        """Temporal embedding with MLP"""
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
