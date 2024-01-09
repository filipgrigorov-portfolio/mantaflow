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
    

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()

        self.channels = channels
        
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        
        self.ln = nn.LayerNorm([channels])
        
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        
        x_ln = self.ln(x)
        
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        
        attention_value = attention_value + x
        
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

class STAtt(nn.Module):
    """ Traditional space-time attention mechanism """
    def __init__(self, input_dim):
        super(STAtt, self).__init__()

        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        # Note: softmax(QK.T / sqrt(d))
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
    
class STEncoder(nn.Module):
    def __init__(self, input_channels, out_features):
        super(STEncoder, self).__init__()

        self.b1 = nn.Sequential(
            Block((input_channels, GRID_SIZE, GRID_SIZE), input_channels, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10)
        )

        self.down1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4, stride=2, padding=1)

        down_size = GRID_SIZE // 2
        self.b2 = nn.Sequential(
            Block((10, down_size, down_size), 10, 20),
            Block((20, down_size, down_size), 20, 20),
            Block((20, down_size, down_size), 20, 20)
        )

        self.down2 = nn.Conv2d(20, 5, 4, 2, 1)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(in_features=5 * (GRID_SIZE // 8)**2, out_features=out_features, bias=True)

    def forward(self, x):
        out = self.b1(x)
        out = self.down1(out)
        out = self.b2(out)
        out = self.down2(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.view(out.size(0), 1, -1)



class SelfAttentionUNet(nn.Module):
    """ Using SelfAttention """
    def __init__(self, output_channels, diffusion_steps, tau_dim):
        super(SelfAttentionUNet, self).__init__()

        time_emb_dim = 64

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(diffusion_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(diffusion_steps, time_emb_dim)
        self.time_embed.requires_grad_(False) # Time embedding is not learnt

        self.tau_embed = nn.Embedding(tau_dim, time_emb_dim)
        self.tau_embed.weight.data = sinusoidal_embedding(tau_dim, time_emb_dim)
        self.tau_embed.requires_grad_(False) # Time embedding is not learnt


        # debug
        # Note: feature extract to time_emb_dim size
        self.st_encoder_den = STEncoder(input_channels=2*1, out_features=time_emb_dim)
        self.st_encoder_vel = STEncoder(input_channels=2*2, out_features=time_emb_dim)
        self.st_attention = STAtt(input_dim=time_emb_dim)


        # Encoder
        self.b1 = nn.Sequential(
            Block(INPUT_SHAPE, INPUT_SHAPE[0], 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.tau_encoding1 = self._make_te(time_emb_dim, 10) # experiment

        self.b2 = nn.Sequential(
            Block((10, GRID_SIZE // 2, GRID_SIZE // 2), 10, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 20),
            Block((20, GRID_SIZE // 2, GRID_SIZE // 2), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.time_encoding1 = self._make_te(time_emb_dim, 20)
        self.attention1 = SelfAttention(channels=20) # Attention ----------------------------------------

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
        self.attention2 = SelfAttention(channels=40) # Attention ----------------------------------------
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

        self.time_encoding2 = self._make_te(dim_in=time_emb_dim, dim_out=20)
        self.attention3 = SelfAttention(channels=20) # Attention ----------------------------------------
        
        self.b_out = nn.Sequential(
            Block((20, GRID_SIZE, GRID_SIZE), 20, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(in_channels=10, out_channels=output_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, time_step, tau_step):
        time_embedding = self.time_embed(time_step)
        n = len(x)

        tau_embedding = self.tau_embed(tau_step)
        tau_encoding1_output = self.time_encoding1(tau_embedding).reshape(n, -1, 1, 1) # [16, 20, 1, 1]

        # [n, 7, h, w]
        dt = x[:, :1, ...]
        vt = x[:, 1:3, ...]
        dt_prev = x[:, 3:4, ...]
        vt_prev = x[:, 4:-1, ...]
        density_stacked = torch.concat([dt_prev, dt], dim=1)
        velocity_stacked = torch.concat([vt_prev, vt], dim=1)
        
        den_temp_encoded = self.st_encoder_den(density_stacked)
        vel_temp_encoded = self.st_encoder_vel(velocity_stacked)

        den_temp_attention = self.st_attention(den_temp_encoded)
        vel_temp_attention = self.st_attention(vel_temp_encoded)

        den_temp_embedding = self.time_encoding1(den_temp_attention).reshape(n, -1, 1, 1)
        vel_temp_embedding = self.time_encoding1(vel_temp_attention).reshape(n, -1, 1, 1)

        temporal_embedding = den_temp_embedding + vel_temp_embedding


        # Encoder
        b1_output = self.b1(x)
        b2_output = self.b2(self.down1(b1_output)) + tau_encoding1_output + temporal_embedding
        
        time_encoding1_output = self.time_encoding1(time_embedding).reshape(n, -1, 1, 1) # [16, 20, 1, 1]
        down2_and_time_encoding_output = self.down2(b2_output) + time_encoding1_output

        attention1_output = self.attention1(down2_and_time_encoding_output)
        b3_output = self.b3(attention1_output)



        # Bottleneck
        down3_output = self.down3(b3_output)
        attention2_output = self.attention2(down3_output)
        b_mid_output = self.b_mid(attention2_output) # [16, 40, 8, 8]
        out4 = torch.cat((b3_output, self.up1(b_mid_output)), dim=1)            # skip connection



        # Decoder        
        b4_output = self.b4(out4)
        out5 = torch.cat((b2_output, self.up2(b4_output)), dim=1)                    # skip connection

        b5_output = self.b5(out5)
        out = torch.cat((b1_output, self.up3(b5_output)), dim=1)                     # skip connection
     

        time_encoding2_output = self.time_encoding2(time_embedding).reshape(n, -1, 1, 1)

        out_and_time_encoding_output = out + time_encoding2_output
        attention3_output = self.attention3(out_and_time_encoding_output)
        out = self.b_out(attention3_output) # [16, 10, 64, 64]


        return self.conv_out(out)    # (N, 1, 64, 64)


    def _make_te(self, dim_in, dim_out):
        """Temporal embedding with MLP"""
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
    

class UNetTransition(nn.Module):
    def __init__(self, output_channels):
        super(UNetTransition, self).__init__()

        # Encoder
        input_shape = (output_channels, GRID_SIZE, GRID_SIZE)
        self.b1 = nn.Sequential(
            Block(input_shape, output_channels, 10),
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
        self.b_out = nn.Sequential(
            Block((20, GRID_SIZE, GRID_SIZE), 20, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10),
            Block((10, GRID_SIZE, GRID_SIZE), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(in_channels=10, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        out1 = self.b1(x)
        out2 = self.b2(self.down1(out1))
        out3 = self.b3(self.down2(out2))

        out_mid = self.b_mid(self.down3(out3))

        # Decoder
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5)

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out)

        out = self.conv_out(out)    # (N, output_channels, 64, 64)

        return out