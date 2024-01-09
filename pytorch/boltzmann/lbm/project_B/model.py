import torch
import torch.nn as nn

# Note: D8 = {I, r, r*r, r*r*r, s, r * s, r * r, * s, r * r * r * s}
# r * r: 180 degree rotation

def rot90(f, k=1):
    """Rotation of 90 degrees, r"""
    return torch.concat([
        f[:, 0, None], #[32, 1]
        torch.roll(f[:, 1:5], k, dims=0), #[32, 4]
        torch.roll(f[:,5: ], k, dims=0) #[32, 4]
    ], dim=-1)

def reflection(f):
    """Reflection along x-axis, s"""
    return torch.concat([
        f[:,0, None], 
        f[:,1, None], 
        f[:,4, None], 
        f[:,3, None], 
        f[:,2, None], 
        f[:,8, None], 
        f[:,7, None], 
        f[:,6, None], 
        f[:,5, None]
    ], dim=-1)

# Note: Rotation and symmetry equivariance
class D4Symmetry(nn.Module):
    def __init__(self):
        super(D4Symmetry, self).__init__()

    def forward(self, x):
        # x [32, 9]
        D8 = torch.stack([
            x, 
            rot90(x, k=1), 
            rot90(x, k=2), 
            rot90(x, k=3), 
            reflection(x), 
            reflection(rot90(x, k=1)), 
            reflection(rot90(x, k=2)), 
            reflection(rot90(x, k=3))], dim=0)
        return D8
    
class D4AntiSymmetry(nn.Module):
    def __init__(self):
        super(D4AntiSymmetry, self).__init__()

    def forward(self, x):
        D8inv = torch.stack([
            x[0], 
            rot90(x[1], k=-1), 
            rot90(x[2], k=-2), 
            rot90(x[3], k=-3), 
            reflection(x[4]), 
            rot90(reflection(x[5]), k=-1), 
            rot90(reflection(x[6]), k=-2), 
            rot90(reflection(x[7]), k=-3)
        ], dim=0)
        return D8inv

class SymAlgReconstruction(nn.Module):
    def __init__(self):
        super(SymAlgReconstruction, self).__init__()

    def forward(self, fpre, fpred):
        df  = fpred - fpre

        df2 = -(df[:,0] + 2 * df[:,3] + df[:,4] + 2 * df[:,6] + 2 * df[:,7])
        df5 = 0.5 * (df[:,0] + 3 * df[:,3] + 2 * df[:,4] + 2 * df[:,6] + 4 * df[:,7] - df[:,1])
        df8 = -0.5 * (df[:,0] + df[:,1]+  df[:,3] + 2 * df[:,4] + 2 * df[:,7])
        
        df = torch.concat([
            df[:, 0, None],
            df[:, 1, None],
            df2[:,None],
            df[:, 3, None],
            df[:, 4, None],
            df5[:,None],
            df[:, 6, None],
            df[:, 7, None],
            df8[:,None]
        ], dim=0)

        return fpre + df

class CollisionNN(nn.Module):
    def __init__(self, in_chs, nhidden=50):
        super(CollisionNN, self).__init__()

        self.d4_symmetry = D4Symmetry()

        self.model = nn.Sequential(
            nn.Linear(in_chs, nhidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(nhidden, nhidden, bias=False),
            nn.LeakyReLU(),
            nn.Linear(nhidden, in_chs, bias=False)
        )

        torch.nn.init.xavier_uniform_(self.model[0].weight.data)
        torch.nn.init.xavier_uniform_(self.model[2].weight.data)
        torch.nn.init.xavier_uniform_(self.model[4].weight.data)
        torch.nn.init.xavier_uniform_(self.model[6].weight.data)

        self.alg_reconstruction = SymAlgReconstruction()
        self.d4_antisymmetry = D4AntiSymmetry()

    def forward(self, x):
        out = self.d4_symmetry(x) # [32, 72]
        fpre_pred = self.model(out)
        out = self.alg_reconstruction(x, fpre_pred)
        out = self.d4_antisymmetry(out)
        out = torch.mean(out, dim=0)
        return out
