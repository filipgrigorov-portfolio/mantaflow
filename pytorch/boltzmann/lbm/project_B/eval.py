import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import stencil, compute_Feq
from model import CollisionNN

# Definitions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_collector(dumpfile, t, ux, uy, rho, nx, ny, dumpit):
    size = nx * ny
    it = t // dumpit
    idx0 =  it * (size)
    idx1 = (it + 1) * (size)
    dumpfile[idx0:idx1, 0] = t
    dumpfile[idx0:idx1, 1] = rho.reshape(size)
    dumpfile[idx0:idx1, 2] = ux.reshape(size)
    dumpfile[idx0:idx1, 3] = uy.reshape(size)

def sol(t, L, F0, nu): 
    return F0 * np.exp(-2 * nu * t / (L / (2 * np.pi))**2  )

def eval():
    """LBM simulation of a Taylor-Green vortex flow, where the BGK operator is replaced by a Neural Network"""
    # Simulation Parameters
    nx      = 32    # grid size along x
    ny      = 32    # grid size along y
    niter   = 1000  # total number of steps
    dumpit  = 100   # collect data every dumpit iterations
    tau     = 1.0   # relaxation time
    u0      = 0.01  # initial velocity amplitude

    # Collect stats
    ndumps   = int(niter // dumpit)
    dumpfile = np.zeros( (ndumps*nx*ny, 4 ) ) 

    # Set Initial conditions
    ix, iy = np.meshgrid(range(nx), range(ny), indexing='ij')
    x = 2.0 * np.pi * (ix / nx)
    y = 2.0 * np.pi * (iy / ny)
    ux =  1.0 * u0 * np.sin(x) * np.cos(y)
    uy = -1.0 * u0 * np.cos(x) * np.sin(y)

    rho = np.ones((nx, ny))

    # Lattice velocities and weights
    Q = 9
    c, w, cs2 = stencil()

    # Lattice 
    feq = np.zeros((nx, ny, Q))
    feq = compute_Feq(feq, rho, ux, uy, c, w, Q, cs2)

    f1 = np.copy(feq)
    f2 = np.copy(feq)

    data_collector(dumpfile, 0, ux, uy, rho, nx, ny, dumpit)
    with torch.no_grad():
        model = CollisionNN(in_chs=Q).to(DEVICE)

        print('Loading pretrained model')
        model.load_state_dict(torch.load("checkpoint_mse.pt"))

    m_initial = np.sum(f1.flatten())

    # Loop on time steps
    for t in range(1, niter):

        # streaming
        for ip in range(Q):
            f1[:, :, ip] = np.roll(np.roll(f2[:, :, ip], c[ip, 0], axis=0), c[ip, 1], axis=1)

        # Calculate density
        rho = np.sum(f1, axis=2)

        # Calculate velocity
        ux = (1.0 / rho) * np.einsum('ijk,k', f1, c[:, 0]) 
        uy = (1.0 / rho) * np.einsum('ijk,k', f1, c[:, 1])                   

        # ML collision step
        # Normalize input data
        fpre = f1.reshape( (nx*ny, Q) )
        norm = np.sum(fpre, axis=1)[:,np.newaxis]
        fpre = fpre / norm

        # NN prediction
        with torch.no_grad():
            fpre = torch.from_numpy(fpre).float().to(DEVICE)
            f2 = model(fpre).detach().cpu().numpy()

        # Rescale output
        f2 = norm * f2
        f2 = f2.reshape((nx, ny, Q))

        # Collect data
        if (t % dumpit) == 0:
            # Keep track of updated fluid attributes every now and then, for plotting later
            data_collector(dumpfile, t, ux, uy, rho, nx, ny, dumpit)
        
    m_final = np.sum(f2.flatten())
    print('Sim ended. Mass err:', np.abs(m_initial - m_final) / m_initial) 

    w = 3.46*3
    h = 2.14*3

    fig = plt.figure(figsize=(w,h))
    ax  = fig.add_subplot(111)

    tLst = np.arange(0, niter, dumpit)
    for i, t in enumerate(tLst):
        ux  = dumpfile[dumpfile[:, 0] == t, 2]
        uy  = dumpfile[dumpfile[:, 0] == t, 3]

        Ft = np.average((ux**2 + uy**2)**0.5) 
        if i == 0:
            F0 = Ft 
            ax.semilogy( t, Ft, 'ob', label='lbm')
        else:
            ax.semilogy( t, Ft, 'ob')

    nu = (tau - 0.5) * cs2

    ax.semilogy(tLst, sol(tLst, nx, F0, nu), linewidth=2.0, linestyle='--', color='r' , label='analytic')

    ax.set_xlabel(r'$t~\rm{[L.U.]}$'      , fontsize=16)
    ax.set_ylabel(r'$\langle |u| \rangle$', fontsize=16, rotation=90, labelpad=0)
    ax.legend(loc='best', frameon=False, prop={'size' : 16})
    ax.tick_params(which="both",direction="in",top="on",right="on",labelsize=14)

    plt.show()
    #plt.savefig(f"time_evolution_vel_field.png")

    X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
    tLst = np.arange(0, niter, dumpit)

    for i, t in enumerate( tLst ):
        fig = plt.figure(figsize=(w, h))
        ax  = fig.add_subplot(111)

        ux  = dumpfile[dumpfile[:,0] == t, 2].reshape((nx, ny))
        uy  = dumpfile[dumpfile[:,0] == t, 3].reshape((nx, ny))
        u = (ux**2 + uy**2)**0.5
        
        im = ax.imshow(u)
        ax.streamplot(X, Y, ux, uy, density = 0.5, color='w')
        fig.colorbar(im, ax=ax, orientation='vertical', pad=0, shrink=0.69)
        ax.set_title(f"Iteration {t}", size=16)
        
        plt.show()
        #plt.savefig(f"vel_field_profile_{t}.png")

if __name__ == "__main__":
    eval()