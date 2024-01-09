import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import CollisionNN

np.random.seed(42)

# Definitions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(eval):
	""" Lattice Boltzmann Simulation """
	
	# Simulation parameters
	Nx = 400    # resolution x-dir
	Ny = 100    # resolution y-dir
	rho0 = 100    # average density
	tau = 1.0    # collision timescale
	Nt = 2000   # number of timesteps (T)
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Lattice speeds / weights
	NL = 9
	idxs = np.arange(NL)
	cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
	cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
	weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1
	
	# Initial Conditions
	F = np.ones((Ny, Nx, NL))
	F += 0.01 * np.random.randn(Ny, Nx, NL) # Add some Gaussian noise

	X, Y = np.meshgrid(range(Nx), range(Ny))
	F[:,:,3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
	rho = np.sum(F, 2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho
	
	# Cylinder boundary
	X, Y = np.meshgrid(range(Nx), range(Ny))
	cylinder = (X - Nx / 4)**2 + (Y - Ny / 2)**2 < (Ny / 4)**2
	
	# Prep figure
	_ = plt.figure(figsize=(8, 4), dpi=80)

	model = None
	if eval:
		model = CollisionNN(in_chs=NL).to(DEVICE)

		print('Loading pretrained model')
		model.load_state_dict(torch.load("checkpoint_mse.pt"))
	
	# Simulation Main Loop
	for it in range(Nt):
		print(it)
		
		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
		
		
		# Set reflective boundaries
		bndryF = F[cylinder,:]
		bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
	
		# Calculate fluid variables (macroscale)
		rho = np.sum(F, axis=2)
		ux  = np.sum(F * cxs, axis=2) / rho
		uy  = np.sum(F * cys, axis=2) / rho
		
		# Collision term
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			Feq[:, :, i] = rho * w * ( 1 + 3 * (cx * ux + cy * uy)  + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2 )

		if eval:
			fpre = F.reshape(Nx * Ny, NL)
			norm = np.sum(fpre, axis=1)[:, np.newaxis]
			fpre = fpre / norm
			with torch.no_grad():
				fpre = torch.from_numpy(fpre).float().to(DEVICE)
				fpost = model(fpre)
				fpost = fpost.detach().cpu().numpy()
				fpost = norm * fpost
				# Note: fi(x + v*dt, t + dt) = fi(x, t) + Omega(fi(x, t))
				F += -(1.0 / tau) * fpost.reshape(Ny, Nx, NL).copy()
		else:
			F += -(1.0 / tau) * (F - Feq) # (100, 400, 9)
		
		# Apply boundary 
		F[cylinder, :] = bndryF

		# Streaming (potentially, approximated by GNNs)
		ux[cylinder] = 0
		uy[cylinder] = 0
		velocity_field = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
		velocity_field[cylinder] = np.nan
		velocity_field = np.ma.array(velocity_field, mask=cylinder)
		
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
			plt.cla()
			plt.imshow(velocity_field, cmap='bwr')
			plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			plt.clim(-.1, .1)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')	
			plt.pause(0.001)
			
	
	# Save figure
	plt.savefig('latticeboltzmann.png', dpi=240)
	plt.show()
	    
	return 0


if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval', action='store_true')

	args = parser.parse_args()

	if args.eval:
		print(f"Evaluation session running")
		main(eval=True)
	else:
		main(eval=False)
	print("Done")