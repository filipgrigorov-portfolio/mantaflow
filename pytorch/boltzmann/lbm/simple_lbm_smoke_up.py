import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate flow past cylinder
for an isothermal fluid

"""

np.random.seed(42)

def main():
	""" Lattice Boltzmann Simulation """
	
	# Simulation parameters (macroscale params)
	Nx                     = 64    # resolution x-dir
	Ny                     = 64    # resolution y-dir
	rho0                   = 10    # average density
	tau                    = 0.6    # collision timescale
	Nt                     = 4000   # number of timesteps (long range)
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Lattice speeds / weights
	NL = 9
	density_indices = np.arange(NL)
	cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
	cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
	weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1
	
	# Initial Conditions -> compute Feq
	F = np.ones((Ny, Nx, NL)) #* rho0 / NL
	F += 0.01 * np.random.randn(Ny, Nx, NL) # Random source

	X, Y = np.meshgrid(range(Nx), range(Ny))
	
	# Periodic source
	F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * Y / Ny * 4))
	
	rho = np.sum(F, 2)
	for idx in density_indices:
		F[:, :, idx] *= rho0 / rho #drho = rho0 / rho
	
	# Prep figure
	fig = plt.figure(figsize=(4, 4), dpi=80)
	
	# Simulation loop
	for it in range(Nt):
		print(it)
		
		# Drift
		for i, cx, cy in zip(density_indices, cxs, cys):
			F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
			F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
		

		# Calculate fluid variables -> used to compute feq
		# rho = SUM(F)
		rho = np.sum(F, 2)
		# ux = SUM(vx * F)
		ux  = np.sum(F * cxs, 2) / rho # momentum
		# uy = SUM(vy * F)
		uy  = np.sum(F * cys, 2) / rho # momentum		
		
		# Collision
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(density_indices, cxs, cys, weights):
			Feq[:, :, i] = rho * w * ( 1 + 3 * (cx * ux + cy * uy)  + (9 / 2) * (cx * ux + cy * uy)**2 - (3 / 2) * (ux**2 + uy**2) )
		
		F += -(1.0 / tau) * (F - Feq)
		
		# Streaming
		velocity_field = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))

		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
			plt.cla()
			plt.imshow(velocity_field, cmap='bwr')
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
  main()