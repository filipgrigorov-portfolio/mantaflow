import argparse
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# NN
import torch
import torch.nn as nn
import torch.optim as optim

# Definitions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# nchw
def sample_batch(replay_buffer, batch_size):
	#print("replay: ", len(replay_buffer)) # debug
	idx = torch.randint(low=0, high=len(replay_buffer) - batch_size - 1, size=(1,))
	seq = [ replay_buffer[idx + offset] for offset in range(0, batch_size) ]
	stacked_data = torch.concat(seq, dim=0)
	#print('stacked: ', stacked_data.size()) # debug
	return stacked_data

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

def time_embed(dim_in, dim_out):
	"""Temporal embedding with MLP"""
	return nn.Sequential(
		nn.Linear(dim_in, dim_out),
		nn.SiLU(),
		nn.Linear(dim_out, dim_out)
	)

class CollisionNN(nn.Module):
	def __init__(self, input_size, chs_in, chs_out, total_sim_time):
		super(CollisionNN, self).__init__()

		self.input_size = input_size * chs_in
		self.chs_in = chs_in
		self.chs_out = chs_out

		self.time_embedding = time_embed(dim_in=total_sim_time, dim_out=32)

		self.model = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=32),
            nn.LeakyReLU(),
        )

		self.head = nn.Sequential(
			nn.Linear(in_features=32, out_features=64),
			nn.Linear(in_features=64, out_features=32),
			nn.SiLU(),
			nn.Linear(in_features=32, out_features=32),
			nn.SiLU(),
            nn.Linear(in_features=32, out_features=input_size * chs_out),
		)

	def forward(self, x, t):
		n, c, h, w = x.shape
		x = x.flatten()
		t_emb = self.time_embedding(t).reshape(-1)
		out = self.model(x)
		out_stacked = out + t_emb
		out = self.head(out_stacked)
		out = out.view(-1, self.chs_out, h, w)
		return out


def main(eval=False, is_nn=False):
	""" Lattice Boltzmann Simulation """
	
	# Simulation parameters
	Nx = 400    # resolution x-dir
	Ny = 100    # resolution y-dir
	rho0 = 100    # average density
	tau = 0.6    # collision timescale
	Nt = 10000   # number of timesteps (T)
	plotRealTime = True # switch on for plotting as the simulation goes along

	replay_buffer = []
	
	# Lattice speeds / weights
	NL = 9
	idxs = np.arange(NL)
	cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
	cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
	weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1
	
	# Initial Conditions
	F = np.ones((Ny, Nx, NL)) #* rho0 / NL
	F += 0.01 * np.random.randn(Ny, Nx, NL)

	X, Y = np.meshgrid(range(Nx), range(Ny))
	F[:,:,3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
	rho = np.sum(F,2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho
	
	# Cylinder boundary
	X, Y = np.meshgrid(range(Nx), range(Ny))
	cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2
	
	# Prep figure
	fig = plt.figure(figsize=(4,2), dpi=80)

	# NN --------------------------------------------------------------------------
	CHS = 3
	BATCH_SIZE = 3
	collision_model = CollisionNN(1 * Nx * Ny, CHS, NL, Nt).to(DEVICE)
	mse = nn.MSELoss().to(DEVICE)
	optimizer = optim.Adam(collision_model.parameters(), lr=1e-4)
	mean_loss = 0.0
	if eval:
		print('Loading pretrained model')
		collision_model.load_state_dict(torch.load("checkpoint.pt"))


	
	# Simulation Main Loop
	for it in range(Nt):
		print(it)
		
		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
		
		
		# Set reflective boundaries
		bndryF = F[cylinder,:]
		bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
	
		
		# Calculate fluid variables (macroscale)
		rho = np.sum(F, axis=2)
		#(h, w)
		ux  = np.sum(F * cxs, axis=2) / rho
		uy  = np.sum(F * cys, axis=2) / rho
		
		# Collision term
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			Feq[:, :, i] = rho * w * ( 1 + 3 * (cx * ux + cy * uy)  + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2 )
		

		# NN collision term (self-supervised)
		rho_nn = torch.from_numpy(rho.copy()[np.newaxis, np.newaxis, :, :]).to(DEVICE).float()
		ux_nn = torch.from_numpy(ux.copy()[np.newaxis, np.newaxis, :, :]).to(DEVICE).float()
		uy_nn = torch.from_numpy(uy.copy()[np.newaxis, np.newaxis, :, :]).to(DEVICE).float()
		x = torch.concat([rho_nn, ux_nn, uy_nn], dim=1) # Concatenate along the channels
		#replay_buffer.append(x)

		t = torch.arange(0, Nt).float().to(DEVICE) / Nt

		collision_term = -(1.0 / tau) * (F - Feq) if not eval else collision_model(x, t)[-1, ...].detach().cpu().numpy()[np.newaxis, ...].squeeze(0).transpose(1, 2, 0)
		
		#if len(replay_buffer) > BATCH_SIZE + 1 and not eval:
		#batch_x = sample_batch(replay_buffer, BATCH_SIZE)
		collision_term_pred = collision_model(x, t)
		loss = mse(collision_term_pred, torch.from_numpy(collision_term.transpose(2, 0, 1)[np.newaxis, :, :, :]).to(DEVICE).float())

		mean_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		F += collision_term # (This can be approximated with a NN)

		if not eval and it % 10 == 9:
			print(f"Loss: {mean_loss / (it + 1)}")
			print("Saving current checkpoint")
			torch.save(collision_model.state_dict(), "checkpoint.pt")
			print("Prediction: ", collision_term_pred.detach().cpu().numpy()[0].max())
			print("Expectation: ", collision_term.max())
		
		# Apply boundary 
		F[cylinder, :] = bndryF

		# Streaming (potentially GNNs???)
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
	plt.savefig('latticeboltzmann.png',dpi=240)
	plt.show()
	    
	return 0



if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--collision', action='store_true')

	args = parser.parse_args()

	if args.train:
		print(f"Training session running")
		main(eval=False)
	elif args.eval:
		print(f"Evaluation session running")
		main(eval=True)
	print("Done")