#******************************************************************************
#
# Simple randomized sim data generation
#
#******************************************************************************
from manta import *
import os, shutil, math, sys, time
import numpy as np
sys.path.append("./tools")
import paramhelpers as ph
import copy

# Definitions
WITH_OBSTACLES = True # Experiment
MAX_NUM_OBSTACLES = 9

def run_simulation():
	# Main params  ----------------------------------------------------------------------#
	steps    = 20 # TOTAL_SIMULATION_TIME
	savedata = True # save data or not
	simNo    = 1000  # start ID
	basePath = "data_20s/"#"test_data/"
	npSeedstr   = "-1"
	showGui  = False  # show UI

	basePath        =     ph.getParam( "basePath",        basePath        )
	npSeedstr       =     ph.getParam( "npSeed"  ,        npSeedstr       )
	npSeed          =     int(npSeedstr)
	ph.checkUnusedParams()

	# Scene settings  ---------------------------------------------------------------------#
	# Note: set debug level for messages (0 off, 1 regular, higher = more, up to 10) 
	setDebugLevel(1)

	# Solver params  ----------------------------------------------------------------------#
	GRID_SIZE = 64
	dim = 2
	offset = 20
	interval = 1

	gs = vec3(GRID_SIZE, GRID_SIZE, 1 if dim == 2 else GRID_SIZE )
	buoy = vec3(0, -1e-3, 0)

	# wlt Turbulence input fluid
	sm = Solver(name='smaller', gridSize = gs, dim=dim)
	sm.timestep = 0.5

	timings = Timings()

	# Simulation Grids  -------------------------------------------------------------------#
	flags    = sm.create(FlagGrid)
	vel      = sm.create(MACGrid)
	density  = sm.create(RealGrid)
	pressure = sm.create(RealGrid)
	phiObs = sm.create(LevelsetGrid)
	fractions = sm.create(MACGrid)

	# open boundaries
	bWidth=1
	flags.initDomain(boundaryWidth=bWidth)
	flags.fillGrid()

	setOpenBound(flags, bWidth,'yY', FlagOutflow | FlagEmpty) 

	# inflow sources ----------------------------------------------------------------------#
	if(npSeed > 0):
		np.random.seed(npSeed)

	# init random density
	noise    = []
	sources  = []

	num_obstacles = np.random.randint(low=0, high=MAX_NUM_OBSTACLES)
	obstacles = []

	noiseN = 12
	nseeds = np.random.randint(10000,size=noiseN)

	cpos = vec3(0.5,0.3,0.5)

	randoms = np.random.rand(noiseN, 8)
	for nI in range(noiseN):
		noise.append( sm.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
		noise[nI].posScale = vec3( GRID_SIZE * 0.1 * (randoms[nI][7] + 1) )
		noise[nI].clamp = True
		noise[nI].clampNeg = 0
		noise[nI].clampPos = 1.0
		noise[nI].valScale = 1.0
		noise[nI].valOffset = -0.01 # some gap
		noise[nI].timeAnim = 0.3
		noise[nI].posOffset = vec3(1.5)
		
		# Randomize offsets for density vorticity
		coff = vec3(0.4) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
		radius_rand = 0.035 + 0.035 * randoms[nI][3]
		upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )
		
		if(dim == 2): 
			coff.z = 0.0
			upz.z = 1.0
		
		sources.append(sm.create(Sphere, center=gs * (cpos + coff), radius=gs.x * radius_rand, scale=upz))

		# Init noise-modulated density inside shape
		densityInflow( flags=flags, density=density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
		print (nI, "centre", gs*(cpos+coff), "radius", gs.x*radius_rand, "other", upz ) 



	# Randomize velocity
	Vrandom = np.random.rand(3)
	v1pos = vec3(0.7 + 0.4 *(Vrandom[0] - 0.5) ) #range(0.5,0.9) 
	v2pos = vec3(0.3 + 0.4 *(Vrandom[1] - 0.5) ) #range(0.1,0.5)
	vtheta = Vrandom[2] * math.pi * 0.5
	velInflow = 0.04 * vec3(math.sin(vtheta), math.cos(vtheta), 0)

	# Obstacles
	for idx in range(num_obstacles):
		random_x = np.random.rand(1)
		random_y = np.random.rand(1)
		obsPos = vec3(random_x, random_y, 0.0)
		obsSize = np.random.uniform(0.05, 0.2)
		obs = sm.create(Sphere, center=gs * obsPos, radius=GRID_SIZE * obsSize)
		phiObs.join(obs.computeLevelset())
		#setObstacleFlags(flags=flags, phiObs=phiObs)
		#flags.fillGrid()
		obstacles.append(obs)
	updateFractions(flags=flags, phiObs=phiObs, fractions=fractions, boundaryWidth=bWidth)
	setObstacleFlags(flags=flags, phiObs=phiObs)
	flags.fillGrid()


	# 2D
	if(dim == 2):
		v1pos.z = v2pos.z = 0.5
		sourcV1 = sm.create(Sphere, center=gs*v1pos, radius=gs.x*0.1, scale=vec3(1))
		sourcV2 = sm.create(Sphere, center=gs*v2pos, radius=gs.x*0.1, scale=vec3(1))
		sourcV1.applyToGrid( grid=vel , value=(-velInflow*float(gs.x)) )
		sourcV2.applyToGrid( grid=vel , value=( velInflow*float(gs.x)) )

	# 3D
	elif(dim == 3):
		VrandomMore = np.random.rand(3)
		vtheta2 = VrandomMore[0] * math.pi * 0.5
		vtheta3 = VrandomMore[1] * math.pi * 0.5
		vtheta4 = VrandomMore[2] * math.pi * 0.5
		for dz in range(1,10,1):
			v1pos.z = v2pos.z = (0.1*dz)
			vtheta_xy = vtheta *(1.0 - 0.1*dz ) + vtheta2 * (0.1*dz)
			vtheta_z  = vtheta3 *(1.0 - 0.1*dz ) + vtheta4 * (0.1*dz)
			velInflow = 0.04 * vec3( math.cos(vtheta_z) * math.sin(vtheta_xy), math.cos(vtheta_z) * math.cos(vtheta_xy),  math.sin(vtheta_z))
			sourcV1 = sm.create(Sphere, center=gs*v1pos, radius=gs.x*0.1, scale=vec3(1))
			sourcV2 = sm.create(Sphere, center=gs*v2pos, radius=gs.x*0.1, scale=vec3(1))
			sourcV1.applyToGrid( grid=vel , value=(-velInflow*float(gs.x)) )
			sourcV2.applyToGrid( grid=vel , value=( velInflow*float(gs.x)) )

	# Setup UI ---------------------------------------------------------------------#
	if (showGui and GUI):
		gui=Gui()
		gui.show()
		gui.pause()

	t = 0

	if savedata:
		folderNo = simNo
		pathaddition = 'simSimple_%04d/' % folderNo
		while os.path.exists(basePath + pathaddition):
			folderNo += 1
			pathaddition = 'simSimple_%04d/' % folderNo

		simPath = basePath + pathaddition
		print("Using output dir '%s'" % simPath) 
		simNo = folderNo
		os.makedirs(simPath)


	# main loop --------------------------------------------------------------------#
	while t < steps + offset:
		curt = t * sm.timestep
		mantaMsg( "Current time t: " + str(curt) +" \n" )
		
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2) # advect density field
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2) # advect velocity field

		# Note: set zero normal velocity boundary condition on walls
		for idx in range(num_obstacles):
			setWallBcs(flags=flags, vel=vel, phiObs=phiObs)
			obstacles[idx].applyToGrid(grid=density, value=0.) # clear smoke inside
		if num_obstacles == 0:
			setWallBcs(flags=flags, vel=vel)

		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)

		# Note: Randomized part
		if 1 and ( t < offset ): 
			vorticityConfinement( vel=vel, flags=flags, strength=0.05 )
		
		# Note: set zero normal velocity boundary condition on walls
		for idx in range(num_obstacles):
			setWallBcs(flags=flags, vel=vel, phiObs=phiObs)
			obstacles[idx].applyToGrid(grid=density, value=0.) # clear smoke inside
		if num_obstacles == 0:
			setWallBcs(flags=flags, vel=vel)

		# Note: Perform pressure projection of the velocity grid perCellCorr: a divergence correction for each cell
		solvePressure(flags=flags, vel=vel, pressure=pressure , cgMaxIterFac=10.0, cgAccuracy=0.0001 )

		# save data
		if savedata and t >= offset and (t - offset) % interval==0:
			tf = (t - offset) / interval
			density.save(simPath + 'density_%04d.uni' % (tf))
			vel.save(simPath + 'vel_%04d.uni' % (tf))
			phiObs.save(simPath + 'boundary_%04d.uni' % (tf))

		sm.step()
		t = t+1

if __name__ == "__main__":
	run_simulation()
