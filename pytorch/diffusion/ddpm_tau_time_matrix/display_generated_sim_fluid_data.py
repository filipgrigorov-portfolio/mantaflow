import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("./tools")
import uniio

# Definitions
DATA_PATH = "data_20s/"
TOTAL_SIMULATION_TIME = 20

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Defining number of rows and columns
    fig = plt.figure(figsize=(15, 15))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    #plt.show()
    plt.savefig(os.path.join(DATA_PATH, title + ".jpg"))

# TODO
def show_quivers(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # x_range = np.arange(0, 64, 1)
    # y_range = np.arange(0, 64, 1)
    # X, Y = np.meshgrid(x_range, y_range)
    # xx = X.flatten()
    # yy = Y.flatten()

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                #plt.quiver(X, Y, images[idx][..., 0], images[idx][..., 1])
                plt.imshow(images[idx])
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    #plt.show()
    plt.savefig(os.path.join(DATA_PATH, title + ".jpg"))

def load_sim_file(data_path, sim, type_name, idx):
    uniPath = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # 100 files per sim
    print(uniPath)
    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr

def load_single_sim_file(data_path):
    print(data_path)
    header, content = uniio.readUni(data_path) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr

def read_uni_files(data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64):
    densities = []
    boundaries = []
    velocities = []

    for sim in range(start_itr, end_itr): 
        if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
            for i in range(0, TOTAL_SIMULATION_TIME):
                densities.append(load_sim_file(data_path, sim, 'density', i))
                boundaries.append(load_sim_file(data_path, sim, 'boundary', i))
                velocities.append(load_sim_file(data_path, sim, 'vel', i))

    num_densities = len(densities)
    num_velocities = len(velocities)
    #if num_densities < 200:
    #    raise("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times...")

    densities = np.reshape( densities, (len(densities), grid_height, grid_width, 1) )
    print("Read uni files (density), total data " + format(densities.shape))

    boundaries = np.reshape( boundaries, (len(boundaries), grid_height, grid_width, 1) )
    print("Read uni files (boundaries), total data " + format(boundaries.shape))

    velocities = np.reshape( velocities, (len(velocities), grid_height, grid_width, 3) )
    print("Read uni files (velocity), total data " + format(velocities.shape))

    show_images(densities, "densities")
    show_images(boundaries, "boundaries")
    show_images(velocities[:, :, :, 0], "velocities_x")
    show_images(velocities[:, :, :, 1], "velocities_y")

def show_single_image(uni_path):
    sample = os.path.join(uni_path, f"density_0003.uni")
    density = load_single_sim_file(sample)
    sample = os.path.join(uni_path, f"vel_0003.uni")
    velocity = load_single_sim_file(sample)

    grid_size = 64
    density = np.reshape( density, (grid_size, grid_size, 1) )
    velocity = np.reshape( velocity, (grid_size, grid_size, 3) )

    # density
    rho = density[:, :, 0]
    print(rho.shape)
    print(rho.max())
    
    # vel x
    vel_x = velocity[:, :, 0]
    print(vel_x.shape)
    print(vel_x.max())
    
    # vel y
    vel_y = velocity[:, :, 1]
    print(vel_y.shape)
    print(vel_y.max())

    io.imwrite("test_vel_x.png", vel_x)
    io.imwrite("test_vel_y.png", vel_y)
    io.imwrite("test_rho.png", rho)


if __name__ == "__main__":
    read_uni_files(data_path=DATA_PATH)
    #show_single_image("test_data/simSimple_1000/")
