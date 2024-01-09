import argparse
import matplotlib.pyplot as plt
import os

from ioutils import generate_next_available_idx, save_np_tensor
from phi.flow import *
from tqdm import tqdm

#TODO: Use the batch dimension of phiflow instead of one by one (todo)
#TODO: Randomize the hot smoke source and the obstacle/s (done)
#TODO: Generalize the type of solid obstacle (todo)

# Definitions
DEBUG = False
OBSTACLE = True
ROOT = 'data'
RUNS = 3
SAVE = True
VISUALIZE = True


class Simulator:
    """ Simulation class to modularize the fluid flow simulation run and ICs/BCs """
    def __init__(self, d0, v0, inflow_location, smoke_intensity, solid_box_location, solid_box_size, bounds, T=16, dt=1):
        self.T = T
        self.dt = dt

        self.smoke = CenteredGrid(d0, extrapolation=extrapolation.BOUNDARY, x=bounds[0], y=bounds[1], bounds=Box(x=bounds[0], y=bounds[1]))
        
        # Dirichlet boundary conditions: ZERO
        self.velocity = StaggeredGrid(v0, extrapolation=extrapolation.ZERO, x=bounds[0], y=bounds[1], bounds=Box(x=bounds[0], y=bounds[1]))

        inflow_location = tensor([inflow_location], batch('inflow_loc'), channel(vector='x,y'))
        self.inflow = smoke_intensity * CenteredGrid(Sphere(center=inflow_location, radius=3), extrapolation.BOUNDARY, x=bounds[0], y=bounds[1], bounds=Box(x=bounds[0], y=bounds[1]))

        if DEBUG:
            print(f"Smoke: {self.smoke.shape}")
            print(f"Velocity: {self.velocity.shape}")
            print(f"Inflow: {self.inflow.shape}")
            print(f"Inflow, spatial only: {self.inflow.shape.spatial}")

            print(self.smoke.values)
            print(self.velocity.values)
            print(self.inflow.values)

        if OBSTACLE:
            x1, y1 = solid_box_location
            bw, bh = solid_box_size
            x2, y2 = min(x1 + bw, bounds[0] - 1), min(y1 + bh, bounds[1] - 1)
            self.obstacle = Obstacle(Box(x=(x1, x2), y=(y1, y2)), velocity=[0, 0], angular_velocity=tensor(0,))
            if DEBUG:
                print(self.obstacle.geometry)

    def step(self, buoyancy_direction=(0, 0.5)):
        smoke_np = self.smoke.values.numpy("inflow_loc,y,x").squeeze(0)[..., np.newaxis]
        velocity_np = self.velocity.at_centers().values.numpy('y,x,vector')
        
        trajectory_smoke = [smoke_np]
        trajectory_velocity = [velocity_np]
        trajectory_obstacle = []
        if OBSTACLE:
            print(self.obstacle.shape)
            H, W = self.smoke.values.numpy('y').shape[0], self.smoke.values.numpy('x').shape[0]
            COMPUTATIONAL_DOMAIN = dict(x=W, y=H)
            obstacle_mask = CenteredGrid(self.obstacle.geometry, ZERO_GRADIENT, **COMPUTATIONAL_DOMAIN)
            trajectory_obstacle = np.tile([obstacle_mask.values.numpy('y,x,vector')], reps=(self.T)).transpose(3, 1, 2, 0)
        
        for t in tqdm(range(0, self.T - 1)):
            self.smoke = advect.mac_cormack(self.smoke, self.velocity, dt=self.dt) + self.inflow
            buoyancy_force = self.smoke * buoyancy_direction @ self.velocity
            self.velocity = advect.semi_lagrangian(self.velocity, self.velocity, dt=self.dt) + buoyancy_force
            if OBSTACLE:
                self.velocity, pressure = fluid.make_incompressible(self.velocity, [self.obstacle])
            else:
                self.velocity, pressure = fluid.make_incompressible(self.velocity)
            
            smoke_np = self.smoke.values.numpy('inflow_loc,y,x').squeeze(0)[..., np.newaxis]
            velocity_np = self.velocity.at_centers().values.numpy('inflow_loc,y,x,vector').squeeze(0)

            trajectory_smoke.append(smoke_np)
            trajectory_velocity.append(velocity_np)

        trajectory_smoke = np.array(trajectory_smoke)
        trajectory_velocity = np.array(trajectory_velocity)

        return np.concatenate([trajectory_smoke, trajectory_velocity, trajectory_obstacle], axis=-1), pressure


def randomly_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=3, help="Number of simulation runs")
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--T', type=int, default=16, help="Number of simulation time steps")
    args = parser.parse_args()

    # Random simulation parameters
    params = {
        "d0": np.zeros(shape=(args.num_runs)),
        "v0": np.zeros(shape=(args.num_runs)),
        "bounds": (args.width, args.height),
        "inflow_location": {
            "x": np.random.uniform(low=0.0, high=(args.width - 1), size=(args.num_runs)),
            "y": np.random.uniform(low=5, high=5, size=(args.num_runs))
        },
        # Note: Supports only one solid for now
        "solid_box_location": {
            "x1": np.random.uniform(low=0.0, high=(args.width // 2), size=(args.num_runs)),
            "y1": np.random.uniform(low=10, high=10, size=(args.num_runs)), # fixed
            "w": np.random.uniform(low=10, high=30, size=(args.num_runs)),  # Note: A long box that is slid sideways <->
            "h": np.random.uniform(low=5, high=5, size=(args.num_runs))   # fixed
        },
        "buoyancy_magnitude": np.random.uniform(low=0.5, high=0.9, size=(args.num_runs)),
        "smoke_intensity": np.random.uniform(low=0.6, high=0.9, size=(args.num_runs))
    }

    for idx in range(args.num_runs):
        sim = Simulator(
            d0=params["d0"][idx], 
            v0=params["v0"][idx],
            inflow_location=(params["inflow_location"]["x"][idx], params["inflow_location"]["y"][idx]),
            smoke_intensity=params["smoke_intensity"][idx],
            solid_box_location=(params["solid_box_location"]["x1"][idx], params["solid_box_location"]["y1"][idx]),
            solid_box_size=(params["solid_box_location"]["w"][idx], params["solid_box_location"]["h"][idx]),
            bounds=params["bounds"],
            T=args.T)
        trajectory, _ = sim.step(buoyancy_direction=(0.0, params["buoyancy_magnitude"][idx]))
        print(f"trajectory shape: {trajectory.shape}")

        dir_name = f"{ROOT}/simSimple_{generate_next_available_idx(ROOT)}"
        if SAVE and not os.path.exists(dir_name):
            print(f"\nCreating \"{dir_name}\"")
            os.mkdir(dir_name)
        
        for t, time_batch in enumerate(trajectory):
            smoke = time_batch[..., :1]
            vel_x = time_batch[..., 1:2]
            vel_y = time_batch[..., 2:-1]
            solid = time_batch[..., -1:]

            if SAVE:
                save_np_tensor(dir_name, smoke, 'density', t)
                save_np_tensor(dir_name, vel_x, 'velocity_x', t)
                save_np_tensor(dir_name, vel_y, 'velocity_y', t)
                save_np_tensor(dir_name, solid, 'solid', t)

            if VISUALIZE:
                h, w, _ = smoke.shape
                tensors = [
                    smoke,
                    vel_x,
                    vel_y,
                    solid
                ]
                C = len(tensors)
                overlay = np.zeros(shape=(h, w * C, 1))
                for i in range(C):
                    overlay[:, i * w : (i + 1) * w, :] = tensors[i]

                plt.imshow(overlay, origin="lower")
                plt.draw()
                plt.pause(0.09)
                plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=3, help="Number of simulation runs")
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--T', type=int, default=16, help="Number of simulation time steps")
    args = parser.parse_args()

    # Definitions
    RUNS = args.num_runs if args.num_runs else RUNS

    sim = Simulator(
        d0=0, 
        v0=0, 
        inflow_location=(4, 5), 
        solid_box_location=(5, 11),
        solid_box_size=(10, 10),
        bounds=(args.width, args.height), 
        T=args.T)
    trajectory, _ = sim.step()
    print(f"trajectory shape: {trajectory.shape}")

    for idx in range(RUNS):
        dir_name = f"{ROOT}/simSimple_{generate_next_available_idx(ROOT)}"
        if SAVE and not os.path.exists(dir_name):
            print(f"\nCreating \"{dir_name}\"")
            os.mkdir(dir_name)
        
        for t, time_batch in enumerate(trajectory):
            smoke = time_batch[..., :1]
            vel_x = time_batch[..., 1:2]
            vel_y = time_batch[..., 2:-1]
            solid = time_batch[..., -1:]

            if SAVE:
                save_np_tensor(dir_name, smoke, 'density', t)
                save_np_tensor(dir_name, vel_x, 'velocity_x', t)
                save_np_tensor(dir_name, vel_y, 'velocity_y', t)
                save_np_tensor(dir_name, solid, 'solid', t)

            if VISUALIZE:
                h, w, _ = smoke.shape
                tensors = [
                    smoke,
                    vel_x,
                    vel_y,
                    solid
                ]
                C = len(tensors)
                overlay = np.zeros(shape=(h, w * C, 1))
                for i in range(C):
                    overlay[:, i * w : (i + 1) * w, :] = tensors[i]
                #overlay[:, : w, :] = smoke
                #overlay[:, w : 2 * w, :] = vel_x
                #overlay[:, 2 * w : 3 * w, :] = vel_y
                #overlay[:, 3 * w :, :] = solid
                plt.imshow(overlay, origin="lower")
                plt.draw()
                plt.pause(0.09)
                plt.clf()


if __name__ == "__main__":
    print("Start")
    randomly_generate()
    print("End")
