import pybullet
from gpd.gym_pybullet_drones.envs.CtrlAviary import (
    CtrlAviary,
)
from gpd.gym_pybullet_drones.control.DSLPIDControl import (
    DSLPIDControl,
)
from gpd.gym_pybullet_drones.utils.enums import DroneModel, Physics
from gpd.gym_pybullet_drones.utils.Logger import Logger
from gpd.gym_pybullet_drones.utils.utils import sync
from gym import spaces
import numpy as np
import torch
from typing import List
import os
from torch import nn, Tensor
import time

DEFAULT_DRONES = DroneModel.CF2X
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_MODEL_DIM = 512
DEFAULT_TASK = "fly in circles around the objective"
min_bound = -100
max_bound = 100


class CustomDroneEnv:
    """
    A custom drone simulation environment that inherits from BaseAviary for physics
    and rendering, but adds custom functionalities.
    """

    def __init__(
        self,
        drone_model=DEFAULT_DRONES,
        num_drones=3,
        transformer=None,
        gui=DEFAULT_GUI,
        model_dim=DEFAULT_MODEL_DIM,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        physics=DEFAULT_PHYSICS,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        simulation=True,
    ):
        """
        Initializes the CustomDroneEnv class.

        Parameters:
        - drone_model: The model of the drone to be used in the simulation. Default is DEFAULT_DRONES.
        - num_drones: The number of drones in the simulation. Default is 3.
        - transformer: The transformer object for linear transformation of observation features. Default is None.
        - gui: A boolean flag indicating whether to enable GUI for visualization. Default is DEFAULT_GUI.
        - model_dim: The dimension of the MIMO transformer model. Default is DEFAULT_MODEL_DIM.
        - task: The task to be performed in the simulation. Default is DEFAULT_TASK.
        - record_video: A boolean flag indicating whether to record video of the simulation. Default is DEFAULT_RECORD_VISION.
        - plot: A boolean flag indicating whether to plot the simulation. Default is DEFAULT_PLOT.
        - physics: The physics engine to be used in the simulation. Default is DEFAULT_PHYSICS.
        - obstacles: A boolean flag indicating whether to include obstacles in the simulation. Default is DEFAULT_OBSTACLES.
        - simulation_freq_hz: The frequency of the simulation in Hz. Default is DEFAULT_SIMULATION_FREQ_HZ.
        - control_freq_hz: The frequency of the control loop in Hz. Default is DEFAULT_CONTROL_FREQ_HZ.
        - user_debug_gui: A boolean flag indicating whether to enable user debug GUI. Default is DEFAULT_USER_DEBUG_GUI.
        - duration_sec: The duration of the simulation in seconds. Default is DEFAULT_DURATION_SEC.
        - output_folder: The folder path to save the simulation output. Default is DEFAULT_OUTPUT_FOLDER.
        - colab: A boolean flag indicating whether the simulation is running in a Colab environment. Default is DEFAULT_COLAB.
        - simulation: A boolean flag indicating whether the simulation is running. Default is True.
        """
        # Path to the directory where the URDF files are located
        os.path.join(
            os.getcwd(), "render", "gym_pybullet_drones", "assets"
        )
        self.simulation = simulation
        self.done = {
            "crashed": False,
            "reached_target": False,
        }
        #### Initialize the task instructions ########################
        # Assuming this mapping is defined in your __init__ or a similar setup method
        self.task_to_id = {
            "fly in circles around the objective": 1,
            "fly into objective": 2,
            "hover": 3,
        }  # Example mapping
        # self.CLIENT = pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        #### Initialize the simulation #############################
        H = 0.1
        H_STEP = 0.05
        R = 0.3
        self.START = time.time()
        self.INIT_XYZS = np.array(
            [
                [
                    R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                    R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R,
                    H + i * H_STEP,
                ]
                for i in range(num_drones)
            ]
        )
        self.INIT_RPYS = np.array(
            [
                [0, 0, i * (np.pi / 2) / num_drones]
                for i in range(num_drones)
            ]
        )
        # Initialize a circular trajectory
        self.task = None
        PERIOD = 10
        self.NUM_WP = control_freq_hz * PERIOD
        self.TARGET_POS = np.zeros((self.NUM_WP, 3))
        for i in range(self.NUM_WP):
            self.TARGET_POS[i, :] = (
                R
                * np.cos((i / self.NUM_WP) * (2 * np.pi) + np.pi / 2)
                + self.INIT_XYZS[0, 0],
                R
                * np.sin((i / self.NUM_WP) * (2 * np.pi) + np.pi / 2)
                - R
                + self.INIT_XYZS[0, 1],
                0,
            )
        self.wp_counters = np.array(
            [
                int((i * self.NUM_WP / 6) % self.NUM_WP)
                for i in range(num_drones)
            ]
        )
        #### Create the environment ################################
        self.env = CtrlAviary(
            drone_model=DEFAULT_DRONES,
            num_drones=num_drones,
            initial_xyzs=self.INIT_XYZS,
            initial_rpys=self.INIT_RPYS,
            physics=physics,
            neighbourhood_radius=10,
            pyb_freq=simulation_freq_hz,
            ctrl_freq=control_freq_hz,
            gui=gui,
            record=record_video,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
        )
        # self.client = bullet_client.BulletClient(connection_mode=connection_mode)
        # self.client = pybullet.connect(pybullet.DIRECT)  # Instead of p.GUI
        # if gui:
        #    self.CLIENT = pybullet.connect(pybullet.GUI)  # For manual testing with visualization
        # else:
        #    self.CLIENT = pybullet.connect(pybullet.DIRECT)  # For automated tests without GUI
        self.logger = Logger(
            logging_freq_hz=control_freq_hz,
            num_drones=num_drones,
            output_folder=output_folder,
            colab=colab,
        )
        self.gui = gui
        self.duration_sec = duration_sec
        self.num_drones = num_drones
        self.drone_ids = (
            []
        )  # Initialize an empty list to store drone IDs
        self.model_dim = (
            model_dim  # Dimension of the MIMO transformer model
        )
        self.expansion_layer = nn.Linear(
            21, model_dim
        )  # Projects from 21 features to 512
        self.action = np.zeros((self.num_drones, 4))
        #### Initialize the controllers ############################
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [
                DSLPIDControl(drone_model=drone_model)
                for _ in range(num_drones)
            ]
        else:
            print("no drones")
        # for _ in range(num_drones):
        # Construct the path to the URDF file for the drone model
        # Adjust the file name based on the model you want to load, e.g., 'cf2x.urdf'
        #    urdf_file_path = os.path.join(urdf_base_path, DEFAULT_DRONES.value + ".urdf")
        # Load the drone into the simulation
        #    drone_id = pybullet.loadURDF(urdf_file_path, physicsClientId=self.env)
        #    self.drone_ids.append(drone_id)
        self.mimo_transformer = transformer
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_drones, 7), dtype=np.float32
        )  # Extended action space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_drones, 10),
            dtype=np.float32,
        )  # Extended observation space
        self.obs = None
        # Initialize state attributes as zero arrays
        self.pos = np.zeros(
            (self.num_drones, 3)
        )  # Position (x, y, z)
        self.quat = np.zeros(
            (self.num_drones, 4)
        )  # Quaternion (x, y, z, w)
        self.rpy = np.zeros((self.num_drones, 3))  # Roll, pitch, yaw
        self.vel = np.zeros(
            (self.num_drones, 3)
        )  # Velocity (vx, vy, vz)
        self.ang_v = np.zeros(
            (self.num_drones, 3)
        )  # Angular velocity (wx, wy, wz)
        self.last_clipped_action = np.zeros(
            (self.num_drones, 4)
        )  # Last applied action (e.g., motor speeds)

    def _apply_action(self, drone_id: int, action: list, i: float):
        """
        Applies computed control to the specified drone and logs the action.

        Parameters:
        - drone_id: The ID of the drone to which the action should be applied.
        - action: The computed control action for the drone (ignored in simulation, used in real-world).
        - timestamp: The current simulation time.

        Returns:
        - computed_actions: A list of computed actions for the drone.

        Notes:
        - This method applies the computed control action to the specified drone in the simulation environment.
        - For simulation, the target positions and orientations are used to compute the control actions.
        - For real-world application, the actions are not applied and a placeholder is provided for future implementation.
        """
        #print(type(action))
        computed_actions = []
        if self.simulation:
            # For simulation: Apply actions as per the simulation setup described in the provided script.
            # Assuming 'action' contains target positions and orientations for simulation purposes.
            # Here, we directly use the target positions and orientations defined globally (e.g., TARGET_POS)
            # and compute the control actions as done in the example script.
            #print(f"action: {action}")
            #n_action = action.detach().numpy()
            #print(n_action)
            # The 'computed_action' would then be applied through the simulation environment's step method.
            # This step is typically done for all drones together, so you might adjust this part based on your simulation's API.
            #physics = self.env._physics(action,drone_id)
            computed_actions.append(action)
        else:
            target_pos = self.TARGET_POS[
                self.wp_counters[drone_id], :
            ]
            target_rpy = self.INIT_RPYS[drone_id, :]
            computed_action, _, _ = self.ctrl[
                drone_id
            ].computeControlFromState(
                control_timestep=self.env.CTRL_TIMESTEP,
                state=self.env._getDroneStateVector(drone_id),
                target_pos=target_pos,
                target_rpy=target_rpy,
            )
            # For real-world application, set actions directly.
            # Placeholder for real-world action application logic.
            pass  # Replace this with real-world action application logic later.
        return computed_actions

    def transform_observation(self, observation):
        """Applies a linear transformation to project observation features to model dimensions."""
        observation_tensor = torch.tensor(
            observation, dtype=torch.float32
        )
        # Apply task to the observation tensor
        task_id = [
            self.task_to_id[self.task]
        ]  # Get the numerical ID for the task
        # print(f"Task ID: {task_id}")
        task_tensor = torch.tensor(task_id, dtype=torch.float32)
        # print(f"Task tensor: {task_tensor}")
        # Concatenate the task information with the observation.
        # This requires the task information to be of compatible shape.
        # Combine the observation tensor and task tensor
        observation_tensor = observation_tensor.unsqueeze(
            0
        )  # Now [1, feature_length]
        task_tensor = task_tensor.unsqueeze(
            0
        )  # Now [1, task_length] Making it compatible for concatenation
        # print(f"Observation tensor shape: {observation_tensor.shape}")
        # print(f"Task tensor shape: {task_tensor.shape}")
        # Concatenate along the feature dimension (dim=1)
        combined_tensor = torch.cat(
            [observation_tensor, task_tensor], dim=1
        )  # Concatenate along the second dimension
        transformed_observation = self.expansion_layer(
            combined_tensor.unsqueeze(0)
        )
        # print(transformed_observation.shape)
        return transformed_observation

    def _get_observations(self):
        """Override to collect observations for all drones, formatted as tensors."""
        observations = []
        for drone_id in range(self.num_drones):
            drone_observation = self.env._getDroneStateVector(
                drone_id
            )  # Collect per-drone observations
            observations.append(
                drone_observation
            )  # Append to the list of observations
        return observations

    def generate_action(self, observations):
        """Generates actions for all drones using the MIMO transformer."""
        observations = self._get_observations()
        transformed_observations = []
        for observation in observations:
            transformed_observation = self.transform_observation(
                observation
            )
            transformed_observations.append(
                transformed_observation
            )
        output_tensors = self.mimo_transformer(
            transformed_observations
            )
        print(f"actions: {output_tensors}")
        return output_tensors

    def apply_actions(self, decoded_actions: List, i):
        results = []
        for drone_id, action in enumerate(decoded_actions):
            result = self._apply_action( # Directly apply the action to the drone
                drone_id, action, i
            )  # Adjust this line according to your environment's API
            results.append(result)
        return results

    def step(self, i, decoded_actions):
        # Inside _apply_action or similar method, before using action as a NumPy array
        if isinstance(decoded_actions, torch.Tensor):
            decoded_actions = decoded_actions.detach().numpy()  # This conversion is safe
        self.action = decoded_actions
        """Perform a step in the environment. This will now use generate_and_apply_actions method."""
        print(f"decoded actions: {decoded_actions}")
        self.obs, reward, terminated, truncated, info = self.env.step(
            self.action
        )
        results = self.apply_actions(
            decoded_actions, i
        )
        self.env.render()
        if self.gui:
            sync(i, self.START, self.env.CTRL_TIMESTEP)
        return results
    
    def calculate_reward(self, results: List[dict], task):
        """
        Calculate the reward for the drone's current state, focusing on hover stability.

        Parameters:
        - results: A list of dictionaries containing information about the current state or result of an action for each drone.
        - task: The current task (e.g., "hover") to tailor the reward calculation.

        Returns:
        - reward: A float representing the calculated reward.
        """
        reward = 0  # Initialize reward

        if task == "hover":
            for result in results:
                # Assuming 'result' dictionary contains 'position' and 'velocity' for each drone
                # And TARGET_POS is the desired hover position (could be initial position or a specified point)
                observations = self._get_observations()
                print(f"observations: {observations}")
                print(f"observations shape: {observations.shape}")
                drone_position = np.array(result['position'])  # Current position of the drone
                drone_velocity = np.array(result['velocity'])  # Current velocity of the drone

                # Calculate distance to the target position (could include altitude as part of the position)
                distance_to_target = np.linalg.norm(drone_position - self.TARGET_POS)

                # Calculate the magnitude of the velocity (should be close to 0 for a good hover)
                velocity_magnitude = np.linalg.norm(drone_velocity)

                # Penalize distance to target position and any movement
                # Adjust weights (0.5, 1.0 in this case) as necessary to balance the importance of position vs. velocity
                reward -= 0.5 * distance_to_target + 1.0 * velocity_magnitude

        return reward
    def load_scenario(self, scenario_file):
        # Load scenario configurations here
        pass

    def reset(self, task="hover"):
        # Your reset logic here
        if self.task is None:
            self.task = task
        if self.action is None:
            self.action = np.zeros((self.num_drones, 4))
        if self.obs is None:
            self.obs = self.env.step(self.action)
        observations = (
            self._get_observations()
        )  # Assuming this returns a list of observations
        return observations  # Convert list to NumPy array

    def render(self, mode="human", **kwargs):
        """
        Enhanced rendering with adjustable camera.
        """
        if mode == "human":
            self.client.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
            )

    def _compute_done(self, drone_id):
        # Example condition: episode ends if the drone flies out of bounds
        drone_pos = self.client.getBasePositionAndOrientation(
            drone_id
        )[0]
        return not (min_bound < drone_pos < max_bound)

    def _log_data(self):
        # Log drone states, actions, and environmental conditions here
        pass

    def _actionSpace(self):
        # Example: Define an action space where each action is a continuous value between -1 and 1
        # Adjust the shape as necessary for your specific use case
        return spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_drones, 4),
            dtype=np.float32,
        )

    def _observationSpace(self):
        # Example: Define an observation space with arbitrary bounds
        # Adjust the shape as necessary based on what your environment observes
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_drones, 12),
            dtype=np.float32,
        )
    
    def is_done(self, result):
        """
        Determine if the episode is done based on the drone's state or environment conditions.

        Parameters:
        - result: A dictionary or object containing information about the current state or result of an action.

        Returns:
        - done: A boolean indicating whether the episode is finished.
        """
        # Example: Episode ends if the drone crashes or reaches its target
        done = self.done.get('crashed', False) or self.done.get('reached_target', False)
        return done
    def _computeInfo(self):
        """
        Compute and return additional info about the environment's state.

        This method should return a dictionary containing information
        relevant to the current state of the environment or the simulation.
        """
        info = {}
        for idx, drone_id in enumerate(self.drone_ids):
            # Example: Gather some basic information for each drone
            pos, _ = pybullet.getBasePositionAndOrientation(
                drone_id, physicsClientId=self.env
            )
            vel, _ = pybullet.getBaseVelocity(
                drone_id, physicsClientId=self.env
            )

            # Store information in the dictionary
            info[f"drone_{idx}_position"] = pos
            info[f"drone_{idx}_velocity"] = vel

        # You can add more environment-specific information here as needed
        return info

    def _computeObs(self):
        """
        Compute and return the current observation of the environment.
        This method should return an array or a dictionary of observations
        that match the structure defined in self.observation_space.
        Observations now include position, orientation (Euler angles), linear velocity, and angular velocity.
        """
        observations = []
        for drone_id in self.drone_ids:
            # Retrieve position, orientation (as quaternion), and velocities for each drone
            pos, orn = pybullet.getBasePositionAndOrientation(
                drone_id, physicsClientId=self.env
            )
            lin_vel, ang_vel = pybullet.getBaseVelocity(
                drone_id, physicsClientId=self.env
            )
            # Convert orientation from quaternion to Euler angles for consistency with control inputs
            euler_orn = pybullet.getEulerFromQuaternion(orn)
            # Combine all components into a single observation array for each drone
            drone_obs = np.concatenate(
                [pos, euler_orn, lin_vel, ang_vel]
            )
            observations.append(drone_obs)
        # Return observations in a structured array format that matches your observation space
        # Ensure the observations array is correctly shaped according to your environment's observation_space
        return np.stack(observations)

    def apply_high_level_command(self, command):
        """
        Applies a high-level command to the environment.
        """
        drone_actions = self.mimo_transformer.transform_command(
            command
        )
        for drone_id, action in enumerate(drone_actions):
            # Assuming you have a method to apply individual actions
            self.apply_action(drone_id, action)
