from pybullet_utils import bullet_client
import pybullet
from render.gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from render.gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from render.gym_pybullet_drones.utils.enums import DroneModel, Physics
from render.gym_pybullet_drones.utils.Logger import Logger
from gym import spaces
import numpy as np
from HRTX.hrtx.mimo import MIMOTransformer
import torch
import os

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
        gui=DEFAULT_GUI,
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
    ):
        # Path to the directory where the URDF files are located
        urdf_base_path = os.path.join(
            os.getcwd(), "render", "gym_pybullet_drones", "assets"
        )

        #### Initialize the simulation #############################
        H = 0.1
        H_STEP = 0.05
        R = 0.3
        INIT_XYZS = np.array(
            [
                [
                    R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                    R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R,
                    H + i * H_STEP,
                ]
                for i in range(num_drones)
            ]
        )
        INIT_RPYS = np.array(
            [[0, 0, i * (np.pi / 2) / num_drones] for i in range(num_drones)]
        )
        #### Create the environment ################################
        self.env = CtrlAviary(
            drone_model=DEFAULT_DRONES,
            num_drones=num_drones,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=physics,
            neighbourhood_radius=10,
            pyb_freq=simulation_freq_hz,
            ctrl_freq=control_freq_hz,
            gui=gui,
            record=record_video,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
        )
        # self.CLIENT = pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
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
        self.duration_sec = duration_sec
        self.num_drones = num_drones
        self.drone_ids = []  # Initialize an empty list to store drone IDs
        #### Initialize the controllers ############################
        if drone_model in [DroneModel]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
        # for _ in range(num_drones):

        # Construct the path to the URDF file for the drone model
        # Adjust the file name based on the model you want to load, e.g., 'cf2x.urdf'
        #    urdf_file_path = os.path.join(urdf_base_path, DEFAULT_DRONES.value + ".urdf")
        # Load the drone into the simulation
        #    drone_id = pybullet.loadURDF(urdf_file_path, physicsClientId=self.env)
        #    self.drone_ids.append(drone_id)
        self.mimo_transformer = MIMOTransformer(
            dim=512, depth=6, heads=8, dim_head=64, num_robots=self.num_drones
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_drones, 7), dtype=np.float32
        )  # Extended action space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_drones, 10), dtype=np.float32
        )  # Extended observation space

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
            drone_obs = np.concatenate([pos, euler_orn, lin_vel, ang_vel])
            observations.append(drone_obs)
        # Return observations in a structured array format that matches your observation space
        # Ensure the observations array is correctly shaped according to your environment's observation_space
        return np.stack(observations)

    def apply_high_level_command(self, command):
        """
        Applies a high-level command to the environment.
        """
        drone_actions = self.mimo_transformer.transform_command(command)
        for drone_id, action in enumerate(drone_actions):
            # Assuming you have a method to apply individual actions
            self.apply_action(drone_id, action)

    def _apply_action(self, drone_id, action, timestamp):
        """
        Applies computed PID control to the specified drone and logs the action.

        Parameters:
        - drone_id: The ID of the drone to which the action should be applied.
        - action: The computed control action for the drone.
        - timestamp: The current simulation time.
        """
        # Extract current state of the drone for PID control computation
        pos, orn = pybullet.getBasePositionAndOrientation(
            self.drone_ids[drone_id], physicsClientId=self.env.CLIENT
        )
        lin_vel, ang_vel = pybullet.getBaseVelocity(
            self.drone_ids[drone_id], physicsClientId=self.env.CLIENT
        )
        euler_orn = pybullet.getEulerFromQuaternion(orn)
        current_state = np.concatenate([pos, euler_orn, lin_vel, ang_vel])

        # Compute control actions (assuming action contains target position and orientation)
        pwm_values = self.ctrl[drone_id].computeControl(
            control_timestep=pybullet.getPhysicsEngineParameters(
                physicsClientId=self.env.CLIENT
            )["fixedTimeStep"],
            state=current_state,
            target_pos=action["position"],
            target_rpy=action["orientation"],
        )

        # Apply the PWM values to the drone motors (placeholder for actual motor control command)
        # This step depends on your simulation setup and how motor control is implemented

        # Log the applied action along with the drone's current state
        self.logger.log(
            drone=drone_id,
            timestamp=timestamp,
            state=current_state,
            control=pwm_values,  # or action if you wish to log the target state instead
        )

    def _get_observation(self):
        """Override to collect observations for all drones, formatted as tensors."""
        observations = []
        for drone_id in range(self.num_drones):
            drone_observation = super()._getDroneStateVector(
                drone_id
            )  # Collect per-drone observations
            observations.append(
                torch.tensor(drone_observation).unsqueeze(0)
            )  # Convert to tensors and batch
        return observations

    def generate_and_apply_actions(self):
        """Generates actions for all drones using the MIMO transformer and applies them."""
        observations = (
            self._get_observation()
        )  # Collect observations in the required tensor format
        observations_tensor = torch.stack(
            observations
        )  # Stack observations along a new dimension
        # Ensure observations_tensor is of shape [batch_size, num_drones, feature_size]
        actions = self.mimo_transformer(observations_tensor)
        for drone_id, action in enumerate(actions):
            # Here, translate 'action' into the format expected by CtrlAviary's control methods
            # For example, if using velocity control:
            self.env.setDroneStateVelocity(drone_id, action.tolist())
            # Ensure 'action' is in the appropriate format and scale for the control method used

    def step(self):
        """Perform a step in the environment. This will now use generate_and_apply_actions method."""
        self.generate_and_apply_actions()  # Replace direct action application with MIMO-generated actions
        super().stepSimulation()  # Advances the simulation forward by one timestep

    def load_scenario(self, scenario_file):
        # Load scenario configurations here
        pass

    def reset(self):
        # Your reset logic here
        observations = (
            self._get_observation()
        )  # Assuming this returns a list of observations
        return np.array(observations)  # Convert list to NumPy array

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
        drone_pos = self.client.getBasePositionAndOrientation(drone_id)[0]
        return not (min_bound < drone_pos < max_bound)

    def _get_reward(self, drone_id):
        # Example: simple distance-based reward
        target_pos = np.array([0, 0, 1])  # Assuming a target position
        drone_pos = np.array(self.client.getBasePositionAndOrientation(drone_id)[0])
        distance = np.linalg.norm(target_pos - drone_pos)
        return -distance

    def _log_data(self):
        # Log drone states, actions, and environmental conditions here
        pass

    def _actionSpace(self):
        # Example: Define an action space where each action is a continuous value between -1 and 1
        # Adjust the shape as necessary for your specific use case
        return spaces.Box(low=-1, high=1, shape=(self.num_drones, 4), dtype=np.float32)

    def _observationSpace(self):
        # Example: Define an observation space with arbitrary bounds
        # Adjust the shape as necessary based on what your environment observes
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_drones, 12), dtype=np.float32
        )

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
            vel, _ = pybullet.getBaseVelocity(drone_id, physicsClientId=self.env)

            # Store information in the dictionary
            info[f"drone_{idx}_position"] = pos
            info[f"drone_{idx}_velocity"] = vel

        # You can add more environment-specific information here as needed
        return info
