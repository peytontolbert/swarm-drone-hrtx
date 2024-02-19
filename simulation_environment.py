from pybullet_utils import bullet_client
import pybullet
from render.gym_pybullet_drones.envs.BaseAviary import BaseAviary
from render.gym_pybullet_drones.utils.enums import DroneModel
from render.gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from render.gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym import spaces
import numpy as np
from HRTX.hrtx.mimo import MIMOTransformer
import torch

DEFAULT_DRONES = DroneModel("cf2x")
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
min_bound = -100
max_bound = 100
class CustomDroneEnv(BaseAviary):
    """
    A custom drone simulation environment that inherits from BaseAviary for physics
    and rendering, but adds custom functionalities.
    """

    def __init__(self, 
                num_drones, 
                gui=DEFAULT_GUI,
                record_video=DEFAULT_RECORD_VISION,
                plot=DEFAULT_PLOT,
                physics=DEFAULT_PHYSICS,
                obstacles=DEFAULT_OBSTACLES,
                simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
                control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
                user_debug_gui=DEFAULT_USER_DEBUG_GUI,
                duration_sec=DEFAULT_DURATION_SEC):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        #self.client = bullet_client.BulletClient(connection_mode=connection_mode)
        #self.client = pybullet.connect(pybullet.DIRECT)  # Instead of p.GUI
        
        self.num_drones = num_drones
        if gui:
            self.CLIENT = pybullet.connect(pybullet.GUI)  # For manual testing with visualization
        else:
            self.CLIENT = pybullet.connect(pybullet.DIRECT)  # For automated tests without GUI
        # Pre-generate initial positions to avoid overlaps; assumes max 3 drones for simplicity
        initial_xyzs = np.zeros((num_drones, 3))
        initial_xyzs[:, 2] = 0.1  # Set z-axis (height) to 0.1 for all drones
        initial_xyzs[:, 0] = np.arange(num_drones)  # Spread out drones along the x-axis

        # Initialize the BaseAviary with desired settings
        super(CustomDroneEnv, self).__init__(
            drone_model=DEFAULT_DRONES,
            num_drones=num_drones,   
            neighbourhood_radius=10,
            initial_xyzs=initial_xyzs[:3],
            pyb_freq=simulation_freq_hz,
            ctrl_freq=control_freq_hz,
            gui=gui,
            obstacles=obstacles,
            record=record_video,
            user_debug_gui=user_debug_gui,
        )
        self.mimo_transformer = MIMOTransformer(num_drones=self.num_drones)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_drones, 7), dtype=np.float32
        )  # Extended action space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_drones, 10), dtype=np.float32
        )  # Extended observation space

    def apply_high_level_command(self, command):
        """
        Applies a high-level command to the environment.
        """
        drone_actions = self.mimo_transformer.transform_command(command)
        for drone_id, action in enumerate(drone_actions):
            # Assuming you have a method to apply individual actions
            self.apply_action(drone_id, action)
    
    def _apply_action(self, drone_id, action):
        """
        Applies the given action to the specified drone.

        Parameters:
        - drone_id: The ID of the drone to which the action should be applied.
        - action: The action to apply. This could be motor speeds or some other form of control command.
        """
        # Assuming action is a list of motor speeds for simplicity
        motor_speeds = action

        # Convert motor speeds into PyBullet commands, if necessary
        # This step depends on how your drones are set up in PyBullet and what the action represents
        # For this example, we'll assume the action directly represents motor speeds

        # Apply the motor speeds to the drone
        # This is where you'd call a PyBullet function to set the motor speeds of the drone
        # The exact function call depends on your drone model and how it's set up in PyBullet
        # For example, you might use something like this:
        pybullet.setJointMotorControlArray(bodyUniqueId=self.drone_ids[drone_id],
                                        jointIndices=[0, 1, 2, 3],  # Assuming four motors
                                        controlMode=pybullet.VELOCITY_CONTROL,
                                        targetVelocities=motor_speeds,
                                        physicsClientId=self.CLIENT)
    def _get_observation(self):
        """Override to collect observations for all drones, formatted as tensors."""
        observations = []
        for drone_id in range(self.num_drones):
            drone_observation = super()._getDroneStateVector(drone_id)  # Collect per-drone observations
            observations.append(torch.tensor(drone_observation).unsqueeze(0))  # Convert to tensors and batch
        return observations
    def generate_and_apply_actions(self):
        """Generates actions for all drones using the MIMO transformer and applies them."""
        observations = self._get_observation()  # Collect observations in the required tensor format
        actions = self.mimo_transformer(observations)  # Process observations through MIMO transformer
        
        for drone_id, action in enumerate(actions):
            self._apply_action(drone_id, action.tolist())  # Convert tensors back to list and apply actions
    def step(self):
        """Perform a step in the environment. This will now use generate_and_apply_actions method."""
        self.generate_and_apply_actions()  # Replace direct action application with MIMO-generated actions
        super().stepSimulation()  # Advances the simulation forward by one timestep
    def load_scenario(self, scenario_file):
        # Load scenario configurations here
        pass
    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        super().reset()
        return self._get_observation()
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
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_drones, 12), dtype=np.float32)