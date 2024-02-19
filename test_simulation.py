import unittest
from simulation_environment import (
    CustomDroneEnv,
    DEFAULT_NUM_DRONES,
)  # Ensure correct import path
import numpy as np


class TestCustomDroneEnv(unittest.TestCase):
    def setUp(self):
        """Initialize the CustomDroneEnv environment for testing."""
        self.env = CustomDroneEnv(gui=False, num_drones=DEFAULT_NUM_DRONES)

        self.env.reset()

    def test_initialization(self):
        """Test if the environment initializes correctly."""
        self.assertEqual(
            self.env.num_drones,
            DEFAULT_NUM_DRONES,
            "Incorrect number of drones initialized.",
        )
        self.assertEqual(
            self.env.action_space.shape,
            (DEFAULT_NUM_DRONES, 7),
            "Action space dimensions are incorrect.",
        )
        self.assertEqual(
            self.env.observation_space.shape,
            (DEFAULT_NUM_DRONES, 10),
            "Observation space dimensions are incorrect.",
        )

    def test_step_functionality(self):
        """Test the step function of the environment."""
        example_action = np.zeros(
            (DEFAULT_NUM_DRONES, 7)
        )  # Example zero actions for each drone
        observations, rewards, dones, infos = self.env.step(example_action)

        self.assertEqual(
            observations.shape,
            (DEFAULT_NUM_DRONES, 10),
            "Step function returned incorrect observation shape.",
        )
        self.assertEqual(
            len(rewards),
            DEFAULT_NUM_DRONES,
            "Step function returned incorrect number of rewards.",
        )
        self.assertEqual(
            len(dones),
            DEFAULT_NUM_DRONES,
            "Step function returned incorrect number of dones.",
        )
        self.assertIsInstance(
            infos, dict, "Step function should return an info dictionary."
        )

    def test_drone_movement(self):
        """Test if drones can change state based on an action."""
        self.env.reset()
        # Example action for each drone might involve movement or change in velocity
        actions = [
            [1, 0, 0, 0] for _ in range(self.num_drones)
        ]  # Placeholder action - adjust based on actual action space
        obs, _, _, _ = self.env.step(actions)
        # Assuming the first value in observation indicates a positional change
        for drone_obs in obs:
            self.assertNotEqual(drone_obs[0], 0, "Drone did not move as expected.")

    def test_reset_functionality(self):
        """Test if the environment can be reset correctly."""
        self.env.step(
            np.zeros((DEFAULT_NUM_DRONES, 7))
        )  # Perform an action to change the state
        observations = self.env.reset()

        self.assertEqual(
            observations.shape,
            (DEFAULT_NUM_DRONES, 10),
            "Reset did not return correct observation shape.",
        )
        # Add additional assertions based on expected initial state post-reset, if applicable


if __name__ == "__main__":
    unittest.main()
