import unittest
from simulation_environment import CustomDroneEnv
import numpy as np

class TestCustomDroneEnv(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        self.env = CustomDroneEnv(gui=False, num_drones=3)  # Assuming no GUI needed for testing
        self.env.reset()

    def test_environment_configuration(self):
        """Test the environment's basic configuration."""
        self.assertEqual(self.env.num_drones, 3, "Unexpected number of drones.")
        self.assertTrue(self.env.physics in ["pyb"], "Unexpected physics engine.")
    def test_action_generation_and_application(self):
        """Test that actions generated by the MIMO transformer are applied correctly."""
        # Assuming there's a way to mock or preset the output of the MIMO transformer
        mock_actions = np.random.uniform(-1, 1, (self.env.num_drones, 7))
        # Mock the transformer output here, then:
        self.env.generate_and_apply_actions()
        # Verify that the drones' states have changed in a way that reflects the mock actions
        # This might involve checking positions or velocities of drones to ensure actions were applied
    def test_observation_space(self):
        """Check the observation space's bounds and dimensions."""
        observation_space = self.env.observation_space
        self.assertEqual(observation_space.shape, (3, 10), "Incorrect observation space shape.")
        self.assertTrue((observation_space.high > 1e3).all() or (observation_space.low < -1e3).all(), "Observation space bounds are incorrect.")

    def test_step_outcome(self):
        """Ensure the step method returns expected shapes and types."""
        action = np.zeros((3, 7))  # Assuming zero action for simplicity
        observations, rewards, dones, infos = self.env.step()
        self.assertEqual(observations.shape, (3, 10), "Incorrect observation shape.")
        self.assertEqual(len(rewards), 3, "Incorrect number of rewards.")
        self.assertEqual(len(dones), 3, "Incorrect number of done flags.")
        self.assertIsInstance(infos, dict, "Info should be a dictionary.")

    def test_reset_state(self):
        """Confirm that the environment resets to a valid initial state."""
        observations = self.env.reset()
        self.assertEqual(observations.shape, (3, 10), "Reset observation shape is incorrect.")
        # Further checks can be added here depending on the expected initial state

    def test_drone_dynamics(self):
        """Execute actions and verify that drone states change as expected."""
        initial_observation = self.env.reset()
        action = np.random.uniform(-1, 1, (3, 7))  # Random action
        observations, _, _, _ = self.env.step()
        # This test assumes the first 3 elements of the observation are position; adjust as necessary
        self.assertFalse(np.array_equal(initial_observation[:, :3], observations[:, :3]), "Drones' positions did not change as expected.")

if __name__ == "__main__":
    unittest.main()
