# FILEPATH: /D:/resumewebsite/functioncalling/drone-swarm-fresh/test_example.py

import unittest
import numpy as np
from simulation_environment import CustomDroneEnv
from hrtx.hrtx.mimo import MIMOTransformer


class TestExample(unittest.TestCase):
    def setUp(self):
        """Initialize the CustomDroneEnv environment for testing."""
        self.num_drones = 5
        self.transformer = MIMOTransformer(
            dim=512,
            depth=6,
            heads=8,
            dim_head=64,
            num_robots=self.num_drones,
        )
        self.env = CustomDroneEnv(
            num_drones=self.num_drones,
            gui=False,
            transformer=self.transformer,
            simulation=True,
            control_freq_hz=1 / 48,
            simulation_freq_hz=1 / 240,
        )

    def test_initialization(self):
        """Test if the environment initializes correctly."""
        self.assertEqual(
            self.env.num_drones,
            self.num_drones,
            "Incorrect number of drones initialized.",
        )
        self.assertIsInstance(
            self.env.transformer,
            MIMOTransformer,
            "Transformer is not an instance of MIMOTransformer.",
        )

    def test_step_functionality(self):
        """Test the step function of the environment."""
        control_commands = np.zeros((self.num_drones, 4))
        results = self.env.step(control_commands)
        self.assertIsInstance(
            results, tuple, "Step function should return a tuple."
        )
        self.assertEqual(
            len(results),
            4,
            "Step function should return a tuple of length 4.",
        )

    def test_close_functionality(self):
        """Test if the environment can be closed correctly."""
        self.env.close()
        # Add assertions here based on your specific requirements for the close function


if __name__ == "__main__":
    unittest.main()
