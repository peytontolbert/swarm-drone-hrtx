import torch
from torch import Tensor
from typing import List
import numpy as np


class Tokenizer:
    def __init__(self, num_drones, max_rpm=1000):
        # Assuming output_dim is the dimension of each action tensor for a drone
        self.num_drones = num_drones
        self.max_rpm = max_rpm

    def decode_transformer_outputs(
        self, output_tensors: List[Tensor]
    ):
        # Process each tensor in action_tensor_list to convert to actionable commands
        action_tensor = output_tensors[
            0
        ].squeeze()  # Remove batch dimension
        actions = self._preprocessAction(action_tensor)
        return actions
    
    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return torch.stack(
            [
                torch.clamp(action[i, :], min=0, max=self.max_rpm)
                for i in range(self.num_drones)
            ]
        )
