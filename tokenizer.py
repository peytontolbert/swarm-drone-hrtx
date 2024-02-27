import torch
from torch import Tensor
from typing import List
from torch import nn
import numpy as np


class Tokenizer:
    def __init__(self, num_drones: int, max_rpm: int=500000):
        # Assuming output_dim is the dimension of each action tensor for a drone
        self.num_drones = num_drones
        self.max_rpm = max_rpm
        input_dim=512
        output_dim=4
        self.linear = nn.Linear(input_dim, output_dim)  # Linear layer mapping from 512 to 4

    def decode_transformer_outputs(
        self, output_tensors: List[Tensor]
    ):
        # Scale the actions to the RPM range.
        scaled_action_reshaped = self.linear(output_tensors)
        action_squeezed_scaled = scaled_action_reshaped * self.max_rpm 
        actions = self._preprocessAction(action_squeezed_scaled)
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
                torch.clamp(action[i, :], min=10, max=self.max_rpm)
                for i in range(self.num_drones)
            ]
        )
