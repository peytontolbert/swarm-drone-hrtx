from torch import Tensor
from typing import List
import numpy as np

class Tokenizer:
    def __init__(self, num_drones):
        # Assuming output_dim is the dimension of each action tensor for a drone
        self.num_drones = num_drones

    def decode_transformer_outputs(
        self, output_tensors: List[Tensor]
    ):
        decoded_actions = []
        # Process each tensor in action_tensor_list to convert to actionable commands
        action_tensor = output_tensors[
            0
        ].squeeze()  # Remove batch dimension
        print(f' action_tensor: {action_tensor.shape}')
        for i in range(action_tensor.size(0)):  # Iterate over drones
            # Example decoding process
            # print(f' i, action_tensor: {i}, {action_tensor}')
            action_data = (
                action_tensor[i].detach().cpu().numpy()
            )  # Assuming a simple conversion; adjust as necessary
            target_pos = action_data[
                :3
            ]  # Example: First 3 values are target position
            target_rpy = action_data[
                3:6
            ]  # Next 3 values are target orientation
            drone_id = (
                i  # Assuming the drone ID is the index in the tensor
            )
            decoded_actions.append(
                {
                    "pos": target_pos,
                    "rpy": target_rpy,
                    "drone_id": drone_id,
                }
            )
        # print(f'length of decoded actions: {len(decoded_actions)}')
        return decoded_actions
    