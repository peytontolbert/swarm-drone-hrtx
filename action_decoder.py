from torch import Tensor
from typing import List
class ActionDecoder:
    def __init__(self, num_drones, output_dim):
        # Assuming output_dim is the dimension of each action tensor for a drone
        self.num_drones = num_drones
        self.output_dim = output_dim

    def decode(self, output_tensors: List[Tensor]):
        """Decodes the output tensors into a list of actions for each drone."""
        decoded_actions = []
        for tensor in output_tensors:
            # Assuming the tensor shape is [1, output_dim], where output_dim includes all action details
            # Flatten the tensor and convert to numpy for easier processing
            action_data = tensor.squeeze().detach().numpy()
            # Decode action_data into meaningful control commands
            # This is a placeholder, actual decoding depends on how your data is structured
            target_pos = action_data[:3]  # Example: First 3 values are target position
            target_rpy = action_data[3:6]  # Next 3 values are target orientation (roll, pitch, yaw)
            decoded_actions.append({'pos': target_pos, 'rpy': target_rpy})
        return decoded_actions