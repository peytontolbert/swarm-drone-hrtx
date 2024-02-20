import torch
import torch.nn.functional as F
from torch import nn
from HRTX.hrtx.mimo import MIMOTransformer
import numpy as np
from gym import spaces
def transform_observation(observation_tensor):
    """Applies a linear transformation to project observation features to model dimensions."""
    #observation_tensor = torch.tensor(observation, dtype=torch.float32)
    # Apply task to the observation tensor
    task_id = [task_to_id[task]]  # Get the numerical ID for the task
    print(f"Task ID: {task_id}")
    task_tensor = torch.tensor(task_id, dtype=torch.float32)
    print(f"Task tensor: {task_tensor}")
    # Concatenate the task information with the observation.
    # This requires the task information to be of compatible shape.
    # Combine the observation tensor and task tensor
    observation_tensor = observation_tensor.unsqueeze(0) # Now [1, feature_length]
    task_tensor = task_tensor.unsqueeze(0) # Now [1, task_length] Making it compatible for concatenation
    print(f"Observation tensor shape: {observation_tensor.shape}")
    print(f"Task tensor shape: {task_tensor.shape}")
    # Concatenate along the feature dimension (dim=1)
    combined_tensor = torch.cat([observation_tensor, task_tensor], dim=1)  # Concatenate along the second dimension
    transformed_observation = expansion_layer(combined_tensor.unsqueeze(0))
    print(transformed_observation.shape)
    return transformed_observation
def actionSpace(self):
    """Returns the action space of the environment.

    Returns:
        spaces.Box: An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.
    """
    act_lower_bound = np.zeros((self.NUM_DRONES, 4))
    act_upper_bound = np.full((self.NUM_DRONES, 4), self.MAX_RPM)
    return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

# Input tensor
task = "fly around in circles"
model_dim = 512
# Assuming this mapping is defined in your __init__ or a similar setup method
task_to_id = {'fly around in circles': 1, 'fly into objective': 2, 'hover': 3}  # Example mapping
# Example tokenizer function (very simplistic)
# Embedding layer for textual command processing
expansion_layer = nn.Linear(21, model_dim)  # Projects from 20 features to 512
x = torch.randn(20)
x_transformed = transform_observation(x)
x = [x, x, x]

# Create the model
model = MIMOTransformer(dim=512, depth=6, heads=8, dim_head=64, num_robots=1)

output = model(x)
print(output)