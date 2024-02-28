# FILEPATH: /D:/resumewebsite/functioncalling/drone-swarm-fresh/train.py

import numpy as np
import torch
import torch.nn.functional as F
from simulation_environment import CustomDroneEnv
from tokenizer import Tokenizer
from hrtx.hrtx.mimo import MIMOTransformer
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Initialize the environment and the transformer
num_drones = 2
transformer = MIMOTransformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    num_robots=num_drones,
)
env = CustomDroneEnv(
    num_drones=num_drones,
    gui=True,
    transformer=transformer,
    simulation=True,
    control_freq_hz=48,
    simulation_freq_hz=240,
)
# Initialize some parameters for training
num_episodes = 10
learning_rate = 0.01
discount_factor = 0.99

# Assuming MIMOTransformer has parameters that require gradients
optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
loss_function = None  # Not used directly
def get_desired_state(task, current_step, initial_state, waypoints=None):
    """
    Generate the desired state based on the task.

    Parameters:
    - task: The current task ('hover', 'move_to_location', etc.)
    - current_step: The current step or iteration in the episode.
    - initial_state: The initial state of the drone, used for hovering.
    - waypoints: A list of waypoints for navigation tasks.

    Returns:
    - desired_state: The desired state representation.
    """
    if task == "hover":
        desired_state = initial_state  # For hovering, the desired state is the initial state
        print(f"desired_state: {desired_state}")
    elif task == "move_to_location":
        # Assuming waypoints is a list of states, and current_step can index into it
        # This may need to be adjusted based on how you track progress towards waypoints
        waypoint_index = min(current_step, len(waypoints) - 1)
        desired_state = waypoints[waypoint_index]
    else:
        raise ValueError(f"Unknown task: {task}")

    return desired_state

#loss_function = torch.nn.CrossEntropyLoss()  # Adjust based on your task requirements
loss_function = torch.nn.MSELoss()
task="hover"
# Training loop
for episode in range(num_episodes):
    # Reset the environment and the episode data
    step = 0
    print("resetting environment")
    initial_state = env.reset(task=task)
    print("initial state", initial_state)
    max_rpm = env.env.MAX_RPM
    tokenizer = Tokenizer(num_drones, max_rpm)
    episode_rewards = []
    total_loss = 0
    state = initial_state
    # Loop for each step in the episode
    while True:
        optimizer.zero_grad()
        logits = env.generate_action(state) # Get action probabilities from the transformer
        actions = tokenizer.decode_transformer_outputs(logits)
        print(logits.requires_grad)
        reward, obs, done = env.step(step, actions) # Take a step
        episode_rewards.append(reward) # Store the reward and the gradient
        desired_state=get_desired_state(task, step, initial_state)
        obs_tensor = torch.stack([torch.tensor(drone['position']) for drone in obs])
        desired_state_tensor = torch.stack([torch.tensor(drone['position']) for drone in desired_state])


        loss = loss_function(obs_tensor, desired_state_tensor)
        loss.backward()
        optimizer.step()  # Update model parameters
        # If the episode is done, update the weights of the transformer
        if done:
            env.env.close()
            break
        step += 1
        # Update the state
        state = obs

# Close the environment
env.env.close()
