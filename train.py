# FILEPATH: /D:/resumewebsite/functioncalling/drone-swarm-fresh/train.py

import numpy as np
import torch
import torch.nn.functional as F
from simulation_environment import CustomDroneEnv
from tokenizer import Tokenizer
from hrtx.hrtx.mimo import MIMOTransformer

# Initialize the environment and the transformer
num_drones = 5
transformer = MIMOTransformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    num_robots=num_drones,
)
tokenizer = Tokenizer(num_drones, max_rpm=1000)
env = CustomDroneEnv(
    num_drones=num_drones,
    gui=False,
    transformer=transformer,
    simulation=True,
    control_freq_hz=48,
    simulation_freq_hz=240,
)

# Initialize some parameters for training
num_episodes = 1000
learning_rate = 0.01
discount_factor = 0.99


# Function to calculate the return of an episode
def calculate_return(rewards, discount_factor):
    return sum(
        reward * (discount_factor**i)
        for i, reward in enumerate(rewards)
    )
task="hover"

# Training loop
for episode in range(num_episodes):
    # Reset the environment and the episode data
    step = 0
    state = env.reset(task=task)
    episode_rewards = []
    episode_gradients = []

    # Loop for each step in the episode
    while True:
        # Get the action probabilities from the transformer
        logits = env.generate_action(state)
        probabilities = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(logits=logits)# Create a distribution to sample from
        actions_sampled = m.sample()# Sample actions from the distribution
        # Get the log probabilities of the sampled actions
        log_probs = m.log_prob(actions_sampled)
        print(f"probabilities: {probabilities}")
        print(f"actions_sampled: {actions_sampled}")
        actions = tokenizer.decode_transformer_outputs(logits)
        # Take a step in the environment
        results = env.step(step, actions)
        reward, obs = env.calculate_reward(task, results)
        done = env.is_done(results,task)
        next_state = results

        # Store the reward and the gradient
        episode_rewards.append(reward)
        episode_gradients.append(log_probs)

        # If the episode is done, update the weights of the transformer
        if done:
            episode_return = calculate_return(
                episode_rewards, discount_factor
            )
            for gradient in episode_gradients:
                transformer.weights += (
                    learning_rate * gradient * episode_return
                )
            break
        step += 1
        # Update the state
        state = next_state

# Close the environment
env.close()
