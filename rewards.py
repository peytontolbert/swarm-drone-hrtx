import numpy as np
def calculate_hover_reward(self, nth_drone, target_position):
    """
    Calculate the reward for maintaining a hover position.

    Parameters:
    - nth_drone: Index of the drone for which to calculate the reward.
    - target_position: The target position (x, y, z) the drone should maintain.

    Returns:
    - reward: A float representing the calculated reward.
    """
    state_vector = self._getDroneStateVector(nth_drone)
    position = state_vector[:3]  # Extract position
    velocity = state_vector[9:12]  # Extract linear velocity
    angular_velocity = state_vector[12:15]  # Extract angular velocity

    # Calculate the distance from the target position
    distance_to_target = np.linalg.norm(position - target_position)

    # Calculate the magnitude of velocity and angular velocity
    velocity_magnitude = np.linalg.norm(velocity)
    angular_velocity_magnitude = np.linalg.norm(angular_velocity)

    # Define weights for each component of the reward
    weight_distance = -1.0  # Negative because we want to minimize distance
    weight_velocity = -0.5  # Negative because we want to minimize velocity
    weight_angular_velocity = -0.5  # Negative because we want to minimize angular velocity

    # Calculate weighted components of the reward
    reward_distance = weight_distance * distance_to_target
    reward_velocity = weight_velocity * velocity_magnitude
    reward_angular_velocity = weight_angular_velocity * angular_velocity_magnitude

    # Sum the components to get the total reward
    total_reward = reward_distance + reward_velocity + reward_angular_velocity

    return total_reward