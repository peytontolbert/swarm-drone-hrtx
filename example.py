import numpy as np
from simulation_environment import CustomDroneEnv
from hrtx.hrtx.mimo import MIMOTransformer
from tokenizer import Tokenizer

def simulate_drones(
    num_drones,
    enable_gui,
    simulation_time_step,
    control_time_step,
    duration_sec,
):
    """
    Simulates a swarm of drones in a custom environment.

    Args:
        num_drones (int): The number of drones in the swarm.
        enable_gui (bool): Flag to enable or disable the GUI for visualization.
        simulation_time_step (float): The time step for the simulation in seconds.
        control_time_step (float): The time step for the control update in seconds.
        duration_sec (int): The total duration of the simulation in seconds.

    Returns:
        None
    """
    # Configuration parameters
    DEFAULT_CONTROL_FREQ_HZ = 48
    transformer = MIMOTransformer(
        dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        num_robots=num_drones,
    )
    tokenizer = Tokenizer(num_drones)
    # Initialize the environment
    env = CustomDroneEnv(
        num_drones=num_drones,
        gui=enable_gui,
        transformer=transformer,
    )
    task = "hover"
    state = env.reset(task=task)
    # Main simulation loop
    current_time = 0.0
    for i in range(0, int(duration_sec * DEFAULT_CONTROL_FREQ_HZ)):
        action_probs = env.generate_action(state)
        actions = tokenizer.decode_transformer_outputs(action_probs)
        results = env.step(i, actions)
        current_time += control_time_step

    # Cleanup and closing of the environment if necessary
    env.close()


# Example usage
simulate_drones(
    num_drones=5,
    enable_gui=True,
    simulation_time_step=1 / 240,
    control_time_step=1 / 48,
    duration_sec=12,
)
