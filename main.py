from example import run_simulation
from simulation_environment import CustomDroneEnv
DEFAULT_NUM_DRONES = 6
DEFAULT_GUI = True
def main():
    simulation = CustomDroneEnv(num_drones=DEFAULT_NUM_DRONES, gui=DEFAULT_GUI)  # Add missing parameters
    while not simulation.is_completed(): # Main simulation loop
        observations = simulation.get_observations() # Get current state observations from the simulation
        actions = simulation.mimo_transformer.transform_command(observations) # Directly use MIMO transformer to generate actions based on observations
        simulation.apply_actions(actions) # Apply the actions to the simulation
        simulation.update() # Update the simulation to the next time step
        simulation.render() # Render the simulation (optional, for visualization)
if __name__ == "__main__":
    main()
    run_simulation(render=True)
