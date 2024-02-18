# main.py
from example import run_simulation

from simulation_environment import DroneSimulation
from llm_control_model import LLMControlModel

def main():
    # Initialize the simulation environment
    simulation = DroneSimulation()

    # Initialize the LLM control model
    control_model = LLMControlModel()

    # Main simulation loop
    while not simulation.is_completed():
        # Get current state observations from the simulation
        observations = simulation.get_observations()

        # Use the LLM control model to determine actions based on observations
        actions = control_model.predict_actions(observations)

        # Apply the actions to the simulation
        simulation.apply_actions(actions)

        # Update the simulation to the next time step
        simulation.update()

        # Render the simulation (optional, for visualization)
        simulation.render()

if __name__ == "__main__":
    main()
    run_simulation(render=True)
