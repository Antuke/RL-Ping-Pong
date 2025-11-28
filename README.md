# Ping Pong Robot (DDPG)

This repository contains a Reinforcement Learning project where a robot arm learns to play ping pong using the **Deep Deterministic Policy Gradient (DDPG)** algorithm. The environment is simulated using **PyBullet**.

## Project Structure

*   `server.py`: The PyBullet simulation environment (server). **(Provided as part of the assignment)**
*   `train.py`: Script to train the DDPG agent.
*   `test.py`: Client script to load a trained model and play against the server.
*   `ddpg.py`: Implementation of the DDPG algorithm (Actor-Critic networks).
*   `models/`: Directory containing trained model checkpoints.

## Installation

Ensure you have the required dependencies installed (PyTorch, PyBullet, NumPy, etc.).

## Usage

### Running the Demo

To see the trained agent in action, you need to run the server and the client in separate terminals.

1.  **Start the Server:**
    ```bash
    python server.py
    ```

2.  **Run the Agent:**
    ```bash
    python test.py
    ```

## Authors

*   [Pietro Martano](https://github.com/pietroemme)
*   [Angelo Molinario](https://github.com/amolinario3)
*   [Massimiliano Ranauro](https://github.com/MassimilianoRanauro)
*   [Antonio Sessa](https://github.com/Antuke)
