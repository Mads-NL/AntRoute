https://youtu.be/UwlfATjSZAc

# Ant Colony Optimization Visualization
This project implements an Ant Colony Optimization (ACO) algorithm and visualizes the process using Manim, a mathematical animation library in Python. The ACO algorithm is inspired by the foraging behavior of ants and is used to solve complex optimization problems, such as finding the shortest path in a graph.

## Visualization Example
[![Watch the video](https://img.youtube.com/vi/UwlfATjSZAc/0.jpg)](https://youtu.be/UwlfATjSZAc)

## Project Overview
The ACO visualization demonstrates how ants collectively explore paths to find the optimal route between multiple points. The simulation features several key components:
- Ant Behavior: Each ant chooses its path based on the distance to other points and the pheromone levels left by previous ants.
- Pheromone Trails: As ants traverse paths, they deposit pheromones, which influence the decisions of subsequent ants, gradually guiding them towards the best routes.
- Dynamic Visualization: The simulation visually represents the movement of ants, their paths, and the fading pheromone trails, creating an engaging experience.

## Key Features
- Real-Time Simulation: Observe how ants navigate through the positions in real time, adjusting their paths based on pheromone levels and distances.
- Customizable Parameters: Modify parameters such as the number of ants, positions, desirability influence, and pheromone decay rates to explore different scenarios and outcomes.
- Animated Trails: Visualize the paths taken by each ant, with trails that dissipate over time, illustrating how past movements influence current decisions.

## How It Works
1. Initialization: The simulation creates a set of random positions for the ants to traverse.
2. Distance Calculation: The distances between all positions are precomputed to facilitate quick access during the simulation.
3. Desirability and Pheromone Levels: The desirability of each path is calculated based on distance, and pheromones are initialized to influence ant decisions.
4. Ant Movement: Ants move between positions based on a probability distribution that considers both desirability and pheromone levels.
5. Pheromone Update: After all ants complete their paths, the pheromone levels are updated based on the distances traveled, with evaporation applied to simulate the decay of pheromones over time.

## Visualization Example
The visualization consists of animated dots representing the ants, lines indicating pheromone trails, and dynamic paths that showcase the exploration process. Each iteration reveals how the ants converge on the shortest path, demonstrating the effectiveness of the ACO algorithm.

## Acknowledgements
- Manim: A fantastic tool for creating mathematical animations that bring concepts to life.