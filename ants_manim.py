from manim import *
import numpy as np

# Represents a position in 2D space with a unique index.
class Position:
    def __init__(self, i, x, y):
        self.i = i
        self.x = x
        self.y = y

    # Calculate the Euclidean distance to another position.
    def distance_to(self, other):
        return np.hypot(self.x - other.x, self.y - other.y)

# Represents an ant that can traverse through positions, tracking its path.
class Ant:
    def __init__(self, initial_index=0, desirability_power=4):
        self.i = initial_index
        self.index_history = [initial_index]
        self.desirability_power = desirability_power

    # Update the ant's current position and record the path.
    def move_index(self, index):
        self.index_history.append(index)
        self.i = index

# Compute a distance matrix for all positions.
def precompute_distances(positions):
    nr_positions = len(positions)
    distances = np.zeros((nr_positions, nr_positions))
    for i in range(nr_positions):
        for j in range(i + 1, nr_positions):
            dist = positions[i].distance_to(positions[j])
            distances[i, j] = distances[j, i] = dist
    return distances

# Compute desirability values based on distances, which guide ant movement.
def precompute_desirabilities(distances, desirability_power):
    nr_positions = distances.shape[0]
    desirabilities = np.zeros((nr_positions, nr_positions))
    for i in range(nr_positions):
        for j in range(nr_positions):
            if i != j and distances[i, j] > 0:
                desirabilities[i, j] = (1 / distances[i, j]) ** desirability_power
    return desirabilities

# Simulate the path traversal for a given ant based on desirability probabilities.
def ant_run(ant, distances, desirabilities, positions):
    nr_positions = len(positions)
    for _ in range(nr_positions - 1):
        current_position = ant.i
        visited_mask = np.isin(np.arange(nr_positions), ant.index_history)
        available_desirabilities = desirabilities[current_position, ~visited_mask]
        available_indices = np.arange(nr_positions)[~visited_mask]
        probabilities = available_desirabilities / available_desirabilities.sum()
        next_position = np.random.choice(available_indices, p=probabilities)
        ant.move_index(next_position)
    ant.index_history.append(ant.index_history[0])  # Return to starting point.
    return ant.index_history

# Calculate the total path distance for an ant's path.
def calculate_total_distance(ant_history, distances):
    return sum(distances[ant_history[i - 1], ant_history[i]] for i in range(1, len(ant_history)))

# Update pheromone levels based on ants' paths to influence future paths.
def update_pheromones(pheromones, ant_histories, distances, evaporation_rate=0.1, deposit_multiplier=20):
    for ant_history in ant_histories:
        total_distance = calculate_total_distance(ant_history, distances)
        pheromone_deposit = (1 / total_distance) * deposit_multiplier

        # Increase pheromone levels along the paths taken by ants.
        for i in range(1, len(ant_history)):
            start = ant_history[i - 1]
            end = ant_history[i]
            pheromones[start, end] += pheromone_deposit
            pheromones[end, start] += pheromone_deposit

    # Apply pheromone evaporation to simulate dissipation over time.
    pheromones *= (1 - evaporation_rate)
    return pheromones

# Manim scene to visualize the ant colony optimization process.
class AntSimulation(Scene):
    def construct(self):
        # Scene dimensions and simulation settings.
        width, height = 10, 6
        nr_positions = 40
        nr_ants = 10
        desirability_power = 2
        tail_length = 0.5
        pheromone_power = 4
        nr_iterations = 10

        # Create positions and initialize distances and pheromones.
        positions = [
            Position(i, *np.random.uniform([0, 0], [width, height])) 
            for i in range(nr_positions)
        ]
        distances = precompute_distances(positions)
        base_desirabilities = precompute_desirabilities(distances, desirability_power)
        pheromones = np.ones_like(base_desirabilities)

        # Visualize the positions as dots.
        dots = [
            Dot(point=[pos.x - width / 2, pos.y - height / 2, 0], radius=0.05, color=BLUE)
            for pos in positions
        ]
        self.play(*[Create(dot) for dot in dots])

        all_ant_histories = []
        all_total_distances = []

        # Run the simulation for the specified number of iterations.
        for iteration in range(nr_iterations):
            # Update desirability values with pheromone influence.
            desirabilities = base_desirabilities * pheromones**pheromone_power

            # Create ants and simulate their paths.
            ants = [Ant(np.random.randint(0, nr_positions), desirability_power) for _ in range(nr_ants)]
            ant_histories = [ant_run(ant, distances, desirabilities, positions) for ant in ants]

            # Store histories for later analysis.
            all_ant_histories.extend(ant_histories)

            # Visualize desirability lines between positions.
            def generate_desire_lines(desirabilities):
                lines = []
                nr_positions = desirabilities.shape[0]
                max_desirability = np.max(desirabilities)
                for i in range(nr_positions):
                    for j in range(i + 1, nr_positions):
                        desirability = desirabilities[i, j]
                        norm_desirability = (desirability / max_desirability)
                        opacity = max(0.03, min(0.95, norm_desirability))
                        stroke_width = 4
                        start = [positions[i].x - width / 2, positions[i].y - height / 2, 0]
                        end = [positions[j].x - width / 2, positions[j].y - height / 2, 0]

                        line = Line(start=start, end=end, color=BLUE, stroke_width=stroke_width)
                        line.set_opacity(opacity)
                        lines.append(line)
                return lines

            desire_lines = generate_desire_lines(pheromones)
            self.play(*[Create(line) for line in desire_lines])

            # Create ant dots and trace their paths with fading tails.
            ant_dots = [
                Dot(
                    point=[positions[ant_history[0]].x - width / 2, positions[ant_history[0]].y - height / 2, 0],
                    color=RED
                )
                for ant_history in ant_histories
            ]
            ant_trails = [
                TracedPath(
                    ant_dot.get_center,
                    stroke_color=YELLOW,
                    stroke_width=2,
                    dissipating_time=tail_length
                )
                for ant_dot in ant_dots
            ]

            # Add ant dots and trails to the scene.
            for ant_dot, ant_trail in zip(ant_dots, ant_trails):
                ant_dot.set_z_index(1)
                self.add(ant_dot, ant_trail)

            total_distances = [0] * nr_ants
            max_steps = max(len(history) for history in ant_histories)

            # Animate the ants' movements step by step.
            for step in range(1, max_steps):
                move_animations = []
                for ant_index, ant_history in enumerate(ant_histories):
                    if step < len(ant_history):
                        start_pos = positions[ant_history[step - 1]]
                        end_pos = positions[ant_history[step]]
                        ant_dot = ant_dots[ant_index]
                        end_point = [end_pos.x - width / 2, end_pos.y - height / 2, 0]
                        move_animation = ant_dot.animate.move_to(end_point)
                        move_animations.append(move_animation)
                        total_distances[ant_index] += distances[ant_history[step - 1], ant_history[step]]
                self.play(*move_animations, run_time=0.3)

            all_total_distances.extend(total_distances)
            best_distance = min(total_distances)
            best_ant_index = total_distances.index(best_distance)

            best_distance_text = Text(
                f"Iteration {iteration + 1}: Best Distance: Ant {best_ant_index + 1} with {np.round(best_distance, 2)}",
                font_size=24
            ).to_edge(UP)
            self.play(Write(best_distance_text), run_time=2)
            self.wait(1)

            # Update pheromone levels based on ant paths.
            pheromones = update_pheromones(pheromones, ant_histories, distances)
            self.play(*[FadeOut(dot) for dot in ant_dots], *[FadeOut(line) for line in desire_lines], FadeOut(best_distance_text))

        overall_best_distance = min(all_total_distances)
        overall_best_ant_index = all_total_distances.index(overall_best_distance)
        overall_best_ant_history = all_ant_histories[overall_best_ant_index]

        # Draw the path of the best-performing ant.
        best_path_lines = [
            Line(
                start=[positions[overall_best_ant_history[i]].x - width / 2, positions[overall_best_ant_history[i]].y - height / 2, 0],
                end=[positions[overall_best_ant_history[i + 1]].x - width / 2, positions[overall_best_ant_history[i + 1]].y - height / 2, 0],
                color=RED,
                stroke_width=4
            )
            for i in range(len(overall_best_ant_history) - 1)
        ]
        self.play(*[Create(line) for line in best_path_lines])
        overall_best_distance_text = Text(
            f"Overall Best Distance: {np.round(overall_best_distance, 2)}",
            font_size=28
        ).to_edge(UP)
        self.play(Write(overall_best_distance_text))
        self.wait(4)
