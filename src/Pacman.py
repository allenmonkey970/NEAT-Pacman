import neat
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import choice, random
from freegames import floor, vector
from turtle import *
from multiprocessing import cpu_count

# ==== CONSTANTS ====

NUM_GENERATIONS = 200  # Number of generations to train NEAT
MEMORY_SIZE = 5        # Number of previous steps to store as memory
POSSIBLE_MOVES = [(5, 0), (-5, 0), (0, 5), (0, -5)]  # Possible movement directions for Pacman
COMBO_BONUS = 8        # Bonus for eating dots in a row
MAZE_CLEAR_BONUS = 500 # Bonus for clearing the maze
EVAL_EPSILON = 0.01     # Probability of random action (exploration) during evaluation
EVAL_MULTI_OBJECTIVE = True  # Whether to use multi-objective fitness

# Pacman and game layout constants
PACMAN_INIT = vector(-40, -80)
GHOSTS_INIT = [
    [vector(-180, 160), vector(5, 0)],
    [vector(-180, -160), vector(0, 5)],
    [vector(100, 160), vector(0, -5)],
    [vector(100, -160), vector(-5, 0)],
]
TILE_LAYOUT = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

# ==== Game and Feature Setup ==== #

state = {'score': 0, 'random_state': 1082}
path = Turtle(visible=False)
writer = Turtle(visible=False)
aim = vector(5, 0)
pacman = PACMAN_INIT.copy()
ghosts = [ [g[0].copy(), g[1].copy()] for g in GHOSTS_INIT ]
tiles = TILE_LAYOUT.copy()

def offset(point):
    """
    Calculate the index in the tile layout for a given position vector.

    Args:
        point (vector): The position vector.

    Returns:
        int: The tile index corresponding to the position.
    """
    x = (floor(point.x, 20) + 200) / 20
    y = (180 - floor(point.y, 20)) / 20
    index = int(x + y * 20)
    return index

def valid(point):
    """
    Check if a given position is valid for movement (not a wall).

    Args:
        point (vector): The position vector to check.

    Returns:
        bool: True if position is valid, False otherwise.
    """
    index = offset(point)
    if tiles[index] == 0:
        return False
    index = offset(point + 19)
    if tiles[index] == 0:
        return False
    return point.x % 20 == 0 or point.y % 20 == 0

def find_nearest_dot(sim_pacman, sim_tiles):
    """
    Find the nearest dot (pellet) to Pacman.

    Args:
        sim_pacman (vector): Pacman's position.
        sim_tiles (list): Current tile layout.

    Returns:
        list: [dx, dy, norm_dist], relative and normalized position/distance to the nearest dot.
    """
    min_dist = float('inf')
    nearest = None
    px, py = sim_pacman.x, sim_pacman.y
    for idx, tile in enumerate(sim_tiles):
        if tile == 1:
            tx = (idx % 20) * 20 - 200
            ty = 180 - (idx // 20) * 20
            dist = abs(px - tx) + abs(py - ty)
            if dist < min_dist:
                min_dist = dist
                nearest = (tx, ty)
    if nearest is None:
        return [0, 0, 0]
    dx = (nearest[0] - px) / 200
    dy = (nearest[1] - py) / 200
    norm_dist = min_dist / 400
    return [dx, dy, norm_dist]

def available_moves(sim_pacman, sim_tiles):
    """
    Get a list of valid moves for Pacman from the current position.

    Args:
        sim_pacman (vector): Pacman's position.
        sim_tiles (list): Tile layout.

    Returns:
        list: List of (dx, dy) tuples for valid directions.
    """
    moves = []
    for dx, dy in POSSIBLE_MOVES:
        pos = vector(sim_pacman.x + dx, sim_pacman.y + dy)
        if valid(pos):
            moves.append((dx, dy))
    return moves

def is_junction(sim_pacman, sim_tiles):
    """
    Check if Pacman is at a junction (more than two possible moves).

    Args:
        sim_pacman (vector): Pacman's position.
        sim_tiles (list): Tile layout.

    Returns:
        bool: True if junction, False otherwise.
    """
    moves = available_moves(sim_pacman, sim_tiles)
    return len(moves) > 2

def is_corridor(sim_pacman, sim_tiles):
    """
    Check if Pacman is in a corridor (exactly two possible moves).

    Args:
        sim_pacman (vector): Pacman's position.
        sim_tiles (list): Tile layout.

    Returns:
        bool: True if corridor, False otherwise.
    """
    moves = available_moves(sim_pacman, sim_tiles)
    return len(moves) == 2

def get_nn_input(sim_pacman, sim_ghosts, sim_tiles, step=0, prev_dot_dist=None, memory=None):
    """
    Prepare and normalize neural network input vector for Pacman.

    Args:
        sim_pacman (vector): Pacman's position.
        sim_ghosts (list): Ghost positions and directions.
        sim_tiles (list): Tile layout.
        step (int, optional): Current step in episode.
        prev_dot_dist (float, optional): Previous distance to nearest dot.
        memory (list, optional): Memory buffer of previous states.

    Returns:
        np.ndarray: Input vector for neural network.
    """
    px, py = sim_pacman.x / 200.0, sim_pacman.y / 200.0

    # Ghosts' info: relative positions and directions
    ghosts_rel = []
    for ghost in sim_ghosts:
        gx, gy = ghost[0].x / 200.0, ghost[0].y / 200.0
        ghosts_rel.extend([gx - px, gy - py, ghost[1].x / 5.0, ghost[1].y / 5.0])

    closest_dot = find_nearest_dot(sim_pacman, sim_tiles)

    num_dots = sim_tiles.count(1) / 100.0
    moves = available_moves(sim_pacman, sim_tiles)
    num_open_dirs = len(moves) / 4.0
    is_junc = 1 if is_junction(sim_pacman, sim_tiles) else 0
    is_corr = 1 if is_corridor(sim_pacman, sim_tiles) else 0
    curr_dot_dist = closest_dot[2]
    delta_dot_dist = 0.0 if prev_dot_dist is None else prev_dot_dist - curr_dot_dist

    # Memory buffer: previous positions and ghost positions
    memory_flat = []
    if memory and len(memory) == MEMORY_SIZE:
        for mem in memory:
            m_px, m_py = mem['pacman']
            memory_flat.extend([m_px / 200.0, m_py / 200.0])
            for g_pos in mem['ghosts']:
                memory_flat.extend([g_pos[0] / 200.0, g_pos[1] / 200.0])
    else:
        memory_flat = [0.0] * (MEMORY_SIZE * (2 + 4 * 2))

    return np.array(
        [px, py] + ghosts_rel + closest_dot + [num_dots, num_open_dirs, is_junc, is_corr, delta_dot_dist] + memory_flat,
        dtype=np.float32
    )

# Update this if you change MEMORY_SIZE or input features.
NN_INPUT_SIZE = (2 + 4*4 + 3 + 5) + MEMORY_SIZE * (2 + 4*2)

def eval_genome_picklable(genome, config):
    """
    Wrapper for parallel evaluation of genomes.

    Args:
        genome: NEAT genome to evaluate.
        config: NEAT configuration.

    Returns:
        float: Fitness score.
    """
    return eval_genome(genome, config, epsilon=EVAL_EPSILON, multi_objective=EVAL_MULTI_OBJECTIVE)

def eval_genome(genome, config, epsilon=0.1, multi_objective=False):
    """
    Simulate a game for a single genome and compute its fitness.

    Args:
        genome: NEAT genome.
        config: NEAT configuration.
        epsilon (float): Exploration rate.
        multi_objective (bool): Use multi-objective fitness.

    Returns:
        float: Fitness value.
    """
    sim_pacman = PACMAN_INIT.copy()
    sim_ghosts = [ [g[0].copy(), g[1].copy()] for g in GHOSTS_INIT ]
    sim_tiles = TILE_LAYOUT.copy()
    score = 0.0
    dots_eaten = 0
    steps_without_progress = 0
    prev_positions = []
    alive = True
    visited = set()
    combo = 0
    max_combo = 0
    min_dist_to_ghost = float('inf')
    max_explore = 0
    prev_dot_dist = None
    memory = []

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for step in range(500):
        # Update memory
        if len(memory) == MEMORY_SIZE:
            memory.pop(0)
        memory.append({
            'pacman': (sim_pacman.x, sim_pacman.y),
            'ghosts': [(g[0].x, g[0].y) for g in sim_ghosts]
        })

        nn_input = get_nn_input(sim_pacman, sim_ghosts, sim_tiles, step, prev_dot_dist, memory)
        output = net.activate(nn_input)
        if random() < epsilon:
            move_idx = np.random.randint(0, 4)
        else:
            move_idx = np.argmax(output)
        dx, dy = POSSIBLE_MOVES[move_idx]
        next_pos = sim_pacman + vector(dx, dy)
        if valid(next_pos):
            sim_pacman.move(vector(dx, dy))
        else:
            score -= 3

        idx = offset(sim_pacman)
        pos_tuple = (sim_pacman.x, sim_pacman.y)

        # Exploration bonus for new positions
        if pos_tuple not in visited:
            visited.add(pos_tuple)
            score += 2.0
            max_explore += 1

        prev_positions.append(pos_tuple)
        if len(prev_positions) > 10:
            prev_positions.pop(0)
        if prev_positions.count(pos_tuple) > 2:
            score -= 10

        # Combo streak bonus for eating consecutive dots
        if idx < len(sim_tiles) and sim_tiles[idx] == 1:
            sim_tiles[idx] = 2
            score += 10
            dots_eaten += 1
            combo += 1
            score += 2 * combo
            if combo > max_combo:
                max_combo = combo
            steps_without_progress = 0
        else:
            steps_without_progress += 1
            combo = 0

        if steps_without_progress > 25:
            score -= 25
            steps_without_progress = 0

        if sim_tiles.count(1) == 0:
            score += MAZE_CLEAR_BONUS
            break

        # Move ghosts
        for ghost in sim_ghosts:
            ghost_pos, ghost_dir = ghost
            if valid(ghost_pos + ghost_dir):
                ghost_pos.move(ghost_dir)
            else:
                options = [vector(5, 0), vector(-5, 0), vector(0, 5), vector(0, -5)]
                plan = choice(options)
                ghost_dir.x = plan.x
                ghost_dir.y = plan.y

        # Ghost collision and proximity penalty
        for ghost_pos, _ in sim_ghosts:
            dist = abs(sim_pacman - ghost_pos)
            if dist < 20:
                score -= 500
                alive = False
                break
            elif dist < 40:
                score -= 1
            score += (dist / 400) * 0.2
            if dist < min_dist_to_ghost:
                min_dist_to_ghost = dist

        # Reward getting closer to the next dot
        curr_dot_dist = find_nearest_dot(sim_pacman, sim_tiles)[2]
        if prev_dot_dist is not None:
            score += (prev_dot_dist - curr_dot_dist) * 10
        prev_dot_dist = curr_dot_dist

        if not alive:
            break

        # Small living bonus for each step
        score += 0.05

    # End-of-episode rewards/penalties
    score += dots_eaten * 5
    score -= (500 - step) * 0.01

    # Optional: multi-objective fitness
    if multi_objective:
        norm_score = max(0, score / 1000.0)
        norm_explore = max_explore / 100.0
        norm_dist = min_dist_to_ghost / 100.0
        return 0.5 * norm_score + 0.25 * norm_explore + 0.25 * norm_dist
    else:
        return score

def eval_population(genomes, config):
    """
    Parallel evaluation of all genomes in a population.

    Args:
        genomes (list): List of (genome_id, genome) tuples.
        config: NEAT configuration.
    """
    from neat.parallel import ParallelEvaluator
    pe = ParallelEvaluator(cpu_count(), eval_genome_picklable)
    pe.evaluate(genomes, config)

def run_neat(config_path="neat_config.txt"):
    """
    Run the NEAT algorithm to train Pacman agents and plot/serialize results.
    Args:
        config_path (str): Path to the NEAT config file.
    Returns:
        float: Fitness of the best genome.
    """
    print("NEAT input size should be:", NN_INPUT_SIZE)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    winner = pop.run(eval_population, NUM_GENERATIONS)
    print('\nBest genome:\n', winner)
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved as best_genome.pkl")

    # Plot fitness history
    if hasattr(stats, "most_fit_genomes"):
        fitness = [g.fitness for g in stats.most_fit_genomes]
        plt.plot(fitness)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Most Fit Genome's Fitness Over Generations")
        plt.savefig("fitness_history.png")
        plt.show()
        print("Fitness history plot saved as fitness_history.png")

    return winner.fitness

def world():
    """
    Render the Pacman world/maze using turtle graphics.
    """
    bgcolor('black')
    path.color('blue')
    for index in range(len(tiles)):
        tile = tiles[index]
        if tile > 0:
            x = (index % 20) * 20 - 200
            y = 180 - (index // 20) * 20
            path.up()
            path.goto(x, y)
            path.down()
            path.begin_fill()
            for count in range(4):
                path.forward(20)
                path.left(90)
            path.end_fill()
            if tile == 1:
                path.up()
                path.goto(x + 10, y + 10)
                path.dot(2, 'white')

def replay_winner(gen_file="best_genome.pkl"):
    """
    Replay a trained genome visually using the turtle graphics environment.

    Args:
        gen_file (str): Path to the pickled genome file.
    """
    global pacman, ghosts, tiles, aim
    state['score'] = 0
    pacman = PACMAN_INIT.copy()
    ghosts = [ [g[0].copy(), g[1].copy()] for g in GHOSTS_INIT ]
    tiles[:] = TILE_LAYOUT.copy()
    aim = vector(5, 0)
    setup(420, 420, 370, 0)
    hideturtle()
    tracer(False)
    writer.goto(160, 160)
    writer.color('white')
    writer.write(state['score'])

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "neat_config.txt"
    )
    with open(gen_file, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    memory = []

    def move():
        """
        Update the game state and the display for each step during replay.
        """
        writer.undo()
        writer.write(state['score'])
        clear()
        # update memory
        if len(memory) == MEMORY_SIZE:
            memory.pop(0)
        memory.append({
            'pacman': (pacman.x, pacman.y),
            'ghosts': [(g[0].x, g[0].y) for g in ghosts]
        })

        nn_input = get_nn_input(pacman, ghosts, tiles, memory=memory)
        output = net.activate(nn_input)
        move_idx = np.argmax(output)
        dx, dy = POSSIBLE_MOVES[move_idx]
        if valid(pacman + vector(dx, dy)):
            pacman.move(vector(dx, dy))
        idx = offset(pacman)
        if tiles[idx] == 1:
            tiles[idx] = 2
            state['score'] += 1
            x = (idx % 20) * 20 - 200
            y = 180 - (idx // 20) * 20
            path.up()
            path.goto(x, y)
            path.down()
            path.begin_fill()
            for count in range(4):
                path.forward(20)
                path.left(90)
            path.end_fill()
        up()
        goto(pacman.x + 10, pacman.y + 10)
        dot(20, 'yellow')
        for point, course in ghosts:
            if valid(point + course):
                point.move(course)
            else:
                options = [
                    vector(5, 0),
                    vector(-5, 0),
                    vector(0, 5),
                    vector(0, -5),
                ]
                plan = choice(options)
                course.x = plan.x
                course.y = plan.y
            up()
            goto(point.x + 10, point.y + 10)
            dot(20, 'red')
        update()
        for point, course in ghosts:
            if abs(pacman - point) < 20:
                print("Game over! Final score:", state['score'])
                return
        if tiles.count(1) == 0:
            print("All dots eaten! Final score:", state['score'])
            return
        ontimer(move, 100)
    world()
    move()
    done()

if __name__ == "__main__":
    print("1. Train and save winner\n2. Replay winner\n3. Replay genome from generation\nType 1, 2 or 3:")
    mode = input().strip()
    if mode == "1":
        run_neat()
    elif mode == "2":
        replay_winner()
    elif mode == "3":
        print("Enter generation number (e.g. 042):")
        gen_num = input().strip()
        replay_winner(f"best_genome_gen{int(gen_num):03d}.pkl")