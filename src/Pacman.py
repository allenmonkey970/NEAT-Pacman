import neat
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from random import choice, random
from freegames import floor, vector
from turtle import *
from multiprocessing import cpu_count

# ==== CONSTANTS ====

TRAIN_UNTIL_CLEAR = True      # When True, training stops only when an agent clears the entire maze
NUM_GENERATIONS = 500  # Number of generations to train NEAT
MEMORY_SIZE = 2        # Number of previous steps to store as memory
LOCAL_GRID_SIZE = 3    # Side length of egocentric wall/dot/ghost grid (LOCAL_GRID_SIZE × LOCAL_GRID_SIZE)
POSSIBLE_MOVES = [(5, 0), (-5, 0), (0, 5), (0, -5)]  # Possible movement directions for Pacman
COMBO_BONUS = 8        # Bonus for eating dots in a row
MAZE_CLEAR_BONUS = 500 # Bonus for clearing the maze
EVAL_EPSILON = 0.0      # No random moves during evaluation — let the network play deterministically
EVAL_MULTI_OBJECTIVE = False  # Whether to use multi-objective fitness
NUM_EVAL_RUNS = 10            # Evaluations per genome (5 seeded for stability + 5 random for generalization)
EXPORT_GIF = True            # When True, replay also saves an animated GIF to outputs/replay.gif
STEP_LIMIT = 4000            # Maximum steps per episode
STAGNATION_THRESHOLD = 75    # Steps without eating a dot before penalty kicks in
STAGNATION_PENALTY = 1       # Penalty applied when stagnation threshold is hit
DEATH_PENALTY = 25           # Penalty per ghost collision (agent respawns instead of dying)
MAX_DEATHS = 8               # Maximum deaths before the episode ends

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
path = None   # initialized in replay_winner to avoid tkinter in worker processes
writer = None # initialized in replay_winner to avoid tkinter in worker processes
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

def find_nearest_dot(sim_pacman, dot_positions):
    min_dist = float('inf')
    nearest = None
    px, py = sim_pacman.x, sim_pacman.y
    for tx, ty in dot_positions:
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

def build_dot_positions(sim_tiles):
    return {((idx % 20) * 20 - 200, 180 - (idx // 20) * 20)
            for idx, tile in enumerate(sim_tiles) if tile == 1}

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

# Recompute when MEMORY_SIZE or LOCAL_GRID_SIZE change.
NN_INPUT_SIZE = (2 + 4*4 + 3 + 5 + 1) + MEMORY_SIZE * (2 + 4*2) + LOCAL_GRID_SIZE * LOCAL_GRID_SIZE

def get_local_grid(sim_pacman, sim_tiles, sim_ghosts):
    """Return a flattened LOCAL_GRID_SIZE×LOCAL_GRID_SIZE egocentric grid centred on Pacman.

    Cell values: 0.0=wall/OOB, 0.25=empty path, 0.5=dot, 1.0=ghost.
    """
    half = LOCAL_GRID_SIZE // 2
    ghost_positions = {(g[0].x, g[0].y) for g in sim_ghosts}
    grid = []
    for row in range(-half, half + 1):
        for col in range(-half, half + 1):
            tx = sim_pacman.x + col * 20
            ty = sim_pacman.y + row * 20
            ix = int((tx + 200) / 20)
            iy = int((180 - ty) / 20)
            if 0 <= ix < 20 and 0 <= iy < 20:
                tile_val = sim_tiles[ix + iy * 20]
                cell = 0.0 if tile_val == 0 else (0.5 if tile_val == 1 else 0.25)
            else:
                cell = 0.0
            if (tx, ty) in ghost_positions:
                cell = 1.0
            grid.append(cell)
    return grid

def get_nn_input(sim_pacman, sim_ghosts, sim_tiles, step=0, prev_dot_dist=None, memory=None, step_history=None, dot_positions=None, dots_remaining=0):
    px, py = sim_pacman.x / 200.0, sim_pacman.y / 200.0

    ghosts_rel = []
    for ghost in sim_ghosts:
        gx, gy = ghost[0].x / 200.0, ghost[0].y / 200.0
        ghosts_rel.extend([gx - px, gy - py, ghost[1].x / 5.0, ghost[1].y / 5.0])

    closest_dot = find_nearest_dot(sim_pacman, dot_positions)

    num_dots = dots_remaining / 100.0
    moves = available_moves(sim_pacman, sim_tiles)
    num_open_dirs = len(moves) / 4.0
    is_junc = 1 if len(moves) > 2 else 0
    is_corr = 1 if len(moves) == 2 else 0
    curr_dot_dist = closest_dot[2]
    delta_dot_dist = 0.0 if prev_dot_dist is None else prev_dot_dist - curr_dot_dist

    cur_pos = (sim_pacman.x, sim_pacman.y)
    revisit_count = (sum(1 for p in step_history if p == cur_pos) / 10.0) if step_history else 0.0

    memory_flat = []
    if memory and len(memory) == MEMORY_SIZE:
        for mem in memory:
            m_px, m_py = mem['pacman']
            memory_flat.extend([m_px / 200.0, m_py / 200.0])
            for g_pos in mem['ghosts']:
                memory_flat.extend([g_pos[0] / 200.0, g_pos[1] / 200.0])
    else:
        memory_flat = [0.0] * (MEMORY_SIZE * (2 + 4 * 2))

    local_grid = get_local_grid(sim_pacman, sim_tiles, sim_ghosts)

    nn_input = np.array(
        [px, py] + ghosts_rel + closest_dot
        + [num_dots, num_open_dirs, is_junc, is_corr, delta_dot_dist, revisit_count]
        + memory_flat + local_grid,
        dtype=np.float32
    )
    assert len(nn_input) == NN_INPUT_SIZE, (
        f"NN input size mismatch: got {len(nn_input)}, expected {NN_INPUT_SIZE}. "
        "Update NN_INPUT_SIZE or get_nn_input if you change MEMORY_SIZE, LOCAL_GRID_SIZE, or input features."
    )
    return nn_input


def eval_genome_picklable(genome, config):
    """
    Wrapper for parallel evaluation of genomes. Runs NUM_EVAL_RUNS episodes and
    averages the scores to reduce noise from stochastic ghost movement.

    Args:
        genome: NEAT genome to evaluate.
        config: NEAT configuration.

    Returns:
        float: Average fitness score across NUM_EVAL_RUNS episodes.
    """
    n_seeded = NUM_EVAL_RUNS // 2
    seeded = [eval_genome(genome, config, epsilon=EVAL_EPSILON, multi_objective=EVAL_MULTI_OBJECTIVE, seed=i)
              for i in range(n_seeded)]
    random_runs = [eval_genome(genome, config, epsilon=EVAL_EPSILON, multi_objective=EVAL_MULTI_OBJECTIVE)
                   for _ in range(NUM_EVAL_RUNS - n_seeded)]
    scores = seeded + random_runs
    return sum(scores) / len(scores)

def eval_genome(genome, config, epsilon=0.1, multi_objective=False, verbose=False, seed=None):
    """
    Simulate a game for a single genome and compute its fitness.

    Returns:
        float or dict: Fitness value, or dict with stats if verbose=True.
    """
    import random as rng_module
    if seed is not None:
        rng_module.seed(seed)

    sim_pacman = PACMAN_INIT.copy()
    sim_ghosts = [ [g[0].copy(), g[1].copy()] for g in GHOSTS_INIT ]
    sim_tiles = TILE_LAYOUT.copy()
    dot_positions = build_dot_positions(sim_tiles)
    total_dots = len(dot_positions)
    dots_remaining = total_dots
    score = 0.0
    dots_eaten = 0
    deaths = 0
    combo = 0
    steps_without_progress = 0
    visited_tiles = set()
    min_dist_to_ghost = float('inf')
    prev_dot_dist = None
    memory = deque(maxlen=MEMORY_SIZE)
    step_history = deque(maxlen=10)
    milestones_hit = set()
    invincible_until = 0

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for step in range(STEP_LIMIT):
        step_history.append((sim_pacman.x, sim_pacman.y))
        memory.append({
            'pacman': (sim_pacman.x, sim_pacman.y),
            'ghosts': [(g[0].x, g[0].y) for g in sim_ghosts]
        })

        nn_input = get_nn_input(sim_pacman, sim_ghosts, sim_tiles, step, prev_dot_dist, memory, step_history, dot_positions, dots_remaining)
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
            score -= 0.5

        idx = offset(sim_pacman)
        tile_pos = (floor(sim_pacman.x, 20), floor(sim_pacman.y, 20))

        if tile_pos not in visited_tiles:
            visited_tiles.add(tile_pos)
            score += 3.0

        if idx < len(sim_tiles) and sim_tiles[idx] == 1:
            sim_tiles[idx] = 2
            dot_positions.discard(((idx % 20) * 20 - 200, 180 - (idx // 20) * 20))
            dots_remaining -= 1
            score += 20
            combo += 1
            if combo > 1:
                score += COMBO_BONUS * min(combo, 10)
            dots_eaten += 1
            steps_without_progress = 0

            pct = dots_eaten / total_dots
            if pct >= 0.25 and 25 not in milestones_hit:
                milestones_hit.add(25)
                score += 50
            if pct >= 0.50 and 50 not in milestones_hit:
                milestones_hit.add(50)
                score += 100
            if pct >= 0.75 and 75 not in milestones_hit:
                milestones_hit.add(75)
                score += 200
            if pct >= 0.90 and 90 not in milestones_hit:
                milestones_hit.add(90)
                score += 300
        else:
            combo = 0
            steps_without_progress += 1

        if steps_without_progress > STAGNATION_THRESHOLD:
            score -= STAGNATION_PENALTY
            steps_without_progress = 0

        if dots_remaining == 0:
            score += MAZE_CLEAR_BONUS
            break

        # Move ghosts every 2nd step (Pacman is 2x faster)
        if step % 2 == 0:
            for ghost in sim_ghosts:
                ghost_pos, ghost_dir = ghost
                if valid(ghost_pos + ghost_dir):
                    ghost_pos.move(ghost_dir)
                else:
                    options = [vector(5, 0), vector(-5, 0), vector(0, 5), vector(0, -5)]
                    plan = choice(options)
                    ghost_dir.x = plan.x
                    ghost_dir.y = plan.y

        hit_ghost = False
        if step >= invincible_until:
            for ghost_pos, _ in sim_ghosts:
                dist = abs(sim_pacman - ghost_pos)
                if dist < 20:
                    score -= DEATH_PENALTY
                    deaths += 1
                    combo = 0
                    hit_ghost = True
                    break
                elif dist < 40:
                    score -= 1
                if dist < min_dist_to_ghost:
                    min_dist_to_ghost = dist

        if hit_ghost:
            if deaths >= MAX_DEATHS:
                break
            sim_pacman = PACMAN_INIT.copy()
            sim_ghosts = [ [g[0].copy(), g[1].copy()] for g in GHOSTS_INIT ]
            invincible_until = step + 15
            steps_without_progress = 0
            prev_dot_dist = None
            memory.clear()
            step_history.clear()
            continue

        curr_dot_dist = find_nearest_dot(sim_pacman, dot_positions)[2]
        if prev_dot_dist is not None:
            score += (prev_dot_dist - curr_dot_dist) * 1
        prev_dot_dist = curr_dot_dist

        score += 0.05

    cleared = dots_remaining == 0

    if verbose:
        return {
            'fitness': score,
            'dots_eaten': dots_eaten,
            'total_dots': total_dots,
            'deaths': deaths,
            'steps_used': step + 1,
            'unique_tiles': len(visited_tiles),
            'cleared': cleared,
        }

    if multi_objective:
        norm_score = score / 1000.0
        norm_explore = len(visited_tiles) / 100.0
        norm_dist = min_dist_to_ghost / 100.0 if min_dist_to_ghost != float('inf') else 0.0
        return 0.5 * norm_score + 0.25 * norm_explore + 0.25 * norm_dist
    else:
        return score

class StatsReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species_set, best_genome):
        stats = eval_genome(best_genome, config, epsilon=0.0, verbose=True)
        pct = stats['dots_eaten'] / stats['total_dots'] * 100
        print(f"  >> Best: {stats['dots_eaten']}/{stats['total_dots']} dots ({pct:.0f}%) | "
              f"{stats['deaths']} deaths | {stats['steps_used']} steps | "
              f"{stats['unique_tiles']} unique tiles | "
              f"{'CLEARED!' if stats['cleared'] else 'not cleared'}")


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

def run_neat(config_path="config/neat_config.txt"):
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
    if TRAIN_UNTIL_CLEAR:
        num_dots = TILE_LAYOUT.count(1)
        config.fitness_threshold = num_dots * 20 + MAZE_CLEAR_BONUS + num_dots * 3 + 650
        print(f"TRAIN_UNTIL_CLEAR: fitness threshold set to {config.fitness_threshold:.1f} ({num_dots} dots)")
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(StatsReporter())
    pop.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix='outputs/neat-checkpoint-'))
    winner = pop.run(eval_population, NUM_GENERATIONS)
    print('\nBest genome:\n', winner)
    with open("outputs/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved as outputs/best_genome.pkl")

    # Plot fitness history
    if hasattr(stats, "most_fit_genomes"):
        fitness = [g.fitness for g in stats.most_fit_genomes]
        plt.plot(fitness)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Most Fit Genome's Fitness Over Generations")
        plt.savefig("outputs/fitness_history.png")
        plt.show()
        print("Fitness history plot saved as outputs/fitness_history.png")

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

def replay_winner(gen_file="outputs/best_genome.pkl", export_gif=False, gif_path="outputs/replay.gif"):
    """
    Replay a trained genome visually using the turtle graphics environment.

    Args:
        gen_file (str): Path to the pickled genome file.
        export_gif (bool): If True, saves the replay as an animated GIF.
        gif_path (str): Output path for the GIF file.
    """
    global pacman, ghosts, tiles, aim, path, writer
    path = Turtle(visible=False)
    writer = Turtle(visible=False)
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
        "config/neat_config.txt"
    )
    with open(gen_file, "rb") as f:
        winner = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    memory = deque(maxlen=MEMORY_SIZE)
    step_history = deque(maxlen=10)
    frames = []
    warmup_frames = [0]  # skip first N frames while the window finishes rendering

    def capture_frame():
        if warmup_frames[0] < 8:
            warmup_frames[0] += 1
            return
        if sys.platform not in ('win32', 'darwin'):
            return
        try:
            from PIL import ImageGrab
        except ImportError:
            return
        cv = getcanvas()
        cv.update_idletasks()
        x = cv.winfo_rootx()
        y = cv.winfo_rooty()
        w = cv.winfo_width()
        h = cv.winfo_height()
        frames.append(ImageGrab.grab(bbox=(x, y, x + w, y + h)))

    def save_gif():
        if not frames:
            return
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
            optimize=False,
        )
        print(f"GIF saved to {gif_path}")

    def move():
        """
        Update the game state and the display for each step during replay.
        """
        writer.undo()
        writer.write(state['score'])
        clear()
        step_history.append((pacman.x, pacman.y))
        memory.append({
            'pacman': (pacman.x, pacman.y),
            'ghosts': [(g[0].x, g[0].y) for g in ghosts]
        })

        nn_input = get_nn_input(pacman, ghosts, tiles, memory=memory, step_history=step_history)
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
        if export_gif:
            capture_frame()
        for point, course in ghosts:
            if abs(pacman - point) < 20:
                print("Game over! Final score:", state['score'])
                if export_gif:
                    save_gif()
                return
        if tiles.count(1) == 0:
            print("All dots eaten! Final score:", state['score'])
            if export_gif:
                save_gif()
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
        replay_winner(export_gif=EXPORT_GIF)
    elif mode == "3":
        print("Enter generation number (e.g. 042):")
        gen_num = input().strip()
        replay_winner(f"outputs/best_genome_gen{int(gen_num):03d}.pkl", export_gif=EXPORT_GIF)