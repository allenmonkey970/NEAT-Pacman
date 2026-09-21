import neat
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from random import choice, random
from freegames import floor
import numba as nb

class Vec:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def copy(self):
        return Vec(self.x, self.y)
    def move(self, other):
        self.x += other.x
        self.y += other.y
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vec(self.x + other, self.y + other)
        return Vec(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)
    def __abs__(self):
        return abs(self.x) + abs(self.y)
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))
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
STEP_LIMIT = 6000            # Maximum steps per episode
STAGNATION_THRESHOLD = 50    # Steps without eating a dot before penalty kicks in
STAGNATION_PENALTY = 3       # Penalty applied when stagnation threshold is hit
DEATH_PENALTY = 50           # Penalty per ghost collision (agent respawns instead of dying)
MAX_DEATHS = 3               # Maximum deaths before the episode ends
SURVIVAL_BONUS = 100         # Bonus per unused death at end of episode (max_deaths - deaths)

# Pacman and game layout constants
PACMAN_INIT = Vec(-40, -80)
GHOSTS_INIT = [
    [Vec(-180, 160), Vec(5, 0)],
    [Vec(-180, -160), Vec(0, 5)],
    [Vec(100, 160), Vec(0, -5)],
    [Vec(100, -160), Vec(-5, 0)],
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

# ==== JIT-compiled simulation ==== #

TILE_LAYOUT_NP = np.array(TILE_LAYOUT, dtype=np.int32)
PACMAN_INIT_XY = np.array([PACMAN_INIT.x, PACMAN_INIT.y], dtype=np.int64)
GHOSTS_INIT_NP = np.array(
    [[g[0].x, g[0].y, g[1].x, g[1].y] for g in GHOSTS_INIT], dtype=np.int64
)
POSSIBLE_MOVES_NP = np.array(POSSIBLE_MOVES, dtype=np.int64)

@nb.njit(cache=True)
def _floor_jit(val, size):
    return ((val + 200) // size) * size - 200

@nb.njit(cache=True)
def _offset_jit(px, py):
    fx = _floor_jit(px, 20)
    fy = _floor_jit(py, 20)
    x = (fx + 200) // 20
    y = (180 - fy) // 20
    return int(x + y * 20)

@nb.njit(cache=True)
def _valid_jit(px, py, tiles):
    idx = _offset_jit(px, py)
    if idx < 0 or idx >= 400 or tiles[idx] == 0:
        return False
    idx2 = _offset_jit(px + 19, py + 19)
    if idx2 < 0 or idx2 >= 400 or tiles[idx2] == 0:
        return False
    return px % 20 == 0 or py % 20 == 0

@nb.njit(cache=True)
def _find_nearest_dot_jit(px, py, tiles):
    min_dist = 999999
    nx, ny = 0, 0
    found = False
    for i in range(400):
        if tiles[i] == 1:
            tx = (i % 20) * 20 - 200
            ty = 180 - (i // 20) * 20
            d = abs(px - tx) + abs(py - ty)
            if d < min_dist:
                min_dist = d
                nx, ny = tx, ty
                found = True
    if not found:
        return 0.0, 0.0, 0.0
    return (nx - px) / 200.0, (ny - py) / 200.0, min_dist / 400.0

@nb.njit(cache=True)
def _nn_forward_jit(inputs, n_total, node_bias, node_response, node_act_type,
                    link_src, link_weight, link_start, link_count, output_indices):
    n_in = len(inputs)
    values = np.zeros(n_total, dtype=np.float64)
    for i in range(n_in):
        values[i] = inputs[i]
    for i in range(len(node_bias)):
        s = 0.0
        st = link_start[i]
        cnt = link_count[i]
        for j in range(cnt):
            s += values[link_src[st + j]] * link_weight[st + j]
        val = node_bias[i] + node_response[i] * s
        act = node_act_type[i]
        if act == 0:
            val = max(0.0, val)
        elif act == 1:
            val = np.tanh(val)
        elif act == 2:
            val = 1.0 / (1.0 + np.exp(-max(-500.0, min(500.0, val))))
        values[n_in + i] = val
    out = np.empty(len(output_indices), dtype=np.float64)
    for i in range(len(output_indices)):
        out[i] = values[output_indices[i]]
    return out

@nb.njit(cache=True)
def _simulate_game_jit(
    tiles_init, pac_ix, pac_iy, ghosts_init,
    n_total, node_bias, node_response, node_act_type,
    link_src, link_weight, link_start, link_count, output_indices,
    seed, step_limit, max_deaths, death_penalty,
    stag_thresh, stag_pen, combo_bonus, maze_clear_bonus, epsilon,
    memory_size, local_grid_size, survival_bonus
):
    if seed >= 0:
        np.random.seed(seed)

    tiles = tiles_init.copy()
    px, py = pac_ix, pac_iy
    ghosts = ghosts_init.copy()
    moves = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]], dtype=np.int64)

    total_dots = 0
    for i in range(400):
        if tiles[i] == 1:
            total_dots += 1
    dots_rem = total_dots

    score = 0.0
    dots_eaten = 0
    deaths = 0
    combo = 0
    stag_count = 0
    inv_until = 0
    prev_dd = -1.0

    visited = np.zeros(400, dtype=nb.boolean)
    n_vis = 0
    mem = np.zeros((memory_size, 10), dtype=np.float64)
    mem_cnt = 0
    mem_i = 0
    sh = np.zeros((10, 2), dtype=np.int64)
    sh_cnt = 0
    sh_i = 0
    m25 = False
    m50 = False
    m75 = False
    m90 = False

    nn_in = np.zeros(56, dtype=np.float64)
    half_g = local_grid_size // 2
    last_step = 0

    for step in range(step_limit):
        last_step = step

        sh[sh_i, 0] = px
        sh[sh_i, 1] = py
        sh_i = (sh_i + 1) % 10
        if sh_cnt < 10:
            sh_cnt += 1

        mem[mem_i, 0] = px / 200.0
        mem[mem_i, 1] = py / 200.0
        for gi in range(4):
            mem[mem_i, 2 + gi * 2] = ghosts[gi, 0] / 200.0
            mem[mem_i, 3 + gi * 2] = ghosts[gi, 1] / 200.0
        mem_i = (mem_i + 1) % memory_size
        if mem_cnt < memory_size:
            mem_cnt += 1

        pxn = px / 200.0
        pyn = py / 200.0
        nn_in[0] = pxn
        nn_in[1] = pyn
        idx = 2
        for gi in range(4):
            nn_in[idx] = ghosts[gi, 0] / 200.0 - pxn
            nn_in[idx+1] = ghosts[gi, 1] / 200.0 - pyn
            nn_in[idx+2] = ghosts[gi, 2] / 5.0
            nn_in[idx+3] = ghosts[gi, 3] / 5.0
            idx += 4
        ddx, ddy, ddd = _find_nearest_dot_jit(px, py, tiles)
        nn_in[idx] = ddx
        nn_in[idx+1] = ddy
        nn_in[idx+2] = ddd
        idx += 3
        nn_in[idx] = dots_rem / 100.0
        idx += 1
        nm = 0
        for mi in range(4):
            if _valid_jit(px + moves[mi, 0], py + moves[mi, 1], tiles):
                nm += 1
        nn_in[idx] = nm / 4.0
        nn_in[idx+1] = 1.0 if nm > 2 else 0.0
        nn_in[idx+2] = 1.0 if nm == 2 else 0.0
        idx += 3
        if prev_dd < 0:
            nn_in[idx] = 0.0
        else:
            nn_in[idx] = prev_dd - ddd
        idx += 1
        rv = 0
        for hi in range(sh_cnt):
            h_idx = (sh_i - 1 - hi) % 10
            if sh[h_idx, 0] == px and sh[h_idx, 1] == py:
                rv += 1
        nn_in[idx] = rv / 10.0
        idx += 1
        if mem_cnt == memory_size:
            for mi in range(memory_size):
                ri = (mem_i - memory_size + mi) % memory_size
                for fi in range(10):
                    nn_in[idx] = mem[ri, fi]
                    idx += 1
        else:
            for _ in range(memory_size * 10):
                nn_in[idx] = 0.0
                idx += 1
        for row in range(-half_g, half_g + 1):
            for col in range(-half_g, half_g + 1):
                tx = px + col * 20
                ty = py + row * 20
                ix_t = (tx + 200) // 20
                iy_t = (180 - ty) // 20
                cell = 0.0
                if 0 <= ix_t < 20 and 0 <= iy_t < 20:
                    tv = tiles[int(ix_t + iy_t * 20)]
                    if tv == 1:
                        cell = 0.5
                    elif tv != 0:
                        cell = 0.25
                is_gh = False
                for gi in range(4):
                    if ghosts[gi, 0] == tx and ghosts[gi, 1] == ty:
                        is_gh = True
                        break
                if is_gh:
                    cell = 1.0
                nn_in[idx] = cell
                idx += 1

        output = _nn_forward_jit(nn_in, n_total, node_bias, node_response,
                                 node_act_type, link_src, link_weight,
                                 link_start, link_count, output_indices)
        if epsilon > 0.0 and np.random.random() < epsilon:
            mi = np.random.randint(0, 4)
        else:
            mi = 0
            bv = output[0]
            for oi in range(1, 4):
                if output[oi] > bv:
                    bv = output[oi]
                    mi = oi
        dx, dy = moves[mi, 0], moves[mi, 1]

        if _valid_jit(px + dx, py + dy, tiles):
            px += dx
            py += dy
        else:
            score -= 0.5

        t_idx = _offset_jit(px, py)
        if 0 <= t_idx < 400 and not visited[t_idx]:
            visited[t_idx] = True
            n_vis += 1
            score += 3.0

        if 0 <= t_idx < 400 and tiles[t_idx] == 1:
            tiles[t_idx] = 2
            dots_rem -= 1
            score += 20.0
            combo += 1
            if combo > 1:
                score += combo_bonus * min(combo, 10)
            dots_eaten += 1
            stag_count = 0
            pct = dots_eaten / total_dots
            if pct >= 0.25 and not m25:
                m25 = True
                score += 50.0
            if pct >= 0.50 and not m50:
                m50 = True
                score += 100.0
            if pct >= 0.75 and not m75:
                m75 = True
                score += 200.0
            if pct >= 0.90 and not m90:
                m90 = True
                score += 300.0
        else:
            combo = 0
            stag_count += 1

        if stag_count > stag_thresh:
            score -= stag_pen
            stag_count = 0

        if dots_rem == 0:
            score += maze_clear_bonus
            break

        if step % 2 == 0:
            for gi in range(4):
                gx, gy = ghosts[gi, 0], ghosts[gi, 1]
                gdx, gdy = ghosts[gi, 2], ghosts[gi, 3]
                if _valid_jit(gx + gdx, gy + gdy, tiles):
                    ghosts[gi, 0] = gx + gdx
                    ghosts[gi, 1] = gy + gdy
                else:
                    r = np.random.randint(0, 4)
                    ghosts[gi, 2] = moves[r, 0]
                    ghosts[gi, 3] = moves[r, 1]

        hit = False
        if step >= inv_until:
            for gi in range(4):
                d = abs(px - ghosts[gi, 0]) + abs(py - ghosts[gi, 1])
                if d < 20:
                    score -= death_penalty
                    deaths += 1
                    combo = 0
                    hit = True
                    break
                elif d < 40:
                    score -= 1.0

        if hit:
            if deaths >= max_deaths:
                break
            px, py = pac_ix, pac_iy
            for gi in range(4):
                ghosts[gi, 0] = ghosts_init[gi, 0]
                ghosts[gi, 1] = ghosts_init[gi, 1]
                ghosts[gi, 2] = ghosts_init[gi, 2]
                ghosts[gi, 3] = ghosts_init[gi, 3]
            inv_until = step + 15
            stag_count = 0
            prev_dd = -1.0
            mem_cnt = 0
            mem_i = 0
            sh_cnt = 0
            sh_i = 0
            continue

        cdd = _find_nearest_dot_jit(px, py, tiles)[2]
        if prev_dd >= 0:
            score += (prev_dd - cdd) * 1.0
        prev_dd = cdd
        score += 0.05

    cleared = 1 if dots_rem == 0 else 0
    score += survival_bonus * (max_deaths - deaths)
    return score, dots_eaten, deaths, last_step + 1, n_vis, cleared

def extract_nn_arrays(net):
    n_inputs = len(net.input_nodes)
    n_nodes = len(net.node_evals)
    n_total = n_inputs + n_nodes
    nmap = {}
    for i, nid in enumerate(net.input_nodes):
        nmap[nid] = i
    for i, (nid, *_) in enumerate(net.node_evals):
        nmap[nid] = n_inputs + i
    node_bias = np.empty(n_nodes, dtype=np.float64)
    node_response = np.empty(n_nodes, dtype=np.float64)
    node_act_type = np.empty(n_nodes, dtype=np.int32)
    act_map = {'relu_activation': 0, 'tanh_activation': 1, 'sigmoid_activation': 2}
    all_src = []
    all_wt = []
    l_start = np.empty(n_nodes, dtype=np.int32)
    l_count = np.empty(n_nodes, dtype=np.int32)
    off = 0
    for i, (nid, act_f, agg_f, bias, resp, links) in enumerate(net.node_evals):
        node_bias[i] = bias
        node_response[i] = resp
        node_act_type[i] = act_map.get(act_f.__name__, 0)
        l_start[i] = off
        l_count[i] = len(links)
        for sid, w in links:
            all_src.append(nmap[sid])
            all_wt.append(w)
        off += len(links)
    link_src = np.array(all_src, dtype=np.int32) if all_src else np.empty(0, dtype=np.int32)
    link_weight = np.array(all_wt, dtype=np.float64) if all_wt else np.empty(0, dtype=np.float64)
    output_idx = np.array([nmap[nid] for nid in net.output_nodes], dtype=np.int32)
    return (n_total, node_bias, node_response, node_act_type,
            link_src, link_weight, l_start, l_count, output_idx)

# ==== Game and Feature Setup ==== #

state = {'score': 0, 'random_state': 1082}
path = None   # initialized in replay_winner to avoid tkinter in worker processes
writer = None # initialized in replay_winner to avoid tkinter in worker processes
aim = Vec(5, 0)
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
        pos = Vec(sim_pacman.x + dx, sim_pacman.y + dy)
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
    if dot_positions is None:
        dot_positions = build_dot_positions(sim_tiles)
        dots_remaining = len(dot_positions)
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
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nn_data = extract_nn_arrays(net)
    result = _simulate_game_jit(
        TILE_LAYOUT_NP, PACMAN_INIT_XY[0], PACMAN_INIT_XY[1], GHOSTS_INIT_NP,
        *nn_data,
        seed if seed is not None else -1,
        STEP_LIMIT, MAX_DEATHS, DEATH_PENALTY,
        STAGNATION_THRESHOLD, STAGNATION_PENALTY,
        COMBO_BONUS, MAZE_CLEAR_BONUS, epsilon,
        MEMORY_SIZE, LOCAL_GRID_SIZE, SURVIVAL_BONUS
    )
    sc, de, dth, su, nv, clr = result
    if verbose:
        return {
            'fitness': sc,
            'dots_eaten': int(de),
            'total_dots': int(np.sum(TILE_LAYOUT_NP == 1)),
            'deaths': int(dth),
            'steps_used': int(su),
            'unique_tiles': int(nv),
            'cleared': bool(clr),
        }
    if multi_objective:
        return 0.5 * (sc / 1000.0) + 0.25 * (nv / 100.0)
    return sc

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
    aim = Vec(5, 0)
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
        if valid(pacman + Vec(dx, dy)):
            pacman.move(Vec(dx, dy))
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
                    Vec(5, 0),
                    Vec(-5, 0),
                    Vec(0, 5),
                    Vec(0, -5),
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