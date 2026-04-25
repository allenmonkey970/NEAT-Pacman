# NEAT-Pacman: Neuroevolutionary Pacman Agent
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-green.svg)

A Pacman agent evolved using [NEAT](https://neat-python.readthedocs.io/en/latest/) (NeuroEvolution of Augmenting Topologies). The agent learns to navigate the maze and eat dots through neuroevolution — both the network topology and weights are optimized over generations. Includes Bayesian hyperparameter tuning via [Optuna](https://optuna.org/) and animated GIF export of the best agent's replay.

---

## Demo

### Trained Agent Replay
![Pacman agent replay](assets/replay.gif)

### Fitness Over Generations
![Fitness history](assets/fitness_history.png)

---

## Project Structure

```
NEAT-Pacman/
├── src/
│   ├── Pacman.py           # Game simulation, NEAT training, and replay logic
│   ├── optimize.py         # Bayesian hyperparameter optimization (Optuna)
│   ├── config/
│   │   └── neat_config.txt # NEAT algorithm configuration
│   └── outputs/            # Generated artifacts (gitignored)
│       ├── best_genome.pkl        # Saved best agent (after training)
│       ├── fitness_history.png    # Fitness-over-generations plot
│       ├── replay.gif             # Animated replay of the best agent
│       └── best_optuna_params.txt # Best Optuna trial results
├── assets/
│   ├── fitness_history.png # Fitness plot (for README display)
│   └── replay.gif          # Replay GIF (for README display)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- Windows or macOS (GIF export uses `PIL.ImageGrab`, which requires a desktop environment)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

All commands are run from the `src/` directory:

```bash
cd src
```

### Train an Agent

```bash
python Pacman.py
```

Select option `1`. NEAT evolves agents in parallel across all CPU cores. Training stops when:
- An agent clears the entire maze (default, `TRAIN_UNTIL_CLEAR = True`), or
- The generation limit (`NUM_GENERATIONS`) is reached.

The best genome is saved to `outputs/best_genome.pkl` and a fitness plot to `outputs/fitness_history.png`.

### Replay the Best Agent

```bash
python Pacman.py
```

Select option `2`. Opens a Turtle graphics window showing the best-trained agent playing Pacman. If `EXPORT_GIF = True`, the replay is also saved as `outputs/replay.gif`.

### Replay a Specific Generation

```bash
python Pacman.py
```

Select option `3` and enter a generation number (e.g. `042`). Loads `outputs/best_genome_gen042.pkl` and replays that checkpoint.

### Tune Hyperparameters (Optional)

```bash
python optimize.py
```

Runs Bayesian optimization over NEAT hyperparameters using Optuna (10 trials by default). Results are saved to `outputs/best_optuna_params.txt`. Apply the best parameters to `config/neat_config.txt` manually.

---

## Configuration

Edit `src/config/neat_config.txt` to control NEAT behaviour:

| Parameter | Description |
|---|---|
| `pop_size` | Population size per generation |
| `compatibility_threshold` | Species separation threshold |
| `conn_add_prob` / `conn_delete_prob` | Connection mutation rates |
| `node_add_prob` / `node_delete_prob` | Node mutation rates |
| `weight_mutate_rate` | Weight mutation rate |

Key constants in `Pacman.py`:

| Constant | Default | Description |
|---|---|---|
| `NUM_GENERATIONS` | `200` | Max training generations |
| `NUM_EVAL_RUNS` | `3` | Episodes averaged per genome (reduces ghost randomness noise) |
| `TRAIN_UNTIL_CLEAR` | `True` | Stop only when maze is fully cleared |
| `EVAL_MULTI_OBJECTIVE` | `False` | Use raw score fitness (recommended) |
| `MEMORY_SIZE` | `2` | Previous steps stored in agent memory |
| `LOCAL_GRID_SIZE` | `3` | Side length of the egocentric wall/dot/ghost grid (3×3) |
| `EVAL_EPSILON` | `0.01` | Probability of random action during evaluation |
| `EXPORT_GIF` | `True` | Save an animated GIF during replay |
| `COMBO_BONUS` | `8` | Bonus for eating dots consecutively |
| `MAZE_CLEAR_BONUS` | `500` | Bonus awarded for clearing the entire maze |

---

## How It Works

### State Representation (56 inputs)

| Feature group | Size | Description |
|---|---|---|
| Pacman position | 2 | Normalized x, y |
| Ghost info | 16 | Relative position + direction for each of 4 ghosts |
| Nearest dot | 3 | Relative dx, dy and normalized distance |
| Navigation flags | 5 | Dot count, open directions, junction flag, corridor flag, dot-distance delta |
| Revisit pressure | 1 | Fraction of the last 10 steps spent on the current tile |
| Memory buffer | 20 | Previous `MEMORY_SIZE` positions (Pacman + 4 ghosts) |
| Local grid | 9 | Egocentric 3×3 cell grid (wall / empty / dot / ghost) |

### Action Selection

The network outputs a value for each of the 4 movement directions; the highest is chosen. A small epsilon (`EVAL_EPSILON = 0.01`) adds exploration during training.

### Fitness Function

| Event | Reward |
|---|---|
| Dot eaten | +15 |
| New tile explored | +2 |
| Step alive | +0.05 |
| Moving toward nearest dot | up to +5 (scaled by distance delta) |
| Maze cleared | +500 |
| Ghost collision | −150 |
| Near ghost (< 40 units) | −1 |
| Invalid move attempt | −3 |
| Stagnation (50 steps without a dot) | −5 |

### Evolution

NEAT adds/removes nodes and connections over generations, organised into species by genome compatibility distance. The population is evaluated in parallel across all CPU cores using `multiprocessing`.

---

## References

- [NEAT-Python Documentation](https://neat-python.readthedocs.io/en/latest/)
- [Optuna Documentation](https://optuna.org/)
- [Freegames Library](https://pypi.org/project/freegames/)
- [Original Pacman game](https://github.com/grantjenks/free-python-games)

---

## License

[MIT License](LICENSE)
