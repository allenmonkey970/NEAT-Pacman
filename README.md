# NEAT-Pacman: Neuroevolutionary Pacman Agent
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-green.svg)

A Pacman agent evolved using [NEAT](https://neat-python.readthedocs.io/en/latest/) (NeuroEvolution of Augmenting Topologies). The agent learns to navigate the maze and eat dots through neuroevolution — both the network topology and weights are optimized over generations. Includes Bayesian hyperparameter tuning via [Optuna](https://optuna.org/).

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
│       ├── best_genome.pkl      # Saved best agent (after training)
│       ├── fitness_history.png  # Fitness-over-generations plot
│       └── best_optuna_params.txt # Best Optuna trial results
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.7+

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

Select option `1`. NEAT will evolve agents across generations. Training stops when:
- An agent clears the entire maze (default, `TRAIN_UNTIL_CLEAR = True`), or
- The generation limit (`NUM_GENERATIONS`) is reached.

The best genome is saved to `outputs/best_genome.pkl` and a fitness plot to `outputs/fitness_history.png`.

### Replay the Best Agent

```bash
python Pacman.py
```

Select option `2`. Opens a Turtle graphics window showing the best-trained agent playing Pacman.

### Tune Hyperparameters (Optional)

```bash
python optimize.py
```

Runs Bayesian optimization over NEAT hyperparameters using Optuna. Results are saved to `outputs/best_optuna_params.txt`. Apply the best parameters to `config/neat_config.txt` manually.

---

## Configuration

Edit `src/config/neat_config.txt` to control NEAT behavior:

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
| `MEMORY_SIZE` | `5` | Previous steps stored in agent memory |

---

## How It Works

- **State Representation**: 76 normalized inputs — Pacman position, ghost positions/directions, nearest dot direction, available moves, junction/corridor flags, distance delta to nearest dot, and a rolling memory buffer.
- **Action Selection**: Network outputs a value for each of 4 moves; the highest is chosen. A small epsilon (`EVAL_EPSILON = 0.01`) adds exploration during training.
- **Fitness Function**: Agents earn `+15` per dot eaten, `+2` for exploring new tiles, `+0.05` per step alive, `+500` for clearing the maze. Dying costs `-150`. Stagnation (50 steps without eating) costs `-5`.
- **Evolution**: NEAT adds/removes nodes and connections over generations, speciated by genome compatibility distance.

---

## References

- [NEAT-Python Documentation](https://neat-python.readthedocs.io/en/latest/)
- [Optuna Documentation](https://optuna.org/)
- [Freegames Library](https://pypi.org/project/freegames/)
- [Original Pacman game](https://github.com/grantjenks/free-python-games)

---

## License

[MIT License](LICENSE)
