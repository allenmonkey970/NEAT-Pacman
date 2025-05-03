# NEAT-Pacman: Neuroevolutionary Pacman Agent

This project implements a Pacman agent evolved using the [NEAT](https://neat-python.readthedocs.io/en/latest/) (NeuroEvolution of Augmenting Topologies) algorithm. The agent learns to play Pacman through simulation, using a feedforward neural network whose architecture and weights are optimized via evolutionary strategies. The project also includes hyperparameter optimization with [Optuna](https://optuna.org/).

## Features

- **Fully playable Pacman** game simulated with a tile-based maze and ghosts
- **NEAT neuroevolution** for reinforcement learning: both topology and weights evolve
- **Parallel training** using Python's `multiprocessing` for faster evolution
- **Replay mode**: visualize the best agent playing Pacman using Turtle graphics
- **Optuna Bayesian optimization**: automatically tune NEAT hyperparameters for best performance
- **Customizable fitness function**: supports both single and multi-objective reward strategies
- **Configurable memory window** for agent state awareness

---

## Project Structure

```
.
├── Pacman.py           # Main game and NEAT training logic
├── neat_config.txt     # NEAT configuration template
├── optuna_opt.py       # Bayesian hyperparameter optimization script (Optuna)
├── best_genome.pkl     # Saved best agent (after training)
├── fitness_history.png # Fitness-over-generations plot (auto-generated)
├── best_bayes_params.txt # Best found Optuna trial hyperparameters (auto-generated)
├── requirements.txt    # Requirements to run Python scripts
└── README.md           # This file
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- [neat-python](https://pypi.org/project/neat-python/)
- numpy
- matplotlib
- optuna
- freegames (for vector/floor operations)
- turtle (usually included with Python)
- [Optuna](https://optuna.org/)

Install dependencies with:

```bash
pip install neat-python numpy matplotlib optuna freegames
```

---

### Basic Usage

#### 1. **Train a Pacman Agent**

Run the main Pacman script and choose "1" to train and evolve an agent:

```bash
python Pacman.py
```

Follow the prompt:

```
1. Train and save winner
2. Replay winner
3. Replay genome from generation
Type 1, 2 or 3:
```

- **1:** Trains the agent using NEAT. The best genome is saved as `best_genome.pkl` and a plot `fitness_history.png` is generated.

#### 2. **Replay the Best Agent**

After training, replay the best agent visually:

```bash
python Pacman.py
```
Select option **2**. This will run the Turtle graphics window and display the agent playing Pacman.

#### 3. **Hyperparameter Optimization (Optional)**

To run Bayesian optimization and find the best NEAT config parameters, execute:

```bash
python optuna_opt.py
```

The script will run multiple trials, updating the config and saving the best found parameters to `best_bayes_params.txt`.

---

## NEAT Configuration

The NEAT config file (`neat_config.txt`) controls all aspects of the neuroevolution, including population size, mutation rates, compatibility thresholds, and network structure. This file can be tuned manually or automatically via Optuna.

Example snippet:

```ini
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1.5
pop_size              = 150
reset_on_extinction   = True

[DefaultGenome]
activation_default      = relu
...
num_inputs              = 76
num_outputs             = 4
...
```

- The input and output sizes are determined by the Pacman state and the number of possible moves.
- For full details, see the comments in `Pacman.py`.

---

## How it Works

- **State Representation**: The neural network receives a vector of normalized game state features, including Pacman's position, ghost positions and directions, nearest dot, and a memory buffer of previous states.
- **Action Selection**: The network outputs a value for each possible move; the highest is chosen (with some random exploration during training).
- **Fitness Evaluation**: Agents are rewarded for eating dots, exploring new areas, surviving longer, and avoiding ghosts. Multi-objective fitness can be enabled.
- **Evolution**: NEAT evolves the population over generations, optimizing both network weights and topology.

---

## Customization

- **Maze layout**: Change the `TILE_LAYOUT` in `Pacman.py`.
- **Agent memory**: Adjust `MEMORY_SIZE` for longer or shorter agent memory.
- **Reward shaping**: Modify the fitness function in `eval_genome`.
- **Hyperparameters**: Tune `neat_config.txt` directly or use Optuna.

---

## References

- [NEAT-Python Documentation](https://neat-python.readthedocs.io/en/latest/)
- [Optuna Documentation](https://optuna.org/)
- [Freegames Library](https://pypi.org/project/freegames/)
- [Turtle Graphics Docs](https://docs.python.org/3/library/turtle.html)

---

## License

[MIT License](https://github.com/allenmonkey970/PacManAi/blob/main/LICENSE)

## Acknowledgments

[Orginal Pacman game](https://github.com/grantjenks/free-python-games)
