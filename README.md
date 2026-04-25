# EIR: Energy Inference Representation

**A Differentiable Planning-as-Inference Compiler for Latent World Models**

EIR is a standalone Intermediate Representation (IR) designed for mixed discrete-continuous optimization, specifically tailored for **Active Inference** and **World Model Planning** (e.g., MuZero, Dreamer architectures). 

Instead of relying on standard Reinforcement Learning algorithms (like PPO or DQN) to blindly guess and learn policies, EIR treats future trajectories as an **Energy Field**. By mathematically modeling latent world dynamics and reward functions as physical constraints (spring-mass systems), EIR uses Simultaneous Perturbation Stochastic Approximation (SPSA) to perform gradient descent over soft continuous relaxations of discrete actions, "pulling" the agent toward the optimal trajectory.

---

## 1. The Core Philosophy: Planning as Physics
EIR compiles the future state-action sequence $z_{t+1 \dots T}$ into a differentiable graph using physical energy terms:

*   **`DynamicsConsistencyFactor`**: Penalizes the deviation from the learned world model: $||z_{t+1} - (A[a_t] z_t + b[a_t])||^2$. This models the world as locally linear latent dynamics, conditioned on a discrete action $a_t$.
*   **`RewardFactor`**: Embeds the reward function directly into the energy landscape.
*   **`FieldBackend`**: The execution engine that resolves the graph. It relaxes discrete action spaces into continuous logits and performs gradient descent to find the action sequence that minimizes the total energy of the graph (maximizing reward while strictly obeying the world model physics).

EIR also supports exact compilation to CP-SAT verification backends (via OR-Tools) for absolute logical constraints.

## 2. Installation

```bash
# Standard install (Energy Field Backend only)
pip install -e .

# With Exact Verification support (OR-Tools CP-SAT)
pip install -e ".[verify]"

# For development
pip install -e ".[dev,verify]"
```

## 3. Usage Example

While EIR is built for complex temporal dynamics, it handles basic logical energy fields seamlessly:

```python
from eir.core import EIRGraph, DiscreteVar
from eir.factors.logic import XorFactor
from eir.backends.field import FieldBackend

# 1. Define Variables
a = DiscreteVar("A", n=2)
b = DiscreteVar("B", n=2)

# 2. Define Factors
# Logic factors support fuzzy evaluation for continuous energy fields
# and exact compilation for CP-SAT verification.
xor = XorFactor([a, b], target=1) # A XOR B = 1

# 3. Build Graph
graph = EIRGraph(vars=[a, b], factors=[xor])

# 4. Solve via Gradient Descent
backend = FieldBackend()
result = backend.solve(graph)

print(result.assignment) 
# Outputs: {'A': 1, 'B': 0} or {'A': 0, 'B': 1}
```

## 4. Why EIR?
Standard Deep RL learns a policy $\pi(a|s)$ via trial and error. EIR allows an agent to **deliberate**. When faced with a novel situation, an agent equipped with EIR can instantiate a local energy graph of its predicted future, relax the graph to find the lowest-energy (safest/highest reward) path, and act with mathematical certainty based on its internal world model.

---
**Author:** Pantelis Christou  
**License:** MIT