# PPO from scratch — DRL Assignment Problem V (Group 174)

Our re-implementation of **PPO-Clip** from Schulman et al., *Proximal Policy Optimization Algorithms* (2017),
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347), plus a small clip-range ablation that mirrors Table 1 of the paper.

## Files
| File | What it does |
|---|---|
| `ppo.py` | Full PPO (Algorithm 1): N parallel actors, GAE, clipped surrogate + value loss + entropy, K epochs of minibatch Adam |
| `run_experiments.py` | Runs the ablation: ε ∈ {none, 0.1, 0.2, 0.3} × 3 seeds, and an "aggressive update" stress test |
| `plot_clip_objective.py` | Re-draws Figure 1 of the paper (L_CLIP vs probability ratio) |
| `plot_results.py` | Learning curves, KL curves and the summary table |
| `render_console.py` | Saves console output as an image for the slides |
| `results/` | Raw CSV logs for every run (step, avg return, approx KL, clip fraction, value loss) |
| `figures/` | All figures used in the PPT |

## Reproduce
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium matplotlib numpy pillow
python ppo.py --env CartPole-v1 --clip 0.2 --seed 1     # single run (~1-2 min on CPU)
python run_experiments.py                               # full ablation (18 runs)
python plot_results.py
python plot_clip_objective.py
```

## Hyper-parameters (and where they come from)
| Param | Ours | Paper |
|---|---|---|
| clip ε | 0.2 (ablated 0.1 / 0.3 / none) | 0.2 best in Table 1 |
| GAE γ, λ | 0.99, 0.95 | 0.99, 0.95 (MuJoCo) |
| epochs K | 10 | 10 (MuJoCo) |
| actors N × horizon T | 4 × 128 | 8 × 128 (Atari) |
| minibatches | 4 (128 samples each) | 4 on Atari (batch 1024, minibatch 32×8 = 256) |
| Adam lr | 2.5e-4, linearly annealed | 2.5e-4 × α (Atari) |
| c1 (value), c2 (entropy) | 0.5, 0.01 | 1, 0.01 (Atari) |
| network | 64-64 tanh, separate actor & critic | 64-64 tanh (MuJoCo) |

Environment: CartPole-v1 (Gymnasium 1.x) — chosen because it trains on a laptop CPU in ~1-2 minutes per run.
The paper used MuJoCo / Roboschool / Atari which needs far more compute.

## References
1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.
2. J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel. *Trust Region Policy Optimization*. ICML 2015.
3. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR 2016.
4. V. Mnih et al. *Asynchronous Methods for Deep Reinforcement Learning* (A3C/A2C). ICML 2016.
5. Z. Wang et al. *Sample Efficient Actor-Critic with Experience Replay* (ACER). ICLR 2017.
6. L. Engstrom et al. *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*. ICLR 2020.
7. S. Huang et al. *The 37 Implementation Details of Proximal Policy Optimization*. ICLR Blog Track, 2022.
8. L. Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022.
9. OpenAI Spinning Up — PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
10. Gymnasium documentation: https://gymnasium.farama.org
