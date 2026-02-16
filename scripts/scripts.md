# Training Guide

## How Training Works

Each training step does the following:

1. **Sample states** from a buffer of previous best solutions (or start blank)
2. **Build a prompt** containing the problem description + the best code so far + its score
3. **Generate N code attempts** per state (this is `group_size`)
4. **Execute each** in a sandboxed subprocess with CPU pinning and a timeout
5. **Score each** using the task's verifier (accuracy, runtime, bound value, etc.)
6. **Compute entropic advantages** within each group — softmax over rewards, so the best generation gets exponentially more weight
7. **Run a policy gradient update** on the model's LoRA weights
8. **Update the state buffer** with any improved solutions
9. **Repeat** with the updated model

### The Entropic Objective (not GRPO)

Standard RL optimizes average performance. This codebase uses the **entropic objective** from the TTT-Discover paper, which optimizes toward the **maximum**:

```
weight(a) = exp(beta * reward(a)) / mean(exp(beta * reward(a')))
advantage(a) = weight(a) - 1
```

With high `beta`, nearly all gradient signal goes to the best generation. This makes the model risk-seeking — good for discovery tasks where you want breakthroughs, not safe averages.

### Key Concepts

- **`group_size`**: How many code attempts the model generates per problem state. These are compared against each other to compute advantages. Minimum useful value is 4, paper uses 8.
- **`groups_per_batch`**: How many different problem states per training step. Total rollouts per step = `group_size * groups_per_batch`.
- **`eval_timeout`**: Seconds each generated program is allowed to run. Must match what the prompt tells the model.
- **`sampler_type`**: How to pick which states to start from.
  - `greedy` — epsilon-greedy over the best solutions found so far
  - `puct` — UCB-style exploration-exploitation (tracks visit counts)
  - `fixed` — always start from the same initial state
- **`initial_exp_type`**: What the model sees on its very first attempt.
  - `none` — blank slate, no seed code
  - `random` — a random initial solution
  - `best_available` — a known good solution to improve from
- **`adv_estimator`**: How advantages are computed.
  - `entropic` — exponential tilt with fixed beta (recommended)
  - `entropic_adaptive_beta` — auto-tunes beta per group
  - `mean_baseline` — simple `reward - mean(rewards)` (like GRPO)

### Loss Function

The advantage is attached to every token the model generated. Tokens from good generations get positive advantage, tokens from bad generations get negative. The training server computes:

```
loss = -sum(advantage[t] * log P_current(token[t]))
```

This is the `importance_sampling` loss (default). PPO is also available via `loss_fn=ppo`.

---

## Task Reference

### Shared Parameters (all tasks)

```
model_name=openai/gpt-oss-120b    # The LLM being trained
lora_rank=32                       # LoRA adapter rank
learning_rate=4e-5                 # Adam learning rate
adv_estimator=entropic             # Entropic advantage (paper default)
adv_estimator_beta=2.0             # Temperature for entropic weighting
max_tokens=26000                   # Max tokens per LLM generation
eval_every=3                       # Evaluate every N steps
save_every=5                       # Checkpoint every N steps
loss_fn=importance_sampling        # Policy gradient loss
wandb_project=discover-ttt         # W&B project name
```

---

### MLE-Bench (Kaggle Competitions)

**What it does**: Model writes a `run()` function that loads CSV data, trains an ML model, and returns predictions. Graded against Kaggle answer keys.

**Reward**: Raw competition metric (accuracy, RMSE, AUC, etc.). For lower-is-better metrics, the score is negated so higher always = better.

**Available competitions**: Any mlebench-prepared competition. Run `mlebench prepare -c <id>` to add new ones.

```bash
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=mle_bench \
    problem_idx=spaceship-titanic \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=none \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=300 \
    num_cpus_per_task=2 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=mle-bench-spaceship-titanic
```

| Parameter | Value | Why |
|---|---|---|
| `eval_timeout=300` | 5 min | Enough time to train a decent sklearn/xgboost model. Matches the `<<<BUDGET_S>>>` in the prompt. |
| `num_cpus_per_task=2` | 2 cores | Allows parallel model fitting (e.g. xgboost uses threads). |
| `initial_exp_type=none` | No seed code | Model starts from scratch — "This is your first attempt." |
| `group_size=8` | 8 attempts | Gives the entropic estimator enough variance. With 2 you get binary signal only. |
| `groups_per_batch=16` | 128 total rollouts | Good balance of gradient quality vs wall-clock time. |

---

### Circle Packing (n=26 and n=32)

**What it does**: Model writes code to pack N non-overlapping circles into a unit square, maximizing the sum of radii.

**Reward**: Linear — sum of radii. Known bests: n=26 → ~2.636, n=32 → ~2.940.

```bash
# n=26
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=cp \
    problem_idx=26 \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=random \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=305 \
    num_cpus_per_task=1 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=cp26

# n=32: change problem_idx=32, wandb_name=cp32
```

| Parameter | Value | Why |
|---|---|---|
| `eval_timeout=305` | ~5 min | 300s code budget + 5s overhead. |
| `num_cpus_per_task=1` | 1 core | Verification is single-threaded geometry checks. |
| `initial_exp_type=random` | Random initial packing | Gives the model something to improve. Starting blank is much harder. |

---

### Autocorrelation Inequality (AC1 and AC2)

**What it does**: Model writes code to construct a sequence that either minimizes an upper bound (AC1) or maximizes a lower bound (AC2) on an autocorrelation inequality.

**Reward**:
- AC1: `scaled_reciprocal_cf` → `5 / (1e-8 + upper_bound)`. Lower bound is better, so reward is inverted.
- AC2: `linear` → raw lower bound value. Higher is directly better.

**Targets**: AC1 → ~1.5030 (upper bound), AC2 → ~0.97 (lower bound).

```bash
# AC1
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=ac1 \
    problem_idx=ac1 \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=random \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=1100 \
    num_cpus_per_task=2 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=ac1

# AC2: change env=ac2, problem_idx=ac2, wandb_name=ac2
```

| Parameter | Value | Why |
|---|---|---|
| `eval_timeout=1100` | ~18 min | These sequences can be long (1000-100000 elements). Optimization code needs time. |
| `num_cpus_per_task=2` | 2 cores | Numerical optimization benefits from threading. |
| `initial_exp_type=random` | Random sequence | Seeds the model with a random starting point to improve from. Use `best_available` to start from the known SOTA sequence. |

---

### Erdos Minimum Overlap

**What it does**: Model writes code to find a function h:[0,1]→[0,1] that minimizes the C5 overlap constant, related to the Erdos minimum overlap problem.

**Reward**: Linear — `1 / (1e-8 + c5_bound)`. Lower C5 bound = higher reward. Target: ~0.3808.

```bash
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=erdos \
    problem_idx=erdos \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=random \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=1100 \
    num_cpus_per_task=2 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=erdos
```

| Parameter | Value | Why |
|---|---|---|
| `eval_timeout=1100` | ~18 min | Numerical optimization over function spaces is slow. |
| `initial_exp_type=random` | Random h-values | Seeds with a random function perturbation around h=0.5. |

---

### Image Denoising

**What it does**: Model writes a denoising function that removes noise from images. Graded on MSE and Poisson loss.

**Reward**: Linear — based on MSE (lower MSE = better). Starting baseline: MSE=0.2316.

```bash
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=denoising \
    problem_idx=denoising \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=none \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=600 \
    num_cpus_per_task=4 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=denoising
```

| Parameter | Value | Why |
|---|---|---|
| `eval_timeout=600` | 10 min | Image processing is CPU-heavy. Needs more time than math tasks. |
| `num_cpus_per_task=4` | 4 cores | Most demanding CPU task. Image operations parallelize well. |

---

### GPU Mode — TriMul (Triton Kernel)

**What it does**: Model writes an optimized Triton GPU kernel for the Triangle Multiplicative Update operation (used in AlphaFold3). Graded on correctness + runtime in microseconds.

**Reward**: `score_scale / runtime_us` — faster kernel = higher reward. Human best: 1371μs. Runs on H100 via Modal.

```bash
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=trimul \
    problem_idx=trimul \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=none \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=1000 \
    num_cpus_per_task=1 \
    gpu_mode_score_scale=3000.0 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=trimul
```

| Parameter | Value | Why |
|---|---|---|
| `gpu_mode_score_scale=3000.0` | Scaling factor | `reward = 3000 / microseconds`. Keeps reward in a reasonable range (~1-3 for competitive kernels). |
| `num_cpus_per_task=1` | 1 core | Execution is remote (Modal GPU), local CPU just orchestrates. |
| `initial_exp_type=none` | No seed | Model starts from the reference PyTorch code shown in the prompt. |

---

### GPU Mode — MLA Decode (Triton Kernel)

**What it does**: Model writes an optimized Triton kernel for Multi-Head Latent Attention decode (DeepSeek-style MLA). Runs on H200 via Modal.

**Reward**: Same as TriMul — `score_scale / runtime_us`. Human best: 1787μs.

```bash
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=mla_decode_nvidia \
    problem_idx=mla_decode_nvidia \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=best_available \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=1000 \
    num_cpus_per_task=1 \
    gpu_mode_score_scale=3000.0 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=mla-decode
```

| Parameter | Value | Why |
|---|---|---|
| `initial_exp_type=best_available` | Seed with known good kernel | MLA is complex — starting from a working but unoptimized kernel helps. |

---

### ALE Bench (ahc039 and ahc058)

**What it does**: Model writes C++ code for competitive programming (AtCoder Heuristic Contest) problems. Graded by the contest judge.

**Reward**: Linear — raw contest score. Higher is better. Targets: ahc039 → ~5000, ahc058 → ~6.5M.

```bash
# ahc039
PYTHONPATH=. python3 -m tinker_cookbook.recipes.ttt.train \
    env=ale_bench \
    problem_idx=ahc039 \
    model_name=openai/gpt-oss-120b \
    sampler_type=greedy \
    initial_exp_type=none \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=4e-5 \
    num_epochs=50 \
    eval_timeout=1000 \
    dataset_timeout=1000 \
    num_cpus_per_task=1 \
    adv_estimator=entropic \
    adv_estimator_beta=2.0 \
    max_tokens=26000 \
    eval_every=3 \
    save_every=5 \
    wandb_project=discover-ttt \
    wandb_name=ahc039

# ahc058: change problem_idx=ahc058, wandb_name=ahc058
```

---

## Quick Reference Table

| Task | `env` | `problem_idx` | `eval_timeout` | `num_cpus` | `initial_exp` | Reward | What's optimized |
|---|---|---|---|---|---|---|---|
| MLE-Bench | `mle_bench` | `spaceship-titanic` | 300 | 2 | `none` | Kaggle metric | Prediction quality |
| Circle Pack 26 | `cp` | `26` | 305 | 1 | `random` | Sum of radii | Packing density |
| Circle Pack 32 | `cp` | `32` | 305 | 1 | `random` | Sum of radii | Packing density |
| AC1 | `ac1` | `ac1` | 1100 | 2 | `random` | 5/(upper bound) | Upper bound (lower=better) |
| AC2 | `ac2` | `ac2` | 1100 | 2 | `random` | Lower bound | Lower bound (higher=better) |
| Erdos | `erdos` | `erdos` | 1100 | 2 | `random` | 1/C5 bound | C5 overlap (lower=better) |
| Denoising | `denoising` | `denoising` | 600 | 4 | `none` | 1/MSE | Image quality |
| TriMul | `trimul` | `trimul` | 1000 | 1 | `none` | 3000/μs | Kernel speed (H100) |
| MLA Decode | `mla_decode_nvidia` | `mla_decode_nvidia` | 1000 | 1 | `best_available` | 3000/μs | Kernel speed (H200) |
| AHC039 | `ale_bench` | `ahc039` | 1000 | 1 | `none` | Contest score | C++ heuristic |
| AHC058 | `ale_bench` | `ahc058` | 1000 | 1 | `none` | Contest score | C++ heuristic |

## Environment Variables

All scripts require these set before running:

```bash
export TINKER_API_KEY="..."      # Tinker training API
export WANDB_API_KEY="..."       # Weights & Biases logging
export WANDB_ENTITY="..."        # W&B team/user name
```

## Resuming Training

Training auto-resumes from the last checkpoint if `log_path` points to an existing run directory. The default `behavior_if_log_dir_exists=resume` handles this. To start fresh, either delete the log directory or specify a new `log_path`.

## Scaling Guidelines

| Scale | `group_size` | `groups_per_batch` | Total rollouts | Notes |
|---|---|---|---|---|
| Debug | 2 | 4 | 8 | Pipeline test only. Too noisy to learn. |
| Single host | 8 | 16 | 128 | Minimum for real training. |
| Paper default | 8 | 64 | 512 | Recommended. Requires multi-host or long wall-clock. |
| Large scale | 16 | 64 | 1024 | More compute, smoother gradients. |
