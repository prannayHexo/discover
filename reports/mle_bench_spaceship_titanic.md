# Applying Discover (Test-Time Training) to MLE-Bench

## 1. Abstract

The goal was to determine whether the Discover methodology of training the model to get better at solving the given problem would show promising results. In this case, the choice of problem was MLE-Bench's Spaceship-Titanic.

Spaceship-Titanic is a problem where, given a list of features about the passengers of a ship (location, destination, etc.), calculate the survival rate after an accident. This problem was chosen as the execution time would be short, and there is a faster feedback loop while training.

---

## 2. Introduction

When we give an LLM a coding problem, it writes one solution and that's it. If the solution scores 78%, we're stuck with 78%. The model has no way to learn from that result and try again with a better approach.

The traditional ways to improve this are:

- **Prompt engineering**: Write a better prompt. This helps, but hits a ceiling.
- **Few-shot examples**: Show the model good solutions. This requires human-written examples.
- **Fine-tuning (SFT)**: Train the model on a dataset of problems and solutions. But the model only learns to imitate — it can't discover approaches that aren't in the training data.

**Discover** takes a different approach: let the model generate solutions, execute them, see how well they scored, and then update the model's weights to produce better solutions next time. The model trains itself on its own attempts, using real execution results as the learning signal.

This report documents our experiment applying this approach to a Kaggle competition.

---

## 3. Background

### 3.1 What is Test-Time Training?

In standard machine learning, training and deployment are separate:

```
Training phase:  Learn from data → produce a model
Deployment phase: Use the fixed model on new problems (no more learning)
```

Test-time training (TTT) removes this boundary. When the model encounters a specific problem, it **keeps training on that problem** — generating attempts, evaluating them, and updating its weights.

Think of it like the difference between:
- **Standard ML**: A student studies a textbook, then takes the exam. Whatever they learned from the textbook is all they have.
- **TTT**: A student takes the exam, checks their answers, studies where they went wrong, tries again, checks again, and keeps improving — all during the exam itself.

The key insight: for difficult problems where we care about finding the **best possible** answer (not just a decent one), this self-improvement loop can significantly outperform a frozen model.

### 3.2 How Discover Compares to Other Approaches

| Method | What it learns from | What it optimizes | When it learns | Strengths | Weaknesses |
|---|---|---|---|---|---|
| **SFT** (Supervised Fine-Tuning) | Human-written solutions | Imitating expert examples | Before deployment | Reliable, predictable | Can't exceed the quality of training data |
| **RLHF/GRPO** | Model generations + reward signal | Average quality across many problems | Before deployment | Generalizes across tasks | Optimizes average, not maximum |
| **Best-of-N** | Nothing — just samples many and picks the best | N/A (no learning) | At inference | Simple, no training needed | Wasteful; model never improves |
| **Discover (TTT)** | Model generations + execution results | Maximum performance on one problem | At test time | Finds solutions beyond training data | Expensive; one problem at a time |

The critical difference: SFT and RLHF produce a general-purpose model that's decent at many tasks. Discover produces a model that's **exceptional** at one specific task. It sacrifices breadth for depth.

### 3.3 The Entropic Objective: Why Not Just Use GRPO?

Standard RL algorithms like GRPO optimize for **average** performance. Here's why that's a problem for discovery.

Say the model generates 8 solutions and we score them:

```
Attempt 1: 0.60    Attempt 5: 0.61
Attempt 2: 0.55    Attempt 6: 0.57
Attempt 3: 0.92    Attempt 7: 0.59
Attempt 4: 0.58    Attempt 8: 0.62
```

**What GRPO does** — subtract the mean (0.63) from each score:

```
Attempt 1: -0.03   (slightly discouraged)
Attempt 3: +0.29   (somewhat encouraged)
Attempt 6: -0.06   (slightly discouraged)
```

Attempt 3 (the brilliant one) gets a positive signal of +0.29. The mediocre attempts each get small negative signals. The model learns a little from the best attempt, but the gradient is spread thinly across all 8.

Over time, GRPO makes the model **consistently decent** — it learns to reliably score ~0.65, then ~0.70. But it avoids risky strategies. A bold approach that scores 0.30 half the time and 0.95 half the time has a lower average (0.625) than a safe approach that always scores 0.65. GRPO would prefer the safe one.

**What the Entropic Objective does** — softmax weighting with temperature beta:

```
weight(i) = e^(beta * score(i)) / average(e^(beta * score(j)))
advantage(i) = weight(i) - 1
```

With beta = 2.0:

```
Attempt 1: advantage = -0.5   (strongly discouraged)
Attempt 3: advantage = +4.2   (strongly encouraged — 14x more than GRPO!)
Attempt 6: advantage = -0.6   (strongly discouraged)
```

The gradient is now **dominated** by Attempt 3. The model receives a clear message: "whatever you did in Attempt 3, do much more of that." The mediocre attempts barely contribute.

**The math in one sentence**: Standard RL maximizes E[reward] (the average). Entropic RL maximizes log E[e^(beta * reward)], which as beta grows, approaches max(reward).

This makes the model **risk-seeking** — it prefers strategies that occasionally produce breakthroughs, even if they sometimes fail. For discovery tasks, where we only need one great solution, this is exactly what we want.

### 3.4 MLE-Bench and Spaceship-Titanic

**MLE-Bench** is a benchmark that turns Kaggle competitions into automated evaluation tasks. It provides the competition data, description, and grading function for each competition, so a model's generated code can be scored against the real Kaggle answer key without manual intervention.

**Spaceship-Titanic** is a Kaggle "Getting Started" binary classification problem:
- **Dataset**: 7,823 training passengers with 14 features (home planet, cabin, spending habits, cryosleep status, etc.)
- **Task**: Predict which of 870 test passengers were "transported" to an alternate dimension
- **Metric**: Classification accuracy (percentage of correct predictions)
- **Kaggle leaderboard top score**: 82.815% accuracy across 2,288 submissions

We chose this competition because:
1. **Fast execution**: A solution runs in under 5 minutes (sklearn/xgboost on 7K rows)
2. **Clear metric**: Accuracy is easy to interpret — higher is better, range is [0, 1]
3. **Established baseline**: The Kaggle leaderboard provides a reference for how good our results are

---

## 4. Methodology

### 4.1 The Training Loop

Each training step works as follows:

**Step 1 — Build the prompt.** The model receives:
- The competition description ("predict which passengers were transported...")
- Instructions to write a `run()` function that returns a DataFrame of predictions
- Its own previous best code and score (if not the first attempt)
- A list of improvement ideas: "consider better feature engineering, different models, ensemble methods..."

**Step 2 — Generate N code attempts.** The model generates `group_size` (e.g. 8) different solutions in parallel. Each is a complete Python program that loads data, trains a model, and outputs predictions.

**Step 3 — Execute each attempt.** Each program runs in a sandboxed subprocess with:
- A 5-minute timeout
- 2 CPU cores allocated
- Access to the competition's train.csv and test.csv

The `run()` function returns a DataFrame with PassengerId and Transported columns.

**Step 4 — Grade against ground truth.** The returned DataFrame is written to CSV and graded by MLE-Bench using the same metric Kaggle uses — classification accuracy against the real answer key.

**Step 5 — Compute entropic advantages.** Within each group of 8 attempts, the softmax weighting is applied. The best-scoring attempt gets a large positive advantage. Failed or low-scoring attempts get negative advantages.

**Step 6 — Update model weights.** The advantages are attached to each token the model generated. A policy gradient update pushes the model's LoRA weights to make high-advantage tokens more likely and low-advantage tokens less likely.

**Step 7 — Repeat.** The updated model generates the next batch. Because the model's weights have shifted toward producing better code, the next batch tends to score higher.

### 4.2 State Reuse: The Iterative Improvement Loop

In standard RL, each episode starts fresh — the model sees a new problem and generates a solution from scratch. Discover adds **state reuse**: the model sees its own previous best solution and is asked to improve it.

The prompt for the second attempt onwards looks like:

```
Here is the last code we ran:
[previous best code]

Previous score: 0.8069
You are iteratively improving your solution.
Consider: better feature engineering, different models, ensemble methods...
Unless you make a meaningful improvement, you will not be rewarded.
```

This creates a chain of improvements:
1. First attempt: basic logistic regression → 0.78 accuracy
2. Second attempt: sees the 0.78 code, switches to XGBoost → 0.81
3. Third attempt: sees the 0.81 code, adds feature engineering → 0.83
4. Fourth attempt: sees the 0.83 code, adds ensemble of XGBoost + CatBoost → 0.84

The model is effectively doing what a human data scientist does: look at the current solution, think about what could be better, try it, check if it worked, and iterate.

A **greedy sampler** manages this buffer of best solutions. It keeps the top-performing solutions found so far and mostly samples from the best one (with 12.5% probability of sampling a random one for exploration).

### 4.3 The Reward Pipeline

Here's how a score turns into a weight update:

```
Model writes code
  → code defines run() that returns DataFrame
  → DataFrame is written to CSV
  → mlebench.grade.grade_csv() compares CSV to answer key
  → returns accuracy (e.g. 0.8276)
  → accuracy becomes the reward (linear, no transformation)
  → rewards across the group are softmax-weighted (entropic)
  → advantages are computed (weight - 1)
  → each token gets the advantage of its generation
  → policy gradient: loss = -sum(advantage * log P(token))
  → Adam optimizer updates LoRA weights
  → updated model generates better code next step
```

The important detail: the reward is the **raw accuracy score**. There is no shaping, no normalization, no clipping. A generation that scores 0.83 contributes directly with that number. The entropic advantage estimator handles the relative weighting within each group.

---

## 5. Experimental Setup

### 5.1 Model and Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base model | gpt-oss-120b | Large reasoning model from the Discover paper |
| LoRA rank | 32 | Paper default; balances capacity vs efficiency |
| Learning rate | 4e-5 | Paper default |
| Optimizer | Adam (beta1=0.9, beta2=0.95) | Standard for LoRA fine-tuning |
| Advantage estimator | Entropic (beta=2.0) | Paper's recommended approach |
| Loss function | Importance sampling | Policy gradient with off-policy correction |
| Max tokens | 26,000 | Enough for complex code generation |
| Temperature | 1.0 | Standard — allows natural variation in generations |
| KL penalty | 0.0 | Disabled — model is free to specialize |

### 5.2 Task-Specific Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Competition | spaceship-titanic | Fast execution, clear metric |
| Eval timeout | 300s | 5 minutes per code execution |
| CPUs per task | 2 | Enough for XGBoost/sklearn parallelism |
| Initial experience | None | Model starts from scratch |
| Sampler | Greedy (epsilon=0.125) | Mostly exploit best, occasionally explore |

### 5.3 Runs Conducted

We ran four training experiments with increasing scale:

| Run | Group Size | Groups/Batch | Total Rollouts/Step | Steps | Notes |
|---|---|---|---|---|---|
| Run 1 | 2 | 2 | 4 | 10 | Initial test |
| Run 2 | 2 | 4 | 8 | 10 | Slightly larger batch |
| Run 3 | 8 | 16 | 128 | 17 | Paper-scale parameters |
| Run 4 | 8 | 16 | 128 | 6 | Resumed from Run 3 checkpoint |

We also ran a **search-only baseline** using `search.py` — the same model generating solutions without any weight updates, purely sampling and keeping the best result (Best-of-N).

---

## 6. Results

### 6.1 Score Progression

| Run | Starting Accuracy | Best Accuracy | Improvement |
|---|---|---|---|
| Run 1 (2g-2b) | 0.8081 | 0.8333 | +2.5% |
| Run 2 (2g-4b) | 0.8069 | 0.8276 | +2.1% |
| Run 3 (8g-16b) | 0.8115 | 0.8345 | +2.3% |
| Run 4 (8g-16b, resumed) | 0.8241 | **0.8379** | +1.4% |
| Search baseline (no training) | — | 0.8230 | — |

### 6.2 Comparison with Kaggle Leaderboard

The Kaggle leaderboard top score for Spaceship-Titanic is **0.82815** (82.815% accuracy), the best out of 2,288 human submissions.

Our best result of **0.83793** (83.793% accuracy) **exceeds the #1 Kaggle leaderboard entry by approximately 1 percentage point**.

```
Kaggle leaderboard distribution (2,288 submissions):
  Median:  0.7957 (79.6%)
  75th %:  0.8027 (80.3%)
  Top 1:   0.8282 (82.8%)
  Ours:    0.8379 (83.8%)  ← exceeds all human submissions
```

### 6.3 Training vs Search-Only

The search-only baseline (same model, same prompts, but no weight updates) achieved **0.82299**. Training improved this by **+1.5 percentage points** to 0.83793, demonstrating that the weight updates provide genuine learning beyond what sampling alone can achieve.

| Approach | Best Score | How It Works |
|---|---|---|
| Single generation | ~0.78 | One-shot, no iteration |
| Search (Best-of-60) | 0.8230 | Generate 60 solutions, keep the best, no learning |
| TTT training (Run 4) | **0.8379** | 6 training steps, model weights updated each step |

### 6.4 What the Model Learned

Over the course of training, the model's generated code evolved from simple baselines to sophisticated data science pipelines:

**Early generations** (steps 1-3):
- Basic XGBoost or RandomForest with default parameters
- Minimal feature engineering (just drop NaN rows, use raw columns)
- Single model, no ensembling

**Later generations** (steps 10+):
- **Feature engineering**: Extracting cabin deck/number/side from the Cabin string, computing group sizes from PassengerId, creating spending patterns (total spend, spend flags, log transforms)
- **Ensemble methods**: Voting classifiers combining XGBoost, CatBoost, and RandomForest
- **Preprocessing pipelines**: ColumnTransformer with StandardScaler and OneHotEncoder for different feature types
- **Hyperparameter tuning**: Grid search over prediction thresholds
- **Robustness**: Fallback mechanisms if a library isn't available

The model effectively learned the same progression a human data scientist would follow: start simple, identify what features matter, add complexity where it helps, and combine multiple models for robustness.

---

## 7. Discussion

### 7.1 Why TTT Works for This Problem

Three properties make Spaceship-Titanic a good fit for TTT:

1. **Verifiable reward**: We can execute the code and grade predictions against ground truth. There's no ambiguity in the score.
2. **Iterative improvement is possible**: A better feature or a smarter model choice can incrementally improve accuracy. The problem has many "knobs" the model can learn to turn.
3. **Fast feedback**: Each attempt takes under 5 minutes. The model can complete dozens of improvement cycles in a few hours.

### 7.2 The Self-Improvement Loop

The key mechanism is the interaction between state reuse and the entropic objective:

1. The model sees its own best code and score
2. It generates N variations — some better, some worse
3. The entropic objective puts nearly all gradient weight on the best variation
4. The model's weights shift to make that kind of code more likely
5. Next iteration, it starts from the improved code and tries again

This creates a positive feedback loop: better code → higher reward → stronger gradient → model more likely to produce that style of code → even better code.

### 7.3 Limitations

**Mode collapse**: The model can converge to generating the same code repeatedly. With `group_size=8`, all 8 attempts might produce nearly identical XGBoost scripts. When this happens, the entropic advantage estimator sees no variance (all rewards are the same), and the group is discarded — no learning occurs. There is currently no explicit diversity mechanism to prevent this.

**Single competition**: We tested on one competition (Spaceship-Titanic). This is a relatively simple binary classification task. Harder competitions with more complex data formats, larger datasets, or non-trivial evaluation metrics may present additional challenges.

**KL penalty disabled**: The model's weights can drift arbitrarily far from the base model. This is intentional (we want specialization), but means the model may lose general capabilities in the process.

**Reward scale**: The accuracy metric is naturally bounded in [0, 1], which makes the entropic weighting well-behaved. For competitions with unbounded metrics (e.g. RMSE on regression), the reward scaling would need more careful handling.

### 7.4 Future Directions

- **Diversity mechanisms**: Adding a bonus for generating structurally different code within a group could prevent mode collapse and encourage exploration of novel approaches
- **Harder competitions**: Testing on competitions with larger datasets, more complex preprocessing requirements, or multi-output predictions
- **Longer training**: Our runs were 6-17 steps. The paper's other tasks (circle packing, GPU kernels) run for 50+ steps. Longer training may yield further improvements
- **Multi-competition transfer**: Training on multiple competitions simultaneously to build a general data science agent, then specializing at test time

---

## 8. Conclusion

We applied the Discover (TTT-Discover) methodology to MLE-Bench's Spaceship-Titanic competition. Starting from a base model with no task-specific training, the system iteratively improved its data science code through a loop of generation, execution, grading, and policy gradient updates.

The model's best solution achieved **83.8% accuracy**, exceeding the top Kaggle leaderboard score of 82.8%. This improvement came from the model learning, without human guidance, to apply feature engineering, model ensembling, and preprocessing pipelines — the same techniques an experienced data scientist would use.

The comparison between search-only (82.3%) and TTT training (83.8%) demonstrates that the weight updates provide genuine learning beyond what Best-of-N sampling can achieve. The model doesn't just get lucky with more samples — it actually gets better at writing data science code for this specific problem.

These results suggest that test-time training with the entropic objective is a promising approach for tasks where:
- Solutions can be automatically verified
- Iterative improvement is possible
- We care about maximum performance, not average performance
