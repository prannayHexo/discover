"""
Inference-only search loop for Discover benchmarks.

Reuses the existing verify/score/sampler infrastructure but replaces
LLM training with plain Anthropic API calls. No gradient updates — just
the search loop (prompt → generate → execute → score → buffer → repeat).

Usage:
    python search.py --env mle_bench --rounds 15 --samples 4 --model claude-sonnet-4-5-20250929
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# Suppress noisy Ray metrics warnings
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("RAY_ENABLE_METRICS", "0")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
logging.getLogger("ray").setLevel(logging.ERROR)

import ray
import anthropic

from tinker_cookbook.recipes.ttt.sampler import create_initial_state, create_sampler
from tinker_cookbook.recipes.ttt.state import ErdosState, MleBenchState
from tinker_cookbook.utils import ml_log


# ---------------------------------------------------------------------------
# Prompt builders (extracted from env classes to avoid tinker import)
# ---------------------------------------------------------------------------

def build_erdos_prompt(state: ErdosState, budget_s: int, num_cpus: int, problem_idx: str = "erdos") -> str:
    from tasks.erdos_min_overlap.prompt import SYSTEM_PROMPT
    hide_code = "state_only" in problem_idx
    has_code = state.code and state.code.strip() and not hide_code

    # Value context
    if state.parent_values and state.value is not None:
        before_bound = -state.parent_values[0]
        after_bound = -state.value
        value_ctx = f"\nHere are the C₅ bounds before and after running the code above (lower is better): {before_bound:.6f} -> {after_bound:.6f}"
        value_ctx += "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    elif state.value is not None:
        value_ctx = f"\nCurrent C₅ bound (lower is better): {-state.value:.6f}"
        value_ctx += "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    elif state.c5_bound is not None:
        value_ctx = f"\nCurrent C₅ bound (lower is better): {state.c5_bound:.6f}"
        value_ctx += "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    else:
        value_ctx = "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."

    if state.observation and state.observation.strip():
        stdout = state.observation.strip()
        if len(stdout) > 500:
            stdout = "...(truncated)\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    prompt = SYSTEM_PROMPT.replace("<<<BUDGET_S>>>", str(budget_s)).replace("<<<CPUS>>>", str(num_cpus))

    h_values_section = ""
    if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
        h_values_section = f"\nYou may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).\nYou are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.\n"

    if has_code:
        clean_code = state.code.strip()
        for prefix in ["```python", "```"]:
            if clean_code.startswith(prefix):
                clean_code = clean_code[len(prefix):].strip()
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3].strip()
        code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively optimizing constructions.{value_ctx}

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.
Unless you make a meaningful improvement, you will not be rewarded.
"""
    else:
        code_section = f"\n{value_ctx}\n\nWrite code to optimize this construction.\n"

    return f"{prompt}\n{h_values_section}{code_section}"


def build_mle_bench_prompt(state: MleBenchState, budget_s: int, num_cpus: int, competition_id: str) -> str:
    from tasks.mle_bench.prompt import create_prompt

    prompt = create_prompt(competition_id, budget_s=budget_s, num_cpus=num_cpus)

    if state.code and state.code.strip():
        clean_code = state.code.strip()
        for prefix in ["```python", "```"]:
            if clean_code.startswith(prefix):
                clean_code = clean_code[len(prefix):].strip()
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3].strip()

        value_ctx = ""
        if state.raw_score is not None:
            value_ctx = f"\nPrevious score: {state.raw_score}"
            value_ctx += f"\nMedal achieved: {state.medal}"
        if state.parent_values and state.value is not None:
            value_ctx += f"\nReward before/after: {state.parent_values[0]:.4f} -> {state.value:.4f}"

        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "...(truncated)\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

        code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively improving your solution.{value_ctx}

Reason about how you could improve the predictions. Consider:
- Better feature engineering
- Different model architectures
- Hyperparameter tuning
- Ensemble methods
- Better data preprocessing
Unless you make a meaningful improvement, you will not be rewarded.
"""
    else:
        code_section = "\nThis is your first attempt. Write code to produce the best predictions you can.\nStart by examining the data to understand the problem, then build a model.\n"

    return f"{prompt}\n{code_section}"


PROMPT_BUILDERS = {
    "erdos": build_erdos_prompt,
    "mle_bench": build_mle_bench_prompt,
}


# ---------------------------------------------------------------------------
# Verify functions (wrappers around existing Ray-based verification)
# ---------------------------------------------------------------------------

def verify(env_type, generation, step, num_cpus, eval_timeout, log_path, state, problem_idx):
    if env_type == "erdos":
        from tinker_cookbook.recipes.ttt.env_erdos import verify_erdos
        return verify_erdos(generation, step, num_cpus, eval_timeout, log_path, state)
    elif env_type == "mle_bench":
        from tinker_cookbook.recipes.ttt.env_mle_bench import verify_mle_bench
        return verify_mle_bench(generation, step, num_cpus, eval_timeout, log_path, problem_idx)
    else:
        raise ValueError(f"Unsupported env: {env_type}")


# ---------------------------------------------------------------------------
# State creation from verification output
# ---------------------------------------------------------------------------

def create_next_state(env_type, step_idx, parsed_code, outs, parent_state):
    parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []

    if env_type == "erdos":
        performance = outs.get("performance")
        if performance is None:
            return None
        return ErdosState(
            timestep=step_idx,
            code=parsed_code,
            value=performance,
            c5_bound=outs.get("c5_bound"),
            construction=outs.get("construction"),
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )
    elif env_type == "mle_bench":
        performance = outs.get("performance")
        if performance is None:
            return None
        return MleBenchState(
            timestep=step_idx,
            code=parsed_code,
            value=performance,
            raw_score=outs.get("raw_score"),
            is_lower_better=outs.get("is_lower_better", False),
            medal=outs.get("medal", "none"),
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str | None:
    m = re.search(r"```python\s+([\s\S]*?)\s*```", response)
    if m:
        return f"```python\n{m.group(1).strip()}\n```"
    return None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(env_type, outs) -> str:
    correctness = outs.get("correctness", 0.0)
    if correctness <= 0:
        msg = outs.get("msg", "unknown error")
        # Truncate long error messages to just the last line
        lines = msg.strip().split('\n')
        short = lines[-1] if lines else msg
        if len(short) > 120:
            short = short[:120] + "..."
        return f"FAIL: {short}"

    if env_type == "erdos":
        c5 = outs.get("c5_bound")
        return f"C5={c5:.6f}" if c5 else "valid (no bound)"
    elif env_type == "mle_bench":
        medal = outs.get("medal", "none")
        raw = outs.get("raw_score")
        return f"score={raw} | {medal}"
    return f"score={outs.get('score', 0):.4f}"


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference-only search for Discover benchmarks")
    parser.add_argument("--env", required=True, choices=["erdos", "mle_bench"], help="Benchmark to run")
    parser.add_argument("--problem_idx", default=None, help="Problem ID (e.g. 'erdos', 'spaceship-titanic')")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Anthropic model to use")
    parser.add_argument("--rounds", type=int, default=15, help="Number of search rounds")
    parser.add_argument("--samples", type=int, default=10, help="Samples per round")
    parser.add_argument("--sampler", default="greedy", choices=["greedy", "puct"], help="Sampler type")
    parser.add_argument("--timeout", type=int, default=300, help="Code execution timeout (seconds)")
    parser.add_argument("--budget_s", type=int, default=1000, help="Time budget passed to generated code")
    parser.add_argument("--num_cpus", type=int, default=2, help="CPUs per task")
    parser.add_argument("--log_path", default="/tmp/discover-search", help="Log directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=26000, help="Max tokens for LLM response")
    parser.add_argument("--wandb_project", default="discover-ttt", help="W&B project (set to '' to disable)")
    parser.add_argument("--wandb_name", default=None, help="W&B run name")
    args = parser.parse_args()

    # Defaults
    if args.problem_idx is None:
        args.problem_idx = {"erdos": "erdos", "mle_bench": "spaceship-titanic"}[args.env]

    os.makedirs(args.log_path, exist_ok=True)

    # Per-run log files with timestamp
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_log_file = os.path.join(args.log_path, f"run_{run_id}.jsonl")
    best_log_file = os.path.join(args.log_path, f"run_{run_id}_best.jsonl")

    # Init wandb + JSONL logging (same as train.py)
    wandb_project = args.wandb_project if args.wandb_project else None
    wandb_name = args.wandb_name or f"search-{args.env}-{args.problem_idx}-{run_id}"
    ml_logger = ml_log.setup_logging(
        log_dir=args.log_path,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        config=vars(args),
        do_configure_logging_module=False,
    )

    # Init Ray for local CPU verification
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, configure_logging=False)

    # Init Anthropic
    client = anthropic.Anthropic()

    # Create sampler + initial state
    sampler = create_sampler(
        args.sampler,
        args.log_path,
        env_type=args.env,
        initial_exp_type="none",
        batch_size=args.samples,
        group_size=1,
    )

    best_value = None
    best_detail = "none"
    total_calls = 0
    total_valid = 0
    total_errors = 0

    print(f"\n{'='*60}")
    print(f"  Discover Search: {args.env} ({args.problem_idx})")
    print(f"  Model: {args.model} | Rounds: {args.rounds} | Samples/round: {args.samples}")
    print(f"  Sampler: {args.sampler} | Timeout: {args.timeout}s | Budget: {args.budget_s}s")
    print(f"  All samples: {run_log_file}")
    print(f"  Best states: {best_log_file}")
    if ml_logger.get_logger_url():
        print(f"  W&B: {ml_logger.get_logger_url()}")
    print(f"{'='*60}\n")

    for round_idx in range(1, args.rounds + 1):
        states = sampler.sample_states(args.samples)
        round_best = None
        round_best_detail = ""

        state_val = states[0].value if states[0].value is not None else "none"
        print(f"Round {round_idx}/{args.rounds} | Sampling {args.samples} candidates from state (value={state_val})")

        for sample_idx, state in enumerate(states, 1):
            total_calls += 1

            # 1. Build prompt
            build_prompt = PROMPT_BUILDERS[args.env]
            prompt = build_prompt(state, args.budget_s, args.num_cpus, args.problem_idx)

            # 2. Call Anthropic (stream to handle long responses)
            t0 = time.time()
            content = ""
            with client.messages.stream(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    content += text
            llm_time = time.time() - t0

            # 3. Parse code
            parsed_code = extract_code(content)
            if not parsed_code:
                total_errors += 1
                # Log failed sample
                sample_log = {
                    "round": round_idx, "sample": sample_idx,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "env": args.env,
                    "problem_idx": args.problem_idx,
                    "llm_time_s": round(llm_time, 1),
                    "verify_time_s": None,
                    "status": "format_error",
                    "result": None,
                    "performance": None,
                    "is_new_best": False,
                    "best_value_so_far": best_value,
                    "parent_state_value": state.value,
                    "parent_state_timestep": state.timestep,
                    "parent_values_history": state.parent_values,
                    "prompt": prompt,
                    "full_response": content,
                    "parsed_code": None,
                    "verify_output": None,
                }
                with open(run_log_file, "a") as f:
                    f.write(json.dumps(sample_log) + "\n")
                print(f"  [{sample_idx}] \u2717 Format error (no code block) [{llm_time:.1f}s LLM]")
                continue

            # 4. Verify via Ray pipeline
            t1 = time.time()
            outs = verify(
                args.env, parsed_code, round_idx * 100 + sample_idx,
                args.num_cpus, args.timeout, args.log_path, state, args.problem_idx,
            )
            verify_time = time.time() - t1

            correctness = outs.get("correctness", 0.0)
            detail = format_result(args.env, outs)

            if correctness > 0:
                total_valid += 1
                performance = outs.get("performance", 0.0)

                # Check if new best
                is_new_best = False
                if best_value is None or performance > best_value:
                    best_value = performance
                    best_detail = detail
                    is_new_best = True
                if round_best is None or performance > round_best:
                    round_best = performance
                    round_best_detail = detail

                # 5. Create next state + update sampler
                next_state = create_next_state(args.env, round_idx, parsed_code, outs, state)
                if next_state is not None:
                    sampler.update_states([next_state], [state], save=False)

                # Log valid sample
                sample_log = {
                    "round": round_idx, "sample": sample_idx,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "env": args.env,
                    "problem_idx": args.problem_idx,
                    "llm_time_s": round(llm_time, 1),
                    "verify_time_s": round(verify_time, 1),
                    "status": "valid",
                    "result": detail,
                    "performance": performance,
                    "is_new_best": is_new_best,
                    "best_value_so_far": best_value,
                    "parent_state_value": state.value,
                    "parent_state_timestep": state.timestep,
                    "parent_values_history": state.parent_values,
                    "prompt": prompt,
                    "full_response": content,
                    "parsed_code": parsed_code,
                    "verify_output": outs,
                }
                with open(run_log_file, "a") as f:
                    f.write(json.dumps(sample_log, default=str) + "\n")

                best_tag = " | NEW BEST" if is_new_best else ""
                print(f"  [{sample_idx}] \u2713 {detail} | reward={performance:.4f}{best_tag} [{llm_time:.1f}s LLM, {verify_time:.1f}s verify]")
            else:
                total_errors += 1
                # Log failed sample
                sample_log = {
                    "round": round_idx, "sample": sample_idx,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "env": args.env,
                    "problem_idx": args.problem_idx,
                    "llm_time_s": round(llm_time, 1),
                    "verify_time_s": round(verify_time, 1),
                    "status": "fail",
                    "result": detail,
                    "performance": None,
                    "is_new_best": False,
                    "best_value_so_far": best_value,
                    "parent_state_value": state.value,
                    "parent_state_timestep": state.timestep,
                    "parent_values_history": state.parent_values,
                    "prompt": prompt,
                    "full_response": content,
                    "parsed_code": parsed_code,
                    "verify_output": outs,
                }
                with open(run_log_file, "a") as f:
                    f.write(json.dumps(sample_log, default=str) + "\n")
                print(f"  [{sample_idx}] \u2717 {detail} [{llm_time:.1f}s LLM, {verify_time:.1f}s verify]")

        # Trim sampler buffers and write best states to JSONL (not experience JSONs)
        if hasattr(sampler, '_lock'):
            with sampler._lock:
                if hasattr(sampler, '_top_states'):
                    sampler._top_states.sort(key=lambda s: s.value if s.value else 0, reverse=True)
                    sampler._top_states = sampler._top_states[:sampler.batch_size]
                    top_states = sampler._top_states
                elif hasattr(sampler, '_states'):
                    top_states = sampler._states
                else:
                    top_states = []
                sampler._current_step = round_idx
        else:
            top_states = []
            sampler._current_step = round_idx

        best_entry = {
            "round": round_idx,
            "best_states": [s.to_dict() for s in top_states],
        }
        with open(best_log_file, "a") as f:
            f.write(json.dumps(best_entry, default=str) + "\n")

        # Log round metrics to wandb
        round_metrics = {
            "search/best_value": best_value if best_value is not None else 0.0,
            "search/round_best": round_best if round_best is not None else 0.0,
            "search/total_valid": total_valid,
            "search/total_errors": total_errors,
            "search/total_calls": total_calls,
        }
        ml_logger.log_metrics(round_metrics, step=round_idx)

        print(f"  Best this round: {round_best_detail or 'none'}")
        print()

    # Summary
    ml_logger.close()
    print(f"{'='*60}")
    print(f"  DONE")
    print(f"  Best: {best_detail}")
    print(f"  Total API calls: {total_calls} | Valid: {total_valid} | Errors: {total_errors}")
    print(f"  All samples: {run_log_file}")
    print(f"  Best states: {best_log_file}")
    if ml_logger.get_logger_url():
        print(f"  W&B: {ml_logger.get_logger_url()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
