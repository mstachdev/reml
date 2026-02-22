"""
Ablation evaluation for REML thesis validation.

Runs the critical ablation experiment with 3 conditions on a held-out eval task:
  1. REML-composed: RL policy selects layers from a trained pool
  2. Random-from-trained-pool: random valid layers (no duplicates) from the same trained pool
  3. Random-from-untrained-pool: random valid layers from a fresh Xavier-init pool

Each condition: compose a 5-layer network, train with Adam for N gradient steps,
record MSE loss curve. Run each condition K times, compute mean +/- std.

Interpretation guide:
  - REML >> random-trained >> untrained  =>  both pool and policy contribute
  - REML ~= random-trained >> untrained  =>  pool training does the work, not the policy
  - All three similar                    =>  neither pool nor policy contributes much

Usage (standalone):
    python src/ablation_eval.py --timesteps 2000 --epochs 1 --n_tasks 7

Usage (from notebook / import):
    from ablation_eval import run_ablation
    results = run_ablation(config_overrides={"timesteps": 2000})
"""

import os
import sys
import copy
import json
import random
import argparse
import datetime
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reml import (
    get_default_config,
    set_seed,
    generate_tasks,
    setup_path,
    LayerPool,
    InnerNetworkTask,
    REML,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compose_network_from_pool(layer_pool, indices, config):
    """Build a torch.nn.ModuleList from pool using given layer indices.

    indices should be a list of pool indices:
      [input_layer_idx, hidden_1_idx, ..., hidden_k_idx, output_layer_idx]
    Total length == config['n_layers_per_network']
    """
    layers = torch.nn.ModuleList([copy.deepcopy(layer_pool.layers[i]) for i in indices])
    return layers


def _pick_random_valid_indices(layer_pool, config):
    """Pick random, non-duplicate, structurally-valid layer indices from a pool.

    Returns a list of indices of length n_layers_per_network:
      - first index  -> an input layer  (in_features == 1)
      - last index   -> an output layer (out_features == 1)
      - middle indices -> hidden layers  (in_features != 1 and out_features != 1)
    """
    n_layers = config["n_layers_per_network"]  # 5 by default
    n_hidden = n_layers - 2  # 3 hidden

    input_indices = [i for i, l in enumerate(layer_pool.layers) if l.in_features == 1]
    output_indices = [i for i, l in enumerate(layer_pool.layers) if l.out_features == 1]
    hidden_indices = [
        i
        for i, l in enumerate(layer_pool.layers)
        if l.in_features != 1 and l.out_features != 1
    ]

    chosen_input = random.choice(input_indices)
    chosen_output = random.choice([o for o in output_indices if o != chosen_input])
    available_hidden = [
        h for h in hidden_indices if h != chosen_input and h != chosen_output
    ]
    chosen_hidden = random.sample(
        available_hidden, min(n_hidden, len(available_hidden))
    )

    return [chosen_input] + chosen_hidden + [chosen_output]


def evaluate_composed_network(layers, eval_task, config, train_steps=100):
    """Train a composed network on eval_task for train_steps, return loss curve.

    Args:
        layers: torch.nn.ModuleList of layers forming the network
        eval_task: InnerNetworkTask
        train_steps: number of Adam gradient steps

    Returns:
        losses: list of float MSE values at each step
    """
    layers = copy.deepcopy(layers)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(layers.parameters(), lr=config["learning_rate"])

    x_all = eval_task.data.view(-1, 1)
    y_all = eval_task.targets.view(-1, 1)

    losses = []
    for step in range(train_steps):
        opt.zero_grad()
        x = x_all.clone()
        for i in range(len(layers) - 1):
            x = torch.nn.functional.relu(layers[i](x))
        y_hat = layers[-1](x)
        loss = loss_fn(y_all, y_hat)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses


def run_reml_composition(reml_model, layer_pool, eval_task, config):
    """Use the trained RL policy to compose a network for the eval task.

    Returns indices chosen by the policy (list of ints).
    """
    import gymnasium

    env = gymnasium.wrappers.NormalizeObservation(
        __import__("reml", fromlist=["InnerNetwork"]).InnerNetwork(
            eval_task, layer_pool, config=config, epoch=0, calibration=False
        )
    )
    obs, _ = env.reset()
    done = False
    chosen_indices = []
    while not done:
        action, _ = reml_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        chosen_indices.append(int(action))
    # The composed layers are now in the unwrapped InnerNetwork
    return env.unwrapped.layers, chosen_indices


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------


def run_ablation(
    config_overrides=None,
    n_trials=10,
    train_steps=100,
    save_dir=None,
):
    """Run the full 3-condition ablation experiment.

    Args:
        config_overrides: dict of config keys to override (optional)
        n_trials: number of repetitions per condition
        train_steps: Adam gradient steps per evaluation
        save_dir: directory to save results/plots (auto-generated if None)

    Returns:
        dict with keys 'reml', 'random_trained', 'random_untrained',
        each mapping to a (n_trials, train_steps) numpy array of losses.
    """
    # ---- config ----
    config = get_default_config()
    config["use_wandb"] = False  # never use wandb for ablation
    if config_overrides:
        for k, v in config_overrides.items():
            config[k] = v

    seed = config["seed"]
    set_seed(seed, config)

    # ---- output dir ----
    if save_dir is None:
        run_dt = datetime.datetime.now().strftime("%m%d_%H%M")
        save_dir = os.path.join("ablation_results", run_dt)
    os.makedirs(save_dir, exist_ok=True)

    # ---- tasks ----
    print("[ablation] Generating tasks...")
    data, targets, info = generate_tasks(config)
    tasks = [
        InnerNetworkTask(data=data[i], targets=targets[i], info=info[i], config=config)
        for i in range(config["n_tasks"])
    ]
    # hold out one task for evaluation
    eval_task = tasks[-1]
    training_tasks = tasks[:-1]
    print(
        f"[ablation] Training on {len(training_tasks)} tasks, evaluating on task {eval_task.info['i']}"
    )

    # ---- Phase 1: Train REML ----
    print("[ablation] Training REML agent...")
    pool = LayerPool(config=config)
    reml = REML(tasks=training_tasks, layer_pool=pool, run="ablation", config=config)
    reml.train()
    trained_pool = copy.deepcopy(pool)
    print("[ablation] REML training complete.")

    # Save trained pool
    torch.save(pool.layers, os.path.join(save_dir, "trained_pool.pth"))
    reml.model.save(os.path.join(save_dir, "reml_model"))

    # ---- Phase 2: Create untrained pool (fresh Xavier init, same structure) ----
    set_seed(seed + 999, config)  # different seed so it's truly fresh
    untrained_pool = LayerPool(config=config)
    set_seed(seed, config)  # reset

    # ---- Phase 3: Run evaluations ----
    results = {
        "reml": [],
        "random_trained": [],
        "random_untrained": [],
    }

    for trial in range(n_trials):
        print(f"[ablation] Trial {trial + 1}/{n_trials}")

        # Condition 1: REML-composed from trained pool
        set_seed(seed + trial, config)
        composed_layers, chosen_idx = run_reml_composition(
            reml.model, trained_pool, eval_task, config
        )
        losses = evaluate_composed_network(
            composed_layers, eval_task, config, train_steps
        )
        results["reml"].append(losses)
        print(f"  REML-composed     final_loss={losses[-1]:.6f}  indices={chosen_idx}")

        # Condition 2: Random composition from trained pool
        set_seed(seed + trial + 1000, config)
        rand_indices = _pick_random_valid_indices(trained_pool, config)
        rand_layers = compose_network_from_pool(trained_pool, rand_indices, config)
        losses = evaluate_composed_network(rand_layers, eval_task, config, train_steps)
        results["random_trained"].append(losses)
        print(
            f"  Random-trained    final_loss={losses[-1]:.6f}  indices={rand_indices}"
        )

        # Condition 3: Random composition from untrained pool
        set_seed(seed + trial + 2000, config)
        rand_indices_u = _pick_random_valid_indices(untrained_pool, config)
        rand_layers_u = compose_network_from_pool(
            untrained_pool, rand_indices_u, config
        )
        losses = evaluate_composed_network(
            rand_layers_u, eval_task, config, train_steps
        )
        results["random_untrained"].append(losses)
        print(
            f"  Random-untrained  final_loss={losses[-1]:.6f}  indices={rand_indices_u}"
        )

    # Convert to numpy
    for key in results:
        results[key] = np.array(results[key])  # (n_trials, train_steps)

    # ---- Save results ----
    results_serializable = {k: v.tolist() for k, v in results.items()}
    with open(os.path.join(save_dir, "ablation_results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    # ---- Save config ----
    config_serializable = {
        k: v for k, v in config.items() if not isinstance(v, defaultdict)
    }
    with open(os.path.join(save_dir, "ablation_config.json"), "w") as f:
        json.dump(config_serializable, f, indent=2)

    # ---- Plot ----
    plot_ablation_results(results, train_steps, save_dir)

    print(f"\n[ablation] Results saved to {save_dir}/")
    print_summary(results)

    return results


def plot_ablation_results(results, train_steps, save_dir):
    """Generate and save comparison plot."""
    steps = np.arange(1, train_steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "reml": "#2196F3",
        "random_trained": "#FF9800",
        "random_untrained": "#F44336",
    }
    labels = {
        "reml": "REML-composed (RL policy)",
        "random_trained": "Random from trained pool",
        "random_untrained": "Random from untrained pool",
    }

    # Left: full learning curves with std bands
    for key in ["reml", "random_trained", "random_untrained"]:
        mean = results[key].mean(axis=0)
        std = results[key].std(axis=0)
        ax1.plot(steps, mean, color=colors[key], label=labels[key], linewidth=2)
        ax1.fill_between(steps, mean - std, mean + std, color=colors[key], alpha=0.2)

    ax1.set_xlabel("Adam Gradient Steps")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Ablation: Learning Curves on Held-Out Task")
    ax1.legend(fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Right: bar chart of final losses
    final_means = [
        results[k][:, -1].mean() for k in ["reml", "random_trained", "random_untrained"]
    ]
    final_stds = [
        results[k][:, -1].std() for k in ["reml", "random_trained", "random_untrained"]
    ]
    bar_labels = [
        "REML\n(RL policy)",
        "Random\n(trained pool)",
        "Random\n(untrained pool)",
    ]
    bar_colors = [colors["reml"], colors["random_trained"], colors["random_untrained"]]

    bars = ax2.bar(
        bar_labels,
        final_means,
        yerr=final_stds,
        color=bar_colors,
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("Final MSE Loss")
    ax2.set_title(f"Final Loss After {train_steps} Steps (mean +/- std)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "ablation_plot.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[ablation] Plot saved to {save_dir}/ablation_plot.png")


def print_summary(results):
    """Print a concise summary of ablation results."""
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for key, label in [
        ("reml", "REML-composed (RL policy)"),
        ("random_trained", "Random from trained pool"),
        ("random_untrained", "Random from untrained pool"),
    ]:
        final = results[key][:, -1]
        print(f"  {label:40s}  {final.mean():.6f} +/- {final.std():.6f}")

    # Interpretation
    reml_final = results["reml"][:, -1].mean()
    rand_trained_final = results["random_trained"][:, -1].mean()
    rand_untrained_final = results["random_untrained"][:, -1].mean()

    print("\nINTERPRETATION:")
    ratio_policy = rand_trained_final / (reml_final + 1e-10)
    ratio_pool = rand_untrained_final / (rand_trained_final + 1e-10)

    if ratio_policy > 1.5 and ratio_pool > 1.5:
        print("  -> Both pool training AND RL policy contribute meaningfully.")
        print("     The thesis claim is SUPPORTED.")
    elif ratio_policy < 1.2 and ratio_pool > 1.5:
        print(
            "  -> Pool training helps, but RL policy adds little beyond random selection."
        )
        print(
            "     The meta-learning benefit comes from the shared trained pool, not composition."
        )
    elif ratio_policy > 1.5 and ratio_pool < 1.2:
        print("  -> RL policy helps, but trained vs untrained pool matters little.")
        print(
            "     Surprising result — the policy learned useful composition independent of pool quality."
        )
    else:
        print("  -> Neither pool training nor RL policy contributes significantly.")
        print(
            "     The Adam optimizer equalizes all conditions during inner-loop training."
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REML Ablation Evaluation")
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=2000,
        help="RL timesteps for REML training",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=1, help="REML training epochs"
    )
    parser.add_argument(
        "--n_tasks", type=int, default=7, help="Number of sinusoidal tasks"
    )
    parser.add_argument(
        "--n_trials", type=int, default=10, help="Repetitions per ablation condition"
    )
    parser.add_argument(
        "--train_steps", type=int, default=100, help="Adam steps per evaluation"
    )
    parser.add_argument("--seed", type=int, default=41, help="Random seed")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_ablation(
        config_overrides={
            "timesteps": args.timesteps,
            "epochs": args.epochs,
            "n_tasks": args.n_tasks,
            "seed": args.seed,
        },
        n_trials=args.n_trials,
        train_steps=args.train_steps,
        save_dir=args.save_dir,
    )
