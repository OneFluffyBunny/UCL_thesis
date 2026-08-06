"""Grow a saved DNA and render a grid image of every intermediate growth stage.

Re-implements the same loop as train_backend.grow_network() (propagate ->
add nodes -> update weights, per growth cycle), but keeps a snapshot after
each sub-phase instead of only the final (W, network_state). Only supports
node_based_growth (not node_pairs_based_growth).
"""
import argparse
import copy
import warnings

import numpy as np
import torch
import yaml
import networkx as nx
from matplotlib import pyplot

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from NDP import make_numpy_mlp, bfs_diameter, W_to_nx, propagate_features, predict_new_nodes, update_weights, add_new_nodes, generate_initial_graph
from train_backend import build_initial_network_state
from utils import nx_layout, io_self_edge_mask_dims


def grow_with_stages(evolved_parameters: np.ndarray, config: dict):
    """Returns a list of (label, W, network_state) snapshots, from the seed
    graph through every sub-phase of every growth cycle."""
    if config["shared_intial_graph_bool"]:
        W = config["shared_intial_graph"].copy()
    else:
        W = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], seed=None, io_dims=io_self_edge_mask_dims(config))

    network_state = copy.deepcopy(build_initial_network_state(config, evolved_parameters))

    n1 = config["nb_params_coevolve_initial_embeddings"]
    n2 = n1 + config["nb_params_growth_model"]
    n3 = n2 + config["nb_params_feature_transformation"]
    n4 = n3 + config["nb_params_mlp_weight_values"]

    mlp_growth_model = make_numpy_mlp(
        evolved_parameters[n1:n2],
        input_dim=config["input_size_growth_model"],
        hidden_dims=config["mlp_growth_hidden_layers_dims"],
        output_dim=1,
        has_bias=config["growth_model_bias"],
        last_activated=config["growth_model_last_layer_activated"],
    )
    mlp_feature_transformation = None
    if config["NN_transform_node_embedding_during_growth"]:
        mlp_feature_transformation = make_numpy_mlp(
            evolved_parameters[n2:n3],
            input_dim=config["node_embedding_size"],
            hidden_dims=config["mlp_embedding_transform_hidden_layers_dims"],
            output_dim=config["node_embedding_size"],
            has_bias=config["transform_model_bias"],
            last_activated=config["transform_model_last_layer_activated"],
        )
    mlp_weight_values = None
    if config["node_based_growth"] and not config["binary_connectivity"]:
        mlp_weight_values = make_numpy_mlp(
            evolved_parameters[n3:n4],
            input_dim=2 * config["node_embedding_size"],
            hidden_dims=config["mlp_weight_values_hidden_layers_dims"],
            output_dim=1,
            has_bias=config["mlp_weight_values_bias"],
            last_activated=config["mlp_weight_values_last_layer_activated"],
        )

    try:
        diameter = bfs_diameter(W)
    except Exception:
        diameter = int(np.sqrt(W.shape[0]))

    stages = [("Seed graph", W.copy(), network_state.copy())]

    for cycle in range(config["number_of_growth_cycles"]):
        network_state = propagate_features(
            network_state=network_state,
            W=W,
            network_thinking_time=diameter + config["network_thinking_time_extra_growth"],
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=None,
            feature_transformation_model=mlp_feature_transformation,
        )
        stages.append((f"Cycle {cycle + 1}: propagate", W.copy(), network_state.copy()))

        new_nodes_predictions = predict_new_nodes(
            mlp_growth_model, network_state, config["node_embedding_size"], growth_threshold=config.get("growth_threshold", 0.0)
        )
        prev_n = W.shape[0]
        W, network_state = add_new_nodes(
            W=W,
            network_state=network_state,
            node_embeddings_concatenated_dict=None,
            new_nodes_predictions=new_nodes_predictions,
            node_based_growth=config["node_based_growth"],
            node_pairs_based_growth=config["node_pairs_based_growth"],
            binary_connectivity=config["binary_connectivity"],
            undirected=config["undirected"],
        )
        stages.append((f"Cycle {cycle + 1}: add nodes", W.copy(), network_state.copy()))

        if W.shape[0] != prev_n:
            try:
                diameter = bfs_diameter(W)
            except Exception:
                diameter = int(np.sqrt(W.shape[0]))

        if W.shape[0] > 1 and not config["binary_connectivity"] and mlp_weight_values is not None:
            W = update_weights(W=W, network_state=network_state, model=mlp_weight_values, undirected=config["undirected"])
            stages.append((f"Cycle {cycle + 1}: update weights", W.copy(), network_state.copy()))

        if config["prunning_phase"]:
            W[np.abs(W) <= config["prunning_threshold"]] = 0
            stages.append((f"Cycle {cycle + 1}: prune", W.copy(), network_state.copy()))

    return stages


def _draw_stage(ax, W, config, title):
    G = W_to_nx(W, config["undirected"])
    n = len(G)

    if "Network" in config["environment"]:
        obs_dim, act_dim = None, None
    elif "gate" in config["environment"]:
        obs_dim, act_dim = 2, 2
    else:
        obs_dim, act_dim = config.get("observation_dim", 0), config.get("action_dim", 0)

    if obs_dim is not None:
        color_map = ["indianred" if node < obs_dim else ("slategray" if obs_dim <= node < obs_dim + act_dim else "white") for node in G.nodes()]
    else:
        color_map = ["white"] * n

    pos = nx_layout(G, config["layout"])
    edges = list(G.edges())
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    edge_widths = [max(0.4, abs(w) * 2) for w in edge_weights]
    edge_colors = ["steelblue" if w >= 0 else "crimson" for w in edge_weights]

    nx.draw_networkx(
        G, ax=ax, pos=pos, with_labels=False, node_size=50, node_color=color_map,
        edgecolors="black", linewidths=0.4, arrows=config["arrows"], width=edge_widths, edge_color=edge_colors,
    )
    ax.set_title(f"{title}\n{n} nodes, {len(edges)} edges", fontsize=9)
    ax.set_axis_off()


def render_growth_stages_png(stages: list, config: dict, out_path: str, run_id: str = ""):
    """Draw the (label, W, network_state) stage list as a grid PNG and save it."""
    n_stages = len(stages)
    ncols = 4
    nrows = int(np.ceil(n_stages / ncols))
    fig, axes = pyplot.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (title, W, network_state) in zip(axes, stages):
        _draw_stage(ax, W, config, title)
    for ax in axes[n_stages:]:
        ax.set_axis_off()

    fig.suptitle(f"{config['environment']} — growth stages" + (f", run {run_id}" if run_id else ""), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_path, dpi=150)
    pyplot.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a grid of intermediate brain-growth stages for a saved DNA")
    parser.add_argument("--run-id", type=str, required=True, help="saved_models/<run-id> to load config.yml + solution_best.npy from")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: saved_models/<run-id>/growth_stages.png)")
    args = parser.parse_args()

    run_dir = f"saved_models/{args.run_id}"
    with open(f"{run_dir}/config.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    solution_best = np.load(f"{run_dir}/solution_best.npy")

    if config["node_pairs_based_growth"]:
        raise NotImplementedError("growth_stages.py only supports node_based_growth")

    stages = grow_with_stages(solution_best, config)
    print(f"Captured {len(stages)} growth stages")

    out_path = args.out or f"{run_dir}/growth_stages.png"
    render_growth_stages_png(stages, config, out_path, run_id=args.run_id)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
