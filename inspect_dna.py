"""Load a saved DNA, grow it, and print/snapshot the resulting brain."""
import sys
import warnings
import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from NDP import MLP, generate_initial_graph
from train_backend import grow_network, snapshot_graph_png


def setup_config(config, seed=42):
    config["seed"] = seed
    config["profile"] = False

    extra_hidden = 0 if config["extra_nodes"] == -1 else config["extra_nodes"]
    if "gate" in config["environment"]:
        config["observation_dim"] = 2
        config["action_dim"] = 2
        config["min_network_size"] = 4
        config["initial_network_size"] = config["observation_dim"] + config["action_dim"] + extra_hidden
    else:
        from utils import dimensions_env
        obs_dim, act_dim, _ = dimensions_env(config["environment"])
        config["observation_dim"] = obs_dim
        config["action_dim"] = act_dim
        config["initial_network_size"] = obs_dim + act_dim + extra_hidden
        config["min_network_size"] = obs_dim + act_dim

    has_io_roles = "Network" not in config["environment"]
    config["has_io_roles"] = has_io_roles
    config["nb_params_coevolve_initial_embeddings"] = ((2 if has_io_roles else 1) * config["node_embedding_size"]) if config["coevolve_initial_embeddings"] else 0
    config["input_size_growth_model"] = config["node_embedding_size"] * 2 if config["node_pairs_based_growth"] else config["node_embedding_size"]

    mlp_g = MLP(config["input_size_growth_model"], 1, config["mlp_growth_hidden_layers_dims"], torch.nn.Tanh(), config["growth_model_last_layer_activated"], config["growth_model_bias"])
    config["nb_params_growth_model"] = torch.nn.utils.parameters_to_vector(mlp_g.parameters()).detach().numpy().shape[0]

    if config["NN_transform_node_embedding_during_growth"]:
        mlp_t = MLP(config["node_embedding_size"], config["node_embedding_size"], config["mlp_embedding_transform_hidden_layers_dims"], torch.nn.Tanh(), config["transform_model_last_layer_activated"], config["transform_model_bias"])
        config["nb_params_feature_transformation"] = torch.nn.utils.parameters_to_vector(mlp_t.parameters()).detach().numpy().shape[0]
    else:
        config["nb_params_feature_transformation"] = 0

    if config["node_based_growth"] and not config["binary_connectivity"]:
        mlp_w = MLP(2 * config["node_embedding_size"], 1, config["mlp_weight_values_hidden_layers_dims"], torch.nn.Tanh(), config["mlp_weight_values_last_layer_activated"], config["mlp_weight_values_bias"])
        config["nb_params_mlp_weight_values"] = torch.nn.utils.parameters_to_vector(mlp_w.parameters()).detach().numpy().shape[0]
    else:
        config["nb_params_mlp_weight_values"] = 0

    if config["shared_intial_graph_bool"]:
        config["shared_intial_graph"] = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], seed)

    return config


if __name__ == "__main__":
    conf_path = sys.argv[1] if len(sys.argv) > 1 else "run_experiment.yaml"
    dna_path  = sys.argv[2] if len(sys.argv) > 2 else None
    out_png   = sys.argv[3] if len(sys.argv) > 3 else "graph_inspect.png"

    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = setup_config(config)

    if dna_path is None:
        import pathlib
        dna_path = str(pathlib.Path(conf_path).parent / "solution_best.npy")

    dna = np.load(dna_path)
    print(f"DNA: {len(dna)} parameters")

    W, network_state = grow_network(dna, config)
    n_nodes = W.shape[0]
    n_edges = int(np.count_nonzero(W))
    print(f"Brain: {n_nodes} nodes, {n_edges} edges")

    config["_path"] = "."
    snapshot_graph_png(W, network_state, config, out_png)
