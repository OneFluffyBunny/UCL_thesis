import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import networkx as nx
from scipy import stats, sparse
from numpy.random import default_rng
from matplotlib import pyplot
import copy
import time
import powerlaw
from scipy.stats import kstest
from typing import List, Tuple, Dict, Union, Optional, Callable

from NDP import (MLP, NumpyMLP, torch_mlp_to_numpy, make_numpy_mlp,
                  generate_initial_graph, bfs_diameter, W_to_nx,
                  propagate_features, query_pairs_of_node_embeddings,
                  predict_new_nodes, update_weights, add_new_nodes)
from optimizers import CMAES
from utils import dimensions_env, animate_graph, seed_python_numpy_torch_cuda, environment_max_reward, nx_layout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_grad_enabled(False)


import contextlib

@contextlib.contextmanager
def _timed(label, timings, active):
    if active:
        t = time.perf_counter()
        yield
        timings[label] = timings.get(label, 0.0) + time.perf_counter() - t
    else:
        yield


def build_initial_network_state(config: dict, evolved_parameters: np.ndarray) -> np.ndarray:
    """Construct the initial node-embedding matrix for the seed graph.

    For I/O tasks the seed graph has the input nodes at indices 0..obs_dim-1, the
    output nodes at obs_dim..obs_dim+action_dim-1, and any extra hidden seed nodes
    after that. Input and output nodes are given distinct *role* embeddings so the
    (weight-shared) growth program can tell sensory from motor neurons and grow
    asymmetric structure around them. Hidden seed nodes keep a neutral (zero)
    embedding.

    Graph-property tasks ("Network" in the environment name) have no I/O roles and
    retain the original single-embedding / ones / random behaviour.

    Returns:
        np.ndarray of shape (initial_network_size, node_embedding_size).
    """
    E = config["node_embedding_size"]
    n_total = config["initial_network_size"]

    if config.get("has_io_roles", "Network" not in config["environment"]):
        obs_dim = config["observation_dim"]
        act_dim = config["action_dim"]
        if config["coevolve_initial_embeddings"]:
            in_role = evolved_parameters[:E]
            out_role = evolved_parameters[E : 2 * E]
        else:
            # Fixed, distinct role constants so the program can still distinguish I/O
            in_role = np.full(E, 1.0)
            out_role = np.full(E, -1.0)
        state = np.zeros((n_total, E))
        state[:obs_dim] = in_role
        state[obs_dim : obs_dim + act_dim] = out_role
        return state

    # Graph-property tasks: original behaviour
    if config["coevolve_initial_embeddings"]:
        return np.expand_dims(evolved_parameters[:E], axis=0)
    elif (not config["shared_intial_embedding"]) and config["initial_embeddings_random"]:
        return np.random.default_rng(None).uniform(-1, +1, (n_total, E))
    elif config["shared_intial_embedding"] and config["initial_embeddings_random"]:
        return config["initial_network_state"]
    else:
        return np.ones((n_total, E))


def env_rollout(W: np.ndarray, config: dict, render=False, animate_graph_rollout: bool = False, solution_id: str = None, seed: int = None) -> float:
    if animate_graph_rollout:
        graph = W_to_nx(W, config["undirected"])

    try:
        diameter = bfs_diameter(W)
    except:
        diameter = int(np.sqrt(W.shape[0]))
        print(f"WARNING: Graph is not connected due to prunning. Diameter manually set to {diameter}.")
    policy_connectivity = W

    # Instatiate the environment
    if render:
        env = gym.make(config["environment"], render_mode="rgb_array")
        env = RecordVideo(env=env, video_folder=config["_path"] + "/env_renders/" + solution_id + str(time.time()))
    else:
        env = gym.make(config["environment"])

    # Resize and normilise input for pixel environments
    if config["pixel_env"] == True:
        raise NotImplementedError

    observation, info = env.reset(seed=seed, options={})
    done = False
    episodeReward = 0
    network_state = np.zeros(policy_connectivity.shape[0])
    network_thinking_time = diameter + config["network_thinking_time_extra_rollout"]
    timestep = 0
    while not done:
        # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
        if isinstance(env.observation_space, gym.spaces.Discrete):
            observation = (observation == torch.arange(env.observation_space.n)).float()
        # Swap axes to the correct order for pytorch
        if config["pixel_env"]:
            raise NotImplementedError

        # Visualise the network's information flow
        if animate_graph_rollout:
            animate_graph(
                G=graph,  # graph is already a NetworkX object for visualisation
                network_state=network_state,
                celluloid_camera=config["celluloid_camera"],
                layout=config["layout"],
                arrows=config["arrows"],
                nodes_role_dims=(config["observation_dim"], config["action_dim"]),
                font_size=8,
                print_labels=True,
                roullout=True,
                rollout_timestep=timestep,
            )

        # Represent observation as a feature vector (node embedding) of the network
        network_state[: config["observation_dim"]] = observation
        persistent_observation = observation if config["persistent_observation_rollout"] else None

        # Let the network update its internal state
        network_state = propagate_features(
            network_state=network_state,
            W=policy_connectivity,
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=persistent_observation,
            feature_transformation_model=None,
            use_torch=config.get("use_torch", False),
        )

        # Select action from the output nodes (fixed indices: right after the input nodes)
        action = network_state[config["observation_dim"] : config["observation_dim"] + config["action_dim"]]

        # Bound the action or convert it to a discrete action
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            action = np.argmax(action)

        # Forward step in the envionrment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if "Bullet" in config["environment"]:
            reward = env.unwrapped.rewards[1]  # Distance walked
        # Save reward
        episodeReward += reward

        # Render the environment
        if render:
            env.render()

        timestep += 1

    # Close the env so RecordVideo flushes the .mp4 to disk (otherwise no video is written)
    env.close()
    return episodeReward


def fitness_functional(config: dict, render=False, animate_graph_growth=False, animate_graph_rollout=False, solution_id=None, checksum=False, return_stats=False) -> Callable[np.ndarray, float]:  # type: ignore
    profile = config.get("profile", False)

    def fitness(evolved_parameters: np.array):
        """Evaluate an agent in its environment.

        Returns a scalar reward normally, or (reg_reward, raw_reward, n_nodes)
        when return_stats=True (used by the optimizer for richer logging).
        """
        timings = {}

        # To average out the growth process stochasticity
        mean_reward = 0
        for _ in range(config["nb_growth_evals"]):
            if config["nb_growth_evals"] > 1:
                config["seed"] = None

            seed_python_numpy_torch_cuda(config["seed"])

            # Define networks
            if config["shared_intial_graph_bool"]:
                W = config["shared_intial_graph"].copy()
            else:
                W = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], seed=None)

            # Initialise the network state (role-based seed embeddings for I/O tasks)
            initial_network_state = build_initial_network_state(config, evolved_parameters)

            # Slice indices into the flat parameter vector
            n1 = config["nb_params_coevolve_initial_embeddings"]
            n2 = n1 + config["nb_params_growth_model"]
            n3 = n2 + config["nb_params_feature_transformation"]
            n4 = n3 + config["nb_params_mlp_weight_values"]

            use_torch = config.get("use_torch", False)
            _tp = time.perf_counter() if profile else 0

            if use_torch:
                mlp_growth_model = MLP(
                    input_dim=config["input_size_growth_model"],
                    output_dim=1,
                    hidden_layers_dims=config["mlp_growth_hidden_layers_dims"],
                    last_layer_activated=config["growth_model_last_layer_activated"],
                    activation=torch.nn.Tanh(),
                    bias=config["growth_model_bias"],
                )
                torch.nn.utils.vector_to_parameters(
                    torch.tensor(evolved_parameters[n1:n2], dtype=torch.float64, requires_grad=False),
                    mlp_growth_model.parameters(),
                )
                if config["NN_transform_node_embedding_during_growth"]:
                    mlp_feature_transformation = MLP(
                        input_dim=config["node_embedding_size"],
                        output_dim=config["node_embedding_size"],
                        hidden_layers_dims=config["mlp_embedding_transform_hidden_layers_dims"],
                        last_layer_activated=config["transform_model_last_layer_activated"],
                        activation=torch.nn.Tanh(),
                        bias=config["transform_model_bias"],
                    )
                    torch.nn.utils.vector_to_parameters(
                        torch.tensor(evolved_parameters[n2:n3], dtype=torch.float64, requires_grad=False),
                        mlp_feature_transformation.parameters(),
                    )
                if config["node_based_growth"] and not config["binary_connectivity"]:
                    mlp_weight_values = MLP(
                        input_dim=2 * config["node_embedding_size"],
                        output_dim=1,
                        hidden_layers_dims=config["mlp_weight_values_hidden_layers_dims"],
                        last_layer_activated=config["mlp_weight_values_last_layer_activated"],
                        activation=torch.nn.Tanh(),
                        bias=config["mlp_weight_values_bias"],
                    )
                    torch.nn.utils.vector_to_parameters(
                        torch.tensor(evolved_parameters[n3:n4], dtype=torch.float64, requires_grad=False),
                        mlp_weight_values.parameters(),
                    )
            else:
                # Build NumpyMLPs directly from parameter slices — no torch object creation
                mlp_growth_model = make_numpy_mlp(
                    evolved_parameters[n1:n2],
                    input_dim=config["input_size_growth_model"],
                    hidden_dims=config["mlp_growth_hidden_layers_dims"],
                    output_dim=1,
                    has_bias=config["growth_model_bias"],
                    last_activated=config["growth_model_last_layer_activated"],
                )
                if config["NN_transform_node_embedding_during_growth"]:
                    mlp_feature_transformation = make_numpy_mlp(
                        evolved_parameters[n2:n3],
                        input_dim=config["node_embedding_size"],
                        hidden_dims=config["mlp_embedding_transform_hidden_layers_dims"],
                        output_dim=config["node_embedding_size"],
                        has_bias=config["transform_model_bias"],
                        last_activated=config["transform_model_last_layer_activated"],
                    )
                if config["node_based_growth"] and not config["binary_connectivity"]:
                    mlp_weight_values = make_numpy_mlp(
                        evolved_parameters[n3:n4],
                        input_dim=2 * config["node_embedding_size"],
                        hidden_dims=config["mlp_weight_values_hidden_layers_dims"],
                        output_dim=1,
                        has_bias=config["mlp_weight_values_bias"],
                        last_activated=config["mlp_weight_values_last_layer_activated"],
                    )

            if profile: timings["mlp_build"] = timings.get("mlp_build", 0.0) + time.perf_counter() - _tp

            network_state = copy.deepcopy(initial_network_state)
            obs_action_dim_tuple = None if "Network" in config["environment"] else (2, 2) if "gate" in config["environment"] else (config["observation_dim"], config["action_dim"])

            if render or checksum:
                ns = np.sum(network_state)
                gs = np.sum(W)
                print(f"\nChecksum of initial network state before growth: {ns}")
                print(f"Checksum of graph G before growth: {gs}")
                print(f"The final grown graph has {W.shape[0]} nodes and {int(np.count_nonzero(W))} edges.")
                config["checksum_best_State_grown"] = ns
                config["checksum_best_Graph_grown"] = gs

            # Compute diameter once; recompute only when graph grows
            try:
                diameter = bfs_diameter(W)
            except Exception:
                diameter = int(np.sqrt(W.shape[0]))

            for growth_cycle_nb in range(config["number_of_growth_cycles"]):
                # Draw the graph

                if animate_graph_growth and growth_cycle_nb == 0:
                    animate_graph(
                        G=W_to_nx(W, config["undirected"]),
                        network_state=network_state,
                        celluloid_camera=config["celluloid_camera"],
                        layout=config["layout"],
                        arrows=config["arrows"],
                        nodes_role_dims=obs_action_dim_tuple,
                        print_labels=True,
                        growth_cycle=growth_cycle_nb,
                    )

                network_thinking_time = diameter + config["network_thinking_time_extra_growth"]

                # Local propagation of node features — i.e thinking time
                if profile: _tp = time.perf_counter()
                network_state = propagate_features(
                    network_state=network_state,
                    W=W,
                    network_thinking_time=network_thinking_time,
                    recurrent_activation_function=config["recurrent_activation_function"],
                    additive_update=config["additive_update"],
                    persistent_observation=None,
                    feature_transformation_model=mlp_feature_transformation if config["NN_transform_node_embedding_during_growth"] else None,
                    use_torch=use_torch,
                )
                if profile: timings["propagate_growth"] = timings.get("propagate_growth", 0.0) + time.perf_counter() - _tp

                if animate_graph_growth:
                    animate_graph(
                        G=W_to_nx(W, config["undirected"]),
                        network_state=network_state,
                        celluloid_camera=config["celluloid_camera"],
                        layout=config["layout"],
                        arrows=config["arrows"],
                        nodes_role_dims=obs_action_dim_tuple,
                        print_labels=True,
                        growth_cycle=growth_cycle_nb,
                    )

                # Query node/edges embeddings
                if profile: _tp = time.perf_counter()
                if config["node_pairs_based_growth"]:
                    # Query pairs of node embeddings
                    node_embeddings_concatenated_dict, embeddings_for_growth_model = query_pairs_of_node_embeddings(
                        W=W,
                        network_state=network_state,
                        self_link_allowed=config["self_link_allowed_during_querying"],
                    )
                elif config["node_based_growth"]:
                    embeddings_for_growth_model = network_state
                    node_embeddings_concatenated_dict = None
                elif config["edge_based_growth"]:
                    raise NotImplementedError
                if profile: timings["query_pairs"] = timings.get("query_pairs", 0.0) + time.perf_counter() - _tp

                # Predict new nodes
                if profile: _tp = time.perf_counter()
                new_nodes_predictions = predict_new_nodes(mlp_growth_model, embeddings_for_growth_model, config["node_embedding_size"], use_torch=use_torch, growth_threshold=config.get("growth_threshold", 0.0))
                if profile: timings["predict_grow"] = timings.get("predict_grow", 0.0) + time.perf_counter() - _tp

                # Add new nodes and increase the network_state vector accordingly
                if profile: _tp = time.perf_counter()
                prev_n = W.shape[0]
                W, network_state = add_new_nodes(
                    W=W,
                    network_state=network_state,
                    node_embeddings_concatenated_dict=node_embeddings_concatenated_dict,
                    new_nodes_predictions=new_nodes_predictions,
                    node_based_growth=config["node_based_growth"],
                    node_pairs_based_growth=config["node_pairs_based_growth"],
                    binary_connectivity=config["binary_connectivity"],
                    undirected=config["undirected"],
                )
                if profile: timings["add_nodes"] = timings.get("add_nodes", 0.0) + time.perf_counter() - _tp
                if W.shape[0] != prev_n:
                    if profile: _tp = time.perf_counter()
                    try:
                        diameter = bfs_diameter(W)
                    except Exception:
                        diameter = int(np.sqrt(W.shape[0]))
                    if profile: timings["bfs_diameter"] = timings.get("bfs_diameter", 0.0) + time.perf_counter() - _tp

                if animate_graph_growth:
                    animate_graph(
                        G=W_to_nx(W, config["undirected"]),
                        network_state=network_state,
                        celluloid_camera=config["celluloid_camera"],
                        layout=config["layout"],
                        arrows=config["arrows"],
                        nodes_role_dims=obs_action_dim_tuple,
                        print_labels=True,
                        growth_cycle=growth_cycle_nb,
                    )

                if W.shape[0] > 1 and not config["binary_connectivity"]:
                    if profile: _tp = time.perf_counter()
                    W = update_weights(W=W, network_state=network_state, model=mlp_weight_values, undirected=config["undirected"], use_torch=use_torch)
                    if profile: timings["update_weights"] = timings.get("update_weights", 0.0) + time.perf_counter() - _tp
                    if animate_graph_growth:
                        animate_graph(
                            G=W_to_nx(W, config["undirected"]),
                            network_state=network_state,
                            celluloid_camera=config["celluloid_camera"],
                            layout=config["layout"],
                            arrows=config["arrows"],
                            nodes_role_dims=obs_action_dim_tuple,
                            print_labels=True,
                            growth_cycle=growth_cycle_nb,
                        )

                if config["prunning_phase"]:
                    W[np.abs(W) <= config["prunning_threshold"]] = 0
                    if animate_graph_growth:
                        animate_graph(
                            G=W_to_nx(W, config["undirected"]),
                            network_state=network_state,
                            celluloid_camera=config["celluloid_camera"],
                            layout=config["layout"],
                            arrows=config["arrows"],
                            nodes_role_dims=obs_action_dim_tuple,
                            print_labels=True,
                            growth_cycle=growth_cycle_nb,
                        )

            if render or checksum:
                ns = np.sum(network_state)
                gs = np.sum(W)
                print(f"Checksum of network state after growth: {ns}")
                print(f"Checksum of graph G after growth: {gs}")
                print(f"The final grown graph has {W.shape[0]} nodes and {int(np.count_nonzero(W))} edges.")
                config["checksum_best_State_grown"] = ns
                config["checksum_best_Graph_grown"] = gs

            if animate_graph_growth:
                figWeights, axWeights = pyplot.subplots(1, 1, figsize=(12, 8))
                axWeights.set_title("Weights distributions")
                edges = np.array(np.nonzero(W)).T
                axWeights.hist(np.array([W[i, j] for i, j in edges]), bins=20)
                figWeights.savefig(config["_path"] + "/weights_" + solution_id + ".png")

            # Run the environment
            if not animate_graph_growth:  # Here we don't want to render the environment if we are just animating the graph growth when calling visualise_graph()
                # If the graph is too small, we don't want to evaluate it and return bad score directly
                if W.shape[0] < config["min_network_size"]:
                    if render:
                        print("\nNetwork too small")
                    if profile:
                        total = sum(timings.values()) or 1e-9
                        print("\n=== Fitness Eval Timing Breakdown (network too small — no env eval) ===")
                        for k, v in sorted(timings.items(), key=lambda x: -x[1]):
                            print(f"  {k:25s}: {v*1000:8.3f}ms  ({100*v/total:5.1f}%)")
                        print(f"  {'TOTAL':25s}: {total*1000:8.3f}ms")
                        print("======================================================================")
                    if config["maximise"]:
                        return W.shape[0] - config["min_network_size"]
                    else:
                        return config["min_network_size"] - W.shape[0]

                mean_episode_reward = 0
                for _ in range(config["nb_episode_evals"]):
                    if profile: _tp = time.perf_counter()
                    if config["environment"] == "SmallWorldNetwork":
                        episode_reward = small_world_ness_fitness(G=W_to_nx(W, config["undirected"]), niter=5, nrand=10, seed=config["seed"], sigma=config["sigma"], omega=config["omega"], render=render)
                    elif "gate" in config["environment"]:
                        episode_reward = bool_gates_fitness(W=W, config=config, render=render, animate_graph_rollout=animate_graph_rollout)
                    elif config["environment"] == "ScaleFreeNetwork":
                        episode_reward = scalefree_fitness(G=W_to_nx(W, config["undirected"]), ks_test=config["ks_test"], render=render)
                    else:
                        seed_env_eval = int(np.random.default_rng(config["env_seed"]).integers(2**32, size=1)[0])
                        animate_graph_rollout_ = True if (_ == 0 and animate_graph_rollout) else False
                        episode_reward = env_rollout(W=W, config=config, render=render, animate_graph_rollout=animate_graph_rollout_, solution_id=solution_id, seed=seed_env_eval)
                    if profile: timings["env_eval"] = timings.get("env_eval", 0.0) + time.perf_counter() - _tp
                    mean_episode_reward += episode_reward

                if render:
                    print(f"\n(mean) Episode reward for {config['nb_episode_evals']} env rollouts: {mean_episode_reward / config['nb_episode_evals']}")

                mean_reward += mean_episode_reward / config["nb_episode_evals"]

        mean_reward /= config["nb_growth_evals"]
        if render:
            print(f"\n-------\n\n(mean) Episodes reward for {config['nb_episode_evals']*config['nb_growth_evals']} runs: {mean_reward}")
            print("\n---------------------------------------------\n")

        raw_reward = mean_reward  # reward before any size penalties
        n_nodes = W.shape[0]

        if config["fewer_edges"]:
            env_max_reward = environment_max_reward(config["environment"])
            sparsity_penalty = (np.count_nonzero(W) / n_nodes ** 2) * env_max_reward
            mean_reward -= sparsity_penalty

        if config["fewer_nodes"]:
            env_max_reward = environment_max_reward(config["environment"])
            nb_nodes_penalty = 10 * n_nodes * env_max_reward
            mean_reward -= nb_nodes_penalty

        size_reg = config.get("size_regularisation")
        reg_warmup = config.get("size_reg_warmup")
        apply_reg = reg_warmup is None or config.get("current_gen", 0) < reg_warmup

        if apply_reg:
            if size_reg in ("io_ratio", "both"):
                seed_size = config.get("observation_dim", 0) + config.get("action_dim", 0)
                if seed_size > 0:
                    alpha = config.get("size_reg_alpha", 1.0)
                    mean_reward -= alpha * max(0, n_nodes / seed_size - 1)

            if size_reg in ("io_edges", "both"):
                seed_size = config.get("observation_dim", 0) + config.get("action_dim", 0)
                # Baseline = fully-connected seed graph (seed_size^2 edges), so only
                # edges added beyond the seed are penalised. Using obs*act (=32 for
                # LunarLander) was too aggressive — it penalised the seed itself.
                baseline_edges = seed_size ** 2
                if baseline_edges > 0:
                    alpha_edges = config.get("size_reg_alpha_edges", 1.0)
                    n_edges = int(np.count_nonzero(W))
                    mean_reward -= alpha_edges * max(0, n_edges / baseline_edges - 1)

        if config["balanced_weights"]:
            env_max_reward = environment_max_reward(config["environment"])
            mean_weights = abs(W[np.nonzero(W)].mean())
            unbalance_penalty = mean_weights * env_max_reward
            mean_reward -= unbalance_penalty

        if profile:
            total = sum(timings.values()) or 1e-9
            print("\n=== Fitness Eval Timing Breakdown ===")
            for k, v in sorted(timings.items(), key=lambda x: -x[1]):
                print(f"  {k:25s}: {v*1000:8.3f}ms  ({100*v/total:5.1f}%)")
            print(f"  {'TOTAL':25s}: {total*1000:8.3f}ms")
            print("=====================================")

        if return_stats:
            return mean_reward, raw_reward, n_nodes
        return mean_reward

    return fitness


def bool_gates_fitness(W: np.ndarray, config: dict, render=False, animate_graph_rollout: bool = False):
    """
    Returns a scalar measure of how many elements of the boolean gate truth table are correctly predicted by the graph G.
    """
    # Truth table
    if config["environment"] == "XOR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 1, 1, 0])
    elif config["environment"] == "NAND_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([1, 1, 1, 0])
    elif config["environment"] == "AND_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 0, 0, 1])
    elif config["environment"] == "OR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 1, 1, 1])
    elif config["environment"] == "NOR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([1, 0, 0, 0])
    elif config["environment"] == "XNOR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([1, 0, 0, 1])

    if animate_graph_rollout:
        graph = W_to_nx(W, config["undirected"])

    try:
        diameter = bfs_diameter(W)
    except:
        diameter = int(np.sqrt(W.shape[0]))
        print(f"WARNING: Graph is not connected due to prunning. Diameter manually set to {diameter}.")
    policy_connectivity = W

    network_state = np.zeros(policy_connectivity.shape[0])
    network_thinking_time = diameter + config["network_thinking_time_extra_rollout"]
    timestep = 0
    fitness = 0
    for idx, x in enumerate(X):
        # Visualise the network's information flow
        if animate_graph_rollout:
            animate_graph(
                G=graph,
                network_state=network_state,
                celluloid_camera=config["celluloid_camera"],
                layout=config["layout"],
                arrows=config["arrows"],
                nodes_role_dims=(2, 2),
                font_size=8,
                print_labels=True,
                roullout=True,
                rollout_timestep=timestep,
            )

        # Represent observation as a feature vector (node embedding) of the network
        network_state[:2] = x
        persistent_observation = x if config["persistent_observation_rollout"] else None

        # Let the network update its internal state
        network_state = propagate_features(
            network_state=network_state,
            W=policy_connectivity,
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=persistent_observation,
            feature_transformation_model=None,
            use_torch=config.get("use_torch", False),
        )

        # Select action from the output nodes (fixed indices 2:4, right after the 2 input nodes)
        bool_prediction = np.argmax(network_state[2:4])

        if bool_prediction == Y[idx]:
            fitness += 1

        timestep += 1

    if render:
        print(f"{config['environment']} gate fitness: {fitness}")
    return fitness


def small_world_ness_fitness(G: nx.Graph, niter=5, nrand=10, seed=None, sigma=True, omega=False, render=False):
    """
    High fitness value means the graph has small-worldness.
    Returns a scalar measure of small-world-ness of a graph G.
    Omega ought to be close to zero and/or sigma > 1
    """
    if sigma:
        sigma_ = nx.sigma(G, niter=niter, nrand=nrand, seed=seed)
    else:
        sigma_ = 0
    if omega:
        abs_omega_inv_ = 1 / abs(nx.omega(G, niter=niter, nrand=nrand, seed=seed))
    else:
        abs_omega_inv_ = 0

    if render:
        s = nx.sigma(G, niter=niter, nrand=nrand, seed=seed)
        o = nx.omega(G, niter=niter, nrand=nrand, seed=seed)
        print(f"\nSigma (smallworldness if > 1): {s}")
        print(f"Omega (smallworldness if ≈ 0): {o}")

    return abs_omega_inv_ + sigma_


def scalefree_fitness(G: nx.Graph, ks_test=False, render=False):
    """
    High fitness value means the graph is scale-free.
    If ks_test:
        Returns 1 minus Kolmogorov-Smirnov test statistic for powerlaw [0,1]
    else:
        Returns loglikelihood ratio R, and its p-value p a graph G being scale-free. [-inf, inf]
        G is scale-free if R > 0 and p < 0.05.
    """

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    fit = powerlaw.Fit(degree_sequence, xmin=1)

    if render:
        pyplot.figure(figsize=(10, 6))
        fig1 = fit.plot_pdf(color="b", linewidth=2, label="data")
        fit.power_law.plot_pdf(color="g", linestyle="--", ax=fig1, label="powerlaw")
        pyplot.legend()

        pyplot.figure(figsize=(10, 6))
        fig2 = fit.plot_ccdf(linewidth=3, color="black", label="data distribution")
        fit.power_law.plot_ccdf(ax=fig2, color="red", linestyle="--", label="powerlaw")
        fit.lognormal.plot_ccdf(ax=fig2, color="green", linestyle="--", label="lognormal")
        fit.stretched_exponential.plot_ccdf(ax=fig2, color="blue", linestyle="--", label="stretched_exponential")
        fit.exponential.plot_ccdf(ax=fig2, color="pink", linestyle="--", label="exponential")
        pyplot.legend()
        pyplot.show()

    if ks_test:
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        test, p = kstest(degree_sequence, "powerlaw", args=(alpha, xmin), N=len(degree_sequence))
        return 1 - test
    else:
        # returns the cummulative sums of  loglikelihood ratio between each pair of distribution fits
        R1, p1 = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
        R2, p2 = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        R3, p3 = fit.distribution_compare("power_law", "stretched_exponential", normalized_ratio=True)
        R4, p4 = fit.distribution_compare("power_law", "lognormal_positive", normalized_ratio=True)
        return R1 + R2 + R3 + R4


def train_model(config):
    if "Network" in config["environment"]:
        config["min_network_size"] = config["min_size_grownNetwork"]
        if config["extra_nodes"] == -1:
            config["initial_network_size"] = 1
        else:
            config["initial_network_size"] = config["extra_nodes"]
    elif "gate" in config["environment"]:
        # Boolean gates: 2 input bits, 2 output nodes (argmax over the two)
        config["observation_dim"] = 2
        config["action_dim"] = 2
        config["min_network_size"] = 4
        extra_hidden = 0 if config["extra_nodes"] == -1 else config["extra_nodes"]
        config["initial_network_size"] = config["observation_dim"] + config["action_dim"] + extra_hidden
    else:
        # Figure out environment dimennsions and type
        observation_dim, action_dim, pixel_env = dimensions_env(config["environment"])
        config["observation_dim"] = observation_dim
        config["action_dim"] = action_dim
        config["pixel_env"] = pixel_env
        if pixel_env:
            raise NotImplementedError
        else:
            # Seed graph always contains the input + output nodes; extra_nodes adds hidden seeds
            extra_hidden = 0 if config["extra_nodes"] == -1 else config["extra_nodes"]
            config["initial_network_size"] = observation_dim + action_dim + extra_hidden
        config["min_network_size"] = observation_dim + action_dim

    # Find number of trainable parameters.
    # I/O tasks evolve two role embeddings (input-role, output-role); graph-property
    # ("Network") tasks keep the single coevolved embedding.
    has_io_roles = "Network" not in config["environment"]
    config["has_io_roles"] = has_io_roles
    if config["coevolve_initial_embeddings"]:
        config["nb_params_coevolve_initial_embeddings"] = (2 if has_io_roles else 1) * config["node_embedding_size"]
    else:
        config["nb_params_coevolve_initial_embeddings"] = 0

    config["input_size_growth_model"] = config["node_embedding_size"] * 2 if config["node_pairs_based_growth"] else config["node_embedding_size"]
    mlp_growth_model = MLP(
        input_dim=config["input_size_growth_model"],
        output_dim=1,
        hidden_layers_dims=config["mlp_growth_hidden_layers_dims"],
        last_layer_activated=config["growth_model_last_layer_activated"],
        activation=torch.nn.Tanh(),
        bias=config["growth_model_bias"],
    )
    config["nb_params_growth_model"] = torch.nn.utils.parameters_to_vector(mlp_growth_model.parameters()).detach().numpy().shape[0]

    if config["NN_transform_node_embedding_during_growth"]:
        mlp_feature_transformation = MLP(
            input_dim=config["node_embedding_size"],
            output_dim=config["node_embedding_size"],
            hidden_layers_dims=config["mlp_embedding_transform_hidden_layers_dims"],
            last_layer_activated=config["transform_model_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["transform_model_bias"],
        )
        config["nb_params_feature_transformation"] = torch.nn.utils.parameters_to_vector(mlp_feature_transformation.parameters()).detach().numpy().shape[0]
    else:
        config["nb_params_feature_transformation"] = 0

    if config["node_based_growth"] and not config["binary_connectivity"]:
        output_dim_mlp = 1 if config["undirected"] else 2
        mlp_weight_values = MLP(
            input_dim=2 * config["node_embedding_size"],
            output_dim=1,
            hidden_layers_dims=config["mlp_weight_values_hidden_layers_dims"],
            last_layer_activated=config["mlp_weight_values_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["mlp_weight_values_bias"],
        )
        config["nb_params_mlp_weight_values"] = torch.nn.utils.parameters_to_vector(mlp_weight_values.parameters()).detach().numpy().shape[0]
    else:
        config["nb_params_mlp_weight_values"] = 0

    config["nb_trainable_parameters"] = (
        config["nb_params_coevolve_initial_embeddings"] + config["nb_params_growth_model"] + config["nb_params_feature_transformation"] + config["nb_params_mlp_weight_values"]
    )
    print(f"The growth model has {config['nb_trainable_parameters']} trainable parameters")

    # Generate initial graph (stored as numpy adjacency matrix)
    if config["shared_intial_graph_bool"]:
        config["shared_intial_graph"] = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], config["seed"])

    # Generate initial node embedding
    if not config["coevolve_initial_embeddings"] and config["shared_intial_embedding"] and config["initial_embeddings_random"]:
        config["initial_network_state"] = np.random.default_rng(config["seed"]).uniform(-1, +1, (config["initial_network_size"], config["node_embedding_size"]))

    fitness = fitness_functional(config)
    fitness_with_stats = fitness_functional(config, return_stats=True)

    # Run optimiser
    if config["optimizer"] == "CMAES":
        solution_best, solution_centroid, early_stopping_executed, logger = CMAES(config, fitness, fitness_with_stats)
    else:
        raise NotImplementedError

    return solution_best, solution_centroid, early_stopping_executed, logger


def grow_network(evolved_parameters: np.ndarray, config: dict):
    """Grow a network from evolved_parameters and return (W, network_state) after all growth cycles."""
    seed_python_numpy_torch_cuda(config["seed"])

    if config["shared_intial_graph_bool"]:
        W = config["shared_intial_graph"].copy()
    else:
        W = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], seed=None)

    initial_network_state = build_initial_network_state(config, evolved_parameters)

    n1 = config["nb_params_coevolve_initial_embeddings"]
    n2 = n1 + config["nb_params_growth_model"]
    n3 = n2 + config["nb_params_feature_transformation"]
    n4 = n3 + config["nb_params_mlp_weight_values"]

    use_torch = config.get("use_torch", False)
    if use_torch:
        mlp_growth_model = MLP(
            input_dim=config["input_size_growth_model"],
            output_dim=1,
            hidden_layers_dims=config["mlp_growth_hidden_layers_dims"],
            last_layer_activated=config["growth_model_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["growth_model_bias"],
        )
        torch.nn.utils.vector_to_parameters(
            torch.tensor(evolved_parameters[n1:n2], dtype=torch.float64, requires_grad=False),
            mlp_growth_model.parameters(),
        )
        mlp_feature_transformation = None
        if config["NN_transform_node_embedding_during_growth"]:
            mlp_feature_transformation = MLP(
                input_dim=config["node_embedding_size"],
                output_dim=config["node_embedding_size"],
                hidden_layers_dims=config["mlp_embedding_transform_hidden_layers_dims"],
                last_layer_activated=config["transform_model_last_layer_activated"],
                activation=torch.nn.Tanh(),
                bias=config["transform_model_bias"],
            )
            torch.nn.utils.vector_to_parameters(
                torch.tensor(evolved_parameters[n2:n3], dtype=torch.float64, requires_grad=False),
                mlp_feature_transformation.parameters(),
            )
        mlp_weight_values = None
        if config["node_based_growth"] and not config["binary_connectivity"]:
            mlp_weight_values = MLP(
                input_dim=2 * config["node_embedding_size"],
                output_dim=1,
                hidden_layers_dims=config["mlp_weight_values_hidden_layers_dims"],
                last_layer_activated=config["mlp_weight_values_last_layer_activated"],
                activation=torch.nn.Tanh(),
                bias=config["mlp_weight_values_bias"],
            )
            torch.nn.utils.vector_to_parameters(
                torch.tensor(evolved_parameters[n3:n4], dtype=torch.float64, requires_grad=False),
                mlp_weight_values.parameters(),
            )
    else:
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

    network_state = copy.deepcopy(initial_network_state)
    try:
        diameter = bfs_diameter(W)
    except Exception:
        diameter = int(np.sqrt(W.shape[0]))

    for _ in range(config["number_of_growth_cycles"]):
        network_state = propagate_features(
            network_state=network_state,
            W=W,
            network_thinking_time=diameter + config["network_thinking_time_extra_growth"],
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=None,
            feature_transformation_model=mlp_feature_transformation,
            use_torch=use_torch,
        )

        if config["node_pairs_based_growth"]:
            node_embeddings_concatenated_dict, embeddings_for_growth_model = query_pairs_of_node_embeddings(
                W=W, network_state=network_state, self_link_allowed=config["self_link_allowed_during_querying"]
            )
        else:
            embeddings_for_growth_model = network_state
            node_embeddings_concatenated_dict = None

        new_nodes_predictions = predict_new_nodes(mlp_growth_model, embeddings_for_growth_model, config["node_embedding_size"], use_torch=use_torch, growth_threshold=config.get("growth_threshold", 0.0))
        prev_n = W.shape[0]
        W, network_state = add_new_nodes(
            W=W,
            network_state=network_state,
            node_embeddings_concatenated_dict=node_embeddings_concatenated_dict,
            new_nodes_predictions=new_nodes_predictions,
            node_based_growth=config["node_based_growth"],
            node_pairs_based_growth=config["node_pairs_based_growth"],
            binary_connectivity=config["binary_connectivity"],
            undirected=config["undirected"],
        )
        if W.shape[0] != prev_n:
            try:
                diameter = bfs_diameter(W)
            except Exception:
                diameter = int(np.sqrt(W.shape[0]))

        if W.shape[0] > 1 and not config["binary_connectivity"] and mlp_weight_values is not None:
            W = update_weights(W=W, network_state=network_state, model=mlp_weight_values, undirected=config["undirected"], use_torch=use_torch)

        if config["prunning_phase"]:
            W[np.abs(W) <= config["prunning_threshold"]] = 0

    return W, network_state


def snapshot_graph_png(W: np.ndarray, network_state: np.ndarray, config: dict, save_path: str):
    """Save a static PNG of the final grown graph. Nodes are coloured by role; edge width encodes |weight|; edge colour encodes sign (blue=excitatory, red=inhibitory)."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    G = W_to_nx(W, config["undirected"])
    n = len(G)

    if "Network" in config["environment"]:
        obs_dim, act_dim = None, None
    elif "gate" in config["environment"]:
        obs_dim, act_dim = 2, 2
    else:
        obs_dim, act_dim = config.get("observation_dim", 0), config.get("action_dim", 0)

    if obs_dim is not None:
        color_map = [
            "indianred" if node < obs_dim else ("slategray" if obs_dim <= node < obs_dim + act_dim else "white")
            for node in G.nodes()
        ]
    else:
        color_map = ["white"] * n

    pos = nx_layout(G, config["layout"])
    labels = {i: str(i) for i in range(n)}

    edges = list(G.edges())
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    # Floor the width so weak (but present) edges stay visible; sign is shown by colour
    edge_widths = [max(1.2, abs(w) * 4) for w in edge_weights]
    edge_colors = ["steelblue" if w >= 0 else "crimson" for w in edge_weights]

    fig, ax = pyplot.subplots(figsize=(14, 10))
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        labels=labels,
        edgecolors="black",
        font_size=8,
        node_size=600,
        node_color=color_map,
        arrows=config["arrows"],
        width=edge_widths,
        edge_color=edge_colors,
    )
    ax.set_title(f"{config['environment']} — best DNA  |  {n} nodes, {len(edges)} edges", fontsize=13)
    pyplot.box(False)

    legend_elements = []
    if obs_dim is not None:
        legend_elements += [
            Patch(facecolor="indianred", edgecolor="black", label=f"Input (nodes 0-{obs_dim-1})"),
            Patch(facecolor="white", edgecolor="black", label="Hidden"),
            Patch(facecolor="slategray", edgecolor="black", label=f"Output (nodes {obs_dim}-{obs_dim + act_dim - 1})"),
        ]
    legend_elements += [
        Line2D([0], [0], color="steelblue", linewidth=2, label="Excitatory (weight > 0)"),
        Line2D([0], [0], color="crimson", linewidth=2, label="Inhibitory (weight < 0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    pyplot.close(fig)
    print(f"Graph snapshot saved to {save_path}")


if __name__ == "__main__":
    pass
