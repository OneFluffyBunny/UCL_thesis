import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import networkx as nx
from scipy import stats, sparse
from scipy.sparse.csgraph import connected_components
from numpy.random import default_rng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_grad_enabled(False)


class NumpyMLP:
    """Numpy-only MLP for inference. Replaces torch.nn.Sequential when use_torch=False."""

    def __init__(self, layers):
        # layers: list of (weight_matrix, bias_or_None, activation_fn_or_None)
        self._layers = layers

    def __call__(self, x):
        for W, b, act in self._layers:
            x = x @ W.T
            if b is not None:
                x = x + b
            if act is not None:
                x = act(x)
        return x


def torch_mlp_to_numpy(torch_model):
    """Extract weights from a torch Sequential MLP and return an equivalent NumpyMLP."""
    children = list(torch_model.children())
    layers = []
    for i, layer in enumerate(children):
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().numpy().copy()
            b = layer.bias.detach().numpy().copy() if layer.bias is not None else None
            has_act = (i + 1 < len(children)) and isinstance(children[i + 1], torch.nn.Tanh)
            layers.append((W, b, np.tanh if has_act else None))
    return NumpyMLP(layers)


def make_numpy_mlp(params, input_dim, hidden_dims, output_dim, has_bias, last_activated):
    """Build a NumpyMLP directly from a flat parameter array, bypassing torch entirely.

    Parameter ordering matches torch.nn.utils.parameters_to_vector: for each Linear
    layer, weight (fan_out × fan_in) is packed first, then bias (fan_out) if present.
    """
    layers = []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    offset = 0
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        W = params[offset : offset + fan_out * fan_in].reshape(fan_out, fan_in).copy()
        offset += fan_out * fan_in
        if has_bias:
            b = params[offset : offset + fan_out].copy()
            offset += fan_out
        else:
            b = None
        is_last = i == len(dims) - 2
        act = np.tanh if (not is_last or last_activated) else None
        layers.append((W, b, act))
    return NumpyMLP(layers)


def MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias):
    """This function creates a multi-layer perceptron.

    Args:
        input_dim (int): The input dimension of the MLP.
        output_dim (int): The output dimension of the MLP.
        hidden_layers_dims (list): A list of integers that represent the number of hidden layers and their dimensions.
        growth_model_last_layer_activated (bool): Whether the last layer should be activated.
        activation (torch.nn.functional, optional): The activation function to use. Defaults to tanh.

    Returns:
        torch.nn.Module: The MLP.
    """
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
    layers.append(activation)
    for i in range(1, len(hidden_layers_dims)):
        layers.append(torch.nn.Linear(hidden_layers_dims[i - 1], hidden_layers_dims[i], bias=bias))
        layers.append(activation)
    layers.append(torch.nn.Linear(hidden_layers_dims[-1], output_dim, bias=bias))
    if last_layer_activated:
        layers.append(activation)
    return torch.nn.Sequential(*layers)


# Generate intial graph random matrix
def generate_initial_graph(network_size, sparsity, binary_connectivity, undirected, seed):
    """Generates a random connected initial graph and returns it as a numpy adjacency matrix.

    Args:
        network_size (int): The intial number of nodes in the network.
        sparsity (float): The initial sparsity of the network.
        binary_connectivity (bool): Whether the network has binary weights.
        undirected (bool): Whether the network is undirected.
        seed: Random seed.

    Returns:
        W (np.ndarray): The initial adjacency matrix.
    """
    nb_disjoint_initial_graphs = np.inf
    while nb_disjoint_initial_graphs > 1:
        rng = default_rng(seed)
        if binary_connectivity:
            rvs = stats.uniform(loc=0, scale=1).rvs
            W = np.rint(sparse.random(network_size, network_size, density=sparsity, data_rvs=rvs, random_state=rng).toarray())
        else:
            rvs = stats.uniform(loc=-1, scale=2).rvs
            W = sparse.random(network_size, network_size, density=sparsity, data_rvs=rvs, random_state=rng).toarray()
        adj = ((np.abs(W) + np.abs(W.T)) > 0).astype(float)
        nb_disjoint_initial_graphs, _ = connected_components(adj, directed=False)

    if undirected:
        # Make symmetric: for each (i,j) pair keep whichever entry has larger absolute value
        W = np.where(np.abs(W) >= np.abs(W.T), W, W.T)

    return W


def bfs_diameter(W):
    """Compute the diameter of the graph represented by adjacency matrix W.

    Uses numpy Floyd-Warshall — no scipy call overhead, fast for the small graphs NDP produces.

    Args:
        W (np.ndarray): Adjacency matrix.

    Returns:
        int: Graph diameter (longest shortest path between any two nodes).
    """
    n = W.shape[0]
    if n <= 1:
        return 0
    adj = np.abs(W) > 0
    adj = adj | adj.T  # treat as undirected
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    dist[adj] = 1.0
    for k in range(n):
        dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
    finite = dist[np.isfinite(dist)]
    return int(finite.max()) if len(finite) > 0 else int(np.sqrt(n))


def W_to_nx(W, undirected=True):
    """Convert a numpy adjacency matrix to a NetworkX graph (for visualisation only).

    Args:
        W (np.ndarray): Adjacency matrix.
        undirected (bool): Whether to create an undirected graph.

    Returns:
        nx.Graph or nx.DiGraph
    """
    if undirected:
        return nx.from_numpy_array(W, create_using=nx.Graph)
    else:
        return nx.from_numpy_array(W, create_using=nx.DiGraph)


def propagate_features(
    network_state: np.array,
    W: np.array,
    network_thinking_time: int,
    recurrent_activation_function: str,
    additive_update: bool,
    persistent_observation: np.array = None,
    feature_transformation_model=None,
    use_torch: bool = False,
):
    """
    Propagate the network state through the network.

    Args:
        network_state (np.array): vector of network state, i.e. the features of the nodes / node embeddings
        W (np.array): adjacency matrix
        network_thinking_time (int): number of steps to propagate the network state
        recurrent_activation_function (str)): activation function to use for network state propagation
        additive_update (bool): whether to use additive update or not
        persistent_observation (np.array) : observation array
        use_torch (bool): if True use torch tensors; if False run entirely in numpy (faster for small networks)

    Returns:
        np.array: the updated network state
    """
    activation = np.tanh if recurrent_activation_function == "tanh" else None

    if use_torch:
        with torch.no_grad():
            network_state = torch.tensor(network_state, dtype=torch.float64)
            persistent_observation_t = torch.tensor(persistent_observation, dtype=torch.float64) if persistent_observation is not None else None
            W_t = torch.tensor(W, dtype=torch.float64)
            for _ in range(network_thinking_time):
                if additive_update:
                    network_state += W_t.T @ network_state
                else:
                    network_state = W_t.T @ network_state
                if feature_transformation_model is not None:
                    network_state = feature_transformation_model(network_state)
                elif activation is not None:
                    network_state = activation(network_state)
                if persistent_observation_t is not None:
                    network_state[: persistent_observation_t.shape[0]] = persistent_observation_t
        return network_state.detach().numpy()
    else:
        for _ in range(network_thinking_time):
            if additive_update:
                network_state = network_state + W.T @ network_state
            else:
                network_state = W.T @ network_state
            if feature_transformation_model is not None:
                network_state = feature_transformation_model(network_state)
            elif activation is not None:
                network_state = activation(network_state)
            if persistent_observation is not None:
                network_state[: persistent_observation.shape[0]] = persistent_observation
        return network_state


def query_pairs_of_node_embeddings(W: np.array, network_state: np.array, self_link_allowed: bool = False):
    """This function returns a dictionary (and its array version) of the concatenated node embedd1ings for every node pair.

    Args:
        W (np.array): The adjacency matrix.
        network_state (np.array): The network state.
        self_link_allowed (bool, optional): whether the output array/dict contains concatenated embedding of nodes with self-links. Defaults to False.

    Returns:
        node_embeddings_concatenated_dict (dict): A dictionary of the concatenated node embeddings for every node pair.
        node_embeddings_concatenated_array (np.array): An array of the concatenated node embeddings for every node pair.
    """
    # Make every node-node pair appear only once
    W_abs = np.abs(W)
    links = np.clip(np.tril(W_abs) + np.triu(W_abs).T, 0, 1)
    if not self_link_allowed:
        np.fill_diagonal(links, 0)

    rows, cols = np.where(links > 0)
    arr = np.concatenate([network_state[rows], network_state[cols]], axis=1)
    node_embeddings_concatenated_dict = {
        k: {"from_node": int(rows[k]), "to_node": int(cols[k]), "concatenated_features": arr[k]}
        for k in range(len(rows))
    }
    return node_embeddings_concatenated_dict, arr


def predict_new_nodes(growth_decision_model, embeddings_for_growth_model, node_embedding_size, use_torch=False, growth_threshold=0.0):
    """This function predicts the new nodes based on the concatenated node embeddings.

    Args:
        growth_decision_model: MLP (torch or NumpyMLP) deciding whether a node should spawn a child.
        embeddings_for_growth_model (np.array): Node embeddings to query.
        node_embedding_size (int): Size of each node embedding.
        use_torch (bool): If True, wrap input in a torch tensor before calling the model.
        growth_threshold (float): MLP output must exceed this value to trigger growth (default 0.0).

    Returns:
        new_nodes_predictions (np.array): Boolean array of growth decisions.
    """
    if use_torch:
        with torch.no_grad():
            probs = growth_decision_model(torch.tensor(embeddings_for_growth_model, dtype=torch.float64)).detach().numpy()
    else:
        probs = growth_decision_model(embeddings_for_growth_model)
    return (probs > growth_threshold).squeeze()


def update_weights(W, network_state, model, undirected, use_torch=False):
    """Update edge weights by running all edges through the weight MLP in a single batched forward pass.

    Args:
        W (np.ndarray): Adjacency matrix.
        network_state (np.ndarray): Node embeddings, shape (n, embedding_size).
        model: MLP (torch or NumpyMLP) that maps concat(state_i, state_j) -> scalar weight.
        undirected (bool): If True, write each weight to both (i,j) and (j,i).
        use_torch (bool): If True, wrap input in a torch tensor before calling the model.

    Returns:
        np.ndarray: Updated adjacency matrix.
    """
    if undirected:
        # Upper triangle only — avoids processing each edge twice on a symmetric matrix
        rows, cols = np.where(np.triu(np.abs(W) > 0))
    else:
        rows, cols = np.nonzero(W)

    if len(rows) == 0:
        return W

    pairs = np.concatenate([network_state[rows], network_state[cols]], axis=1)

    if use_torch:
        with torch.no_grad():
            weights = model(torch.tensor(pairs, dtype=torch.float64)).detach().numpy().squeeze(axis=-1)
    else:
        weights = model(pairs).squeeze(axis=-1)

    new_W = W.copy()
    new_W[rows, cols] = weights
    if undirected:
        new_W[cols, rows] = weights  # mirror: both directions get the same weight

    return new_W


def add_new_nodes(
    W,
    network_state,
    node_embeddings_concatenated_dict,
    new_nodes_predictions,
    node_based_growth,
    node_pairs_based_growth,
    binary_connectivity,
    undirected,
):
    """Add new nodes to the graph using numpy matrix operations instead of NetworkX.

    Args:
        W (np.ndarray): Adjacency matrix, shape (n, n).
        network_state (np.ndarray): Node embeddings, shape (n, embedding_size).
        node_embeddings_concatenated_dict (dict): Edge pair dict (used for node_pairs_based_growth).
        new_nodes_predictions (np.ndarray): Boolean array of which nodes/edges trigger growth.
        node_based_growth (bool): Each node independently decides to spawn a child.
        node_pairs_based_growth (bool): Each edge independently decides to insert a new node.
        binary_connectivity (bool): Whether edge weights are binary.
        undirected (bool): Whether the graph is undirected.

    Returns:
        W (np.ndarray): Updated adjacency matrix.
        network_state (np.ndarray): Updated node embeddings.
    """
    n = W.shape[0]

    if new_nodes_predictions.shape == ():
        new_nodes_predictions = new_nodes_predictions.reshape(1)

    if node_pairs_based_growth:
        for idx_edge in node_embeddings_concatenated_dict:
            if new_nodes_predictions[idx_edge]:
                src = node_embeddings_concatenated_dict[idx_edge]["from_node"]
                dst = node_embeddings_concatenated_dict[idx_edge]["to_node"]

                # Neighbors of both endpoints via numpy row lookup
                neighbors = np.unique(np.concatenate([
                    np.where(np.abs(W[src]) > 0)[0],
                    np.where(np.abs(W[:, src]) > 0)[0],
                    np.where(np.abs(W[dst]) > 0)[0],
                    np.where(np.abs(W[:, dst]) > 0)[0],
                ]))

                new_n = n + 1
                new_W = np.zeros((new_n, new_n))
                new_W[:n, :n] = W
                init_w = 1 if binary_connectivity else 0
                new_W[src, n] = init_w
                new_W[n, dst] = init_w
                if undirected:
                    new_W[n, src] = init_w
                    new_W[dst, n] = init_w

                new_state = np.concatenate([network_state, np.expand_dims(np.mean(network_state[neighbors], axis=0), axis=0)])
                W = new_W
                network_state = new_state
                n += 1

    elif node_based_growth:
        grow_nodes = np.where(new_nodes_predictions[:n])[0]
        num_new = len(grow_nodes)
        if num_new == 0:
            return W, network_state

        # Adjacency with self-loops, computed once for the whole matrix
        adj = np.abs(W) > 0
        if not undirected:
            adj = adj | adj.T
        np.fill_diagonal(adj, True)

        new_n = n + num_new
        new_W = np.zeros((new_n, new_n))
        new_W[:n, :n] = W

        # Edge assignments: loop only over growing nodes (num_new, not n)
        for k, src in enumerate(grow_nodes):
            new_idx = n + k
            nbrs = np.where(adj[src])[0]
            new_W[nbrs, new_idx] = 1
            new_W[new_idx, nbrs] = 1

        # Vectorized mean-embedding: one matmul instead of per-node np.mean
        adj_grow = adj[grow_nodes].astype(network_state.dtype)   # (num_new, n)
        row_sums = adj_grow.sum(axis=1, keepdims=True)            # (num_new, 1)
        new_embeddings = (adj_grow @ network_state) / row_sums    # (num_new, embed_dim)

        W = new_W
        network_state = np.concatenate([network_state, new_embeddings], axis=0)

    return W, network_state
