# %%
import random
import torch
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class GraphLayout:
    M: int  # number of nodes/channels
    nodes: torch.Tensor  # long, shape [M]
    edges: list[tuple[int, int]]  # list of directed edges (src, dst)

    R_idx: torch.Tensor  # long, shape [M, R_max]
    R_mask: torch.Tensor  # bool, shape [M, R_max]
    R_deg: torch.Tensor  # long, shape [M]   (#valid receive-neighbors per node)

    self_slot_in_R: torch.Tensor  # long, shape [M]

    T_idx: torch.Tensor  # long, shape [M, T_max]
    T_mask: torch.Tensor  # bool, shape [M, T_max]
    T_deg: torch.Tensor  # long, shape [M]   (#valid transmit-neighbors per node)

    T_slot_in_R: torch.Tensor  # long, shape [M, T_max]

    pair_i_local: torch.Tensor  # long, shape [M, P_max]
    pair_j_local: torch.Tensor  # long, shape [M, P_max]
    pair_mask: torch.Tensor  # bool, shape [M, P_max]
    P_deg: torch.Tensor  # long, shape [M]   (#valid local pairs per node)


def build_pairs(adj_mask: torch.Tensor, device: str):
    iu, ju = torch.triu_indices(adj_mask.size(0), adj_mask.size(1), offset=1)
    m = adj_mask[iu, ju]
    return iu[m].to(device), ju[m].to(device)


def fully_connected_graph(M: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Fully connected graph. returns nodes and edges"""
    nodes = list(range(M))
    edges = [(i, j) for i in range(M) for j in range(M) if i != j]
    return nodes, edges


def ring_graph(M: int, device: str, directed: int = 0) -> torch.Tensor:
    """
    Ring graph (each node connected to two neighbors)
    directed: 0=undirected (i<j), 1=forward only (i→i+1), -1=backward only (i→i-1)
    """
    adj = torch.zeros(M, M, dtype=torch.bool, device=device)
    for i in range(M):
        if directed == 0:
            adj[i, (i - 1) % M] = True
            adj[i, (i + 1) % M] = True
        elif directed == 1:
            adj[i, (i + 1) % M] = True
        elif directed == -1:
            adj[i, (i - 1) % M] = True
    return adj


def get_receive_neighborhood(adj_mask: torch.Tensor, node_idx: int) -> torch.Tensor:
    """
    Get indices of neighbors that node `node_idx` receives from (i.e., j where adj_mask[j,
    node_idx] is True). Returns a 1D tensor of neighbor indices.
    """
    return torch.where(adj_mask[:, node_idx])[0]


def get_transmit_neighborhood(adj_mask: torch.Tensor, node_idx: int) -> torch.Tensor:
    """
    Get indices of neighbors that node `node_idx` transmits to (i.e., j where adj_mask[node_idx,
    j] is True). Returns a 1D tensor of neighbor indices.
    """
    return torch.where(adj_mask[node_idx, :])[0]


def get_receive_neighborhoods(adj_mask: torch.Tensor) -> torch.Tensor:
    """
    Get receive neighborhoods for all nodes. Returns a [M, max_deg] tensor
    where max_deg is the maximum in-degree, and entries are neighbor indices (padded with -1).
    """
    M = adj_mask.shape[0]
    max_deg = adj_mask.sum(dim=0).max().item()
    neighborhoods = torch.full((M, max_deg), -1, dtype=torch.long, device=adj_mask.device)
    for i in range(M):
        neighbors = get_receive_neighborhood(adj_mask, i)
        neighborhoods[i, : len(neighbors)] = neighbors
    return neighborhoods


def get_transmit_neighborhoods(adj_mask: torch.Tensor) -> torch.Tensor:
    """
    Get transmit neighborhoods for all nodes. Returns a [M, max_deg] tensor
    where max_deg is the maximum out-degree, and entries are neighbor indices (padded with -1).
    """
    M = adj_mask.shape[0]
    max_deg = adj_mask.sum(dim=1).max().item()
    neighborhoods = torch.full((M, max_deg), -1, dtype=torch.long, device=adj_mask.device)
    for i in range(M):
        neighbors = get_transmit_neighborhood(adj_mask, i)
        neighborhoods[i, : len(neighbors)] = neighbors
    return neighborhoods


def build_graph_layout(
    M: int,
    edges: Iterable[tuple[int, int]],
    *,
    directed: bool = True,
    include_self: bool = True,
    device: Optional[torch.device | str] = None,
) -> GraphLayout:
    """
    Build padded graph metadata for batched ADMM / local-CR simulation.

    Parameters
    ----------
    M:
        Number of nodes/channels. Nodes are assumed to be 0..M-1.
    edges:
        Iterable of directed edges (src, dst).
        If directed=False, each edge is symmetrized.
    directed:
        If False, every (i, j) also adds (j, i).
    include_self:
        If True, add self-loops (i, i) for all i. This is recommended here,
        because it guarantees that each node owns a local copy of its own channel.
    device:
        Optional torch device for the returned tensors.

    Returns
    -------
    GraphLayout
    """
    if M <= 0:
        raise ValueError("M must be positive.")

    edge_set: set[tuple[int, int]] = set()

    for src, dst in edges:
        if not (0 <= src < M and 0 <= dst < M):
            raise ValueError(f"Invalid edge ({src}, {dst}) for M={M}.")
        edge_set.add((src, dst))
        if not directed:
            edge_set.add((dst, src))

    if include_self:
        for i in range(M):
            edge_set.add((i, i))

    R_lists: list[list[int]] = [[] for _ in range(M)]
    T_lists: list[list[int]] = [[] for _ in range(M)]

    for src, dst in sorted(edge_set):
        T_lists[src].append(dst)
        R_lists[dst].append(src)

    for i in range(M):
        R_lists[i] = sorted(set(R_lists[i]))
        T_lists[i] = sorted(set(T_lists[i]))

    R_deg_list = [len(lst) for lst in R_lists]
    T_deg_list = [len(lst) for lst in T_lists]

    R_max = max(R_deg_list) if M > 0 else 0
    T_max = max(T_deg_list) if M > 0 else 0

    R_idx = torch.full((M, R_max), -1, dtype=torch.long, device=device)
    R_mask = torch.zeros((M, R_max), dtype=torch.bool, device=device)
    T_idx = torch.full((M, T_max), -1, dtype=torch.long, device=device)
    T_mask = torch.zeros((M, T_max), dtype=torch.bool, device=device)

    self_slot_in_R = torch.full((M,), -1, dtype=torch.long, device=device)

    for i in range(M):
        # Receive-neighborhood
        for r, j in enumerate(R_lists[i]):
            R_idx[i, r] = j
            R_mask[i, r] = True
            if j == i:
                self_slot_in_R[i] = r

        # Transmit-neighborhood
        for t, j in enumerate(T_lists[i]):
            T_idx[i, t] = j
            T_mask[i, t] = True

    if include_self and torch.any(self_slot_in_R < 0):
        missing = torch.nonzero(self_slot_in_R < 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"Self-loops were requested, but self_slot_in_R is missing for nodes {missing}."
        )

    R_deg = torch.tensor(R_deg_list, dtype=torch.long, device=device)
    T_deg = torch.tensor(T_deg_list, dtype=torch.long, device=device)

    # Lookups
    slot_map: dict[tuple[int, int], int] = {}
    for holder in range(M):
        for r, owner in enumerate(R_lists[holder]):
            slot_map[(holder, owner)] = r

    T_slot_in_R = torch.full((M, T_max), -1, dtype=torch.long, device=device)

    for owner in range(M):
        for t, holder in enumerate(T_lists[owner]):
            r = slot_map.get((holder, owner), -1)
            if r < 0:
                raise RuntimeError(
                    f"Inconsistent graph layout: owner {owner} transmits to holder {holder}, "
                    f"but holder does not have owner in its receive-neighborhood."
                )
            T_slot_in_R[owner, t] = r

    # pair slots
    pair_lists_i: list[list[int]] = []
    pair_lists_j: list[list[int]] = []
    P_deg_list: list[int] = []

    for i in range(M):
        Ri = R_deg_list[i]
        local_pairs = list(combinations(range(Ri), 2))
        pair_lists_i.append([a for a, _ in local_pairs])
        pair_lists_j.append([b for _, b in local_pairs])
        P_deg_list.append(len(local_pairs))

    P_max = max(P_deg_list) if M > 0 else 0

    pair_i_local = torch.full((M, P_max), -1, dtype=torch.long, device=device)
    pair_j_local = torch.full((M, P_max), -1, dtype=torch.long, device=device)
    pair_mask = torch.zeros((M, P_max), dtype=torch.bool, device=device)

    for i in range(M):
        for p, (a, b) in enumerate(zip(pair_lists_i[i], pair_lists_j[i])):
            pair_i_local[i, p] = a
            pair_j_local[i, p] = b
            pair_mask[i, p] = True

    P_deg = torch.tensor(P_deg_list, dtype=torch.long, device=device)

    # nodes
    nodes = torch.arange(M, dtype=torch.long, device=device)

    return GraphLayout(
        M=M,
        nodes=nodes,
        edges=sorted(edge_set),
        R_idx=R_idx,
        R_mask=R_mask,
        R_deg=R_deg,
        self_slot_in_R=self_slot_in_R,
        T_idx=T_idx,
        T_mask=T_mask,
        T_deg=T_deg,
        T_slot_in_R=T_slot_in_R,
        pair_i_local=pair_i_local,
        pair_j_local=pair_j_local,
        pair_mask=pair_mask,
        P_deg=P_deg,
    )


# Utility functions for debugging and visualization
def plot_network(layout: GraphLayout):
    G = nx.DiGraph() if any(src != dst for src, dst in layout.edges) else nx.Graph()
    G.add_nodes_from(layout.nodes.cpu().tolist())
    G.add_edges_from(layout.edges)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.show()


def normalize_edges_ref(M: int, edges, directed: bool, include_self: bool):
    """Independent reference: normalize edges exactly as spec says (not using layout)."""
    E = set()
    for src, dst in edges:
        assert 0 <= src < M and 0 <= dst < M, (src, dst)
        E.add((src, dst))
        if not directed:
            E.add((dst, src))
    if include_self:
        for i in range(M):
            E.add((i, i))
    return E


def test_neighborhoods_match_edges():
    M = 5
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    directed = False
    include_self = True

    layout = build_graph_layout(
        M, edges, directed=directed, include_self=include_self, device="cpu"
    )

    E = normalize_edges_ref(M, edges, directed=directed, include_self=include_self)

    # R_i = {j : (j,i) in E}
    for i in range(M):
        R_expected = sorted({src for (src, dst) in E if dst == i})
        R_got = layout.R_idx[i][layout.R_mask[i]].cpu().tolist()
        assert R_got == R_expected, f"R[{i}] mismatch: got {R_got}, expected {R_expected}"

    # T_i = {j : (i,j) in E}
    for i in range(M):
        T_expected = sorted({dst for (src, dst) in E if src == i})
        T_got = layout.T_idx[i][layout.T_mask[i]].cpu().tolist()
        assert T_got == T_expected, f"T[{i}] mismatch: got {T_got}, expected {T_expected}"


def build_ref_neighborhoods(M: int, E: set[tuple[int, int]]):
    R_lists = [[] for _ in range(M)]
    T_lists = [[] for _ in range(M)]
    for src, dst in E:
        T_lists[src].append(dst)
        R_lists[dst].append(src)
    for i in range(M):
        R_lists[i] = sorted(set(R_lists[i]))
        T_lists[i] = sorted(set(T_lists[i]))
    return R_lists, T_lists


def test_local_pairs_are_combinations_ref():
    M = 6
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    directed = False
    include_self = True

    layout = build_graph_layout(
        M, edges, directed=directed, include_self=include_self, device="cpu"
    )

    E = normalize_edges_ref(M, edges, directed=directed, include_self=include_self)
    R_ref, _ = build_ref_neighborhoods(M, E)

    for i in range(M):
        Ri = len(R_ref[i])  # reference degree
        expected = list(combinations(range(Ri), 2))

        pi = layout.pair_i_local[i][layout.pair_mask[i]].cpu().tolist()
        pj = layout.pair_j_local[i][layout.pair_mask[i]].cpu().tolist()
        got = list(zip(pi, pj))

        assert got == expected, f"node {i}: pairs mismatch\n got={got}\n exp={expected}"


def test_random_graphs_against_reference():
    for seed in range(20):
        random.seed(seed)
        M = random.randint(2, 15)
        directed = random.choice([True, False])
        include_self = random.choice([True, False])

        edges = []
        for _ in range(random.randint(0, M * (M - 1))):
            a = random.randrange(M)
            b = random.randrange(M)
            if a != b:
                edges.append((a, b))

        layout = build_graph_layout(
            M, edges, directed=directed, include_self=include_self, device="cpu"
        )

        E = normalize_edges_ref(M, edges, directed=directed, include_self=include_self)
        R_ref, T_ref = build_ref_neighborhoods(M, E)

        for i in range(M):
            R_got = layout.R_idx[i][layout.R_mask[i]].cpu().tolist()
            T_got = layout.T_idx[i][layout.T_mask[i]].cpu().tolist()
            assert R_got == R_ref[i]
            assert T_got == T_ref[i]


# %%
if __name__ == "__main__":
    # %%
    print("Testing graph layout construction...")
    test_neighborhoods_match_edges()
    test_local_pairs_are_combinations_ref()
    test_random_graphs_against_reference()
    print("All tests passed.")

    # layout = build_graph_layout(
    #     M=5,
    #     edges=edges,
    #     directed=False,
    #     include_self=True,
    #     device="cpu",
    # )

    # print("M:", layout.M)
    # print("edges:\n", edges)
    # print("R_idx:\n", layout.R_idx)
    # print("T_idx:\n", layout.T_idx)
    # print("self_slot_in_R:\n", layout.self_slot_in_R)
    # print("pair_i_local:\n", layout.pair_i_local)
    # print("pair_j_local:\n", layout.pair_j_local)
    # plot_network(layout)
