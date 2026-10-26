from __future__ import annotations
import numpy as np
from time import process_time
from typing import Tuple, Dict, List, Callable
from .simulation_utils import computeOverlapSaveMatrices, VariableTracker, Graph
from .coding import Connection, Encoder, Decoder  # , ResidualEncoder, ResidualDecoder

SIGNAL = "sig"
LOCAL = "loc"
CONSENSUS = "con"

complextype = np.complex128


def _inexactNewton(node: Node):
    """
    Perform an inexact Newton iteration.

    Args:
        node (Node): The node object containing the necessary variables and parameters.

    Returns:
        None
    """
    CR = node.CR_sn + node.lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    V = 1 / (np.diagonal(CR).reshape(node.N_r * node.L, 1) + node.rho)
    node.x = node.x - node.mu * V * y


def _inexactNewtonSUB(node: Node):
    """
    Perform an inexact Newton iteration with covariance subtraction.

    Args:
        node (Node): The node object containing the necessary variables.

    Returns:
        None
    """
    CR = node.CR_sn - node.CR_n + node.lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    V = 1 / (np.diagonal(CR).reshape(node.N_r * node.L, 1) + node.rho)
    node.x = node.x - node.mu * V * y


def _inexactNewtonGEV(node: Node):
    """
    Perform an inexact Newton iteration solving the GEV problem.

    Args:
        node (Node): The node object containing the necessary data.

    Returns:
        None
    """
    a = node.x.conj().T @ node.CR_sn @ node.x
    b = node.x.conj().T @ node.CR_n @ node.x
    # node.sig = np.clip(
    #     (a + node.gev_reg) / (b + node.gev_reg), 0.0, node.gev_lim
    # ).squeeze()
    node.sig = np.core.umath.maximum(
        np.core.umath.minimum(
            (a + node.gev_reg) / (b + node.gev_reg).squeeze(), node.gev_lim
        ),
        0.0,
    )
    CR = node.CR_sn - node.sig * node.CR_n + node.lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    V = 1 / (np.diagonal(CR).reshape(node.N_r * node.L, 1) + node.rho)
    node.x = node.x - node.mu * V * y


def _BFGSGEVP(node: Node):
    """
    Perform an inexact Newton iteration solving the GEV problem.

    Args:
        node (Node): The node object containing the necessary data.

    Returns:
        None
    """
    # BFGS optimization method
    if node.first_iter:
        node.H = np.eye(node.x.shape[0], dtype=complextype)
        node.prev_grad = np.zeros_like(node.x)

    a = (
        node.x.conj().T @ node.CR_sn @ node.x
    )  # L*N_r^2 + L*N_r multiplications, L*N_r^2 - 1 additions
    b = (
        node.x.conj().T @ node.CR_n @ node.x
    )  # L*N_r^2 + L*N_r multiplications, L*N_r^2 - 1 additions
    node.sig = np.core.umath.maximum(
        np.core.umath.minimum(
            (a + node.gev_reg) / (b + node.gev_reg).squeeze(), node.gev_lim
        ),
        0.0,
    )

    CR = node.CR_sn - node.sig * node.CR_n  # L*N_r^2 additions
    lambd = node.lambd * np.trace(CR) / (np.trace(node.sigma_inv) * node.N_r)
    CR = CR + lambd * node.sigma_inv  # L*N_r^2 additions
    y = (
        CR @ node.x + node.u + node.rho * (node.x - node.B * node.z)
    )  # L*N_r^2 multiplications, L*N_r^2 + 3*L*N_r

    grad = y
    if node.first_iter:
        direction = -grad
        node.first_iter = False
    else:
        direction = -node.H @ grad  # L*N_r^2 multiplications, L*N_r^2 - 1 additions

    # Update step size (simple fixed step)
    x_new = node.x + node.mu * direction  # L*N_r additions

    y_new = (
        CR @ x_new + node.u + node.rho * (x_new - node.B * node.z)
    )  # L*N_r^2 multiplications, L*N_r^2 + 3*L*N_r additions
    grad_new = y_new

    s = x_new - node.x  # L*N_r additions
    y_diff = grad_new - grad  # L*N_r additions

    # ys = np.vdot(y_diff, s)  # L*N_r^2 multiplications
    # if ys > 1e-10:
    #     rho = 1.0 / ys
    #     I = np.eye(node.H.shape[0], dtype=complextype)
    #     A = I - rho * np.outer(
    #         s, y_diff.conj()
    #     )  # L*N_r^2 multiplications + L*N_r^2 additions
    #     node.H = A @ node.H @ A.conj().T + rho * np.outer(
    #         s, s.conj()
    #     )  # 2* L*N_r^3 + L*N_r^2 multiplications + L*N_r^2 additions
    # sHs = np.vdot(s, node.H @ s)  # L*N_r^2 + L*N_r multiplications, L*N_r^2 - 1 additions
    # ys = np.vdot(y_diff, s)  # L*N_r multiplications + L*N_r - 1 additions
    # if ys > 1e-10 and sHs > 1e-10:
    #     a = node.H @ s  # L*N_r^2 multiplications, L*N_r^2 - 1 additions
    #     b = s.conj().T @ node.H  # L*N_r^2 multiplications, L*N_r^2 - 1 additions
    #     A = np.outer(a, b) / sHs  # L*N_r^2 multiplications, L*N_r^2 additions
    #     node.H = (
    #         node.H - A + np.outer(s, s.conj()) / ys
    #     )  # 2*L*N_r^2 multiplications, 2*L*N_r^2 additions
    gamma = np.vdot(s, y_diff)  # L*N_r^2 multiplications, L*N_r^2 - 1 additions
    if gamma > 1e-10:
        a = node.H @ y_diff  # L*N_r^2 multiplications, L*N_r^2 - 1 additions
        b = y_diff.conj().T @ node.H  # L*N_r^2 multiplications, L*N_r^2 - 1 additions
        c = np.vdot(y_diff, a) / gamma  # L*N_r multiplications, L*N_r additions
        W = np.outer(s, s.conj()) / gamma  # L*N_r^2 multiplications
        node.H = (
            node.H - np.outer(a, s.conj()) / gamma - np.outer(s, b) / gamma + (c + 1) * W
        )  # 3*L*N_r^2 multiplications, 3*L*N_r^2 additions

    node.x = x_new
    node.prev_grad = grad_new


def _LSGEVP(node: Node):
    """
    Perform an inexact Newton iteration solving the GEV problem.

    Args:
        node (Node): The node object containing the necessary data.

    Returns:
        None
    """
    a = node.x.conj().T @ node.CR_sn @ node.x
    b = node.x.conj().T @ node.CR_n @ node.x
    # node.sig = np.clip(
    #     (a + node.gev_reg) / (b + node.gev_reg), 0.0, node.gev_lim
    # ).squeeze()
    node.sig = np.core.umath.maximum(
        np.core.umath.minimum(
            (a + node.gev_reg) / (b + node.gev_reg).squeeze(), node.gev_lim
        ),
        0.0,
    )

    CR = node.CR_sn - node.sig * node.CR_n
    lambd = node.lambd * np.trace(CR) / (np.trace(node.sigma_inv) * node.N_r)
    CR = CR + lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    # y = y + (node.u + node.rho * (node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    # V = 1 / (np.diagonal(CR).reshape(node.N_r * node.L, 1) + node.rho)
    V = np.linalg.inv(CR + np.eye(node.N_r * node.L))
    # V = 1
    node.x = node.x - node.mu * V @ y


def _inexactNewtonGEVP(node: Node):
    """
    Perform an inexact Newton iteration solving the GEV problem.

    Args:
        node (Node): The node object containing the necessary data.

    Returns:
        None
    """
    a = node.x.conj().T @ node.CR_sn @ node.x
    b = node.x.conj().T @ node.CR_n @ node.x
    # node.sig = np.clip(
    #     (a + node.gev_reg) / (b + node.gev_reg), 0.0, node.gev_lim
    # ).squeeze()
    node.sig = np.core.umath.maximum(
        np.core.umath.minimum(
            (a + node.gev_reg) / (b + node.gev_reg).squeeze(), node.gev_lim
        ),
        0.0,
    )

    CR = node.CR_sn - node.sig * node.CR_n
    lambd = node.lambd * np.trace(CR) / (np.trace(node.sigma_inv) * node.N_r)
    CR = CR + lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    # y = y + (node.u + node.rho * (node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    V = 1 / (np.diagonal(CR).reshape(node.N_r * node.L, 1) + node.rho)
    # V = 1
    node.x = node.x - node.mu * V * y


def _inexactNewtonGEVPA(node: Node):
    """
    Perform an inexact Newton iteration solving the GEV problem.

    Args:
        node (Node): The node object containing the necessary data.

    Returns:
        None
    """
    a = node.x.conj().T @ node.CR_sn @ node.x
    b = node.x.conj().T @ node.CR_n @ node.x
    # node.sig = np.clip(
    #     (a + node.gev_reg) / (b + node.gev_reg), 0.0, node.gev_lim
    # ).squeeze()
    node.sig = np.core.umath.maximum(
        np.core.umath.minimum(
            (a + node.gev_reg) / (b + node.gev_reg).squeeze(), node.gev_lim
        ),
        0.0,
    )

    CR = node.CR_sn - node.sig * node.CR_n
    lambd = node.lambd * np.trace(CR) / (np.trace(node.sigma_inv) * node.N_r)
    CR = CR + lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    # y = y + (node.u + node.rho * (node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    V = 1 / (np.diagonal(CR).reshape(node.N_r * node.L, 1) + node.rho)
    node.vel = node.vel * 0.9 + 0.1 * V * y
    # V = 1
    node.x = node.x - node.mu * node.vel


def _gradientDescGEVP(node: Node):
    """
    Perform an inexact Newton iteration solving the GEV problem.

    Args:
        node (Node): The node object containing the necessary data.

    Returns:
        None
    """
    a = node.x.conj().T @ node.CR_sn @ node.x
    b = node.x.conj().T @ node.CR_n @ node.x
    # node.sig = np.clip(
    #     (a + node.gev_reg) / (b + node.gev_reg), 0.0, node.gev_lim
    # ).squeeze()
    node.sig = np.core.umath.maximum(
        np.core.umath.minimum(
            (a + node.gev_reg) / (b + node.gev_reg).squeeze(), node.gev_lim
        ),
        0.0,
    )

    CR = node.CR_sn - node.sig * node.CR_n
    lambd = node.lambd * np.trace(CR) / (np.trace(node.sigma_inv) * node.N_r)
    CR = CR + lambd * node.sigma_inv
    y = CR @ node.x  # + node.A.T @ (node.u + node.rho * (node.A @ node.x - node.z))
    # y = y + (node.u + node.rho * (node.x - node.z))
    y = y + (node.u + node.rho * (node.x - node.B * node.z))

    node.x = node.x - node.mu * y


class Node:
    """
    Node of a sensor network.
    """

    id: int  # node/vertex id
    L: int  # filter length
    N_t: int  # number of nodes transmitting to
    N_r: int  # number of channels on node

    transmit: List[Connection]  # list of nodes transmitting from this node
    receive: List[Connection]  # list of nodes transmitting to this node

    signal_encoder: Encoder
    signal_decoder: Dict[int, Decoder]
    local_estimate_encoder: Dict[int, Encoder]
    local_estimate_decoder: Dict[int, Decoder]
    consensus_estimate_encoder: Encoder
    consensus_estimate_decoder: Dict[int, Decoder]

    receive_indices: Dict[int, int]
    receive_index_ranges: Dict[int, range]
    transmit_indices: Dict[int, int]

    M: int  # number of nodes in network
    rho: float  # ADMM penalty parameter
    tau: float  # ADMM penalty parameter scaling factor
    mu: float  # local update step size
    eta: float  # forgetting factor for covariance estimation
    lambd: float  # regularization parameter
    # beta: float  # nesterov acceleration factor

    block: np.ndarray  # signal block (frequency domain)
    x: np.ndarray  # local estimates (primal)
    z: np.ndarray  # consensus estimate (primal)
    z_old: np.ndarray  # consensus estimate (primal)
    primal_res: np.ndarray  # primal residual
    u: np.ndarray  # lagrange multiplier (dual)
    # dual_res: np.ndarray  # dual residual
    y: np.ndarray  # combined local variables (primal + dual)
    CR_sn: np.ndarray  # signal+noise cross-relation matrix estimate
    CR_n: np.ndarray  # noise-only cross-relation matrix estimate

    sigma_inv: np.ndarray  # prior estimate covariance
    A: np.ndarray  # constraint mapping
    B: np.ndarray  # alignment matrix
    B_: np.ndarray  # alignment matrix, reshaped

    vad: bool  # voice activity detection status

    w_da: np.ndarray  # weights for distributed averaging
    n_inst: np.ndarray  # instantaneous norm estimate for distributed averaging
    phi_da: np.ndarray  # norm estimates for distributed averaging
    phi_da_: float  # norm estimate for distributed averaging

    variable_trackers: VariableTracker  # variable tracker
    localStep: Callable  # local update function
    # updatePenalty: Callable  # penalty update function
    vel: np.ndarray  # velocity for gradient descent
    first_iter: bool  # first iteration flag

    def __init__(self, id: int):
        self.id = id
        self.receive = []
        self.transmit = []
        self.variable_trackers = None
        self.sigma_inv = 0
        # self.beta = 0
        # self.updatePenalty = lambda *args: None

    def _resetBuffers(self):
        self.block = np.zeros(shape=(self.L * 2, self.N_r), dtype=complextype)
        self.x = np.ones(shape=(self.L * self.N_r, 1), dtype=complextype) / self.L
        self.u = np.zeros(shape=(self.L * self.N_r, 1), dtype=complextype)
        self.z = np.zeros(shape=(self.L * self.N_r, 1), dtype=complextype)
        self.y = np.zeros_like(self.z)
        self.y_r = np.zeros((self.L, self.N_t), dtype=complextype)
        self.z[self.receive_index_ranges[self.id]] = 1 / np.sqrt(self.L)
        self.B = np.ones((self.N_r * self.L, 1), dtype=complextype)
        self.B_ = self.B.reshape(
            self.N_r, self.L
        ).T  # create a view for some multiplications

        self.w_da = np.ones((1, self.N_r)) / self.N_r
        self.n_inst = np.zeros((self.N_r, 1))
        self.phi_da = np.ones((self.N_r, 1)) / self.M
        self.phi_da_ = 1 / self.M

        self.primal_res = np.zeros_like(self.z)

        self.CR_sn = np.zeros((self.L * self.N_r, self.L * self.N_r), dtype=complextype)
        self.CR_n = np.zeros((self.L * self.N_r, self.L * self.N_r), dtype=complextype)
        self.R = np.zeros((self.L * self.N_r, self.L * self.N_r), dtype=complextype)
        self.vel = np.zeros_like(self.x)
        self.first_iter = True
        self.H = np.eye(self.x.shape[0], dtype=complextype)

    def setDelay(self, delay: np.int64):
        k = np.arange(self.L)
        for i in range(self.N_r):
            self.B_[:, i] = np.exp(-1j * 2 * np.pi * delay * k / self.L)

    def _setSignalFrame(self, signal):
        i = self.receive_indices[self.id]
        self.block[:, i] = signal

    def _encodeSignal(self) -> np.ndarray:
        i = self.receive_indices[self.id]
        data = self.block[:, i, None]
        encoded = self.signal_encoder.apply(data)
        return encoded

    def _decodeSignal(self, from_node: int, encoded: np.ndarray):
        decoded = self.signal_decoder[from_node].apply(encoded)
        i = self.receive_indices[from_node]
        self.block[:, i, None] = decoded

    def _encodeLocal(self, to_node: int) -> np.ndarray:  # , transmission_indices):
        ind = self.receive_index_ranges[to_node]
        i = self.receive_indices[self.id]
        data = self.B_[:, i, None].conj() * self.y[ind]
        # data = self.y[ind]
        # self.local_estimate_encoder[to_node].transmission_indices = transmission_indices
        encoded = self.local_estimate_encoder[to_node].apply(data)
        return encoded

    def _decodeLocal(
        self, from_node: int, encoded: np.ndarray
    ):  # , transmission_indices):
        # self.local_estimate_decoder[from_node].transmission_indices = transmission_indices
        decoded = self.local_estimate_decoder[from_node].apply(encoded)
        i = self.transmit_indices[from_node]
        self.y_r[:, i, None] = decoded

    # def _getLocalTransmissionIndices(self, to_node: int, n: int):
    #     ind = self.receive_index_ranges[to_node]
    #     res = np.abs(self.primal_res[ind]) ** 2
    #     transmission_indices = np.argpartition(res, -n, axis=0).squeeze()[-n:]
    #     return transmission_indices

    def _encodeConsensus(self) -> np.ndarray:  # , transmission_indices):
        ind = self.receive_index_ranges[self.id]
        data = self.z[ind]
        # self.consensus_estimate_encoder.transmission_indices = transmission_indices
        encoded = self.consensus_estimate_encoder.apply(data)
        return encoded

    def _decodeConsensus(self, from_node, encoded):  # , transmission_indices):
        ind = self.receive_index_ranges[from_node]
        # self.consensus_estimate_decoder[from_node].transmission_indices = (
        #     transmission_indices
        # )
        decoded = self.consensus_estimate_decoder[from_node].apply(encoded)
        self.z[ind] = decoded

    # def _getConsensusTransmissionIndices(self, n: int):
    #     ind = self.receive_index_ranges[self.id]
    #     res = np.abs(self.dual_res[ind])
    #     transmission_indices = np.argpartition(res, -n, axis=0).squeeze()[-n:]
    #     # transmission_indices = np.array(range(self.L))
    #     return transmission_indices

    def _updateLocalEstimate(self) -> None:
        self.localStep(self)
        self.y = self.x + 1 / self.rho * self.u
        # self.primal_res = self.rho * (self.x - self.z)

    def _updateSignalPlusNoiseCRMatrix(self) -> None:
        self._constructCrossRelationMatrix(self.CR_sn)
        # self.CR_sn = self.eta * self.CR_sn + (1 - self.eta) * self.R

    def _updateNoiseOnlyCRMatrix(self) -> None:
        self._constructCrossRelationMatrix(self.CR_n)
        # self.CR_n = self.eta * self.CR_n + (1 - self.eta) * self.R

    def _constructCrossRelationMatrix(self, R):
        # R: np.ndarray = np.zeros(
        #     shape=(self.L * self.N_r, self.L * self.N_r), dtype=complextype
        # )
        R *= self.eta
        for i in range(self.N_r):
            for j in range(i, self.N_r):
                R_ij = self._computeRij(i, j)
                R_ij *= 1.0 - self.eta
                if i != j:
                    idx_i = slice(i * self.L, (i + 1) * self.L)
                    idx_j = slice(j * self.L, (j + 1) * self.L)
                    R[idx_j, idx_i] += -R_ij
                    R[idx_i, idx_j] += -R_ij.conj().T
                else:
                    idx_diag = [n for n in range(self.N_r) if n != i]
                    for n in idx_diag:
                        idx_n = slice(n * self.L, (n + 1) * self.L)
                        R[idx_n, idx_n] += R_ij

    def _computeRij(self, i: int, j: int) -> np.ndarray:
        # D_i = np.diagonal(self.block[:, i])
        # D_j = np.diagonal(self.block[:, j])
        # R_ij = self.W_10_Lx2L_fq @ D_i.conj().T @ self.W_R @ D_j @ self.W_10_2LxL_fq
        R_ij: np.ndarray = (
            self.W_10_Lx2L_fq
            @ ((self.block[:, i, None].conj() * self.W_R).T * self.block[:, j, None]).T
            @ self.W_10_2LxL_fq
        )
        return R_ij

    def _updateConsensusEstimate(self):
        ind = self.receive_index_ranges[self.id]
        i = self.transmit_indices[self.id]
        self.y_r[:, i, None] = self.y[ind]
        self.z_old = self.z.copy()
        self.z[ind] = self.y_r.mean(1, keepdims=True)
        # self.dual_res = self.rho * (self.z - self.z_old)

    def _normalizeConsensusEstimate(self):
        n_inst = np.zeros_like(self.n_inst)
        for id, i in self.receive_indices.items():
            ind = self.receive_index_ranges[id]
            n_inst[i, 0] = np.linalg.norm(self.z[ind]) ** 2
        gamma = np.linalg.norm(self.n_inst - n_inst) / (
            np.linalg.norm(self.n_inst) + np.linalg.norm(n_inst)
        )
        self.n_inst = n_inst.copy()
        self.phi_da_ = (
            self.w_da @ (gamma * self.n_inst + (1 - gamma) * self.phi_da)
        ).item()
        ind = self.receive_index_ranges[self.id]
        self.z[ind] = self.z[ind] / np.sqrt(self.phi_da_ * self.M)

    # def _accelerateConsensusEstimate(self):
    #     ind = self.receive_index_ranges[self.id]
    #     self.z[ind] = self.z[ind] + self.beta * (self.z[ind] - self.z_old[ind])

    def _updateLagrangeMultiplier(self):
        # self.u = self.u + self.rho * (self.x[self.A] - self.z)
        # self.primal_res = self.rho * (self.x - self.z)
        self.primal_res = self.rho * (self.x - self.B * self.z)
        self.u = self.u + self.primal_res

    def getEstimate(self) -> np.ndarray:
        return self.x[self.receive_index_ranges[self.id]]


class Network:
    graph: Graph
    L: int  # filter length
    M: int  # number of nodes (channels)
    nodes: Dict[int, Node]  # set of nodes
    # variable_trackers: VariableTracker  # list of variable trackers
    connections: dict[Tuple[int, int], Connection]  # set of connections

    local_iters: int
    frame_iters: int
    da: bool  # use distributed averaging or not

    consensus_type: str  # consensus type
    L_prime: int  # consensus length
    # transmission_indices: np.ndarray  # consensus transmission indices

    def __init__(self, graph: Graph, L: int):
        self.graph = graph
        self.nodes = {}
        self.connections = {}
        self.L = L
        self.M = len(graph.vertices)
        # self.variable_trackers = VariableTracker()
        self.local_iters = 1
        self.frame_iters = 1
        self.da = False
        self.consensus_type = "regular"
        self.L_prime = L
        # self.transmission_indices = np.arange(0, L, L // self.L_prime)
        for m in graph.vertices:
            node = Node(m)
            node.L = L
            node.N_r = int(np.sum(graph.A[:, m]))
            node.N_t = int(np.sum(graph.A[m, :]))
            node.M = self.M
            self.nodes[m] = node

        self._setEncoderDecoders()

        for node in self.nodes.values():
            self._setIndexRanges(node)
            # node._resetBuffers()

        # self.onTransmitCallback: Callable = None

    def _setEncoderDecoders(self):
        for node in self.nodes.values():
            # node.signal_encoder = ResidualEncoder(node, SIGNAL, 2 * self.L)
            node.signal_encoder = Encoder(node, SIGNAL)
            node.signal_decoder = {}
            node.local_estimate_encoder = {}
            node.local_estimate_decoder = {}
            # node.consensus_estimate_encoder = ResidualEncoder(node, CONSENSUS, node.L)
            node.consensus_estimate_encoder = Encoder(node, CONSENSUS)
            node.consensus_estimate_decoder = {}

        for e in self.graph.edges:
            connection = Connection(e, self.nodes[e[0]], self.nodes[e[1]])
            tr: Node = connection.transmitter
            rc: Node = connection.receiver
            # rc.signal_decoder[tr.id] = ResidualDecoder(rc, SIGNAL, 2 * self.L)
            rc.signal_decoder[tr.id] = Decoder(rc, SIGNAL)
            # rc.local_estimate_encoder[tr.id] = ResidualEncoder(rc, LOCAL, rc.L)
            # tr.local_estimate_decoder[rc.id] = ResidualDecoder(tr, LOCAL, tr.L)
            # rc.consensus_estimate_decoder[tr.id] = ResidualDecoder(rc, CONSENSUS, rc.L)
            rc.local_estimate_encoder[tr.id] = Encoder(rc, LOCAL)
            tr.local_estimate_decoder[rc.id] = Decoder(tr, LOCAL)
            rc.consensus_estimate_decoder[tr.id] = Decoder(rc, CONSENSUS)
            self.connections[e] = connection

    def _setIndexRanges(self, node: Node):
        node.receive_indices = {}
        node.receive_index_ranges = {}
        for i, m in enumerate(np.where(self.graph.A[:, node.id])[0]):
            node.receive_indices[m] = i
            node.receive_index_ranges[m] = range(i * node.L, (i + 1) * node.L)

        node.transmit_indices = {}
        for i, m in enumerate(np.where(self.graph.A[node.id, :])[0]):
            node.transmit_indices[m] = i

    def reset(self):
        W_10_Lx2L_fq, W_R, W_10_2LxL_fq = computeOverlapSaveMatrices(self.L)
        for node in self.nodes.values():
            node._resetBuffers()
            node.W_10_Lx2L_fq = W_10_Lx2L_fq
            node.W_R = W_R
            node.W_10_2LxL_fq = W_10_2LxL_fq
        # self.resetTrackers()

    def step(self, signal: np.ndarray, vad: bool) -> float:

        signal_f = np.fft.fft(signal, axis=0)
        self._setSignalFrame(signal_f)
        self._transmitSignals()
        self._setVadStatus(vad)
        self._updateSignalPlusNoiseCRMatrix()
        self._updateNoiseOnlyCRMatrix()

        st = process_time()
        for _ in range(self.frame_iters):
            for _ in range(self.local_iters):
                self._updateLocalEstimates()
            self._transmitLocal()
            self._updateConsensusEstimates()
            self._normalizeConsensusEstimates()
            self._transmitNormEst()
            # self._accelerateConsensusEstimates()
            # self._normalizeConsensusEstimates()
            self._transmitConsensus()
            self._updateLagrangeMultipliers()
        et = process_time()

        # self._trackVariables()

        return (et - st) / self.M

    def _setSignalFrame(self, signal: np.ndarray):
        for node in self.nodes.values():
            node._setSignalFrame(signal[:, node.id])

    def _transmitSignals(self):
        encoded = {}
        for connection in self.connections.values():
            tr: Node = connection.transmitter
            rc: Node = connection.receiver
            if tr.id == rc.id:
                continue
            if tr.id not in encoded:
                encoded[tr.id] = tr._encodeSignal()
            rc._decodeSignal(tr.id, encoded[tr.id])

    def _setVadStatus(self, vad: np.ndarray):
        for node in self.nodes.values():
            node.vad = vad[node.id]

    def _transmitLocal(self):
        for connection in self.connections.values():
            tr: Node = connection.transmitter
            rc: Node = connection.receiver
            if tr.id == rc.id or not tr.vad or not rc.vad:
                continue
            # match self.consensus_type:
            #     case "regular":
            #         self.transmission_indices = np.mod(
            #             self.transmission_indices + 1, self.L
            #         )
            #         connection.transmission_indices = self.transmission_indices
            #     case "block":
            #         self.transmission_indices = np.mod(
            #             self.transmission_indices + self.L_prime, self.L
            #         )
            #         connection.transmission_indices = self.transmission_indices
            #     case "random":
            #         self.transmission_indices = np.random.choice(
            #             self.L, self.L_prime, replace=False
            #         )
            #         connection.transmission_indices = self.transmission_indices
            #     case "max_res":
            #         connection.transmission_indices = rc._getLocalTransmissionIndices(
            #             tr.id, self.L_prime
            #         )
            encoded = rc._encodeLocal(tr.id)
            # encoded = rc._encodeLocal(tr.id, connection.transmission_indices)
            tr._decodeLocal(rc.id, encoded)
            # tr._decodeLocal(rc.id, encoded, connection.transmission_indices)

    def _transmitConsensus(self):
        encoded = {}
        for connection in self.connections.values():
            tr: Node = connection.transmitter
            rc: Node = connection.receiver
            if tr.id == rc.id or not tr.vad or not rc.vad:
                continue
            if tr.id not in encoded:
                # encoded[tr.id] = tr._encodeConsensus(connection.transmission_indices)
                encoded[tr.id] = tr._encodeConsensus()
            # rc._decodeConsensus(tr.id, encoded[tr.id], connection.transmission_indices)
            rc._decodeConsensus(tr.id, encoded[tr.id])

    def _transmitNormEst(self):
        if self.da:
            for connection in self.connections.values():
                tr: Node = connection.transmitter
                rc: Node = connection.receiver
                rc_ind = rc.receive_indices[tr.id]
                rc.phi_da[rc_ind, 0] = tr.phi_da_

    def _updateSignalPlusNoiseCRMatrix(self):
        for node in self.nodes.values():
            if node.vad:
                node._updateSignalPlusNoiseCRMatrix()

    def _updateNoiseOnlyCRMatrix(self):
        for node in self.nodes.values():
            if not node.vad:
                node._updateNoiseOnlyCRMatrix()

    def _updateLocalEstimates(self):
        for node in self.nodes.values():
            if node.vad:
                node._updateLocalEstimate()

    # def setConsensusParameters(self, type: str, L_prime: int = -1):
    #     self.consensus_type = type
    #     if L_prime == -1:
    #         self.L_prime = self.L
    #     else:
    #         self.L_prime = L_prime
    #     match type:
    #         case "regular":
    #             self.transmission_indices = np.arange(0, self.L, self.L // self.L_prime)
    #             print(self.transmission_indices)
    #         case "block":
    #             self.transmission_indices = np.arange(0, self.L_prime)
    #         case "random":
    #             self.transmission_indices = np.random.choice(
    #                 self.L, self.L_prime, replace=False
    #             )
    #         case "max_res":
    #             self.transmission_indices = np.arange(0, self.L_prime)
    #         case _:
    #             raise ValueError("Invalid consensus type")

    def setDelays(self, delays: List[np.int64]):
        for node_id, delay in enumerate(delays):
            self.nodes[node_id].setDelay(delay)

    def setPriorInformation(self, sigma: np.ndarray):
        inv = np.linalg.inv(sigma)
        for node in self.nodes.values():
            node.sigma_inv = np.kron(np.eye(node.N_r), inv)

    def _updateConsensusEstimates(self):
        for node in self.nodes.values():
            if node.vad:
                node._updateConsensusEstimate()

    def _accelerateConsensusEstimates(self):
        for node in self.nodes.values():
            if node.vad:
                node._accelerateConsensusEstimate()

    def _normalizeConsensusEstimates(self):
        if self.da:
            for node in self.nodes.values():
                if node.vad:
                    node._normalizeConsensusEstimate()
        else:  # This is global norm, e.g. broadcasting
            n_z = 0
            for node in self.nodes.values():
                ind = node.receive_index_ranges[node.id]
                n_z += np.linalg.norm(node.z[ind]) ** 2
            n_z = np.sqrt(n_z)
            for node in self.nodes.values():
                if node.vad:
                    ind = node.receive_index_ranges[node.id]
                    node.z[ind] = node.z[ind] / n_z

    def _updateLagrangeMultipliers(self):
        for node in self.nodes.values():
            if node.vad:
                node._updateLagrangeMultiplier()

    def setParameters(self, parameters: Dict, node_id: int = -1):
        if node_id != -1:
            node = self.nodes[node_id]
            for key, val in parameters.items():
                setattr(node, key, val)
        for node in self.nodes.values():
            for key, val in parameters.items():
                setattr(node, key, val)

    # def setVariableTrackers(self, variables: List[str], node_id: int = -1):
    #     if node_id != -1:
    #         node = self.nodes[node_id]
    #         tracker = VariableTracker(node)
    #         for var in variables:
    #             tracker.addVariable(var)
    #         node.variable_trackers = tracker
    #     else:
    #         for node in self.nodes.values():
    #             tracker = VariableTracker(node)
    #             for var in variables:
    #                 tracker.addVariable(var)
    #             node.variable_trackers = tracker

    # def _trackVariables(self):
    #     for node in self.nodes.values():
    #         if node.variable_trackers is not None:
    #             node.variable_trackers.trackVariables()

    # def resetTrackers(self):
    #     for node in self.nodes.values():
    #         if node.variable_trackers is not None:
    #             node.variable_trackers.reset()

    # def getTrackedVariable(self, var_name: str, node_id: int = -1) -> List:
    #     if node_id != -1:
    #         node = self.nodes[node_id]
    #         if var_name in node.variable_trackers.variables:
    #             return node.variable_trackers.tracked_variables[var_name]
    #         else:
    #             return []
    #     else:
    #         tracked_vars = []
    #         for node in self.nodes.values():
    #             if var_name in node.variable_trackers.variables:
    #                 tracked_vars.append(
    #                     node.variable_trackers.tracked_variables[var_name]
    #                 )
    #         return tracked_vars

    def getEstimate(self) -> np.ndarray:
        return np.concatenate(
            [node.getEstimate() for node in self.nodes.values()],
            axis=0,
        )
