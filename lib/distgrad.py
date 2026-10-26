from __future__ import annotations
import numpy as np
import scipy.linalg as sla
from time import process_time
from typing import List, Dict
from .simulation_utils import Graph


class DGDNode:
    id: int
    L: int
    alpha: float
    win_len: int
    N_t: int  # number of nodes transmitting to
    N_r: int  # number of channels on node
    transmit: List[DGDNode]  # list of nodes transmitting from this node
    receive: List[DGDNode]  # list of nodes transmitting to this node

    c: np.ndarray
    phi_: float
    psi_: float
    phi: np.ndarray
    psi: np.ndarray
    Ys = List[np.ndarray]

    signal: np.ndarray

    h: np.ndarray

    receive_indices: Dict[int, int]
    transmit_indices: Dict[int, int]

    def __init__(self, id: int, L: int, win_len: int = None) -> None:
        self.id = id
        self.L = L
        self.win_len = win_len if win_len is not None else 4 * L
        self.N_t = 0
        self.N_r = 0
        self.receive = []
        self.transmit = []
        self.receive_indices = {}
        self.transmit_indices = {}
        self.c = None
        self.h = None
        self.signal = None
        self.Ys = []

    def _constructYMatrix(self, i) -> np.ndarray:
        Y = sla.hankel(
            self.signal[:, i][::-1],
            self.signal[:, i][::-1],
        )
        return Y[: self.win_len - 2 * self.L, : self.L]

    def _setSignalFrame(self, signal, id):
        i = self.receive_indices[id]
        self.signal[:, i] = signal

    def _genYs(self):
        for r_node in self.receive:
            r_i = self.receive_indices[r_node.id]
            self.Ys[r_i] = self._constructYMatrix(r_i)

    def _distributedAveragingInit(self) -> None:
        s_i = self.receive_indices[self.id]
        self.phi_ = np.linalg.norm(self.h[:, s_i]) ** 2
        self.psi_ = 0
        for r_node in self.receive:
            r_i = self.receive_indices[r_node.id]
            self.psi_ += (
                np.linalg.norm(
                    self.Ys[r_i] @ self.h[:, s_i, np.newaxis]
                    - self.Ys[s_i] @ self.h[:, r_i, np.newaxis]
                )
                ** 2
            )

    def _distributedAveragingStep(self) -> None:
        self.phi_ = np.squeeze(self.c.T @ self.phi)
        self.psi_ = np.squeeze(self.c.T @ self.psi)

    def _updateEstimate(self) -> None:
        h_ = self.h[:, self.receive_indices[self.id], np.newaxis]
        e = self._getE()
        h_ += (
            -self.alpha * self.L**2 * self.phi_**2 * e
            + self.alpha * self.L / 2 * self.psi_ * h_
        )
        # self.h[:, self.receive_indices[self.id]] = (
        #     h_  # not sure if necessary, maybe it's a view already
        # )

    def _getE(self) -> np.ndarray:
        e = np.zeros((self.L, 1))
        s_i = self.receive_indices[self.id]
        for r_node in self.receive:
            r_i = self.receive_indices[r_node.id]
            e += (
                self.Ys[r_i].T @ self.Ys[r_i] @ self.h[:, s_i, np.newaxis]
                - self.Ys[r_i].T @ self.Ys[s_i] @ self.h[:, r_i, np.newaxis]
            )
        return e

    def getEstimate(self) -> np.ndarray:
        return self.h[:, self.receive_indices[self.id]]


class DGD:
    graph: Graph
    L: int
    M: int
    nodes: Dict[int, DGDNode]

    def __init__(self, graph: Graph, L: int, win_len: int, alpha: float) -> None:
        self.graph = graph
        self.L = L
        self.M = len(graph.vertices)
        self.nodes = {}
        for m in graph.vertices:
            node = DGDNode(m, L, win_len)
            node.L = L
            node.N_r = int(np.sum(graph.A[:, m]))
            node.N_t = int(np.sum(graph.A[m, :]))
            self.nodes[m] = node

        for e in graph.edges:
            self.nodes[e[0]].transmit.append(self.nodes[e[1]])
            self.nodes[e[1]].receive.append(self.nodes[e[0]])

        for node in self.nodes.values():
            for i, m in enumerate(np.where(self.graph.A[:, node.id])[0]):
                node.receive_indices[m] = i
            for i, m in enumerate(np.where(self.graph.A[node.id, :])[0]):
                node.transmit_indices[m] = i
            node.c = np.ones((node.N_r, 1)) / node.N_r
            node.phi = np.zeros((node.N_r, 1))
            node.psi = np.zeros((node.N_r, 1))
            node.h = np.zeros((node.L, node.N_r))
            node.h[::L] = 1
            node.signal = np.zeros((node.win_len, node.N_r))
            node.alpha = alpha
            node.Ys = [np.zeros((2 * L, L)) for _ in range(node.N_r)]

    def step(self, signal: np.ndarray, vad: np.ndarray, da_R: int = 50) -> float:
        if not np.any(vad):
            return 0
        st = process_time()

        self._setSignalFrame(signal)
        self._transmitSignals()
        self._generateYs()
        self._distributedAveragingInit()
        for _ in range(da_R):
            self._distributedAveragingStep
            self._transmitDistributedAvergingEstimates()
        self._updateEstimate()
        self._transmitEstimates()

        et = process_time()
        return et - st

    def _setSignalFrame(self, signal):
        for node in self.nodes.values():
            assert node.win_len == signal.shape[0]
            node.signal[:, node.receive_indices[node.id]] = signal[:, node.id]

    def _transmitSignals(self):
        for node in self.nodes.values():
            for transmit in node.transmit:
                transmit.signal[:, transmit.receive_indices[node.id]] = node.signal[
                    :, node.receive_indices[node.id]
                ]

    def _generateYs(self):
        for node in self.nodes.values():
            node._genYs()

    def _updateEstimate(self):
        for node in self.nodes.values():
            node._updateEstimate()

    def _transmitEstimates(self):
        for node in self.nodes.values():
            for transmit in node.transmit:
                transmit.h[:, transmit.receive_indices[node.id]] = node.h[
                    :, node.receive_indices[node.id]
                ]

    def _distributedAveragingInit(self):
        for node in self.nodes.values():
            node._distributedAveragingInit()

    def _distributedAveragingStep(self):
        for node in self.nodes.values():
            node._distributedAveragingStep()

    def _transmitDistributedAvergingEstimates(self) -> None:
        for node in self.nodes.values():
            for transmit in node.transmit:
                transmit.phi[transmit.receive_indices[node.id]] = node.phi_
                transmit.psi[transmit.receive_indices[node.id]] = node.psi_

    def getEstimate(self) -> np.ndarray:
        return np.concatenate(
            [np.fft.fft(node.getEstimate(), axis=0) for node in self.nodes.values()]
        ).reshape(-1, 1)
