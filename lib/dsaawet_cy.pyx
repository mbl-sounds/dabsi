from __future__ import annotations
import numpy as np
import scipy.linalg as sla
from time import process_time
from typing import List, Dict
from .simulation_utils_cy import Graph


class DSAAWETNode:
    id: int
    L: int
    alpha: float
    eta: float
    win_len: int
    N_t: int  # number of nodes transmitting to
    N_r: int  # number of channels on node
    transmit: List[DSAAWETNode]  # list of nodes transmitting from this node
    receive: List[DSAAWETNode]  # list of nodes transmitting to this node

    sigma: int
    sigmas: List[int]

    Phi: np.ndarray
    Psi: np.ndarray
    Ys: Dict[int, np.ndarray]

    signal: np.ndarray

    h: np.ndarray
    h_init: np.ndarray

    vad: bool

    receive_indices: Dict[int, int]
    transmit_indices: Dict[int, int]

    def __init__(
        self,
        id: int,
        L: int,
        M: int,
        h_init: np.ndarray,
    ) -> None:
        self.id = id
        self.L = L
        self.M = M
        self.N_t = 0
        self.N_r = 0
        self.receive = []
        self.transmit = []
        self.receive_indices = {}
        self.transmit_indices = {}
        self.h = None
        self.h_init = h_init
        self.signal = None
        self.Phi = None
        self.Psi = None
        self.Ys = {}
        self.sigma = 0
        self.sigmas = []

    def _setSignalFrame(self, signal, id):
        i = self.receive_indices[id]
        self.signal[:, i] = signal

    def _constructYMatrix(self, i) -> np.ndarray:
        Y = sla.hankel(
            self.signal[:, i][::-1][: self.L],
            self.signal[:, i][::-1][self.L - 1 : self.L * 2 - 1],
        )
        return Y

    def _constructPhi(self):
        Phi_ = np.zeros((self.L * self.M * (self.M - 1), self.L * self.M))
        row = 0
        col = 0
        for m, i in self.receive_indices.items():
            self.Ys[m] = self._constructYMatrix(i)
        for i in range(self.M - 1):
            for j in range(i + 1, self.M):
                if j in self.receive_indices.keys() and i in self.receive_indices.keys():
                    Phi_[
                        row * self.L : (row + 1) * self.L,
                        col * self.L : (col + 1) * self.L,
                    ] = self.Ys[j]
                    Phi_[
                        row * self.L : (row + 1) * self.L,
                        (j) * self.L : (j + 1) * self.L,
                    ] = -self.Ys[i]
                row += 1
            col += 1
        eta = 0.98
        Phi_ = Phi_.T @ Phi_
        self.Phi = self.Phi * eta + Phi_ * (1 - eta)

    def _constructPsi(self):
        Psi_ = np.zeros((self.L * self.M * (self.M - 1), self.L * self.M))
        row = 0
        col = 0
        for m, i in self.receive_indices.items():
            self.Ys[m] = self._constructYMatrix(i)
        for i in range(self.M - 1):
            for j in range(i + 1, self.M):
                if j in self.receive_indices.keys() and i in self.receive_indices.keys():
                    Psi_[
                        row * self.L : (row + 1) * self.L,
                        col * self.L : (col + 1) * self.L,
                    ] = self.Ys[j]
                    Psi_[
                        row * self.L : (row + 1) * self.L,
                        (j) * self.L : (j + 1) * self.L,
                    ] = -self.Ys[i]
                row += 1
            col += 1
        eta = 0.98
        Psi_ = Psi_.T @ Psi_
        self.Psi = self.Psi * eta + Psi_ * (1 - eta)

    def _updateEstimate(self) -> None:
        h_ = self.h[:, self.receive_indices[self.id], np.newaxis]
        a = 1 / self.N_r
        sigma = np.max(self.sigmas)
        if self.sigmas[self.receive_indices[self.id]] < sigma:
            h_prime = self.h_init.copy()
        else:
            h_n = np.zeros((self.L * self.M, 1))
            for r_node in self.receive:
                r_i = self.receive_indices[r_node.id]
                sigma_j = self.sigmas[r_i]
                h__ = self.h[:, r_i, np.newaxis] if sigma_j >= sigma else self.h_init
                h_n += a * h__
            h_prime = h_n - self.alpha * (self.Phi - self.Psi) @ h_

        n_h_prime = np.linalg.norm(h_prime)
        if n_h_prime > 1:
            h_ = self.h_init.copy()
            self.sigma = sigma + 1
        else:
            h_ = h_prime
        self.h[:, self.receive_indices[self.id], np.newaxis] = h_

    def getEstimate(self) -> np.ndarray:
        return self.h[
            self.id * self.L : (self.id + 1) * self.L,
            self.receive_indices[self.id],
        ]


class DSAAWET:
    graph: Graph
    L: int
    M: int
    nodes: Dict[int, DSAAWETNode]

    def __init__(
        self,
        graph: Graph,
        L: int,
        h_init: np.ndarray,
        win_len: int,
        alpha: float,
    ) -> None:
        self.graph = graph
        self.L = L
        self.M = len(graph.vertices)
        self.nodes = {}
        for m in graph.vertices:
            node = DSAAWETNode(m, L, self.M, h_init)
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
            node.h = np.zeros((node.L * self.M, node.N_r))
            node.h[::L] = 1
            node.signal = np.zeros((win_len, node.N_r))
            node.alpha = alpha
            node.sigma = 0
            node.sigmas = [0 for _ in range(node.N_r)]
            node.Psi = np.zeros((node.L * self.M, node.L * self.M))
            node.Phi = np.zeros((node.L * self.M, node.L * self.M))

    def step(self, signal: np.ndarray, vad: np.ndarray) -> float:
        st = process_time()

        self._setSignalFrame(signal)
        self._setVadStatus(vad)
        self._transmitSignals()
        self._constructPhi()
        self._constructPsi()
        self._updateEstimate()
        self._transmitEstimates()
        self._transmitSigmas()

        et = process_time()
        return et - st

    def _setSignalFrame(self, signal):
        for node in self.nodes.values():
            # assert node.win_len == signal.shape[0]
            node.signal[:, node.receive_indices[node.id]] = signal[:, node.id]

    def _setVadStatus(self, vad):
        for node in self.nodes.values():
            node.vad = vad[node.id]

    def _transmitSignals(self):
        for node in self.nodes.values():
            for transmit in node.transmit:
                transmit.signal[:, transmit.receive_indices[node.id]] = node.signal[
                    :, node.receive_indices[node.id]
                ]

    def _constructPhi(self):
        for node in self.nodes.values():
            if node.vad:
                node._constructPhi()

    def _constructPsi(self):
        for node in self.nodes.values():
            if not node.vad:
                node._constructPsi()

    def _updateEstimate(self):
        for node in self.nodes.values():
            if node.vad:
                node._updateEstimate()

    def _transmitEstimates(self):
        for node in self.nodes.values():
            if node.vad:
                for transmit in node.transmit:
                    transmit.h[:, transmit.receive_indices[node.id]] = node.h[
                        :, node.receive_indices[node.id]
                    ]

    def _transmitSigmas(self) -> None:
        for node in self.nodes.values():
            # for receive in node.receive:
            #     receive.sigmas[receive.transmit_indices[node.id]] = node.sigma
            if node.vad:
                for transmit in node.transmit:
                    transmit.sigmas[transmit.receive_indices[node.id]] = node.sigma

    def getEstimate(self) -> np.ndarray:
        return np.concatenate(
            [np.fft.fft(node.getEstimate(), axis=0) for node in self.nodes.values()]
        ).reshape(-1, 1)
