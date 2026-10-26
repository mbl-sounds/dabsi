import numpy as np
from typing import List
from lib.simulation_utils_cy import Graph
from time import process_time


class MCALG:
    L: int  # filter length
    N: int  # number of channels
    buffer_size: int  # buffer size
    R_buffer: np.ndarray  # buffer for R_xx
    buffer_ind: int  # buffer index
    h: np.ndarray  # estimated impulse response

    def __init__(self, L: int, N: int, buffer_size: int = None) -> None:
        self.L = L
        self.N = N
        self.buffer_size = buffer_size if buffer_size is not None else L
        self.R_buffer = np.zeros(shape=(self.buffer_size, N, N, L, L))
        self.buffer_ind = 0

    def compute_X(self, x) -> float:
        X = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                e = (
                    x[:, i, None][::-1].T @ self.h[(j * self.L) : (j * self.L + self.L)]
                    - x[:, j, None][::-1].T @ self.h[(i * self.L) : (i * self.L + self.L)]
                )
                X += e**2
        return X

    def compute_X_prior(self, x) -> float:
        X = 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                e = (
                    x[:, i, None][::-1].T
                    @ self.h_old[(j * self.L) : (j * self.L + self.L)]
                    - x[:, j, None][::-1].T
                    @ self.h_old[(i * self.L) : (i * self.L + self.L)]
                )
                X += e**2
        return X

    def construct_Rxp(self, x):
        R_xp = np.zeros(shape=(self.L * self.N, self.L * self.N))
        for i in range(self.N):
            for j in range(self.N):
                R_xx = self.compute_Rxx(i, j, x)
                if i != j:
                    R_xp[
                        j * self.L : j * self.L + self.L,
                        i * self.L : i * self.L + self.L,
                    ] = -R_xx
                else:
                    for n in range(self.N):
                        if i != n:
                            R_xp[
                                n * self.L : n * self.L + self.L,
                                n * self.L : n * self.L + self.L,
                            ] += R_xx
        self.buffer_ind += 1
        if self.buffer_ind >= self.buffer_size:
            self.buffer_ind = 0
        return R_xp

    def compute_Rxx(self, i, j, x):
        self.R_buffer[self.buffer_ind, i, j, :, :] = (
            x[:, i, None][::-1] @ x[:, j, None][::-1].T
        )
        R_xx = np.mean(self.R_buffer[:, i, j, :, :], axis=0)
        return R_xx

    def getEstimateT(self) -> np.ndarray:
        return self.h

    def getEstimate(self) -> np.ndarray:
        h = np.zeros((self.L * self.N, 1), dtype=np.complex128)
        for i in range(self.N):
            h[i * self.L : (i + 1) * self.L] = np.fft.fft(
                self.h[i * self.L : (i + 1) * self.L],
                axis=0,
            )
        return h


class MCLMS(MCALG):
    def __init__(self, L: int, N: int, mu: float, buffer_size: int = None) -> None:
        MCALG.__init__(self, L, N, buffer_size)
        self.mu = mu
        self.h = np.zeros(shape=(L * N, 1))
        self.h[0::L] = 1
        self.h = self.h / np.sqrt(N)

    def step(self, x: np.ndarray, vad: np.ndarray) -> None:
        if not np.any(vad):
            return 0
        st = process_time()
        X = self.compute_X(x)  # error function Χ
        R_xp = self.construct_Rxp(x)  # construct matrix R_x+
        h_tilde = self.h - 2 * self.mu * (R_xp @ self.h - X * self.h)
        self.h = h_tilde / np.linalg.norm(h_tilde)
        et = process_time()
        return et - st


class VSSUCMCLMS(MCALG):
    def __init__(self, L: int, N: int, buffer_size: int = None) -> None:
        MCALG.__init__(self, L, N, buffer_size)
        self.h = np.ones(shape=(L * N, 1))

    def step(self, x: np.ndarray, vad: np.ndarray) -> None:
        if not np.any(vad):
            return 0
        st = process_time()
        R_xp = self.construct_Rxp(x)  # construct matrix R_x+

        delta_h = (2 * R_xp @ self.h) / (self.h.T @ self.h)
        mu = (self.h.T @ delta_h) / (delta_h.T @ delta_h)
        # update step
        self.h = self.h - mu * delta_h
        et = process_time()
        return et - st


class MCN(MCALG):
    def __init__(
        self,
        L: int,
        N: int,
        mu: float,
        eta: float,
        x_vars: np.ndarray = None,
        buffer_size: int = None,
    ) -> None:
        MCALG.__init__(self, L, N, buffer_size)
        self.mu = mu
        self.eta = eta
        self.h = np.zeros(shape=(L * N, 1))
        self.h[0::L] = 1
        self.h = self.h / np.sqrt(N)
        self.R_xp_ = (
            np.diag(np.repeat(x_vars, L))
            if x_vars is not None
            else np.eye(L * N) / (N * L)
        )

    def step(self, x: np.ndarray, vad: np.ndarray) -> float:
        if not np.any(vad):
            return 0
        X = self.compute_X(x)  # error function Χ
        R_xp = self.construct_Rxp(x)  # construct matrix R_x+
        st = process_time()
        self.R_xp_ = self.eta * self.R_xp_ + (1 - self.eta) * R_xp
        # update step
        W = (
            2 * self.R_xp_
            - 4 * self.h @ self.h.T @ self.R_xp_
            - 4 * self.R_xp_ @ self.h @ self.h.T
        )
        h_tilde = self.h - self.mu * np.linalg.solve(W, R_xp @ self.h - X * self.h)
        self.h = h_tilde / np.linalg.norm(h_tilde)
        et = process_time()
        return et - st


class MCQN(MCALG):
    def __init__(
        self,
        L: int,
        N: int,
        mu: float,
        eta: float,
        lambd: float,
        x_vars: np.ndarray = None,
        buffer_size: int = None,
    ) -> None:
        MCALG.__init__(self, L, N, buffer_size)
        self.mu = mu
        self.lambd = lambd
        self.eta = eta
        self.h = np.zeros(shape=(L * N, 1))
        self.h[0::L] = 1
        self.h = self.h / np.sqrt(N)
        self.h_old = np.zeros(shape=(L * N, 1))
        self.R_xp_ = (
            np.diag(np.repeat(x_vars, L))
            if x_vars is not None
            else np.eye(L * N) / (N * L)
        )
        self.V_inv = np.eye(self.L * self.N)

    def step(self, x: np.ndarray, vad: np.ndarray) -> float:
        if not np.any(vad):
            return 0
        st = process_time()
        X_post = self.compute_X(x)  # error function Χ
        X_prior = self.compute_X_prior(x)  # error function Χ
        R_xp = self.construct_Rxp(x)  # construct matrix R_x+
        self.R_xp_ = self.eta * self.R_xp_ + (1 - self.eta) * R_xp

        dh = self.h - self.h_old
        g = self.R_xp_ @ dh + X_post * self.h - X_prior * self.h_old + dh * self.lambd
        self.h_old = self.h.copy()

        v = dh.T @ g
        A = np.eye(self.L * self.N) - (g @ dh.T) / v

        self.V_inv = A.T @ self.V_inv @ A + (dh @ dh.T) / v

        h_tilde = self.h - self.mu * self.V_inv @ (self.R_xp_ @ self.h - X_post * self.h)
        self.h = h_tilde / np.linalg.norm(h_tilde)
        et = process_time()
        return et - st


class SPMCQN:
    graph: Graph
    L: int
    N: int
    sub_alg: List[MCQN]
    index_sets: List[np.ndarray]

    def __init__(
        self,
        graph: Graph,
        L: int,
        N: int,
        mu: float,
        eta: float,
        lambd: float,
        x_vars: np.ndarray = None,
        buffer_size: int = None,
    ) -> None:
        self.graph = graph
        self.L = L
        self.N = N
        self.sub_alg = []
        self.index_sets = []
        self.self_index = []

        for m in graph.vertices:
            index_set = np.where(graph.A[:, m])[0]
            self.index_sets.append(index_set)
            self_index = np.where(index_set == m)[0][0]
            self.self_index.append(self_index)
            N_m = int(np.sum(graph.A[:, m]))
            alg = MCQN(L, N_m, mu, eta, lambd, x_vars, buffer_size)
            self.sub_alg.append(alg)

    def step(self, x: np.ndarray, vad: np.ndarray) -> float:
        start_time = process_time()
        for i, m in enumerate(self.graph.vertices):
            x_m = x[:, self.index_sets[i]]
            vad_m = vad[self.index_sets[i]]
            self.sub_alg[i].step(x_m, vad_m)
        return process_time() - start_time

    def getEstimate(self) -> np.ndarray:
        h = np.zeros((self.L * self.N, 1), dtype=np.complex128)
        for i, m in enumerate(self.graph.vertices):
            h[m * self.L : (m + 1) * self.L] = self.sub_alg[i].getEstimate()[
                self.self_index[i] * self.L : (self.self_index[i] + 1) * self.L
            ]
        return h


# ###############################################################
# FREQUENCY DOMAIN ALGORITHMS


class FMCALG:
    """
    This class represents the FMCALG algorithm.

    Attributes:
        mu (float): The step size.
        h (np.ndarray): The estimated impulse response.
        L (int): The filter length.
        N (int): The number of channels.
    """

    mu: float
    h: np.ndarray
    L: int
    N: int

    def __init__(self, L: int, N: int) -> None:
        """
        Initializes a new instance of the FMCALG class.

        Args:
            L (int): The filter length.
            N (int): The number of channels.
        """
        self.L = L
        self.N = N

    def DFT_matrix(self, N):
        """
        Computes the DFT matrix.

        Args:
            N (int): The size of the DFT matrix.

        Returns:
            np.ndarray: The DFT matrix.
        """
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp(-2 * np.pi * 1j / N)
        W = np.power(omega, i * j) / np.sqrt(N)
        return W

    def getEstimate(self) -> np.ndarray:
        """
        Returns the estimated impulse response.

        Returns:
            np.ndarray: The estimated impulse response.
        """
        return np.fft.fft(self.h.T).reshape(-1, 1)


class CMCFLMS(FMCALG):
    def __init__(self, L: int, N: int, mu: float) -> None:
        FMCALG.__init__(self, L, N)
        self.mu = mu
        self.h = np.zeros((self.L, self.N))
        self.h[1, :] = 1
        self.h = self.h / np.sqrt(self.N)

    def step(self, x: np.ndarray, vad: np.ndarray) -> None:
        if not np.any(vad):
            return 0
        # compute fq domain h_m
        h_mf = np.fft.fft(np.concatenate([self.h, np.zeros(self.h.shape)]), axis=0)
        x_f = np.fft.fft(x, axis=0)
        dh_mnf = np.zeros((2 * self.L, self.N), dtype=np.complex128)
        st = process_time()
        for i in range(self.N):
            for j in range(self.N):
                e_f = x_f[:, i] * h_mf[:, j] - x_f[:, j] * h_mf[:, i]
                e_2L = np.fft.ifft(e_f)
                e_2L[: self.L] = 0
                e_01f = np.fft.fft(e_2L)
                dh_mnf[:, j] += x_f[:, i].conj() * e_01f

        for i in range(self.N):
            dh_m2L = np.real(np.fft.ifft(self.mu * dh_mnf[:, i]))
            dh_m = dh_m2L[: self.L]
            # update
            h_tilde = self.h[:, i] - dh_m
            self.h[:, i] = (h_tilde) / np.linalg.norm(h_tilde)
        et = process_time()
        return et - st


class NMCFLMS(FMCALG):
    eta: float  # forgetting factor
    zeta: float  # regularization parameter

    def __init__(self, L: int, N: int, mu: float, eta: float, zeta: float) -> None:
        FMCALG.__init__(self, L, N)
        self.mu = mu
        self.eta = eta
        self.zeta = zeta
        self.h = np.zeros((self.L, self.N))
        self.h[1, :] = 1
        self.h = self.h / np.sqrt(self.N)
        self.P_nn = None

    def step(self, x: np.ndarray, vad: np.ndarray) -> float:
        if not np.any(vad):
            return 0
        # compute fq domain h_m

        h_mf = np.fft.fft(np.concatenate([self.h, np.zeros(self.h.shape)]), axis=0)
        x_f = np.fft.fft(x, axis=0)
        dh_nf = np.zeros((2 * self.L, self.N), dtype=np.complex128)
        P_nn_ = np.zeros((2 * self.L, self.N), dtype=np.complex128)
        st = process_time()
        for i in range(self.N):
            for j in range(self.N):
                e_f = x_f[:, i] * h_mf[:, j] - x_f[:, j] * h_mf[:, i]
                e_2L = np.fft.ifft(e_f)
                e_2L[: self.L] = 0
                e_01f = np.fft.fft(e_2L)
                dh_nf[:, j] += x_f[:, i].conj() * e_01f
                if i != j:
                    P_nn_[:, i] += x_f[:, j].conj() * x_f[:, j]

        self.P_nn = (
            P_nn_ if self.P_nn is None else self.eta * self.P_nn + (1 - self.eta) * P_nn_
        )

        P_nninv = 1 / (self.P_nn + self.zeta)

        for i in range(self.N):
            dh_2L = np.real(np.fft.ifft(self.mu * P_nninv[:, i] * dh_nf[:, i]))
            dh_m = dh_2L[: self.L]
            # update
            h_tilde = self.h[:, i] - dh_m
            self.h[:, i] = (h_tilde) / np.linalg.norm(h_tilde)
        et = process_time()
        return et - st

class SPNMCFLMS:
    graph: Graph
    L: int
    N: int
    sub_alg: List[NMCFLMS]
    index_sets: List[np.ndarray]
    self_index: List[int]

    def __init__(
        self,
        graph: Graph,
        L: int,
        N: int,
        mu: float,
        eta: float,
        lambd: float,
    ) -> None:
        self.graph = graph
        self.L = L
        self.N = N
        self.sub_alg = []
        self.index_sets = []
        self.self_index = []

        for m in graph.vertices:
            index_set = np.where(graph.A[:, m])[0]
            self.index_sets.append(index_set)
            self_index = np.where(index_set == m)[0][0]
            self.self_index.append(self_index)
            N_m = int(np.sum(graph.A[:, m]))
            alg = NMCFLMS(L, N_m, mu, eta, lambd)
            self.sub_alg.append(alg)

    def step(self, x: np.ndarray, vad: np.ndarray) -> float:
        start_timem = process_time()
        for i, m in enumerate(self.graph.vertices):
            x_m = x[:, self.index_sets[i]]
            vad_m = vad[self.index_sets[i]]
            self.sub_alg[i].step(x_m, vad_m)
        return process_time() - start_timem

    def getEstimate(self) -> np.ndarray:
        h = np.zeros((self.L * self.N, 1), dtype=np.complex128)
        for i, m in enumerate(self.graph.vertices):
            h[m * self.L : (m + 1) * self.L] = self.sub_alg[i].getEstimate()[
                self.self_index[i] * self.L : (self.self_index[i] + 1) * self.L
            ]
        return h


class RNMCFLMS(FMCALG):
    lambd: float  # coupling parameter

    def __init__(self, L, N, mu, eta, zeta, lambd) -> None:
        FMCALG.__init__(self, L, N)
        self.mu = mu
        self.eta = eta
        self.zeta = zeta
        self.lambd = lambd
        self.h = np.zeros((self.L, self.N))
        self.h[1, :] = 1
        self.h = self.h / np.sqrt(self.N)
        self.P_nn = np.zeros((2 * self.L, self.N))
        self.first = True

    def step(self, x: np.ndarray, vad: np.ndarray) -> None:
        if not np.any(vad):
            return 0
        # compute fq domain h_m
        h_mf = np.fft.fft(np.concatenate([self.h, np.zeros(self.h.shape)]), axis=0)
        x_f = np.fft.fft(x, axis=0)
        dh_nf = np.zeros((2 * self.L, self.N), dtype=np.complex128)
        P_nn_ = np.zeros((2 * self.L, self.N), dtype=np.complex128)
        st = process_time()
        for i in range(self.N):
            for j in range(self.N):
                e_f = x_f[:, i] * h_mf[:, j] - x_f[:, j] * h_mf[:, i]
                e_2L = np.fft.ifft(e_f)
                e_2L[: self.L] = 0
                e_01f = np.fft.fft(e_2L)
                dh_nf[:, j] += x_f[:, i].conj() * e_01f
                if i != j:
                    P_nn_[:, i] += x_f[:, j].conj() * x_f[:, j]

        self.P_nn = P_nn_ if self.first else self.eta * self.P_nn + (1 - self.eta) * P_nn_
        self.first = False

        P_nninv = 1 / (self.P_nn + self.zeta)

        # penalty term
        q = 2 / np.abs(h_mf) ** 2
        ddJ_p = q * (h_mf)
        # gradient in f
        ddJ_f = dh_nf
        # coupling parameter
        β = np.abs(
            (
                self.lambd
                * ddJ_p.T.reshape(1, 2 * self.L * self.N).conj()
                @ ddJ_f.T.reshape(2 * self.L * self.N, 1)
            )
            / np.sum(np.abs(ddJ_p) ** 2)
        ).squeeze()

        for i in range(self.N):
            dh_2L = np.real(
                np.fft.ifft(
                    self.mu * (P_nninv[:, i] * dh_nf[:, i] + β * ddJ_p[:, i]),
                    axis=0,
                )
            )
            dh_m = dh_2L[: self.L]
            # update
            h_tilde = self.h[:, i] - dh_m
            self.h[:, i] = (h_tilde) / np.linalg.norm(h_tilde)
        et = process_time()
        return et - st


# class LPRNMCFLMS(FMCALG):
#     def __init__(self, rho, lambd, zeta, eta, p, L, N) -> None:
#         FMCALG.__init__(self, L, N)
#         self.rho = rho
#         self.lambd = lambd
#         self.zeta = zeta
#         self.eta = eta
#         self.p = p
#         self.h = np.zeros((self.L, self.N))
#         self.h[1, :] = 1
#         self.h = self.h / np.sqrt(self.N)
#         self.P_nn = np.zeros((2 * self.L, self.N))
#         self.S = np.ones((2 * self.L, self.N))
#         self.first = True

#     def step(self, x:np.ndarray, vad:np.ndarray) -> None:
# if not np.any(vad):
#     return 0
#         # compute fq domain h_m
#         h_f = np.fft.fft(np.concatenate([self.h, np.zeros(self.h.shape)]), axis=0)
#         x_f = np.fft.fft(x, axis=0)
#         dh_nf = np.zeros((2 * self.L, self.N), dtype=np.complex128)
#         P_nn_ = np.zeros((2 * self.L, self.N), dtype=np.complex128)
#         for i in range(self.N):
#             for j in range(self.N):
#                 e_f = x_f[:, i] * h_f[:, j] - x_f[:, j] * h_f[:, i]
#                 e_2L = np.fft.ifft(e_f)
#                 e_2L[: self.L] = 0
#                 e_01f = np.fft.fft(e_2L)
#                 dh_nf[:, j] += x_f[:, i].conj() * e_01f
#                 if i != j:
#                     P_nn_[:, i] += x_f[:, j].conj() * x_f[:, j]

#         self.P_nn = (
#             P_nn_ if self.first else self.lambd * self.P_nn + (1 - self.lambd) * P_nn_
#         )
#         self.first = False

#         S_inv = 1 / (self.S + self.zeta)

#         # penalty gradient
#         ddJ_p = self.p * S_inv * np.abs(h_f) ** (self.p - 1) * np.exp(1j * np.angle(h_f))
#         # gradient
#         ddJ_f = S_inv * dh_nf
#         # coupling parameter
#         β = np.abs(
#             (
#                 self.eta
#                 * ddJ_p.T.reshape(1, 2 * self.L * self.N).conj()
#                 @ ddJ_f.T.reshape(2 * self.L * self.N, 1)
#             )
#             / np.sum(np.abs(ddJ_p) ** 2)
#         )
#         H = np.abs(h_f) ** (self.p - 2)
#         self.S = self.P_nn - β * self.p**2 * H

#         # update
#         h_f = h_f - self.rho * ddJ_f + self.rho * β * ddJ_p
#         h_2L = np.real(np.fft.ifft(h_f, axis=0))
#         self.h = h_2L[: self.L, :]
#         self.h = self.h / np.linalg.norm(self.h)


# class PCLPRNMCFLMS(FMCALG):
#     def __init__(self, rho, lambd, zeta, eta, p, L, N) -> None:
#         FMCALG.__init__(self, L, N)
#         self.rho = rho
#         self.lambd = lambd
#         self.zeta = zeta
#         self.eta = eta
#         self.p = p
#         self.h = np.zeros((self.L, self.N))
#         self.h[1, :] = 1
#         self.h = self.h / np.sqrt(self.N)
#         self.P_nn = np.zeros((2 * self.L, self.N))
#         self.S = np.ones((2 * self.L, self.N))
#         self.first = True

#     def step(self, x:np.ndarray, vad:np.ndarray) -> None:
# if not np.any(vad):
#     return 0
#         # compute fq domain h_m
#         h_f = np.fft.fft(np.concatenate([self.h, np.zeros(self.h.shape)]), axis=0)
#         x_f = np.fft.fft(x, axis=0)
#         dh_nf = np.zeros((2 * self.L, self.N), dtype=np.complex128)
#         P_nn_ = np.zeros((2 * self.L, self.N), dtype=np.complex128)
#         for i in range(self.N):
#             for j in range(self.N):
#                 e_f = x_f[:, i] * h_f[:, j] - x_f[:, j] * h_f[:, i]
#                 e_2L = np.fft.ifft(e_f)
#                 e_2L[: self.L] = 0
#                 e_01f = np.fft.fft(e_2L)
#                 dh_nf[:, j] += x_f[:, i].conj() * e_01f
#                 if i != j:
#                     P_nn_[:, i] += x_f[:, j].conj() * x_f[:, j]

#         self.P_nn = (
#             P_nn_ if self.first else self.lambd * self.P_nn + (1 - self.lambd) * P_nn_
#         )
#         self.first = False

#         Q_1 = np.zeros((2 * self.L, 2 * self.L))
#         Q_1[:-1, 1:] = np.eye(2 * self.L - 1)
#         Q_2 = np.zeros((2 * self.L, 2 * self.L))
#         Q_2[1:, :-1] = np.eye(2 * self.L - 1)
#         Q_3 = np.zeros((2 * self.L, 2 * self.L))
#         Q_3[1:-1, :] = 1
#         Q = (8 * np.eye(2 * self.L) - 4 * Q_1 - 4 * Q_2) * Q_3

#         S_inv = 1 / (self.S + self.zeta)

#         # penalty gradient
#         ddJ_g = 1j * (h_f / np.abs(h_f) ** 2) * (Q @ np.angle(h_f))
#         ddJ_p = self.p * S_inv * np.abs(h_f) ** (self.p - 1) * np.exp(1j * np.angle(h_f))
#         # gradient
#         ddJ_f = S_inv * dh_nf
#         # coupling parameters
#         β = np.abs(
#             (
#                 self.eta
#                 * np.linalg.pinv(
#                     np.array(
#                         [
#                             ddJ_p.T.reshape(2 * self.L * self.N, 1),
#                             -ddJ_g.T.reshape(2 * self.L * self.N, 1),
#                         ]
#                     )
#                 )
#                 @ ddJ_f.T.reshape(2 * self.L * self.N, 1)
#             )
#         )
#         H = np.abs(h_f) ** (self.p - 2)
#         self.S = self.P_nn - β[0] * self.p**2 * H

#         # update
#         h_f = h_f - self.rho * ddJ_f + self.rho * (β[0] * ddJ_p - β[1] * ddJ_g)
#         h_2L = np.real(np.fft.ifft(h_f, axis=0))
#         self.h = h_2L[: self.L, :]
#         self.h = self.h / np.linalg.norm(self.h)


# class EPCLPRNMCFLMS(FMCALG):
#     def __init__(self, rho, lambd, zeta, eta, p, M, L, N) -> None:
#         FMCALG.__init__(self, L, N)
#         self.rho = rho
#         self.lambd = lambd
#         self.zeta = zeta
#         self.eta = eta
#         self.p = p
#         self.h = np.zeros((self.L, self.N))
#         self.h[1, :] = 1
#         self.h = self.h / np.sqrt(self.N)
#         self.P_nn = np.zeros((2 * self.L, self.N))
#         self.S = np.ones((2 * self.L, self.N))
#         self.first = True
#         self.J_s = np.zeros((M,))

#         self.J_ = np.array([])
#         self.J_s_ = np.array([])

#     def step(self, x:np.ndarray, vad:np.ndarray) -> None:
# if not np.any(vad):
#     return 0
#         # compute fq domain h_m
#         h_f = np.fft.fft(np.concatenate([self.h, np.zeros(self.h.shape)]), axis=0)
#         x_f = np.fft.fft(x, axis=0)
#         dh_nf = np.zeros((2 * self.L, self.N), dtype=np.complex128)
#         P_nn_ = np.zeros((2 * self.L, self.N), dtype=np.complex128)
#         J = 0
#         self.J_s = self.J_s[1:]
#         for i in range(self.N):
#             for j in range(self.N):
#                 e_f = x_f[:, i] * h_f[:, j] - x_f[:, j] * h_f[:, i]
#                 e_2L = np.fft.ifft(e_f)
#                 e = np.abs(np.sum(e_2L**2))
#                 e_2L[: self.L] = 0
#                 J += e
#                 e_01f = np.fft.fft(e_2L)
#                 dh_nf[:, j] += x_f[:, i].conj() * e_01f
#                 if i != j:
#                     P_nn_[:, i] += x_f[:, j].conj() * x_f[:, j]

#         self.J_s = np.append(self.J_s, J)
#         self.P_nn = (
#             P_nn_ if self.first else self.lambd * self.P_nn + (1 - self.lambd) * P_nn_
#         )
#         self.first = False

#         Q_1 = np.zeros((2 * self.L, 2 * self.L))
#         Q_1[:-1, 1:] = np.eye(2 * self.L - 1)
#         Q_2 = np.zeros((2 * self.L, 2 * self.L))
#         Q_2[1:, :-1] = np.eye(2 * self.L - 1)
#         Q_3 = np.zeros((2 * self.L, 2 * self.L))
#         Q_3[1:-1, :] = 1
#         Q = (8 * np.eye(2 * self.L) - 4 * Q_1 - 4 * Q_2) * Q_3

#         S_inv = 1 / (self.S + self.zeta)

#         # penalty gradient
#         ddJ_g = 1j * (h_f / np.abs(h_f) ** 2) * (Q @ np.angle(h_f))
#         ddJ_p = self.p * S_inv * np.abs(h_f) ** (self.p - 1) * np.exp(1j * np.angle(h_f))
#         # gradient
#         ddJ_f = S_inv * dh_nf
#         # coupling parameters
#         β = np.abs(
#             (
#                 self.eta
#                 * np.linalg.pinv(
#                     np.array(
#                         [
#                             ddJ_p.T.reshape(2 * self.L * self.N, 1),
#                             -ddJ_g.T.reshape(2 * self.L * self.N, 1),
#                         ]
#                     )
#                 )
#                 @ ddJ_f.T.reshape(2 * self.L * self.N, 1)
#             )
#         )
#         H = np.abs(h_f) ** (self.p - 2)
#         self.S = self.P_nn - β[0] * self.p**2 * H

#         # const functions
#         # if self.J_s.mean() > J:
#         #     print("lol")

#         self.J_ = np.append(self.J_, J)
#         self.J_s_ = np.append(self.J_s_, self.J_s.mean())

#         # update
#         h_f = h_f - self.rho * ddJ_f + self.rho * (β[0] * ddJ_p - β[1] * ddJ_g)
#         h_2L = np.real(np.fft.ifft(h_f, axis=0))
#         self.h = h_2L[: self.L, :]
#         self.h = self.h / np.linalg.norm(self.h)

