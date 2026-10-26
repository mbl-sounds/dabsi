import os
import time
import tarfile
import zipfile
import numpy as np
import soundfile as sf
import scipy.signal as sig
from typing import Callable
import pyroomacoustics as pr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from urllib.request import urlretrieve
from scipy.ndimage import median_filter
from typing import Tuple, Dict, List, Generator


class Graph:
    """
    Small class to define a graph.

    Attributes:
        vertices (List[int]): List of vertex indices.
        edges (List[Tuple[int, int]]): List of edge tuples.
        A (np.ndarray): Adjacency matrix of the graph.
        W (np.ndarray): Weight matrix of the graph.

    Methods:
        __init__(self, M: int, E: list) -> None: Initializes the Graph object.
        _computeAdjacencyMatrix(self) -> None: Computes the adjacency matrix and weight matrix of the graph.
    """

    vertices: List[int]
    edges: List[Tuple[int, int]]
    positions: np.ndarray
    A: np.ndarray
    W: np.ndarray

    def __init__(self, M: int, E: list) -> None:
        """
        Initializes the Graph object.

        Args:
            M (int): The number of vertices in the graph.
            E (list): The list of edge tuples representing the connections between vertices.
        """
        self.vertices = [*range(M)]
        self.edges = E
        self._computeAdjacencyMatrix()

    def _computeAdjacencyMatrix(self) -> None:
        """
        Computes the adjacency matrix and weight matrix of the graph.
        """
        M = len(self.vertices)
        self.A = np.zeros((M, M))
        for e in self.edges:
            self.A[e[0], e[1]] = 1
        self.W = np.apply_along_axis(lambda row: row / sum(row), 1, self.A)

    def computeCRCoverage(self) -> np.ndarray:

        pass


class VariableTracker:
    node: object
    variables: Dict[str, Callable]
    tracked_variables: Dict[str, List]

    def __init__(self, node):
        self.node = node
        self.variables = {}
        self.tracked_variables = {}

    def addVariable(self, var_name: str, callback: Callable = lambda x: x):
        self.variables[var_name] = callback
        self.tracked_variables[var_name] = []

    def trackVariables(self):
        # print("tracking variables for node", self.node)
        for var_name, callback in self.variables.items():
            if not hasattr(self.node, var_name):
                self.tracked_variables[var_name].append(np.nan)
            else:
                self.tracked_variables[var_name].append(
                    callback(getattr(self.node, var_name))
                )

    def reset(self):
        for var_name in self.tracked_variables.keys():
            self.tracked_variables[var_name] = []


def figureSizeFromPt(width_pt, fraction=1, subplots=(1, 1), ratio=(5**0.5 - 1) / 2):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches (this is not DPI!!)
    latex_inches_per_pt = 1 / 72.27

    # # Golden ratio to set aesthetic figure height
    # golden_ratio =

    # Figure width in inches
    fig_width_in = fig_width_pt * latex_inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def NPM(a: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Computes the Normalized Projection Misalignment (NPM) for two vectors a and b.

    Parameters
    ----------
    a: np.ndarray
            Column vector
    b: np.ndarray
            Column vector

    Returns
    -------
    NPM: np.float64
            Normalized Projection Misalignment
    """
    if b.shape[1] > 1:
        b = b.T.reshape(-1, 1)
    e = b - (b.conj().T @ a) / (a.conj().T @ a) * a
    return np.linalg.norm(e) / np.linalg.norm(b)


H_TRUE: np.ndarray = None
F_OPT: np.ndarray = None


def NMSE(a: np.ndarray, b: np.ndarray) -> np.float64:
    """
    Computes the Normalized Mean Squared Error (NMSE) for two vectors a and b.

    Parameters
    ----------
    a: np.ndarray
            Column vector
    b: np.ndarray
            Column vector

    Returns
    -------
    NMSE: np.float64
            Normalized Mean Squared Error
    """
    return np.square(np.abs(a - b)).sum() / np.square(np.abs(a)).sum()


def NFPM(h_est: np.ndarray, h_true: np.ndarray, L: int, M: int, Df: int = 199):
    """
    Computes the Normalized Filter-Projection Misalignment (NFPM) for two vectors a and b
    in the time domain.

    Parameters
    ----------
    a: np.ndarray
            Column vector
    b: np.ndarray
            Column vector
    L: int
            length of single IR
    M: int
            number of channels
    Df: int
            length of common filter

    Returns
    -------
    NFPM: np.float64
            Normalized Filter-Projection Misalignment
    """
    global H_TRUE
    global F_OPT
    assert h_true.shape == h_est.shape, "estimate and true dont have the same length"
    assert h_est.shape[0] == M * L, "estimate is not correct length M*L"
    assert h_true.shape[0] == M * L, "ground truth is not correct length M*L"

    if np.any(H_TRUE != h_true) or F_OPT is None:
        # print("Recalculating F_OPT")
        H = []
        for i in range(M):
            H_i = np.zeros((L + 2 * Df, 2 * Df + 1))
            for n in range(2 * Df + 1):
                H_i[n : n + L, n, None] = h_true[i * L : (i + 1) * L]
            H.append(H_i)
        H = np.concatenate(H)
        H_TRUE = h_true
        F_OPT = H @ np.linalg.inv(H.T @ H) @ H.T

    h_tilde = []
    for i in range(M):
        h_i_tilde = np.pad(
            h_est[i * L : (i + 1) * L],
            ((Df, Df), (0, 0)),
        )
        h_tilde.append(h_i_tilde)

    h_tilde = np.concatenate(h_tilde)

    nfpm = np.linalg.norm(h_tilde - F_OPT @ h_tilde) / np.linalg.norm(h_tilde)
    return nfpm


H_TRUEF: np.ndarray = None
F_OPTF: np.ndarray = None


def NFPMf(h_est: np.ndarray, h_true: np.ndarray, get_f: bool = False) -> np.float64:
    global H_TRUEF
    global F_OPTF
    M: np.int64 = h_true.shape[1]
    if (
        F_OPTF is None
        or H_TRUEF is None
        or H_TRUEF.shape != h_true.shape
        or np.any(H_TRUEF != h_true)
    ):
        # print("Recalculating F_OPT")
        H = np.concatenate(
            [np.diag(h_true[:, m]) for m in range(M)],
            axis=0,
        )
        H_TRUEF = h_true
        F_OPTF = H @ np.linalg.inv(H.conj().T @ H) @ H.conj().T
    nfpm = np.linalg.norm(h_est - F_OPTF @ h_est) / np.linalg.norm(h_est)
    if get_f:
        H = np.concatenate(
            [np.diag(h_true[:, m]) for m in range(M)],
            axis=0,
        )
        return nfpm, np.linalg.inv(H.conj().T @ H) @ H.conj().T @ h_est
    return nfpm


H_TRUES: np.ndarray = None
F_OPTS: np.complex128 = None


def NFPMs(h_est: np.ndarray, h_true: np.ndarray, get_f: bool = False):
    global H_TRUES
    global F_OPTS
    M = h_true.shape[1]
    if (
        F_OPTS is None
        or H_TRUES is None
        or H_TRUES.shape != h_true.shape
        or np.any(H_TRUES != h_true)
    ):
        # print("Recalculating F_OPT")
        h_tilde = np.concatenate(
            [h_true[:, m, np.newaxis] for m in range(M)],
            axis=0,
        )
        H_TRUES = h_true
        F_OPTS = h_tilde @ h_tilde.conj().T / np.linalg.norm(h_tilde) ** 2

    h_hat = h_est
    nfpm = np.linalg.norm(h_hat - F_OPTS @ h_hat) / np.linalg.norm(h_hat)
    if get_f:
        h_tilde = np.concatenate(
            [h_true[:, m, np.newaxis] for m in range(M)],
            axis=0,
        )
        return nfpm, h_tilde.conj().T @ h_hat / np.linalg.norm(h_tilde) ** 2
    return nfpm


MARKERS = list(Line2D.markers.keys())
LINESSTYLES = list(Line2D.lineStyles.keys())
LINECOLORS = list(mcolors.TABLEAU_COLORS.keys())

# LINECOLORS = list(mcolors.XKCD_COLORS.keys())


def plotCommonFilterf(
    h_hat: np.ndarray,
    h_true: np.ndarray,
    K_prime: List[int] = None,
    figsize=(8, 2.5),
    fontdict={"size": 7, "family": "serif"},
):
    L = h_true.shape[0]
    _, f = NFPMf(h_hat, h_true, True)
    _, f_tilde = NFPMs(h_hat, h_true, True)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].set_title("Magnitude")
    axs[1].set_title("Phase")
    axs[0].plot(20 * np.log10(np.abs(f)), label="NFPMf")
    pp = 20 * np.log10(np.abs(f_tilde))
    axs[0].hlines(
        pp,
        0,
        L,
        linestyles="dashed",
        color="tab:orange",
        label="NFPMs",
    )
    if K_prime is not None:
        axs[0].vlines(
            K_prime,
            pp - 10,
            pp + 10,
            "tab:red",
            linestyles="dotted",
            label=r"$\mathcal{K}$'",
        )
    axs[0].grid()
    axs[0].legend(
        fontsize=7,
        loc="lower right",
    )
    axs[0].set_ylabel("Magnitude [dB]")
    axs[0].set_xlabel("Frequency bins")
    axs[0].set_ylim((pp - 10, pp + 10))

    axs[1].plot(np.angle(f), label="NFPMf")
    axs[1].hlines(
        np.angle(f_tilde),
        0,
        L,
        linestyles="dashed",
        color="tab:orange",
        label="NFPMs",
    )
    if K_prime is not None:
        axs[1].vlines(
            K_prime,
            -np.pi,
            np.pi,
            "tab:red",
            linestyles="dotted",
            label=r"$\mathcal{K}$'",
        )
    axs[1].set_ylim((-np.pi, np.pi))
    axs[1].grid()
    axs[1].legend(
        loc="lower right",
        prop=fontdict,
    )
    axs[1].set_ylabel("Phase [rad]")
    axs[1].set_xlabel("Frequency bins")

    plt.tight_layout()
    return fig, axs


def plotCommonFiltersf(
    estimates: List[np.ndarray],
    h_true: np.ndarray,
    K_prime: List[int] = None,
    figsize=(8, 2.5),
    fontdict={"size": 7, "family": "serif"},
):
    L = h_true.shape[0]
    fs = []
    for h_hat in estimates:
        _, f = NFPMf(h_hat, h_true, True)
        # _, f_tilde = NFPMs(h_hat, h_true, True)
        fs.append(f / np.linalg.norm(f) * np.sqrt(L))

    fs = np.asarray(fs).squeeze()

    f_mean = np.mean(fs, axis=0)
    f_std = np.std(fs, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].set_title("Magnitude")
    axs[1].set_title("Phase")
    pp = 20 * np.log10(np.abs(f_mean))
    ppm = np.mean(pp)
    ppstd = 20 * np.log10(np.abs(f_std))
    axs[0].plot(pp, label=r"mean $\vert \mathrm{f} \vert$", color="tab:blue")
    axs[0].plot(pp + ppstd, ":", color="tab:blue")
    axs[0].plot(pp - ppstd, ":", color="tab:blue")
    # axs[0].hlines(
    #     pp,
    #     0,
    #     L,
    #     linestyles="dashed",
    #     color="tab:orange",
    #     label="NFPMs",
    # )
    if K_prime is not None:
        axs[0].vlines(
            K_prime,
            ppm - 10,
            ppm + 10,
            "tab:red",
            linestyles="dotted",
            label=r"$\mathcal{K}$'",
        )
    axs[0].grid()
    axs[0].legend(
        fontsize=7,
        loc="lower right",
    )
    axs[0].set_ylabel("Magnitude [dB]")
    axs[0].set_xlabel("Frequency bins")
    # axs[0].set_ylim((ppm - 10, ppm + 10))

    tt = np.angle(f_mean)
    ttstd = np.angle(f_std)

    axs[1].plot(np.angle(f_mean), label=r"mean $\angle \mathrm{f}$", color="tab:blue")
    axs[1].plot(tt + ttstd, ":", color="tab:blue")
    axs[1].plot(tt - ttstd, ":", color="tab:blue")
    # axs[1].hlines(
    #     np.angle(f_tilde),
    #     0,
    #     L,
    #     linestyles="dashed",
    #     color="tab:orange",
    #     label="NFPMs",
    # )
    if K_prime is not None:
        axs[1].vlines(
            K_prime,
            -np.pi,
            np.pi,
            "tab:red",
            linestyles="dotted",
            label=r"$\mathcal{K}$'",
        )
    # axs[1].set_ylim((-np.pi, np.pi))
    axs[1].grid()
    axs[1].legend(
        loc="lower right",
        prop=fontdict,
    )
    axs[1].set_ylabel("Phase [rad]")
    axs[1].set_xlabel("Frequency bins")

    plt.tight_layout()
    return fig, axs


MISALIGNMENT_TYPES = ["NPM", "NPMA", "NPMM", "NFPM"]


def plotMisalignmentCurve(
    alg_ids,
    misalignment,
    misalignment_types=["NPM", "NPMA"],
    title=None,
    xlabel="Frames",
    ylabel="Misalignment [dB]",
    figsize=(5, 3),
    fontdict={"size": 7, "family": "serif"},
    max_columns=3,
    cumulator=np.nanmean,
    markevery=1000,
):
    if type(alg_ids) is not list:
        alg_ids = [alg_ids]
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontdict=fontdict)
    if xlabel is not None:
        plt.xlabel(xlabel, fontdict=fontdict)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict=fontdict)
    lines = []
    for i, alg_id in enumerate(alg_ids):
        alg_lines = []
        misalignment[alg_id] = np.array(misalignment[alg_id])
        for j, misalignment_type in enumerate(misalignment_types):
            marker = MARKERS[j]
            if len(misalignment_types) > 1:
                label = f"{alg_id} {misalignment_type}"
            else:
                label = alg_id
            (line,) = plt.plot(
                20
                * np.log10(
                    sig.medfilt(
                        cumulator(
                            misalignment[alg_id][
                                :, :, MISALIGNMENT_TYPES.index(misalignment_type)
                            ],
                            axis=0,
                        ),
                        201,
                    )
                ),
                label=label,
            )
            alg_lines.append(line)
        lines.append(alg_lines)
    lines = np.array(lines)
    lines = lines.reshape(-1).tolist()
    # plt.grid()
    ncols = np.minimum(len(alg_ids) * len(misalignment_types), max_columns)
    # plt.legend(
    #     # lines,
    #     ncols=ncols,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.0),
    #     prop=fontdict,
    #     columnspacing=0.6,
    #     # handlelength=1.2,
    # )
    plt.xticks(fontsize=fontdict["size"])
    plt.yticks(fontsize=fontdict["size"])
    # plt.tight_layout()
    return fig


def getMisalignment(
    h_est: np.ndarray,
    h_true: np.ndarray,
    types: list = ["NPM", "NPMA", "NPMM"],
    L: int = 128,
    # ) -> Tuple[np.float64, np.float64]:
) -> List[np.float64]:
    """
    Calculate the misalignment between the estimated and true values of h.

    Args:
        h_est (np.ndarray): The estimated values of h.
        h_true (np.ndarray): The true values of h.

    Returns:
        Tuple[np.float64, np.float64, np.float64]: A tuple containing three misalignment values:
            - npm: Misalignment calculated using NPM method.
            - nfpms: Misalignment calculated using NFPMs method.
            - nfpmf: Misalignment calculated using NFPMf method.
    """
    ret = []
    npms = []
    for m in range(0, h_true.shape[0], L):
        npms.append(NPM(h_est[m : (m + L)], h_true[m : (m + L)]))
    for type in types:
        if type == "NPM":
            npm = NPM(h_est, h_true)
            ret.append(npm)
        if type == "NPMA":
            npma = np.mean(npms)
            ret.append(npma)
        if type == "NPMM":
            npmm = np.min(npms)
            ret.append(npmm)
        if type == "NFPM":
            nfpmf = NFPMf(h_est, h_true)
            ret.append(nfpmf)
    return ret


def plotEstimatesNW(
    nw,
    rirs_f,
    K_prime: List[int] = None,
    figsize=(8, 2.5),
    fontdict={"size": 7, "family": "serif"},
):
    M = rirs_f.shape[1]
    figsize = (figsize[0], figsize[1] * M)
    fig, axs = plt.subplots(M, 2, figsize=figsize)
    axs[0][0].set_title("Magnitude")
    axs[0][1].set_title("Phase")
    for m in range(M):
        true_ = rirs_f[:, m] / np.linalg.norm(rirs_f[:, m])
        pp = 20 * np.log10(np.abs(true_))
        axs[m][0].plot(pp, "k--", label=f"true")
        for node in nw.nodes.values():
            if m in node.receive_indices.keys():
                est_ = node.x[node.receive_index_ranges[m]] / np.linalg.norm(
                    node.x[node.receive_index_ranges[m]]
                )
                axs[m][0].plot(20 * np.log10(np.abs(est_)), label=f"est. node {node.id}")
        if K_prime is not None:
            axs[m][0].vlines(
                K_prime,
                np.mean(pp) - 10,
                np.mean(pp) + 10,
                "tab:red",
                linestyles="dotted",
                label=r"$\mathcal{K}$'",
            )
        axs[m][0].set_ylim((np.mean(pp) - 10, np.mean(pp) + 10))
        axs[m][0].legend(
            prop=fontdict,
            loc="lower right",
        )
        axs[m][0].grid()
        axs[m][0].set_ylabel("Magnitude [dB]", fontdict=fontdict)
        # axs[m][0].set_ylim((-50, 10))
        true_ = rirs_f[:, m] / np.linalg.norm(rirs_f[:, m])
        phases_ = np.angle(true_)
        axs[m][1].plot(phases_, "k--", label=f"true")
        for node in nw.nodes.values():
            if m in node.receive_indices.keys():
                est_ = node.x[node.receive_index_ranges[m]] / np.linalg.norm(
                    node.x[node.receive_index_ranges[m]]
                )
                phases_ = np.angle(est_)
                axs[m][1].plot(phases_, label=f"est. node {node.id}")
        if K_prime is not None:
            axs[m][1].vlines(
                K_prime,
                -np.pi,
                np.pi,
                "tab:red",
                linestyles="dotted",
                label=r"$\mathcal{K}$'",
            )
        axs[m][1].legend(
            prop=fontdict,
            loc="lower right",
        )
        axs[m][1].grid()
        axs[m][1].set_ylabel("Phase [rad]", fontdict=fontdict)
        axs[m][1].set_ylim((-np.pi, np.pi))
    axs[m][0].set_xlabel("Frequency bins", fontdict=fontdict)
    axs[m][1].set_xlabel("Frequency bins", fontdict=fontdict)
    axs[m][0].tick_params(axis="both", labelsize=fontdict["size"])
    axs[m][1].tick_params(axis="both", labelsize=fontdict["size"])
    plt.tight_layout()
    return fig, axs


def generateNonStationaryNoise(
    num_samples: int, fs: float, rng=np.random.default_rng()
) -> np.ndarray:
    """
    Generates WGN with randomized segments of variance mimicking the envelope of speech
    (not that well though)

    Parameters
    ----------
    num_samples: int
            number of signal samples to generate
    fs: float
            sampling rate
    rng: Generator
            random number generator

    Returns
    -------
    signal: np.ndarray
            generated signal
    """
    # TODO formant filters
    # TODO glottal pulse train
    # "silence" in the beginning
    length = int(0.1 * fs)
    signal = np.zeros((length,))
    while length < num_samples:
        dur = int(rng.uniform(0.01 * fs, 0.2 * fs))
        # amp = rng.uniform(0.0, 1.0)
        # freq = rng.uniform(200, 6000)
        noise = rng.uniform(0.0001, 0.1)
        # noise = 0
        length += dur
        # t = np.linspace(0, (dur) / fs, dur) * freq * 2 * np.pi
        env_part = rng.normal(size=(dur,), scale=noise)  # + np.sin(t) * amp
        signal = np.concatenate([signal, env_part])

    signal = signal[:num_samples].reshape(num_samples, 1)

    return signal / signal.max()


def generateRandomIRs(
    L, N, a, cov=None, rng=np.random.default_rng(), freq=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates random impulse responses WGN

    Parameters
    ----------
    L: int
            length of IRs
    N: int
            number of channels/IRs
    a: Array-like
            target norm values of IRs (must have N elements)
    rng: Generator
            The random generator to be used

    Returns
    -------
    h: np.ndarray
            LxN array containing the generated IRs
    hf: np.ndarray
            LxN array containing the L-sample FFT of generated IRs
    """
    h = np.zeros((L, N))
    hf = np.zeros((L, N), dtype=np.complex128)
    Omega = np.linalg.cholesky(cov) if cov is not None else None
    for n in range(N):
        if cov is None:
            h_ = rng.normal(size=(L, 1))
        else:
            h_ = rng.multivariate_normal(np.zeros(L), cov, size=1).T
        # h_ = h_ / np.linalg.norm(h_) * a[n]
        if freq:
            hf[:, n, None] = h_
            pp = np.fft.ifft(h_, n=L, axis=0)
            print(np.linalg.norm(pp.imag))
            h[:, n, None] = np.fft.ifft(h_, n=L, axis=0)
        else:
            h[:, n, None] = h_
            hf[:, n, None] = np.fft.fft(h_, n=L, axis=0)
    return h, hf


def _direct_sound_index(rir: np.ndarray) -> int:
    rir = np.asarray(rir, dtype=np.float64)
    if rir.size == 0:
        return 0
    return int(np.argmax(np.square(rir)))


def trim_rir_before_direct_sound(
    rir: np.ndarray,
    src_fs: float,
    target_fs: float,
    target_length: int,
) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float64)
    if rir.size == 0:
        return np.zeros(target_length, dtype=np.float64)

    peak_idx = _direct_sound_index(rir)
    rir = rir[peak_idx:]

    if rir.size == 0:
        return np.zeros(target_length, dtype=np.float64)

    if src_fs != target_fs:
        num_samples = max(1, int(np.round(rir.size * target_fs / src_fs)))
        rir = sig.resample(rir, num_samples)

    rir = rir[:target_length]
    if rir.size < target_length:
        pad_len = target_length - rir.size
        rir = np.pad(rir, (0, pad_len), mode="constant", constant_values=0.0)
    return rir


def removeInitialDelayFromRIRs(rirs: np.ndarray, L: int, all: bool = False) -> np.ndarray:
    M = len(rirs)
    dels = []
    for m in range(M):
        rir = np.asarray(rirs[m][0], dtype=np.float64)
        dels.append(_direct_sound_index(rir))
    aligned_rirs = np.zeros((L, M))
    if all:
        for m in range(M):
            prop_delay = dels[m]
            rir_ = rirs[m][0][prop_delay : prop_delay + L]
            aligned_rirs[:, m] = np.pad(
                rir_,
                (0, L - len(rir_)),
                mode="constant",
                constant_values=0,
            )
    else:
        prop_delay = min(dels) if dels else 0
        for m in range(M):
            rir_ = rirs[m][0][prop_delay : prop_delay + L]
            aligned_rirs[:, m] = np.pad(
                rir_,
                (0, L - len(rir_)),
                mode="constant",
                constant_values=0,
            )
    if np.max(np.abs(aligned_rirs)) > 0:
        aligned_rirs = aligned_rirs / np.max(np.abs(aligned_rirs))
    return aligned_rirs


def generateRandomRoomAndGraph(
    roomsize_range: Tuple[float, float],
    L: int,
    num_nodes: int,
    num_closest: int,
    fs: int,
    max_receive: int = 5,
    directed: bool = False,
    margin: float = 0.5,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[pr.ShoeBox, Graph]:
    roomsize = rng.uniform(*roomsize_range, size=3)

    rt60 = L / fs * 5
    while True:
        try:
            e_absorption, max_order = pr.inverse_sabine(rt60, roomsize)
            break
        except ValueError:
            rt60 *= 1.1
    max_order = 50
    # materials = {
    #     "ceiling": rng.uniform(0.05, 0.5),
    #     "floor": rng.uniform(0.05, 0.5),
    #     "north": rng.uniform(0.05, 0.5),
    #     "south": rng.uniform(0.05, 0.5),
    #     "east": rng.uniform(0.05, 0.5),
    #     "west": rng.uniform(0.05, 0.5),
    # }
    # pr.Material(
    #     e_absorption
    # )
    # absorption = rng.uniform(0.05, 0.5, size=6)

    # Create the room
    room = pr.ShoeBox(
        roomsize,
        fs=fs,
        # absorption=absorption,
        materials=pr.Material(e_absorption),
        max_order=max_order,
        use_rand_ism=True,
        max_rand_disp=0.08,
    )

    node_pos, edges, src_pos = generateRandomWSNTop(
        num_nodes,
        roomsize,
        num_closest=num_closest,
        max_receive=max_receive,
        num_sources=1,
        directed=directed,
        rng=rng,
        margin=margin,
    )

    room.add_microphone_array(node_pos.T)
    room.add_source(src_pos[0, :])
    room.compute_rir()
    # delays = []
    # for node_i, src in enumerate(room.rir):
    #     delays_ = []
    #     for s_i, rir in enumerate(src):
    #         d = np.linalg.norm(node_pos[node_i] - src_pos[s_i]) / room.c * fs
    #         delays_.append(d)
    #     delays.append(np.min(delays_))
    # min_delay = np.min(delays)
    # for node_i, src in enumerate(room.rir):
    #     for s_i, rir in enumerate(src):
    #         rir[:] = np.roll(
    #             rir, -int(np.floor(min_delay + 39))
    #         )  # 40 because the fractional delay filter is 81 long
    for node_i, src in enumerate(room.rir):
        for s_i, rir in enumerate(src):
            d = np.linalg.norm(node_pos[node_i] - src_pos[s_i]) / room.c * fs
            rir[:] = np.roll(
                rir, -int(np.floor(d + 40))
            )  # 40 because the fractional delay filter is 81 long
    # edges.sort()
    graph = Graph(num_nodes, edges)
    graph.positions = node_pos
    return room, graph


def generateRandomDynamicRoomAndGraph(
    roomsize_range: Tuple[float, float],
    L: int,
    num_nodes: int,
    num_closest: int,
    fs: int,
    dynamic_parts: List[Dict[int, Dict]] = [{}],
    max_receive: int = 5,
    directed: bool = False,
    margin: float = 0.5,
    rng: np.random.Generator = np.random.default_rng(),
) -> List[Tuple[pr.ShoeBox, Graph]]:
    roomsize = rng.uniform(*roomsize_range, size=3)

    rt60 = L / fs * 5
    while True:
        try:
            e_absorption, max_order = pr.inverse_sabine(rt60, roomsize)
            break
        except ValueError:
            rt60 *= 1.1

    # absorption = {
    #     "ceiling": rng.uniform(0.05, 0.5),
    #     "floor": rng.uniform(0.05, 0.5),
    #     "north": rng.uniform(0.05, 0.5),
    #     "south": rng.uniform(0.05, 0.5),
    #     "east": rng.uniform(0.05, 0.5),
    #     "west": rng.uniform(0.05, 0.5),
    # }
    # absorption = rng.uniform(0.05, 0.5, size=6)
    max_order = 50

    node_pos, edges, src_pos = generateRandomWSNTop(
        num_nodes,
        roomsize,
        num_closest=num_closest,
        max_receive=max_receive,
        num_sources=1,
        directed=directed,
        rng=rng,
        margin=margin,
    )

    rooms = []
    for i, part_info in enumerate(dynamic_parts):
        room = pr.ShoeBox(
            roomsize,
            fs=fs,
            # absorption=absorption,
            materials=pr.Material(e_absorption),
            max_order=max_order,
            use_rand_ism=True,
            max_rand_disp=0.08,
        )
        for node_id, part in part_info.items():
            while True:
                phi = rng.uniform(0, 2 * np.pi)
                theta = rng.uniform(0, np.pi)
                movement = part["movement_speed"] * np.array(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ]
                )
                if np.all(node_pos[node_id] + movement < roomsize - margin) and np.all(
                    node_pos[node_id] + movement > 0 + margin
                ):
                    break
            node_pos[node_id] += movement
        room.add_microphone_array(node_pos.T)
        room.add_source(src_pos[0, :])
        room.compute_rir()
        # delays = []
        # for node_i, src in enumerate(room.rir):
        #     delays_ = []
        #     for s_i, rir in enumerate(src):
        #         d = np.linalg.norm(node_pos[node_i] - src_pos[s_i]) / room.c * fs
        #         delays_.append(d)
        #     delays.append(np.min(delays_))
        # min_delay = np.min(delays)
        # for node_i, src in enumerate(room.rir):
        #     for s_i, rir in enumerate(src):
        #         rir[:] = np.roll(
        #             rir, -int(np.floor(min_delay + 39))
        #         )  # 40 because the fractional delay filter is 81 long
        for node_i, src in enumerate(room.rir):
            for s_i, rir in enumerate(src):
                d = np.linalg.norm(node_pos[node_i] - src_pos[s_i]) / room.c * fs
                rir[:] = np.roll(
                    rir, -int(np.floor(d + 40))
                )  # 40 because the fractional delay filter is 81 long

        graph = Graph(num_nodes, edges)
        graph.positions = node_pos.copy()
        rooms.append((room, graph))

    return rooms


def plotTopology2D(graph: Graph, roomsize: np.ndarray, src_pos: np.ndarray):
    fig_top = plt.figure()
    ax = fig_top.add_subplot(111)
    for edge in graph.edges:
        ax.arrow(
            graph.positions[edge[0]][0],
            graph.positions[edge[0]][1],
            graph.positions[edge[1]][0] - graph.positions[edge[0]][0],
            graph.positions[edge[1]][1] - graph.positions[edge[0]][1],
            color="k",
            alpha=0.8,
            lw=1,
            width=0.04,
            length_includes_head=True,
            # arrow_length_ratio=0.1,
        )
        # ax.arrow(
        #     [graph.positions[edge[0]][0], graph.positions[edge[1]][0]],
        #     [graph.positions[edge[0]][1], graph.positions[edge[1]][1]],
        #     [graph.positions[edge[0]][2], graph.positions[edge[1]][2]],
        #     "k-",
        # )
    ax.scatter(graph.positions[:, 0], graph.positions[:, 1])
    ax.scatter(src_pos[0], src_pos[1], marker="x", color="r")
    ax.set_aspect("equal")
    ax.set_xlim([0, roomsize[0]])
    ax.set_ylim([0, roomsize[1]])
    return fig_top


def plotTopology3D(graph: Graph):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for edge in graph.edges:
        ax.quiver(
            graph.positions[edge[0]][0],
            graph.positions[edge[0]][1],
            graph.positions[edge[0]][2],  # <-- starting point of vector
            graph.positions[edge[1]][0] - graph.positions[edge[0]][0],
            graph.positions[edge[1]][1] - graph.positions[edge[0]][1],
            graph.positions[edge[1]][2]
            - graph.positions[edge[0]][2],  # <-- directions of vector
            color="k",
            alpha=0.8,
            lw=1,
            arrow_length_ratio=0.1,
        )
        # ax.arrow(
        #     [graph.positions[edge[0]][0], graph.positions[edge[1]][0]],
        #     [graph.positions[edge[0]][1], graph.positions[edge[1]][1]],
        #     [graph.positions[edge[0]][2], graph.positions[edge[1]][2]],
        #     "k-",
        # )
    ax.scatter(*graph.positions.T)
    return fig, ax


def generateRandomWSNTop(
    num_nodes: int,
    room_dim: list | np.ndarray,
    num_closest: int = 0,
    max_receive: int = 5,
    margin: float = 0.5,
    num_sources: int = 0,
    directed: bool = False,
    rng=np.random.default_rng(),
) -> Tuple[np.ndarray, list] | Tuple[np.ndarray, list, np.ndarray]:
    """
    generates a random WASN network topology within a room

    Parameters
    ----------
    num_nodes: int
            Number of nodes in the network
    room_dim: list, ndarray
            room dimensions within which the nodes are placed
    num_closest: int
            Number of minumum connections per node (closest neighbors)
            If no value given, no connections generated
    margin: float
            minimum distance from room walls
    rng: Generator
            random number generator


    Returns
    -------
    node_pos: ndarray
            Node positions in 3d space
    network_edges: list
            list of edges (connections) between nodes
    """
    node_pos = np.asarray(
        [
            rng.uniform(0.0 + margin, room_dim[0] - margin, size=(num_nodes,)),
            rng.uniform(0.0 + margin, room_dim[1] - margin, size=(num_nodes,)),
            rng.uniform(0.0 + margin, room_dim[2] - margin, size=(num_nodes,)),
        ]
    ).T

    network_edges = generateNetworkEdges(num_nodes, node_pos, num_closest, max_receive)

    if num_sources > 0:
        src_pos = np.asarray(
            [
                rng.uniform(0.0 + margin, room_dim[0] - margin, size=(num_sources,)),
                rng.uniform(0.0 + margin, room_dim[1] - margin, size=(num_sources,)),
                rng.uniform(0.0 + margin, room_dim[2] - margin, size=(num_sources,)),
            ]
        ).T

        return (node_pos, network_edges, src_pos)
    return (node_pos, network_edges)


def generateNetworkEdges(
    num_nodes: int, node_pos: np.ndarray, num_closest: int, max_receive: int
) -> List[Tuple[int, int]]:
    max_receive = int(np.minimum(max_receive, num_nodes - 1))
    network_edges = []
    transmit = {}
    receive = {}
    for i in range(num_nodes):
        receive[i] = [i]
        transmit[i] = []
    # first "ring" of connections
    connected_nodes = [0]
    i = 0
    while len(connected_nodes) < num_nodes:
        network_edges.append((i, i))
        d = []
        di = []
        for j in [x for x in range(num_nodes) if x not in connected_nodes and x != i]:
            d.append(np.linalg.norm(node_pos[i, :] - node_pos[j, :]))
            di.append(j)
        idx = int(np.argmin(np.asarray(d)))
        idx = di[idx]
        if (i, idx) not in network_edges and idx not in connected_nodes:
            network_edges.append((i, idx))
            connected_nodes.append(idx)
            transmit[i].append(idx)
            receive[idx].append(i)
            i = idx

    network_edges.append((idx, idx))
    network_edges.append((connected_nodes[-1], connected_nodes[0]))
    transmit[connected_nodes[-1]] = [connected_nodes[0]]
    receive[connected_nodes[0]].append(connected_nodes[-1])

    for i in range(num_nodes):
        d = []
        for j in range(num_nodes):
            d.append(np.linalg.norm(node_pos[i, :] - node_pos[j, :]))
        idx = np.argsort(np.asarray(d))
        for j in idx:
            if (i, j) not in network_edges and len(receive[j]) < max_receive:
                network_edges.append((i, j))
                transmit[i].append(j)
                receive[j].append(i)
            if len(transmit[i]) >= num_closest:
                break

    return network_edges


def getPart(index, part_len) -> int:
    return int(np.floor(index / part_len))


def getNoisySignal(
    signal: np.ndarray,
    IRs: np.ndarray,
    SNR: np.float64,
    rng=np.random.default_rng(),
    noise_cov: np.ndarray = None,
) -> np.ndarray:
    """
    Convolves clean signal with IRs and adds noise at SNR.

    Parameters
    ----------
    signal: np.ndarray
            clean signal (single channel)
    IRs: np.ndarray
            LxM array containing IRs
    a: Array-like
            target norm values of IRs (must have N elements)
    rng: Generator
            The random generator to be used

    Returns
    -------
    noisy_signal: np.ndarray
            convolved and noisy signals
    """
    N_s = signal.shape[0]
    L = IRs.shape[0]
    M = IRs.shape[1]

    noise_cov = np.eye(M * L) if noise_cov is None else noise_cov

    convolved_signal = np.zeros(shape=(N_s + L - 1, M))
    for m in range(M):
        convolved_signal[:, m] = np.convolve(signal.squeeze(), IRs[:, m].squeeze())
    covar_s = np.var(convolved_signal.T)
    n_var = 10 ** (-SNR / 10) * covar_s / M
    noise_cov = noise_cov / np.trace(noise_cov) * n_var
    noise = None
    if n_var > 0:
        noise = generateMCNoise(N_s + L - 1, L, M, noise_cov=noise_cov, rng=rng)
        # mic_snrs = np.var(convolved_signal, axis=0) / np.var(noise, axis=0)
        noisy_signal = convolved_signal + noise
    else:
        # mic_snrs = np.ones(M) * np.inf
        noisy_signal = convolved_signal
    return noisy_signal, noise, noise_cov


def signalToWGN(
    signal,
    nr_samples_part,
    dyn_parts,
    L,
    M,
    N_i,
    rng=np.random.default_rng(),
    plots=False,
):
    rirs_f_all = []
    leading_zeros_all = []
    mic_signals = np.zeros((nr_samples_part * len(dyn_parts) + L - 1, M))

    # Generate random network topology
    node_pos = rng.uniform(0.0, 1.0, size=(M, 3))
    edges = generateNetworkEdges(M, node_pos, num_closest=N_i, max_receive=N_i)
    graph = Graph(M, edges)
    graph.positions = node_pos

    for i, part in enumerate(dyn_parts):
        # Sample random IRs from normal distribution
        rirs_ = rng.normal(loc=0.0, scale=1.0, size=(L, M))
        if i == 0:
            rirs = rirs_
        else:
            for j in part.keys():
                rirs[:, j] = rirs_[:, j]

        # Compute leading zeros
        leading_zeros = list(np.argmax(rirs != 0, axis=0))
        leading_zeros_all.append(leading_zeros)

        # Compute FFT of IRs
        rirs_f = np.fft.fft(rirs, axis=0)
        rirs_f_all.append(rirs_f)

        # Convolve signal with RIRs
        signal_part = signal[i * nr_samples_part : (i + 1) * nr_samples_part]
        signal__ = convolveSignalWithRIRs(signal_part, rirs)
        mic_signals[
            i * nr_samples_part : (i + 1) * nr_samples_part + L - 1, :
        ] += signal__

        if plots:
            plotTopology2D(graph, roomsize=np.array([1.0, 1.0, 1.0]), src_pos=np.zeros(3))
            for m in range(M):
                plt.figure()
                plt.plot(rirs_[:, m], label=f"IR {m}")
                plt.legend()
                plt.show()

    return rirs_f_all, leading_zeros_all, mic_signals, graph


def signalToRoom(
    signal,
    nr_samples_part,
    dyn_parts,
    L,
    M,
    N_i,
    fs,
    roomsize,
    rng=np.random.default_rng(),
    plots=False,
):
    rirs_f_all = []
    leading_zeros_all = []
    mic_signals = np.zeros((nr_samples_part * len(dyn_parts) + L - 1, M))
    rooms = generateRandomDynamicRoomAndGraph(
        roomsize, L, M, N_i, fs, dyn_parts, margin=0.1, rng=rng
    )
    if plots:
        figt, axst = plt.subplots(M, 1, figsize=(10, 8))
        figf, axsf = plt.subplots(M, 1, figsize=(10, 8))
    for i, ((room, graph), part) in enumerate(zip(rooms, dyn_parts)):
        rirs_ = removeInitialDelayFromRIRs(room.rir, L, False)
        signal_power = np.mean(rirs_**2, axis=0, keepdims=True)
        noise_power = signal_power / (10 ** (80 / 10))
        noise = rng.normal(scale=np.sqrt(noise_power), size=rirs_.shape)
        rirs_ += noise * (rirs_ != 0)
        if i == 0:
            rirs = rirs_
        else:
            for j in part.keys():
                rirs[:, j] = rirs_[:, j]
        rirs = rirs / np.linalg.norm(rirs) * M
        leading_zeros = list(np.argmax(rirs != 0, axis=0))
        leading_zeros_all.append(leading_zeros)
        rirs_f = np.fft.fft(rirs, axis=0)
        if plots:
            print("Leading zeros:", leading_zeros)
            plotTopology2D(
                graph, roomsize=room.shoebox_dim, src_pos=room.sources[0].position
            )
            for j in range(M):
                axst[j].plot(rirs[:, j], label=f"Room {i}")
                axsf[j].plot(20 * np.log10(np.abs(rirs_f[: L // 2, j])))
        rirs_f_all.append(rirs_f)
        signal__ = convolveSignalWithRIRs(
            signal[i * nr_samples_part : (i + 1) * nr_samples_part], rirs
        )
        mic_signals[
            i * nr_samples_part : (i + 1) * nr_samples_part + L - 1, :
        ] += signal__
    if plots:
        axst[0].legend()
        for ax in axst:
            ax.set_ylim((-1, 1))
        figt.tight_layout()
        figf.tight_layout()
        plt.show()
    return rirs_f_all, leading_zeros_all, mic_signals, graph


def convolveSignalWithRIRs(signal: np.ndarray, rirs: np.ndarray) -> np.ndarray:
    """
    Convolve a signal with a set of Room Impulse Responses (RIRs).

    Parameters:
        signal (np.ndarray): The input signal array of shape (N_s,).
        rirs (np.ndarray): The RIRs array of shape (L, M), where L is the length of each RIR
            and M is the number of RIRs.
    Returns:
        np.ndarray: The convolved signal array of shape (N_s, M), normalized by the maximum
            absolute value.
    """

    N_s = signal.shape[0]
    M = rirs.shape[1]
    L = rirs.shape[0]
    convolved_signal = np.zeros(shape=(N_s + L - 1, M))
    for m in range(M):
        convolved_signal[:, m] = np.convolve(
            signal.squeeze(), rirs[:, m].squeeze(), mode="full"
        )
    convolved_signal = convolved_signal / np.abs(convolved_signal).max()
    return convolved_signal


def generateNoiseCovarianceMatrix(
    N: int, alpha: float = 1.0, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """
    Generates a random noise covariance matrix.
    (A positive definite matrix with a unit norm.)

    Parameters:
    - N (int): The size of the noise covariance matrix.
    - rng (np.random.Generator): The random number generator.

    Returns:
    - sigma (np.ndarray): The noise covariance matrix.
    """
    c = rng.normal(size=(N, N))
    sigma = c.T @ c
    sigma = sigma * alpha + np.eye(N) * (1 - alpha)
    sigma = sigma / np.linalg.norm(sigma)
    return sigma


def estimateSignalCovarianceMatrix(
    signal: np.ndarray, L: int, samples: int = 1000
) -> np.ndarray:
    """
    Estimates the signal covariance matrix from a signal.

    Parameters:
    - signal (np.ndarray): The signal from which the covariance matrix is estimated.
    - L (int): The length of each segment used for covariance estimation.

    Returns:
    - R (np.ndarray): The signal covariance matrix.
    """
    for k in range(0, signal.shape[0] - L, L):
        s_ = signal[k : k + L, :].reshape(-1)
        if k == 0:
            R = np.outer(s_, s_)
        else:
            R += np.outer(s_, s_)
    return R / (k - 1)


def generateMCNoise(
    n: int,
    L: int,
    M: int,
    noise_cov: np.ndarray,
    noise_mean: np.ndarray = None,
    noise_dist: str = "normal",
    rng=np.random.default_rng(),
) -> np.ndarray:
    assert noise_cov.shape == (
        M * L,
        M * L,
    ), f"The noise covariance has to be (M*L, M*L)=\
        ({M*L}, {M*L})"
    noise_mean = np.zeros((noise_cov.shape[0],)) if noise_mean == None else noise_mean
    assert noise_mean.shape == (
        M * L,
    ), f"The noise mean has to be (M*L, )=\
        ({M*L}, )"
    # size = noise_cov.shape[0]
    # noise = np.zeros((n, M))
    samples = int(n / L) + 1
    if noise_dist == "normal":
        X = rng.normal(size=(samples, M * L)).T
        for nn in range(X.shape[0]):
            X[nn] = X[nn] - X[nn].mean()

        # L_inv = np.linalg.cholesky(np.cov(X))
        # L_inv = np.linalg.inv(L_inv)
        # X = np.dot(L_inv, X)

        L_ = np.linalg.cholesky(noise_cov)
        X = np.dot(L_, X)
        for nn in range(X.shape[0]):
            X[nn] = X[nn] + noise_mean[nn]
        noise = []
        for m in range(M):
            noise.append(X[m * L : (m + 1) * L, :].T.reshape((samples * L,)))
        noise = np.asarray(noise)
    else:
        raise ValueError("Noise distribution not supported.")

    return noise[:, :n].T


def getNoiseFromSNR(
    signal, vad, SNR, M, L, alpha=1.0, ir_type="room", rng=np.random.default_rng()
):
    """
    Generate noise based on the given Signal-to-Noise Ratio (SNR).

    Parameters:
        signal (np.ndarray): The input signal array.
        vad (np.ndarray): Voice activity detection array.
        SNR (float): Desired Signal-to-Noise Ratio.
        M (int): Number of microphones.
        L (int): Length of the signal segments.
        rng (np.random.Generator, optional): Random number generator. Defaults to
            np.random.default_rng().
    Returns:
        np.ndarray: Generated noise array.
    """

    signal_vad = signal[np.max(vad, axis=1), :]
    # sigma_s = estimateSignalCovarianceMatrix(signal_vad, L)
    sigma_s = np.var(signal_vad)
    nr_samples = signal.shape[0]
    if ir_type == "room":
        SNR = SNR + 3
    tr_v = sigma_s * 10 ** (
        -SNR / 10
    )  # we add 3 dB because it seems like this compensates for the unorthodox noise generation.
    sigma_v = generateNoiseCovarianceMatrix(
        M * L,
        alpha=alpha,
        rng=rng,
    )
    noise = generateMCNoise(
        nr_samples,
        L,
        M,
        noise_cov=sigma_v,
        rng=rng,
    )
    # sigma_v_est = estimateSignalCovarianceMatrix(noise, L)
    sigma_v_est = np.var(noise)
    noise = (
        noise
        / np.sqrt(sigma_v_est)
        * np.sqrt(tr_v * signal.shape[0] / signal_vad.shape[0])
    )
    # noise = (
    #     noise
    #     / np.sqrt(np.trace(sigma_v_est))
    #     * np.sqrt(tr_v * signal.shape[0] / signal_vad.shape[0])
    # )
    return noise


def getMicSNRs(signal, vad, noise):
    mic_snrs = np.var(signal[np.max(vad, axis=1), :], axis=0) / np.var(
        noise[np.max(vad, axis=1), :], axis=0
    )

    return mic_snrs


def computeDFTMatrix(N: int):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2 * np.pi * 1j / N)
    W = np.power(omega, i * j) / np.sqrt(N)
    return W


def computeOverlapSaveMatrices(L: int):
    F_LxL = computeDFTMatrix(L)
    F_2Lx2L = computeDFTMatrix(2 * L)
    W_01_Lx2L = np.concatenate([np.zeros(shape=(L, L)), np.eye(L)]).T
    W_10_2LxL = np.concatenate([np.eye(L), np.zeros(shape=(L, L))])
    W_01_Lx2L_fq = F_LxL @ W_01_Lx2L @ np.linalg.inv(F_2Lx2L)
    W_10_2LxL_fq = F_2Lx2L @ W_10_2LxL @ np.linalg.inv(F_LxL)
    W_10_Lx2L = np.concatenate([np.eye(L), np.zeros(shape=(L, L))]).T
    # W_01_2LxL = np.concatenate([np.zeros(shape=(L, L)), np.eye(L)])
    W_10_Lx2L_fq = F_LxL @ W_10_Lx2L @ np.linalg.inv(F_2Lx2L)
    # W_01_2LxL_fq = F_2Lx2L @ W_01_2LxL @ np.linalg.inv(F_LxL)
    W_R = W_01_Lx2L_fq.conj().T @ W_01_Lx2L_fq
    return W_10_Lx2L_fq, W_R, W_10_2LxL_fq


# "https://zenodo.org/records/7389996/files/MYRiAD_V2_econ.zip?download=1"


def downloadMyriad(
    path: str = ".myriad",
    remove_archive: bool = False,
):
    """
    Downloads the MYRiAD dataset.
    """

    def printProgress(count, blockSize, totalSize):
        percent = int(count * blockSize * 100 / totalSize)

    url = "https://zenodo.org/records/7389996/files/MYRiAD_V2_econ.zip?download=1"
    filename = "MYRiAD_V2_econ.zip"
    download_dir = ".temp"

    if os.path.exists(path):
        print("MYRiAD already exists. Skipping download and extraction.")
        return

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    if not os.path.exists(os.path.join(download_dir, filename)):
        print("Downloading MYRiAD dataset...")
        urlretrieve(url, os.path.join(download_dir, filename), printProgress)
    else:
        print("MYRiAD already downloaded.")
    if not os.path.exists(path):
        print("Extracting MYRiAD dataset...")
        with zipfile.ZipFile(os.path.join(download_dir, filename), "r") as zip:
            zip.extractall(path)
    if os.path.exists(filename) and remove_archive:
        os.remove(filename)


SRC = [
    "S0_1",
    "S0_2",
    "S30_1",
    "S-30_1",
    "S45_2",
    "S-45_2",
    "S60_1",
    "S-60_1",
    "S90_1",
    "S-90_1",
]
HA_SET = ["BTELB", "BTELF", "BTERB", "BTERF"]
EXT_SET = ["XM1", "XM2", "XM3", "XM4", "XM5"]
SRC_POS = {
    "S0_1": [0, 0.97, 1.3],
    "S0_2": [0, 1.97, 1.3],
    "S30_1": [-0.485, 0.840044641670906, 1.3],
    "S-30_1": [0.485, 0.840044641670905, 1.3],
    "S45_2": [-1.3930003589375, 1.3930003589375, 1.3],
    "S-45_2": [1.3930003589375, 1.3930003589375, 1.3],
    "S60_1": [-0.840044641670906, 0.485, 1.3],
    "S-60_1": [0.840044641670906, 0.485, 1.3],
    "S90_1": [-0.97, 0, 1.3],
    "S-90_1": [0.97, 0, 1.3],
}

MIC_POS = {
    "BTELF": [-0.08, -0.03, 1.336],
    "BTELB": [-0.08, -0.042, 1.334],
    "BTERF": [0.08, -0.03, 1.336],
    "BTERB": [0.08, -0.042, 1.334],
    "XM1": [0, 0.01, 1],
    "XM2": [0, 0.5, 1],
    "XM3": [0, 0.99, 1],
    "XM4": [-0.353553390593274, 0.353553390593274, 1],
    "XM5": [0.353553390593274, 0.353553390593274, 1],
}


def getRandomMyriadRIRsAndTopology(
    path: str = ".myriad",
    L: int = 64,
    fs: float = 4000,
    num_externals: int = 0,
    num_closest: int = 1,
    max_receive: int = 1,
    rng: np.random.Generator = np.random.default_rng(),
):
    global SRC
    global HA_SET
    global EXT_SET
    global MIC_POS

    subpath = "MYRiAD_V2_econ/audio/SAL"
    src = rng.choice(SRC)
    path_ = os.path.join(path, subpath, src)
    ext = rng.choice(EXT_SET, size=num_externals, replace=False)
    # load hearing aids RIRs
    rirs = []
    pos = []
    for ha in HA_SET:
        rir_, fs_ = sf.read(os.path.join(path_, ha + "_RIR.wav"))
        d = np.linalg.norm(np.subtract(MIC_POS[ha], SRC_POS[src])) / 340 * fs_
        rir_ = np.roll(rir_, -int(np.floor(d) + 20))
        rir_ = trim_rir_before_direct_sound(rir_, fs_, fs, L)
        rirs.append([rir_])
        pos.append(MIC_POS[ha])
    # load external RIRs
    for ex in ext:
        rir_, fs_ = sf.read(os.path.join(path_, ex + "_RIR.wav"))
        d = np.linalg.norm(np.subtract(MIC_POS[ex], SRC_POS[src])) / 340 * fs_
        rir_ = np.roll(rir_, -int(np.floor(d) + 20))
        rir_ = trim_rir_before_direct_sound(rir_, fs_, fs, L)
        rirs.append([rir_])
        pos.append(MIC_POS[ex])

    # rirs = np.asarray(rirs).T

    pos = np.asarray(pos)
    # generate topology
    # num_nodes = rirs.shape[1]
    num_nodes = len(rirs)
    edges = generateNetworkEdges(
        num_nodes, pos, num_closest=num_closest, max_receive=max_receive
    )
    graph = Graph(num_nodes, edges)
    graph.positions = pos
    return rirs, graph


def signalToMeas(
    signal,
    nr_samples_part,
    dyn_parts,
    L,
    N_i,
    fs,
    num_externals,
    myriad_path,
    rng=np.random.default_rng(),
    plots=False,
):

    rirs_f_all = []
    leading_zeros_all = []
    M = num_externals + 4
    mic_signals = np.zeros((nr_samples_part * len(dyn_parts) + L - 1, M))

    if plots:
        figt, axst = plt.subplots(M, 1, figsize=(10, 8))
        figf, axsf = plt.subplots(M, 1, figsize=(10, 8))
    # Generate random network topology
    for i, part in enumerate(dyn_parts):
        rirs_, graph = getRandomMyriadRIRsAndTopology(
            myriad_path,
            L,
            fs,
            num_externals,
            num_closest=N_i,
            max_receive=N_i,
            rng=rng,
        )
        rirs_ = removeInitialDelayFromRIRs(rirs_, L, False)
        signal_power = np.mean(rirs_**2, axis=0, keepdims=True)
        noise_power = signal_power / (10 ** (20 / 10))
        noise = rng.normal(scale=np.sqrt(noise_power), size=rirs_.shape)
        rirs_ += noise * (rirs_ != 0)
        # Sample random IRs from normal distribution
        if i == 0:
            rirs = rirs_
        else:
            for j in part.keys():
                rirs[:, j] = rirs_[:, j]

        # Compute leading zeros
        leading_zeros = list(np.argmax(rirs != 0, axis=0))
        leading_zeros_all.append(leading_zeros)

        # Compute FFT of IRs
        rirs_f = np.fft.fft(rirs, axis=0)
        rirs_f_all.append(rirs_f)

        if plots:
            print("Leading zeros:", leading_zeros)
            # plotTopology2D(
            #     graph, roomsize=, src_pos=room.sources[0].position
            # )
            for j in range(M):
                axst[j].plot(rirs[:, j], label=f"Room {i}")
                axsf[j].plot(20 * np.log10(np.abs(rirs_f[: L // 2, j])))
        # Convolve signal with RIRs
        signal_part = signal[i * nr_samples_part : (i + 1) * nr_samples_part]
        signal__ = convolveSignalWithRIRs(signal_part, rirs)
        mic_signals[
            i * nr_samples_part : (i + 1) * nr_samples_part + L - 1, :
        ] += signal__

    if plots:
        axst[0].legend()
        for ax in axst:
            ax.set_ylim((-1, 1))
        figt.tight_layout()
        figf.tight_layout()
        plt.show()

    return rirs_f_all, leading_zeros_all, mic_signals, graph


def downloadLibrispeech(
    path: str = ".librispeech",
    dataset: str = "train-clean-100",
    remove_archive: bool = False,
):
    """
    Downloads the Librispeech dataset.
    """

    # url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    def printProgress(count, blockSize, totalSize):
        percent = int(count * blockSize * 100 / totalSize)
        # print(f"\rDownloading: {percent}%", end="")

    def members(tf: tarfile.TarFile):
        l = len("LibriSpeech/")
        for member in tf.getmembers():
            if member.path.startswith("LibriSpeech/"):
                member.path = member.path[l:]
                yield member

    url = f"https://www.openslr.org/resources/12/{dataset}.tar.gz"
    filename = f"{dataset}.tar.gz"
    download_dir = ".temp"

    if os.path.exists(path) and os.path.exists(os.path.join(path, dataset)):
        print("Librispeech already exists. Skipping download and extraction.")
        print(os.getpid())
        return

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    if not os.path.exists(os.path.join(download_dir, filename)):
        print("Downloading Librispeech dataset...")
        urlretrieve(url, os.path.join(download_dir, filename), printProgress)
    else:
        print("Librispeech already downloaded.")
    if not os.path.exists(path):
        print("Extracting Librispeech dataset...")
        with tarfile.open(os.path.join(download_dir, filename)) as tar:
            tar.extractall(path, members=members(tar))
    if os.path.exists(filename) and remove_archive:
        os.remove(filename)


def librispeechSampleRandomizer(
    path: str = ".librispeech",
    dataset: str = "train-clean-100",
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Randomly samples audio files from the specified folder path.

    Args:
        folder_path (str, optional): The path to the folder containing the audio files. Defaults to ".librispeech/train-clean-100".
        rng (numpy.random.Generator, optional): The random number generator. Defaults to np.random.default_rng().

    Yields:
        str: The path to a randomly selected audio file.
    """
    path_ = os.path.join(path, dataset)
    files = {}
    speakers = os.listdir(path_)
    speakers.sort()
    rng.shuffle(speakers)
    for speaker in speakers:
        files = []
        speaker_path = os.path.join(path_, speaker)
        for root, _, file in os.walk(speaker_path):
            files_ = [os.path.join(root, f) for f in file if f.endswith(".flac")]
            files_.sort()
            rng.shuffle(files_)
            files += files_
        yield files


def _loadSample(path: str, fs: float, len: int = None) -> np.ndarray:
    signal, fs_ = sf.read(path)
    if len is not None:
        signal = signal[:len]
    signal = signal / np.abs(signal).max() * 1.0
    signal = signal - signal.mean()
    num_samples = int(signal.shape[0] * fs / fs_)
    signal = sig.resample(signal, num_samples)
    return signal


def _prepareSpeechSample(
    sample: np.ndarray,
    threshold: float = -60,
    pre_padding: int = 0,
    partitions: int = 8,
    part_rand: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    sample_split = np.array_split(sample, partitions)
    interleaved = [np.zeros((pre_padding,))]
    for part in sample_split:
        if not part_rand:
            len_ = part.shape[0]
        else:
            len_ = rng.integers(part.shape[0] // 8, part.shape[0] // 1.5)
        interleaved.append(part)
        interleaved.append(np.zeros(len_))
    new_sample = np.concatenate(interleaved)
    sq_sample = np.square(new_sample)
    sq_sample = np.convolve(sq_sample, np.ones(100), "same")
    factor = 10 ** (threshold / 20)
    binary = sq_sample < factor
    new_sample[binary] = 0
    return new_sample


def getSpeechSample(
    N: int,
    fs: float,
    paths: List[str],
    vad_threshold: float,
    padding: Tuple[int, int] = (2000, 2000),
    partitions: int = 8,
    part_rand: bool = False,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """
    Load a random sample, concatenated from a list of recordings to fill target length N
    from the Librispeech database.

    Args:
        N (int): The desired length of the speech sample.
        fs (float): The sampling rate of the speech sample.
        paths (List[str]): List of paths to the Librispeech database.
        vad_threshold (float): The VAD threshold for speech activity detection.
        padding (Tuple[int, int], optional): The amount of padding to add to the speech sample. Defaults to (2000, 2000).
        partitions (int, optional): The number of partitions to divide the speech sample into for VAD. Defaults to 8.
        part_rand (bool, optional): Whether to randomly select partitions for VAD. Defaults to False.
        rng (np.random.Generator, optional): The random number generator. Defaults to np.random.default_rng().

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the generated speech sample and the voice activity detection (VAD) signal.

    """
    signal = np.zeros((padding[0],))
    for path in paths:
        signal_ = _loadSample(path, fs)
        signal_ = _prepareSpeechSample(
            signal_,
            vad_threshold,
            padding[0],
            partitions,
            part_rand,
            rng,
        )
        signal = np.concatenate((signal, signal_))
        if signal.shape[0] >= N:
            break
    if signal.shape[0] < N:
        signal = np.tile(signal, N // signal.shape[0] + 1)
    signal = signal / np.abs(signal).max()
    signal = signal[:N]
    if padding[1] > 0:
        signal[-padding[1] :] = 0
    return signal[:, np.newaxis]


def getWhitenoiseSample(
    N: int,
    partitions: int = 8,
    part_rand: bool = False,
    amp_rand: Tuple[float, float] = (1.0, 1.0),
    rng=np.random.default_rng(),
) -> np.ndarray:
    """
    Generate a whitenoise sample of length N.

    Parameters:
        N (int): The length of the whitenoise sample.
        partitions (int, optional): The number of partitions to divide the sample into. Defaults to 8.
        part_rand (int, optional): The maximum random deviation from the partition length. Defaults to 0.
        amp_rand (bool, optional): Whether to apply random amplitude scaling. Defaults to False.
        rng (numpy.random.Generator, optional): The random number generator. Defaults to np.random.default_rng().

    Returns:
        tuple: A tuple containing the generated whitenoise sample and the voice activity detection (VAD) array.
    """
    part_len = N // partitions
    if rng.choice([False, True]):
        vad_ = [False, True] * 2 * partitions
    else:
        vad_ = [True, False] * 2 * partitions

    signal = np.empty((0,))
    part = 0
    while len(signal) < N:
        if not part_rand:
            len_ = part_len
        else:
            len_ = rng.integers(part_len // 100, part_len)
        if amp_rand[0] != amp_rand[1]:
            amp_ = rng.uniform(amp_rand[0], amp_rand[1])
        else:
            amp_ = amp_rand[0]
        if vad_[part]:
            signal = np.concatenate((signal, rng.normal(size=len_) * amp_))
        else:
            signal = np.concatenate((signal, np.zeros(len_)))
        part += 1
    signal = signal / np.abs(signal).max()
    return signal[:N]


def getVADFromCleanSignal(
    signal: np.ndarray, threshold: float, smoothing_length: int = 100
) -> np.ndarray:
    """
    Computes the Voice Activity Detection (VAD) from a clean signal.
    This is a very rudimentary approach specifically for this simulation
    which relies on the knowledge of the clean signal
    and should be replaced by a more sophisticated method in practice.

    Args:
        signal (np.ndarray): The input signal.
        threshold (float): The threshold value in decibels (dB) for VAD.
        smoothing_length (int, optional): The length of the smoothing window. Defaults to 100.

    Returns:
        np.ndarray: The VAD array where 1 represents voice activity and 0 represents silence.
    """
    M = signal.shape[1]
    sq_signal = np.square(signal)
    smoothed_sq_signal = np.zeros_like(sq_signal)
    # smoothed_sq_signal = median_filter(sq_signal, size=(smoothing_length,), axes=0)
    for m in range(M):
        smoothed_sq_signal[:, m] = np.convolve(
            sq_signal[:, m].squeeze(), np.ones(smoothing_length), "same"
        )
    factor = 10 ** (threshold / 10)
    vad = smoothed_sq_signal > factor
    vad = median_filter(vad, size=(smoothing_length,), axes=0)
    return vad


# %%
def runAlgorithm(
    alg_id, alg, misalignment, estimates, noisy_signal, vad, rirs_f, frame_size
):
    run_error = []
    nr_samples = noisy_signal.shape[0]
    L = alg.L
    hopsize = L
    t = 0
    start_time = time.time()
    for k in range(0, nr_samples - frame_size, hopsize):
        t += alg.step(
            noisy_signal[k : k + frame_size, :],
            vad[k + frame_size // 2, :],
        )
        h_hat = alg.getEstimate()
        mis = getMisalignment(h_hat, rirs_f)
        run_error.append(mis)
    t = t / k
    t_run = time.time() - start_time
    misalignment[alg_id].append(run_error)
    print(
        "[",
        time.asctime(),
        "]",
        "Algorithm:",
        alg_id,
        "Error:",
        run_error[-1],
        "Comp. time:",
        t,
        "Run time:",
        t_run,
    )
    estimates[alg_id].append(h_hat)
    return t


# %%
def runAlgorithmDyn(
    alg_id,
    alg,
    misalignment,
    estimates,
    noisy_signal,
    vad,
    delays,
    rirs_f,
    frame_size,
    part_len,
):
    types = ["NPM", "NPMA", "NPMM"]
    run_error = [[1] * len(types)]
    nr_samples = noisy_signal.shape[0]
    L = alg.L
    hopsize = L
    t = 0
    start_time = time.time()
    last_part = -1
    for k in range(0, nr_samples - frame_size, hopsize):
        # print("Frame:", k)
        part = getPart(k, part_len)
        if part != last_part:
            try:
                print("Setting delays for part", part)
                alg.setDelays(delays[part])
            except AttributeError:
                pass
            last_part = part
        t += alg.step(
            noisy_signal[k : k + frame_size, :],
            vad[k + frame_size // 2, :],
        )
        h_hat = alg.getEstimate()
        mis = getMisalignment(h_hat, rirs_f[part].T.reshape(-1, 1), types, L)
        run_error.append(mis)
    t = t / k
    t_run = time.time() - start_time
    misalignment[alg_id].append(run_error)
    print(
        "[",
        time.asctime(),
        "]",
        "Algorithm:",
        alg_id,
        "Error:",
        run_error[-1],
        "Comp. time:",
        t,
        "Run time:",
        t_run,
    )
    estimates[alg_id].append(h_hat)
    return t