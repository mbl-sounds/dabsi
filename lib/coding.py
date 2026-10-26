from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Callable

# from .admm import *
# from .huffman import (
#     Node as HuffmanNode,
#     huffman_decode,
#     _encode as huffman_encode,
#     _bits_from_bytes,
# )

# self.var_a = 0.98
# self.var_b = 0.98
# self.var_c = 0.98

# self.res_local_var = None
# self.last_local_var = None
# self.res_consensus_var = None
# self.last_consensus_var = None
# self.res_local_mean = None
# self.res_consensus_mean = None
# self.res_local_var_hist = []
# self.res_consensus_var_hist = []

# self.enc_normalizer = None
# self.dec_normalizer = None

# self.residuals = []
# self.bit_buffer = 0

# self.local_enc_normalizer = np.ones((len(self.receive),))
# self.local_dec_normalizer = np.ones((len(self.transmit),))
# self.local_enc_normalizer_hist = []
# self.local_dec_normalizer_hist = []
# self.consensus_enc_normalizer = 1
# self.consensus_dec_normalizer = np.ones((len(self.receive),))
# self.consensus_enc_normalizer_hist = []
# self.consensus_dec_normalizer_hist = []

# self.res_local_var = np.ones((len(self.receive),)) * 1e-7
# self.last_local_var = np.ones((len(self.receive),)) * 1e-7
# self.res_consensus_var = 1e-7
# self.last_consensus_var = 1e-7
# self.res_local_mean = np.zeros((len(self.receive),))
# self.res_consensus_mean = 0

# self.res_local_var_hist = []
# self.res_consensus_var_hist = []
# self.residuals = []
# self.bit_buffer = 0

# def setCodeBook(
#     self,
#     tree,
#     table,
#     bins,
#     centers,
#     ref_var,
#     smoothing,
# ):
#     node
#     for node in self.nodes.values():
#         node.setCodebook(
#             tree,
#             table,
#             bins,
#             centers,
#             ref_var,
#             smoothing,
#         )


class Connection:
    id: Tuple[int, int]
    # transmitter
    # receiver

    def __init__(self, edge: Tuple[int, int], transmitter, receiver):
        self.id = edge
        self.transmitter = transmitter
        self.receiver = receiver


class Coder:
    """
    The Coder class represents an encoder object that can be used to encode data using a specified function.

    Attributes:
        f (Callable): The encoding function.
        parameters (Dict): The parameters for the encoding function.
        in_buffer (np.ndarray): The input buffer for encoding.
        out_buffer (np.ndarray): The output buffer for encoding.

    Methods:
        __init__(self, node, var_name: str) -> None: Initializes the Coder object.
        apply(self, data) -> None: Encodes the data using the specified function.

    """

    # node
    var_name: str

    def __init__(self, node, var_name: str) -> None:
        """
        Initializes the Coder object.

        Args:
            node (Node): The node object.
            var_name (str): The variable name.

        """
        self.node = node
        self.var_name = var_name

    def apply(self, data):
        """
        Encodes the data using the specified function.

        Args:
            data (np.ndarray): The data to be encoded.

        """
        return data


class Encoder(Coder):
    """
    A class that represents an encoder.

    Attributes:
        node (Node): The node object associated with the encoder.
        var_name (str): The variable name of the encoder.

    Methods:
        __init__(self, node, var_name: str) -> None: Initializes the Encoder object.
    """

    def __init__(self, node, var_name: str) -> None:
        super().__init__(node, var_name)

    def apply(self, data):
        """
        Encodes the data using the specified function.

        Args:
            data (np.ndarray): The data to be encoded.

        """
        return data


class Decoder(Coder):
    """
    A class representing a decoder.

    Attributes:
        node (Node): The node object associated with the decoder.
        var_name (str): The variable name of the decoder.
    """

    def __init__(self, node, var_name: str) -> None:
        super().__init__(node, var_name)

    def apply(self, data):
        """
        Decodes the data using the specified function.

        Args:
            data (np.ndarray): The data to be decoded.

        """
        return data


# class ResidualEncoder(Coder):
#     data_q: np.ndarray
#     transmission_indices: np.ndarray

#     def __init__(self, node, var_name: str, L: int) -> None:
#         """
#         Initialize a ResidualEncoder object.

#         Args:
#             node (Node): The node object.
#             var_name (str): The variable name.
#             L (int): The length of the data.

#         Returns:
#             None
#         """
#         super().__init__(node, var_name)
#         self.data_q = np.zeros((L, 1), dtype=np.complex128)
#         self.transmission_indices = np.array(range(L))

#     def apply(self, data):
#         """
#         Apply the ResidualEncoder to the given data.

#         Args:
#             data: The input data.

#         Returns:
#             res_q: The result of applying the ResidualEncoder to the data.
#         """

#         res = data[self.transmission_indices] - self.data_q[self.transmission_indices]
#         res_q = res
#         self.data_q[self.transmission_indices] = (
#             self.data_q[self.transmission_indices] + res_q
#         )
#         return res_q


# class ResidualDecoder(Coder):
#     data_q: np.ndarray
#     transmission_indices: np.ndarray
#     """
#     ResidualDecoder is a class that represents a decoder for residual data.

#     Attributes:
#         data_q (np.ndarray): An array to store the decoded data.

#     Methods:
#         __init__(self, node, var_name: str, L: int) -> None:
#             Initializes a ResidualDecoder instance.
#         apply(self, data):
#             Applies the decoder to the given data.

#     """

#     def __init__(self, node, var_name: str, L: int) -> None:
#         super().__init__(node, var_name)
#         self.data_q = np.zeros((L, 1), dtype=np.complex128)
#         self.transmission_indices = np.array(range(L))

#     def apply(self, data, ind=None):
#         """
#         Applies the decoder to the given data.

#         Args:
#             data: The data to be decoded.

#         Returns:
#             The decoded data.

#         """

#         self.data_q[self.transmission_indices] += data
#         return self.data_q


# # class ResidualPartialEncoder(Coder):
# #     data_q: np.ndarray
# #     N: int
# #     transmission_indices: np.ndarray

# #     def __init__(self, node, var_name: str, L: int, N: int) -> None:
# #         """
# #         Initialize a ResidualEncoder object.

# #         Args:
# #             node (Node): The node object.
# #             var_name (str): The variable name.
# #             L (int): The length of the data.

# #         Returns:
# #             None
# #         """
# #         super().__init__(node, var_name)
# #         self.data_q = np.zeros((L, 1), dtype=np.complex128)
# #         self.N = N

# #     def apply(self, data):
# #         """
# #         Apply the ResidualEncoder to the given data.

# #         Args:
# #             data: The input data.

# #         Returns:
# #             res_q: The result of applying the ResidualEncoder to the data.
# #         """
# #         res = data - self.data_q
# #         self.transmission_indices = np.argpartition(
# #             np.abs(res) / np.abs(data), -self.N, axis=0
# #         ).squeeze()[-self.N :]
# #         print(self.transmission_indices)
# #         # self.transmission_indices = np.random.choice(data.shape[0], self.N, replace=False)
# #         res_q = res[self.transmission_indices]
# #         self.data_q[self.transmission_indices] = self.data_q[self.transmission_indices] + res_q
# #         return res_q


# # class ResidualPartialDecoder(Coder):
# #     data_q: np.ndarray
# #     N: int
# #     transmission_indices: np.ndarray
# #     """
# #     ResidualDecoder is a class that represents a decoder for residual data.

# #     Attributes:
# #         data_q (np.ndarray): An array to store the decoded data.

# #     Methods:
# #         __init__(self, node, var_name: str, L: int) -> None:
# #             Initializes a ResidualDecoder instance.
# #         apply(self, data):
# #             Applies the decoder to the given data.

# #     """

# #     def __init__(self, node, var_name: str, L: int, N: int) -> None:
# #         super().__init__(node, var_name)
# #         self.data_q = np.zeros((L, 1), dtype=np.complex128)
# #         self.N = N

# #     def apply(self, data):
# #         """
# #         Applies the decoder to the given data.

# #         Args:
# #             data: The data to be decoded.

# #         Returns:
# #             The decoded data.

# #         """

# #         self.data_q[self.transmission_indices] += data
# #         return self.data_q


# class ResidualHuffmanCoder(Coder):
#     """
#     A class that represents a Residual Huffman Coder.

#     Attributes:
#         tree (HuffmanNode): The Huffman tree used for encoding and decoding.
#         table (Dict[int, str]): The lookup table for encoding symbols.
#         bins (np.ndarray): The bins used for digitization.
#         centers (np.ndarray): The centers of the bins.
#         ref_var (float): The reference variance.
#         smoothing (float): The smoothing factor.
#         p (float): The probability factor.

#         res_var (float): The residual variance.
#         last_var (float): The last variance.
#         dig_var (float): The digitized variance.
#         normalizer (float): The normalizing factor.

#         data_q (np.ndarray): The data queue.
#         bit_buffer (int): The bit buffer.
#     """

#     tree: HuffmanNode
#     table: Dict[int, str]
#     bins: np.ndarray
#     centers: np.ndarray
#     ref_var: float
#     smoothing: float
#     p: float

#     res_var: float
#     last_var: float
#     dig_var: float
#     normalizer: float

#     bit_buffer: int

#     def __init__(
#         self,
#         node,
#         var_name: str,
#         L: int,
#         tree,
#         table,
#         bins,
#         centers,
#         ref_var,
#         smoothing,
#         p,
#     ) -> None:
#         super().__init__(node, var_name, L)
#         self.tree = tree
#         self.table = table
#         self.bins = bins
#         self.centers = centers
#         self.ref_var = ref_var
#         self.smoothing = smoothing
#         self.p = p
#         self.mirror_bin = int((len(centers) - 1) / 2)
#         self.res_var = np.ones((self.node.N_r,)) * ref_var
#         self.last_var = np.ones((self.node.N_r,)) * ref_var
#         self.dig_var = np.ones((len(self.node.receive),)) * ref_var
#         self.normalizer = 1
#         self.data_q = np.zeros((self.node.L, 1), dtype=np.complex128)
#         self.bit_buffer = 0

#     def digitize(self, vec: np.ndarray) -> List:
#         """
#         Digitizes the given vector.

#         Args:
#             vec (np.ndarray): The vector to be digitized.

#         Returns:
#             List: The digitized vector.
#         """
#         x = np.concatenate([vec.real, vec.imag]).squeeze()
#         return list(np.searchsorted(self.bins, x, side="left") - 1)

#     def dedigitize(self, digitized) -> np.ndarray:
#         """
#         Dedigitizes the given digitized vector.

#         Args:
#             digitized: The digitized vector.

#         Returns:
#             np.ndarray: The dedigitized vector.
#         """
#         nn = self.centers[digitized]
#         pp: np.ndarray = nn[: self.node.L] + 1j * nn[self.node.L :]
#         return pp.reshape(self.node.L, 1)


# class ResidualHuffmanEncoder(ResidualHuffmanCoder):
#     """
#     A class that represents a residual Huffman encoder.

#     Attributes:
#         node (Node): The root node of the Huffman tree.
#         var_name (str): The name of the variable.
#         L (int): The number of bits used to represent each symbol.
#         tree: The Huffman tree.
#         table: The Huffman encoding table.
#         bins: The number of bins used for digitization.
#         centers: The centers of the bins used for digitization.
#         ref_var: The reference variance.
#         smoothing: The smoothing factor.
#         p: The power factor.

#     Methods:
#         _hufencode(digitized): Encodes the digitized data using Huffman encoding.
#         apply(data): Applies the residual Huffman encoding to the given data.

#     """

#     def __init__(
#         self,
#         node,
#         var_name: str,
#         L: int,
#         tree,
#         table,
#         bins,
#         centers,
#         ref_var,
#         smoothing,
#         p,
#     ) -> None:
#         super().__init__(
#             node,
#             var_name,
#             L,
#             tree,
#             table,
#             bins,
#             centers,
#             ref_var,
#             smoothing,
#             p,
#         )

#     def _hufencode(self, digitized) -> List:
#         """
#         Encodes the digitized data using Huffman encoding.

#         Args:
#             digitized: The digitized data.

#         Returns:
#             The encoded data as a list.

#         """
#         encoded = huffman_encode(digitized, self.table)
#         return list(encoded)

#     def apply(self, data) -> List:
#         """
#         Applies the residual Huffman encoding to the given data.

#         Args:
#             data: The input data.

#         Returns:
#             The encoded data as a list.

#         """
#         # compute residual
#         residual = data - self.data_q
#         # value to symbol
#         digitized = self.digitize(residual / self.normalizer)
#         # compute local quantized residual
#         res_q = self.dedigitize(digitized)
#         # estimate variance
#         self.dig_var = self.dig_var * self.smoothing + (1 - self.smoothing) * np.var(
#             np.concatenate([res_q.real, res_q.imag]).squeeze()
#         )
#         # compute local quantized estimate
#         self.data_q = self.data_q + res_q * self.normalizer
#         # determine scaling update for next frame
#         self.normalizer = np.power(self.dig_var / self.ref_var, self.p)
#         # encode
#         encoded = self._hufencode(digitized)
#         self.bit_buffer += len(list(_bits_from_bytes(encoded)))
#         return encoded


# class ResidualHuffmanDecoder(ResidualHuffmanCoder):
#     """
#     A class that represents a decoder for residual Huffman coding.

#     Args:
#         node (Node): The root node of the Huffman tree.
#         var_name (str): The name of the variable.
#         L (int): The number of levels in the Huffman tree.
#         tree: The Huffman tree.
#         table: The Huffman table.
#         bins: The number of bins.
#         centers: The centers of the bins.
#         ref_var: The reference variance.
#         smoothing: The smoothing factor.
#         p: The power factor.

#     Attributes:
#         dig_var (float): The digitized variance.
#         normalizer (float): The scaling factor.

#     Methods:
#         _hufdecode(encoded): Decodes the input using Huffman decoding.
#         apply(data): Applies the decoding process to the input data.

#     """

#     def __init__(
#         self,
#         node,
#         var_name: str,
#         L: int,
#         tree,
#         table,
#         bins,
#         centers,
#         ref_var,
#         smoothing,
#         p,
#     ) -> None:
#         super().__init__(
#             node,
#             var_name,
#             L,
#             tree,
#             table,
#             bins,
#             centers,
#             ref_var,
#             smoothing,
#             p,
#         )

#     def _hufdecode(self, encoded) -> List:
#         """
#         Decodes the input using Huffman decoding.

#         Args:
#             encoded: The encoded input.

#         Returns:
#             List: The decoded output.

#         """
#         decoded = huffman_decode(encoded, self.tree)
#         return list(decoded)

#     def apply(self, data) -> np.ndarray:
#         """
#         Applies the decoding process to the input data.

#         Args:
#             data: The input data.

#         Returns:
#             np.ndarray: The scaled residual.

#         """
#         # decode
#         digitized = self._hufdecode(data)
#         # symbol to value
#         res_q = self.dedigitize(digitized)
#         # estimate variance
#         self.dig_var = self.dig_var * self.smoothing + (1 - self.smoothing) * np.var(
#             np.concatenate([res_q.real, res_q.imag]).squeeze()
#         )
#         # update local
#         # self.xy_r[:, i, None] = (
#         #     self.xy_r[:, i, None] + res_q * self.local_dec_normalizer[i]
#         # )
#         # scaled residual
#         res_q_scaled = res_q * self.normalizer
#         # determine scaling update for next frame
#         self.normalizer = np.power(self.dig_var / self.ref_var, self.p)
#         return res_q_scaled


# class TransmissionLoss:
#     """
#     Represents a transmission loss calculation.

#     Attributes:
#         f (Callable): The function used for transmission loss calculation.
#         parameters (Dict): The parameters used by the function.

#     Methods:
#         __init__(self, f: Callable = lambda x, y: x) -> None: Initializes the TransmissionLoss object.
#         apply(self) -> np.ndarray: Applies the transmission loss calculation to the given data.
#         setFunction(self, f: Callable, parameters: Dict) -> None: Sets the function and parameters for transmission loss calculation.
#     """

#     encoder: Coder
#     decoder: Decoder
#     f: Callable
#     parameters: Dict
#     in_buffer: np.ndarray | float
#     out_buffer: np.ndarray | float

#     def __init__(self, f: Callable = lambda s, x, y: x, parameters: Dict = {}) -> None:
#         """
#         Initializes the TransmissionLoss object.

#         Args:
#             f (Callable, optional): The function used for transmission loss calculation. Defaults to lambda x, y: x.
#         """
#         self.f = f
#         self.parameters = parameters

#     def _apply(self) -> np.ndarray:
#         """
#         Applies the transmission loss calculation to the given data.

#         Args:
#             data (np.ndarray): The data to apply the transmission loss calculation to.
#             data_parameters (Dict): The parameters for the data.

#         Returns:
#             np.ndarray: The result of the transmission loss calculation.
#         """
#         self.out_buffer = self.f(self, self.in_buffer, self.parameters)

#     def setFunction(self, f: Callable, parameters: Dict) -> None:
#         """
#         Sets the function and parameters for transmission loss calculation.

#         Args:
#             f (Callable): The function used for transmission loss calculation.
#             parameters (Dict): The parameters used by the function.
#         """
#         self.f = f
#         self.parameters = parameters
