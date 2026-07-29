import struct

import numpy as np
import pytest

import mojo_corto as corto
from mojo_corto.codec import _dictionary, _probabilities


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"x",
        b"x" * 10000,
        b"abracadabra" * 1000,
        bytes([0, 1, 0, 2, 0, 1, 0, 3]) * 1000,
        bytes(range(255)) * 3,
    ],
)
def test_tunstall_roundtrip(data):
    block = corto.tunstall_compress(data)
    assert corto.tunstall_decompress(block) == data
    assert corto.tunstall_decompress(block.to_bytes()) == data


def test_tunstall_header_matches_outstream_layout():
    data = np.frombuffer(b"mississippi" * 100, dtype=np.uint8)
    block = corto.tunstall_compress(data)
    symbols, probabilities = _probabilities(data)
    assert block.payload[0] == symbols.size
    pairs = np.frombuffer(
        block.payload[1 : 1 + symbols.size * 2], dtype=np.uint8
    ).reshape(-1, 2)
    np.testing.assert_array_equal(pairs[:, 0], symbols)
    np.testing.assert_array_equal(pairs[:, 1], probabilities)
    size, compressed_size = struct.unpack_from("<II", block.payload, 1 + 2 * symbols.size)
    assert size == data.size
    assert compressed_size == len(block.payload) - (1 + 2 * symbols.size + 8)


def test_tunstall_low_entropy_optimized_dictionary():
    data = np.frombuffer(b"a" * 100000 + b"b" * 10 + b"c" * 3, dtype=np.uint8)
    symbols, probabilities = _probabilities(data)
    index, lengths, table = _dictionary(symbols, probabilities)
    assert index.size == lengths.size == 256
    assert lengths.max() >= 16
    assert np.all(index >= 0)
    assert np.all(index + lengths <= table.size)
    assert corto.tunstall_decompress(corto.tunstall_compress(data)) == data.tobytes()


def test_tunstall_rejects_upstream_symbol_count_overflow():
    with pytest.raises(ValueError, match="256"):
        corto.tunstall_compress(bytes(range(256)))


def test_tunstall_rejects_truncation():
    block = corto.tunstall_compress(b"geometry" * 100)
    with pytest.raises(ValueError, match="truncated"):
        corto.tunstall_decompress(block.payload[:-1])
    with pytest.raises(ValueError, match="trailing"):
        corto.tunstall_decompress(block.payload + b"\x00")
