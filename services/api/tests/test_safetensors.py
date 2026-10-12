import json
import struct
from pathlib import Path

import pytest
from wasifyai_art_api.safetensors import MAX_HEADER_BYTES, InvalidSafetensors, read_header

from tests.helpers import write_safetensors


def test_reads_header_and_whitelisted_metadata(tmp_path: Path) -> None:
    p = write_safetensors(
        tmp_path / "a.safetensors",
        {"modelspec.title": "Kaia", "ss_output_name": "kaia", "secret": "ignored"},
    )
    h = read_header(p)
    assert h.tensor_count == 2
    assert h.metadata == {"modelspec.title": "Kaia", "ss_output_name": "kaia"}
    assert len(h.header_sha256) == 64


def test_trigger_phrase_wins_over_tag_frequency(tmp_path: Path) -> None:
    freq = json.dumps({"ds": {"kaia_x": 50, "1girl": 80, "smile": 10}})
    p = write_safetensors(tmp_path / "a.safetensors", {"ss_tag_frequency": freq})
    assert read_header(p).trigger_candidates == ["1girl", "kaia_x", "smile"]
    p = write_safetensors(
        tmp_path / "b.safetensors", {"ss_tag_frequency": freq, "modelspec.trigger_phrase": "kaia_x"}
    )
    assert read_header(p).trigger_candidates == ["kaia_x"]


def test_rejects_short_file(tmp_path: Path) -> None:
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"abc")
    with pytest.raises(InvalidSafetensors):
        read_header(p)


def test_rejects_huge_header_length_without_reading_it(tmp_path: Path) -> None:
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1) + b"{}")
    with pytest.raises(InvalidSafetensors, match="not acceptable"):
        read_header(p)


def test_rejects_non_json_and_truncated(tmp_path: Path) -> None:
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", 4) + b"nope")
    with pytest.raises(InvalidSafetensors, match="JSON"):
        read_header(p)
    good = write_safetensors(tmp_path / "g.safetensors")
    good.write_bytes(good.read_bytes()[:-4])
    with pytest.raises(InvalidSafetensors, match="truncated"):
        read_header(good)
