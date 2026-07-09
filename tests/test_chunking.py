from __future__ import annotations

from ai_code_review.agent.chunking import chunk_file
from ai_code_review.models import FileDiff


def test_chunk_small_patch_returns_single_chunk() -> None:
    patch = "@@ -1 +1 @@\n-x\n+y\n"
    f = FileDiff(path="a.py", patch=patch)
    chunks = chunk_file(f)
    assert chunks == [patch]


def test_chunk_empty_patch_returns_empty() -> None:
    assert chunk_file(FileDiff(path="a.py", patch="")) == []


def test_chunk_large_patch_splits() -> None:
    line = "+some changed content line\n"
    big_patch = "@@ -1 +1,2000 @@\n" + line * 1000
    f = FileDiff(path="big.py", patch=big_patch)
    chunks = chunk_file(f, max_chars=2000)
    assert len(chunks) > 1
    # No chunk should exceed max_chars by more than the overlap.
    for c in chunks:
        assert len(c) <= 2400


def test_chunk_overlaps() -> None:
    line = "+x" + ("a" * 80) + "\n"
    big_patch = "@@ -1 +1,500 @@\n" + line * 500
    f = FileDiff(path="big.py", patch=big_patch)
    chunks = chunk_file(f, max_chars=2000)
    if len(chunks) >= 2:
        # Subsequent chunk should start before the previous ended.
        assert chunks[1][:10] in chunks[0]
