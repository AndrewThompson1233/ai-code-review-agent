from __future__ import annotations

from ..models import FileDiff

# A single hunk larger than this is split into overlapping windows so we stay
# inside typical context budgets. 8k chars ~= 2k tokens, leaves headroom for
# the system prompt + JSON output.
_CHUNK = 8000
_OVERLAP = 400


def chunk_file(file_diff: FileDiff, *, max_chars: int = _CHUNK) -> list[str]:
    """Split a file's patch into reviewable windows.

    Each window is a slice of the patch text small enough to send to the LLM
    alongside the system prompt. Windows overlap by `_OVERLAP` chars so a
    defect that straddles the boundary is still caught.
    """
    patch = file_diff.patch
    if not patch:
        return []
    if len(patch) <= max_chars:
        return [patch]

    chunks: list[str] = []
    i = 0
    while i < len(patch):
        end = min(i + max_chars, len(patch))
        # Try to break on a hunk boundary to avoid splitting a `@@` header.
        if end < len(patch):
            boundary = patch.rfind("\n@@", i, end)
            if boundary > i + _OVERLAP:
                end = boundary
        chunks.append(patch[i:end])
        if end >= len(patch):
            break
        i = end - _OVERLAP
        if i <= 0:
            i = end
    return chunks
