from __future__ import annotations

import pytest

from ai_code_review.errors import DiffError
from ai_code_review.git.diff import filter_reviewable, parse_diff

_SIMPLE_DIFF = """diff --git a/foo.py b/foo.py
index 1111111..2222222 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def foo():
-    return 1
+    return 2
+    return None
"""


def test_parse_basic_diff() -> None:
    files = parse_diff(_SIMPLE_DIFF)
    assert len(files) == 1
    f = files[0]
    assert f.path == "foo.py"
    assert f.added == 2
    assert f.removed == 1
    assert f.added_lines == [2, 3]
    assert "diff --git" in f.patch
    assert "return 2" in f.patch


def test_parse_empty_diff() -> None:
    assert parse_diff("") == []
    assert parse_diff("   \n  ") == []


def test_parse_binary_file() -> None:
    diff = """diff --git a/logo.png b/logo.png
new file mode 100644
index 0000000..1111111
Binary files /dev/null and b/logo.png differ
"""
    files = parse_diff(diff)
    assert files[0].is_binary is True
    assert files[0].added_lines == []


def test_parse_new_file() -> None:
    diff = """diff --git a/new.py b/new.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def hello():
+    return "hi"
"""
    files = parse_diff(diff)
    assert files[0].is_new is True
    assert files[0].added_lines == [1, 2]


def test_parse_deleted_file() -> None:
    diff = """diff --git a/gone.py b/gone.py
deleted file mode 100644
index 1111111..0000000
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def hello():
-    return "hi"
"""
    files = parse_diff(diff)
    assert files[0].is_deleted is True


def test_parse_multiple_files() -> None:
    diff = _SIMPLE_DIFF + "\n" + _SIMPLE_DIFF.replace("foo.py", "bar.py")
    files = parse_diff(diff)
    assert len(files) == 2
    assert {f.path for f in files} == {"foo.py", "bar.py"}


def test_parse_rename() -> None:
    diff = """diff --git a/old.py b/new.py
similarity index 90%
rename from old.py
rename to new.py
"""
    files = parse_diff(diff)
    assert files[0].path == "new.py"
    assert files[0].old_path == "old.py"


def test_parse_bad_header_raises() -> None:
    with pytest.raises(DiffError):
        parse_diff("diff --git broken\n")


def test_parse_path_with_space() -> None:
    # git appends a tab after the path in +++/--- lines when it contains
    # special chars. The parser must strip it.
    diff = (
        "diff --git a/sub dir/mod.py b/sub dir/mod.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/sub dir/mod.py\t\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )
    files = parse_diff(diff)
    assert files[0].path == "sub dir/mod.py"
    assert files[0].is_new is True
    assert files[0].added_lines == [1]


def test_parse_quoted_path() -> None:
    diff = (
        'diff --git "a/file\\"quote.py" "b/file\\"quote.py"\n'
        '--- a/file"quote.py\n'
        '+++ b/file"quote.py\n'
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    files = parse_diff(diff)
    # Quoted form: git escapes the inner quote.
    assert files[0].path == 'file"quote.py'


def test_parse_multiple_hunks() -> None:
    diff = (
        "diff --git a/multi.py b/multi.py\n"
        "--- a/multi.py\n"
        "+++ b/multi.py\n"
        "@@ -1,2 +1,2 @@\n"
        " a\n"
        "-b\n"
        "+B\n"
        "@@ -10,2 +10,2 @@\n"
        " c\n"
        "-d\n"
        "+D\n"
    )
    files = parse_diff(diff)
    assert files[0].added == 2
    assert files[0].removed == 2
    # Added lines from both hunks, with correct line numbers.
    assert files[0].added_lines == [2, 11]


def test_parse_no_newline_marker() -> None:
    diff = (
        "diff --git a/nn.py b/nn.py\n"
        "--- a/nn.py\n"
        "+++ b/nn.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
    )
    files = parse_diff(diff)
    assert files[0].added == 1
    assert files[0].removed == 1
    assert files[0].added_lines == [1]


def test_filter_reviewable_skips_binary_and_locks() -> None:
    binary = parse_diff("diff --git a/x.png b/x.png\nBinary files differ\n")
    src = parse_diff(_SIMPLE_DIFF)
    lockfile = parse_diff(
        "diff --git a/package-lock.json b/package-lock.json\n"
        "--- a/package-lock.json\n+++ b/package-lock.json\n@@ -1 +1 @@\n-1\n+2\n"
    )
    reviewable, skipped = filter_reviewable(binary + src + lockfile, max_files=50)
    assert [f.path for f in reviewable] == ["foo.py"]
    assert "x.png" in skipped
    assert "package-lock.json" in skipped


def test_filter_reviewable_caps_files() -> None:
    diffs = [_SIMPLE_DIFF.replace("foo.py", f"f{i}.py") for i in range(10)]
    files = parse_diff("\n".join(diffs))
    reviewable, skipped = filter_reviewable(files, max_files=3)
    assert len(reviewable) == 3
    assert len(skipped) == 7


def test_ignored_paths() -> None:
    for path in [
        "node_modules/x.py",
        "vendor/y.py",
        "third_party/z.py",
        "dist/bundle.min.js",
        "build/out.py",
        "__pycache__/mod.cpython-312.pyc",
        "package-lock.json",
        "Cargo.lock",
        "go.sum",
    ]:
        f = parse_diff(_SIMPLE_DIFF.replace("foo.py", path))[0]
        reviewable, _ = filter_reviewable([f], max_files=10)
        assert reviewable == [], f"expected {path} to be ignored"
