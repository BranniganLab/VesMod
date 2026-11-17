"""Shared filesystem utilities work without importing command-line modules."""

import subprocess
import sys


def test_filesystem_utilities_work_without_cli_imports(tmp_path):
    """Discover inputs, map outputs, and clean a manifest in a fresh process."""
    script = '''
from pathlib import Path
import sys

from vesmod.io.input_selection import select_input_files
from vesmod.io.path_mapping import map_output_path, relative_selected_path
from vesmod.io.artifacts import remove_manifest_artifacts, validate_managed_artifacts

source_root = Path("input")
source_root.mkdir()
source = source_root / "sample.npy"
source.touch()
files, root = select_input_files(source_root, ".npy", recursive=False)
assert files == [source.resolve()]
assert relative_selected_path(files[0], root) == Path("sample.npy")
output = map_output_path(files[0], root, Path("output"), suffix=".json")
output.parent.mkdir()
output.touch()
unrelated = output.parent / "unrelated.json"
unrelated.touch()
artifacts = [output.name]
assert validate_managed_artifacts(output.parent, artifacts, {".json"}, "test") == [output]
remove_manifest_artifacts(
    output.parent,
    {"artifacts": artifacts},
    manifest_key="artifacts",
    manifest_name="test",
    allowed_suffixes={".json"},
    metadata_files=(),
)
assert not output.exists()
assert unrelated.is_file()
assert not any(name == "vesmod.cli" or name.startswith("vesmod.cli.") for name in sys.modules)
'''
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)
