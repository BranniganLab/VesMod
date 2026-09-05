"""Shared path helpers for VesMod command-line interfaces."""

from pathlib import Path


def validate_managed_artifacts(
    output_dir: Path,
    artifacts: object,
    allowed_suffixes: set[str],
    manifest_name: str,
) -> list[Path]:
    """Validate relative manifest paths before any destructive operation."""
    if not isinstance(artifacts, list) or any(
        not isinstance(item, str) for item in artifacts
    ):
        raise ValueError(
            f"Existing {manifest_name} has no valid artifact manifest; "
            "refusing to remove files."
        )

    root = output_dir.expanduser().resolve()
    resolved = []
    for item in artifacts:
        relative = Path(item)
        artifact = (root / relative).resolve()
        if (
            relative.is_absolute()
            or relative == Path(".")
            or artifact == root
            or not artifact.is_relative_to(root)
            or artifact.suffix not in allowed_suffixes
        ):
            raise ValueError(
                f"Existing {manifest_name} contains an unsafe artifact path; "
                "refusing to remove files."
            )
        resolved.append(artifact)
    return resolved


def remove_manifest_artifacts(
    output_dir: Path,
    provenance: object,
    *,
    manifest_key: str,
    manifest_name: str,
    allowed_suffixes: set[str],
    metadata_files: tuple[str, ...],
) -> None:
    """Remove only validated artifacts and fixed metadata owned by a batch."""
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Existing {manifest_name} is invalid; refusing to remove files."
        )
    paths = validate_managed_artifacts(
        output_dir,
        provenance.get(manifest_key),
        allowed_suffixes,
        manifest_name,
    )
    for path in paths:
        if path.is_file():
            path.unlink()
    for filename in metadata_files:
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def _display_path(path: Path) -> str:
    """Return a stable absolute path for command-line messages."""
    return str(path.expanduser().resolve())


def _relative_input_path(path: Path, input_path: Path) -> Path:
    """Return one selected file relative to the user-selected input root."""
    resolved_path = path.expanduser().resolve()
    resolved_input = input_path.expanduser().resolve()
    if resolved_path == resolved_input:
        return Path(resolved_path.name)
    return resolved_path.relative_to(resolved_input)
