"""Validation and cleanup of files recorded in batch artifact manifests."""

import json
from pathlib import Path


def validate_managed_artifacts(
    output_dir: Path,
    artifacts: object,
    allowed_suffixes: set[str],
    manifest_name: str,
) -> list[Path]:
    """Validate relative manifest paths before any destructive operation.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory containing managed output files.
    artifacts : object
        Manifest entries, required to be a list of relative path strings.
    allowed_suffixes : set[str]
        File suffixes permitted in the artifact manifest.
    manifest_name : str
        Description of the manifest used in validation errors.

    Returns
    -------
    list[pathlib.Path]
        Resolved artifact paths contained within the output directory.

    Raises
    ------
    ValueError
        If the manifest is malformed or contains an unsafe path.
    """
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
    """Remove validated artifacts and fixed metadata owned by a batch.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory containing managed output files.
    provenance : object
        Batch provenance, required to be a dictionary.
    manifest_key : str
        Provenance key containing the artifact path list.
    manifest_name : str
        Description of the manifest used in validation errors.
    allowed_suffixes : set[str]
        File suffixes permitted in the artifact manifest.
    metadata_files : tuple[str, ...]
        Fixed metadata filenames owned by the calling command.

    Returns
    -------
    None
        Deletes existing managed files after validating the artifact list.

    Raises
    ------
    ValueError
        If provenance or its artifact manifest fails validation.
    """
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


def record_manifest_artifacts(
    output_dir: Path,
    managed_artifacts: set[Path],
    *,
    provenance_filename: str,
    manifest_key: str,
) -> None:
    """Record managed output paths in an existing provenance file.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory containing managed output files and provenance.
    managed_artifacts : set[pathlib.Path]
        Output paths to record relative to the output directory.
    provenance_filename : str
        Filename of the existing JSON provenance file.
    manifest_key : str
        Provenance key to replace with the sorted artifact path list.

    Returns
    -------
    None
        Updates the provenance file while retaining its other fields.

    Raises
    ------
    ValueError
        If an artifact resolves outside the output directory or JSON is invalid.
    OSError
        If the provenance file cannot be read or written.
    """
    provenance_path = output_dir / provenance_filename
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[manifest_key] = sorted(
        str(path.resolve().relative_to(output_dir.resolve()))
        for path in managed_artifacts
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
