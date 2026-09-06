"""Reusable random-access frame sources for arrays and microscopy videos."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import nd2
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FrameSource(Protocol):
    """A bounded-memory, random-access sequence of two-dimensional frames."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(frames, height, width)``."""

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return source and selection metadata."""

    def __len__(self) -> int:
        """Return the number of selected frames."""

    def __getitem__(self, index: int) -> NDArray[np.number]:
        """Read one selected two-dimensional frame."""

    def __iter__(self) -> Iterator[NDArray[np.number]]:
        """Iterate without materializing the complete video."""


class ArrayFrameSource:
    """Random-access frames backed by an in-memory or memory-mapped array."""

    def __init__(
        self,
        frames: NDArray[np.number],
        *,
        owns_frames: bool = False,
    ) -> None:
        if not isinstance(frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray or FrameSource.")
        if frames.ndim != 3:
            raise IndexError("frames must be a 3D array.")
        self._frames = frames
        self._owns_frames = owns_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(frames, height, width)``."""
        return self._frames.shape

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return metadata describing the array-backed source."""
        return {"kind": "array", "dtype": str(self._frames.dtype)}

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> NDArray[np.number]:
        return self._frames[index]

    def __setitem__(self, index: int, value) -> None:
        """Preserve ordinary mutable-array behavior for in-memory sources."""
        self._frames[index] = value

    def __iter__(self) -> Iterator[NDArray[np.number]]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """Release an array opened and owned by this source."""
        if self._owns_frames:
            self._frames = None
            self._owns_frames = False

    def __enter__(self) -> "ArrayFrameSource":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


class ND2FrameSource:
    """Lazy ND2 frames with explicit non-time axis selection.

    Parameters
    ----------
    path : str or Path
        ND2 acquisition to open.
    axis_selection : mapping, optional
        Index selected for every non-spatial, non-time axis whose size exceeds
        one. Axis names follow ``nd2.ND2File.sizes`` (for example ``P``, ``Z``,
        or ``C``). Ambiguous acquisitions are rejected rather than silently
        selecting a position, z-plane, or channel.
    """

    def __init__(
        self,
        path: str | Path,
        axis_selection: Mapping[str, int] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._file = nd2.ND2File(self.path)
        self._selection = {
            str(axis).upper(): int(index)
            for axis, index in (axis_selection or {}).items()
        }
        try:
            self._validate_selection()
            self._sequence_indices = self._select_sequence_indices()
        except (KeyError, TypeError, ValueError):
            self._file.close()
            raise

    def _validate_selection(self) -> None:
        sizes = self._file.sizes
        unknown = set(self._selection) - set(sizes)
        if unknown:
            raise ValueError(
                "Unknown ND2 selection axes: " + ", ".join(sorted(unknown))
            )
        ambiguous = [
            axis
            for axis, size in sizes.items()
            if axis not in {"T", "Y", "X"}
            and size > 1
            and axis not in self._selection
        ]
        if ambiguous:
            raise ValueError(
                "ND2 contains multiple values for axis/axes "
                f"{', '.join(ambiguous)}; provide an explicit selection."
            )
        for axis, index in self._selection.items():
            if index < 0 or index >= sizes[axis]:
                raise ValueError(
                    f"ND2 {axis} selection {index} is outside 0..{sizes[axis] - 1}."
                )

    def _select_sequence_indices(self) -> tuple[int, ...]:
        selection_without_channel = {
            axis: index
            for axis, index in self._selection.items()
            if axis != "C"
        }
        return tuple(
            raw_index
            for raw_index, loop_index in enumerate(self._file.loop_indices)
            if all(
                loop_index.get(axis, 0) == selected
                for axis, selected in selection_without_channel.items()
            )
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return selected ``(frames, height, width)`` dimensions."""
        return (
            len(self),
            int(self._file.sizes["Y"]),
            int(self._file.sizes["X"]),
        )

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return source identity, dimensions, and explicit axis selection."""
        return {
            "kind": "nd2",
            "path": str(self.path),
            "sizes": dict(self._file.sizes),
            "axis_selection": dict(self._selection),
        }

    def __len__(self) -> int:
        return len(self._sequence_indices)

    def __getitem__(self, index: int) -> NDArray[np.number]:
        if index < 0 or index >= len(self):
            raise IndexError(f"frame index must be between 0 and {len(self) - 1}.")
        frame = np.asarray(self._file.read_frame(self._sequence_indices[index]))
        channel_count = int(self._file.sizes.get("C", 1))
        if channel_count > 1:
            channel = self._selection["C"]
            if frame.shape[0] != channel_count:
                raise ValueError("Unexpected channel layout returned by ND2 reader.")
            frame = frame[channel]
        if frame.ndim != 2:
            raise ValueError("Selected ND2 frame is not two-dimensional.")
        return frame

    def __iter__(self) -> Iterator[NDArray[np.number]]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """Close the underlying ND2 file handle."""
        self._file.close()

    def __enter__(self) -> "ND2FrameSource":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def as_frame_source(
    frames: FrameSource | NDArray[np.number],
) -> FrameSource:
    """Normalize an existing frame source or three-dimensional NumPy array."""
    if isinstance(frames, np.ndarray):
        return ArrayFrameSource(frames)
    if isinstance(frames, FrameSource):
        return frames
    raise TypeError("frames must be a numpy ndarray or FrameSource.")


def open_frame_source(
    path: str | Path,
    axis_selection: Mapping[str, int] | None = None,
) -> ArrayFrameSource | ND2FrameSource:
    """Open an ND2 or memory-mapped NumPy video as a shared frame source."""
    source_path = Path(path).expanduser().resolve()
    suffix = source_path.suffix.lower()
    if suffix == ".nd2":
        return ND2FrameSource(source_path, axis_selection=axis_selection)
    if suffix == ".npy":
        return ArrayFrameSource(
            np.load(source_path, allow_pickle=False, mmap_mode="r"),
            owns_frames=True,
        )
    raise ValueError(f"Unsupported video source type: {source_path}")
