"""On-demand frame access for file-backed video sources."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import nd2
import numpy as np
from numpy.typing import NDArray


_ND2_READER_MEMORY_BUDGET_BYTES = 100 * 1024**2


class OnDemandFrameSequence:
    """Random-access frames loaded from a file-backed source when requested.

    The current implementation supports ND2 acquisitions and memory-mapped
    NumPy ``.npy`` files. Additional on-demand backends can be incorporated or
    split into format-specific subclasses when their behavior warrants it.

    Parameters
    ----------
    path : str or Path
        File-backed video sequence to open.
    axis_selection : mapping, optional
        For ND2 input, the index selected for every non-spatial, non-time axis
        whose size exceeds one. Axis names follow ``nd2.ND2File.sizes`` (for
        example ``P``, ``Z``, or ``C``). Ambiguous acquisitions are rejected
        rather than silently selecting a position, z-plane, or channel.
    """

    def __init__(
        self,
        path: str | Path,
        axis_selection: Mapping[str, int] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._suffix = self.path.suffix.lower()
        self._file = None
        self._array = None
        self._bytes_since_reopen = 0
        self._selection = {
            str(axis).upper(): int(index)
            for axis, index in (axis_selection or {}).items()
        }

        if self._suffix == ".nd2":
            self._open_nd2()
            try:
                self._validate_selection()
                self._sequence_indices = self._select_sequence_indices()
            except (KeyError, TypeError, ValueError):
                self.close()
                raise
            return

        if self._suffix == ".npy":
            if self._selection:
                raise ValueError("axis_selection is only supported for ND2 input.")
            array = np.load(self.path, allow_pickle=False, mmap_mode="r")
            if array.ndim != 3:
                raise IndexError("frames must be a 3D array.")
            self._array = array
            self._sequence_indices = tuple(range(array.shape[0]))
            return

        raise ValueError(f"Unsupported video source type: {self.path}")

    def _open_nd2(self) -> None:
        self._file = nd2.ND2File(self.path)
        self._bytes_since_reopen = 0

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

    def _reopen(self) -> None:
        """Recycle the ND2 reader so accessed file-backed pages can be released."""
        if self._suffix != ".nd2":
            return
        self._file.close()
        self._open_nd2()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return selected ``(frames, height, width)`` dimensions."""
        if self._suffix == ".npy":
            return self._array.shape
        return (
            len(self),
            int(self._file.sizes["Y"]),
            int(self._file.sizes["X"]),
        )

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return source identity and backend-specific metadata."""
        if self._suffix == ".npy":
            return {
                "kind": "npy",
                "path": str(self.path),
                "dtype": str(self._array.dtype),
            }
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
        if self._suffix == ".npy":
            return self._array[index]

        if self._bytes_since_reopen >= _ND2_READER_MEMORY_BUDGET_BYTES:
            self._reopen()

        raw_frame = np.asarray(self._file.read_frame(self._sequence_indices[index]))
        self._bytes_since_reopen += raw_frame.nbytes

        channel_count = int(self._file.sizes.get("C", 1))
        if channel_count > 1:
            channel = self._selection["C"]
            if raw_frame.shape[0] != channel_count:
                raise ValueError("Unexpected channel layout returned by ND2 reader.")
            raw_frame = raw_frame[channel]
        frame = np.array(raw_frame, copy=True)
        if frame.ndim != 2:
            raise ValueError("Selected ND2 frame is not two-dimensional.")
        return frame

    def __iter__(self) -> Iterator[NDArray[np.number]]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        """Release resources owned by the file-backed sequence."""
        if self._suffix == ".nd2" and self._file is not None:
            self._file.close()
            self._file = None
        if self._suffix == ".npy":
            self._array = None

    def __enter__(self) -> "OnDemandFrameSequence":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def as_frame_source(
    frames: OnDemandFrameSequence | NDArray[np.number],
) -> OnDemandFrameSequence | NDArray[np.number]:
    """Validate supported frame input without wrapping resident NumPy arrays."""
    if isinstance(frames, np.memmap):
        raise TypeError(
            "Pass the backing path to OnDemandFrameSequence rather than a "
            "bare memory-mapped array."
        )
    if isinstance(frames, np.ndarray):
        if frames.ndim != 3:
            raise IndexError("frames must be a 3D array.")
        return frames
    if isinstance(frames, OnDemandFrameSequence):
        return frames
    raise TypeError(
        "frames must be a numpy ndarray or OnDemandFrameSequence."
    )


def open_frame_source(
    path: str | Path,
    axis_selection: Mapping[str, int] | None = None,
) -> OnDemandFrameSequence:
    """Open a supported file-backed video for on-demand frame access."""
    return OnDemandFrameSequence(path, axis_selection=axis_selection)
