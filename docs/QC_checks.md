# Writing a VesEdge QC check

Each registered check returns a `QCResult` defined in
[`qc_result.py`](../vesmod/VesEdge/qc/qc_result.py). The runner validates that
result and applies its decisions and diagnostics to the corresponding
detections. Every enabled check must return a result, including a check that
accepts every frame.

Implement [`QCCheck`](../vesmod/VesEdge/qc/qc_check.py) in a module under
`vesmod/VesEdge/qc/checks/`, or `qc/experimental/` for an experimental method.
Define the check's `name`, `config_type`, and `run()` method, then import and
instantiate it in [`qc_registry.py`](../vesmod/VesEdge/qc/qc_registry.py).

## Standard result fields

| Field | Required contents |
| --- | --- |
| `check_name` | The registered check's name. |
| `frame_indices` | The zero-based source-frame indices of every supplied detection, in the same order. Indices must be nonnegative, unique, and increasing. |
| `rejected` | One boolean per supplied detection. `True` rejects that frame for this check. |
| `diagnostics` | A mapping of declared measurement names to columns, each containing one value per supplied detection. Defaults to an empty mapping. |
| `trajectory_flags` | A collection of string reasons to reject the entire recording. Defaults to an empty collection. |
| `extras` | A mapping of declared names to additional aggregate measurements. Defaults to an empty mapping. |

`rejected_count` is calculated from `rejected`. A recording-level rejection
can leave all individual frame decisions accepted.

The runner checks that frame indices match the supplied detections and
that the measurement names match the check's declarations. Returning `None`,
an arbitrary object, or a result for another check is an error.

## Declaring measurements

Declare per-frame diagnostic names and additional aggregate field names on
the check class:

```python
diagnostic_fields = ("score",)
extra_result_fields = ("reference",)
```

The returned mappings must contain exactly those names. Use `None` for an
unavailable measurement. This keeps the saved structure explicit even when
a measurement cannot be calculated.

After calculating scores, a check could return:

```python
return QCResult(
    check_name=self.name,
    frame_indices=tuple(edge.frame_index for edge in detections),
    rejected=tuple(bool(score > config.threshold) for score in scores),
    diagnostics={"score": scores},
    extras={"reference": reference},
)
```

Measurements support JSON values including null, booleans, finite numbers,
strings, lists, and mappings with string keys. NumPy scalar numbers are
normalized to Python numbers. Nonfinite numbers are represented by `None`,
which becomes JSON null. Convert NumPy arrays to lists explicitly.
Arbitrary class instances are rejected.

Document what each measurement means and its units in the check module.

## Inputs to run

`run(detections, config, frames)` receives successful `EdgeDetection` objects,
this check's validated settings, and a NumPy video array, an on-demand frame
source, or `None`. The detections retain their original frame indices, which
may have gaps where extraction failed.

Set `requires_source_video_frames = True` when images are needed. Retrieve a
source image using `frames[detection.frame_index]`. The CLI opens the source
video when an enabled check requires it. Python callers provide the frames
to `edges.run_qc(config, frames=frames)`.

The result is authoritative. The runner adds or removes this check's frame
rejection flag, records its diagnostics, and combines recording-level flags.
Existing geometric helper functions also record their measurements on
detections. A check using such a helper can collect those declared fields with:

```python
return self.make_result(detections, extras={"reference": reference})
```

If a check fails or returns an invalid result during `edges.run_qc()`, the
previous QC state is restored.

## Saving and loading results

`QCResult.to_dict()` produces the standard JSON structure.
`check.result_from_dict(data)` reconstructs it and verifies the check name
and declared measurement fields.

`VesicleQCResult` combines the check results, evaluated source-frame indices,
configuration, and recording-level flags. Its `to_dict()` and `from_dict()`
methods save and load that structure. The `accepted_frame_indices` property
derives the final accepted selection, including runs with every check disabled.

## Command-line integration

`QCCheck` provides defaults for configuration construction, dataclass
serialization, the `enabled` flag, and a `<check_name>_rejected` summary column.
Override `add_cli_arguments()` and `config_from_args()` to expose settings.

To write additional files, declare `(suffix, writer)` pairs in
`diagnostic_writers`. A writer receives `(output_path, edges)` and reads
its result with `edges.qc_result.get_result(check_name)`.

Override both `summary_defaults` and `summary_values()` to provide different
summary columns. These methods must handle disabled checks and failed runs.

The [registration test](../tests/unit_tests/test_qc_runner.py) demonstrates an
image-based check with declared diagnostics and extras, command-line
execution, JSON round-tripping, and reapplication of recorded settings.
