# VesEdge GIF command

GIF rendering is a separate stage from edge extraction and quality control.
This lets the same checkpoint produce several visualizations without repeating
edge extraction.

## Render recursively

Always double-quote each checkpoint path or pattern. This is safe for ordinary
paths and ensures that VesEdge, rather than the shell, expands wildcard
patterns and applies `--recursive`.

Given extraction checkpoints under `checkpoints/`:

```text
checkpoints/
├── condition_a/ND Acquisition.npz
└── condition_b/ND Acquisition.npz
```

the following command mirrors that structure under `gifs/`:

```bash
vesedge gif "checkpoints/" --recursive --output-dir gifs/ --style edges
```

## Available styles

Render the source video without annotations:

```bash
vesedge gif "checkpoints/" \
    --recursive \
    --output-dir gifs/original/ \
    --style original
```

Render all successfully detected edges in green:

```bash
vesedge gif "checkpoints/" \
    --recursive \
    --output-dir gifs/edges/ \
    --style edges
```

Render accepted edges in green and QC-rejected edges in red:

```bash
vesedge gif "checkpoints/" \
    --recursive \
    --output-dir gifs/qc/ \
    --style qc \
    --qc-dir results/new_QC/
```

The QC directory must be the output of `vesedge qc`. For recursive input, the
command pairs each checkpoint and filtered array by relative path and filename:

```text
checkpoints/condition_a/ND Acquisition.npz
results/new_QC/condition_a/ND Acquisition.npy
```

It reads `results/new_QC/vesedge_qc.json`, reapplies that recorded
configuration to the checkpoint, and verifies that the reconstructed accepted
radii match the paired `.npy`. This recovers source-frame QC status without
assuming that rows in the filtered array correspond to consecutive frames.

## Source videos

Each checkpoint normally records its original `.nd2` or `.npy` source
path. The command uses that path first. If the recorded path no longer exists,
it also checks for a source file with the same name beside the checkpoint.

A missing source or QC pair is reported with the full checkpoint path. In a
recursive batch, that failure does not stop other checkpoints from rendering.

## Existing output

Existing GIFs are skipped by default. Replace them with:

```bash
vesedge gif "checkpoints/" \
    --recursive \
    --output-dir gifs/ \
    --style edges \
    --overwrite
```

`vesedge extract` now writes checkpoints only. Run `vesedge gif` afterward
whenever a visualization is needed.
