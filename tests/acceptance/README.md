# End-to-end acceptance references

`test_end_to_end_acceptance.py` runs a real vesicle image sequence through the
public scientific pipeline:

1. extract edges and downsample them to 120 angular samples;
2. save and reload the QC-independent `.npz` checkpoint;
3. run curvature and area QC;
4. save the accepted radii in microns and reload the `.npy` file in EdgeMod;
5. calculate the fluctuation spectrum; and
6. fit bending modulus and surface tension over modes 3–7.

The `.npz` files in this directory are reviewed canonical outputs, not test
inputs. Integer identities and configuration metadata must match exactly.
Floating-point scientific outputs use tight tolerances; the nonlinear fit uses
a slightly wider relative tolerance to allow harmless optimizer differences.

Each entry in `ACCEPTANCE_CASES` explicitly pairs an input filename under
`tests/sample_vesicle_videos/` with a generic reference filename here. The
generic names distinguish these files from ordinary VesEdge checkpoints and
make adding a second biologically distinct video straightforward.

To intentionally approve a scientific-output change, run:

```bash
python -m pytest tests/test_end_to_end_acceptance.py \
    --update-acceptance-reference
```

Review the code change and the reported test results before committing the new
binary reference. Updating a reference should be a deliberate scientific
decision, never an automatic response to a failing test.
