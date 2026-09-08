"""Regression tests for the shared radial baseline implementation."""
import numpy as np
from vesmod.VesEdge.contour_geometry import fit_radial_baseline
from vesmod.VesEdge.vesicle_video_utils import zero_out_all_but_lowest_n_modes


def test_shared_baseline_matches_legacy_extraction_projection():
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    radii = 40 + 3 * np.cos(theta) + 1.5 * np.sin(2 * theta) + 0.4 * np.cos(17 * theta)
    np.testing.assert_allclose(
        fit_radial_baseline(radii, order=7).values,
        zero_out_all_but_lowest_n_modes(radii, n=7), rtol=0, atol=1e-12)
