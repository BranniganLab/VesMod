#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 12:27:16 2025.

@author: js2746
"""
from pathlib import Path
from scipy import ndimage
from skimage import filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
from .vesicle_video_utils import wrap_image_to_polar, isolate_region_of_array, zero_out_all_but_lowest_n_modes, convert_to_cartesian


def extract_edge_from_frame(frame, debug_path=None):
    """
    Extract vesicle edge from frame of vesicle video.

    Parameters
    ----------
    frame : numpy ndarray
        The 2D array of intensity values from a vesicle video frame.
    debug_path : pathlib Path, optional
        If not None, output debug images to debug_path. The default is None.

    Returns
    -------
    r_vals : numpy ndarray
        1D array of distances from center_of_mass to vesicle edge. Evenly spaced
        in theta from 0 to 2pi.
    center_of_mass : tuple
        The Cartesian coordinates of the approximate vesicle center.

    """
    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.mkdir(parents=True, exist_ok=True)
        _make_debug_image(frame, debug_path)
        return None, None

    # step 1: find internal vesicle point
    center_of_mass = approximate_vesicle_com(frame)

    # step 2: naive refinement of edge region
    polar_sobel, scaling_factor = wrap_image_to_polar(filters.sobel(frame), center_of_mass)
    avg = np.mean(np.argmax(polar_sobel, axis=1))
    vertically_masked_polar_sobel = isolate_region_of_array(polar_sobel, avg, 0.25)
    max_of_masked_region = np.argmax(vertically_masked_polar_sobel, axis=1)

    # step 3: FFT-informed refinement of edge region
    approx_edge = zero_out_all_but_lowest_n_modes(max_of_masked_region, n=7)

    # wrap original image to polar
    original_frame_polar, _ = wrap_image_to_polar(frame, center_of_mass)

    # step 4: horizontal Sobel filter and apply FFT-informed mask
    horizontal_sobel = filters.sobel(original_frame_polar, axis=1)
    gauss_blur = ndimage.gaussian_filter(horizontal_sobel, sigma=2)
    fft_masked_horizontal_sobel = isolate_region_of_array(gauss_blur, approx_edge, 0.05, True)
    max_sobel = np.nanargmax(fft_masked_horizontal_sobel, axis=1)

    r_vals = np.array(max_sobel) / scaling_factor

    return r_vals, center_of_mass


def _make_debug_image(frame, output_path):
    """
    Make a debug image that shows each step in the process.

    Parameters
    ----------
    frame : numpy ndarray
        The 2D array of intensity values from a vesicle video frame.
    output_path : pathlib Path
        Output debug images to output_path.

    Returns
    -------
    None.

    """
    _, axes = plt.subplots(2, 4, figsize=(12, 6), layout='constrained')
    # plt.axis('off')
    axes[0][0].imshow(frame, cmap='gray')
    axes[1][1].imshow(frame, cmap='gray')
    axes[1][2].imshow(frame, cmap='gray')
    axes[1][3].imshow(frame, cmap='gray')
    axes[1][0].set_visible(False)

    center_of_mass = approximate_vesicle_com(frame, debug_path=output_path)
    polar_image, scaling_factor = wrap_image_to_polar(filters.sobel(frame), center_of_mass)
    axes[0][1].imshow(polar_image * 100, cmap='gray', vmin=0, vmax=.4)
    axes[0][0].scatter(center_of_mass[1], center_of_mass[0], color='tab:red', marker="+", s=20)

    masked_polar_image0 = isolate_region_of_array(polar_image, np.mean(np.argmax(polar_image, axis=1)), 0.35, False)
    masked_polar_image_nan = isolate_region_of_array(polar_image, np.mean(np.argmax(polar_image, axis=1)), 0.35, True)
    max_of_masked_region = np.argmax(masked_polar_image0, axis=1)
    axes[0][1].plot(np.argmax(polar_image, axis=1), np.arange(0, max_of_masked_region.shape[0]))

    bad_x, bad_y = convert_to_cartesian((center_of_mass[1], center_of_mass[0]), np.argmax(polar_image, axis=1) / scaling_factor)
    axes[1][1].plot(bad_x, bad_y, color='tab:blue')

    ifft = zero_out_all_but_lowest_n_modes(max_of_masked_region, n=7)
    axes[0][2].imshow(masked_polar_image_nan, cmap='gray', vmin=0, vmax=.004)
    axes[0][2].plot(max_of_masked_region, np.arange(0, polar_image.shape[0]), color='tab:orange')

    bad_x2, bad_y2 = convert_to_cartesian((center_of_mass[1], center_of_mass[0]), max_of_masked_region / scaling_factor)
    axes[1][2].plot(bad_x2, bad_y2, color='tab:orange')

    og_polar_image, _ = wrap_image_to_polar(frame, center_of_mass)
    horizontal_sobel = filters.sobel(og_polar_image, axis=1)
    polar_image_masked_blurred = isolate_region_of_array(ndimage.gaussian_filter(horizontal_sobel, sigma=2), ifft, 0.05, True)
    axes[0][3].imshow(polar_image_masked_blurred, cmap='gray')

    max_sobel = np.nanargmax(polar_image_masked_blurred, axis=1)
    axes[0][3].plot(max_sobel, np.arange(0, horizontal_sobel.shape[0]), color='red')

    x_vals, y_vals = convert_to_cartesian((center_of_mass[1], center_of_mass[0]), np.array(max_sobel) / scaling_factor)
    axes[1][3].plot(x_vals, y_vals, color='red')

    axes[0][0].set_xlabel("i")
    axes[0][0].set_ylabel("j")
    axes[1][1].set_xlabel("i")
    axes[1][1].set_ylabel("j")
    axes[1][2].set_xlabel("i")
    axes[1][2].set_ylabel("j")
    axes[1][3].set_xlabel("i")
    axes[1][3].set_ylabel("j")

    axes[0][1].set_xlabel(r"$\rho$")
    axes[0][1].set_ylabel(r"$\theta$")
    axes[0][2].set_xlabel(r"$\rho$")
    axes[0][2].set_ylabel(r"$\theta$")
    axes[0][3].set_ylabel(r"$\theta$")
    axes[0][3].set_xlabel(r"$\rho$")
    plt.savefig(output_path / "edge_extraction_debug.pdf")
    plt.clf()
    plt.close()


def approximate_vesicle_com(frame, sigma=10, debug_path=None):
    """
    Find the center of mass of a vesicle within an image frame.

    Workflow: sobel filter -> gaussian blur -> otsu threshold -> find centroid.

    Parameters
    ----------
    frame : 2D numpy ndarray
        The image frame containing your vesicle.
    sigma : float, OPTIONAL
        Sigma value for the amount of gaussian blurring. Default is 10.
    debug_path : pathlib Path, OPTIONAL
        If not None, make an image that shows each step of this process and save\
        it to this path.

    Returns
    -------
    tuple
        The coordinates for the approximate center of mass of your vesicle.

    """
    sobel_filter = filters.sobel(frame)
    blurred = ndimage.gaussian_filter(sobel_filter, sigma=sigma)
    greater_than_otsu_threshold = (blurred > filters.threshold_otsu(blurred)).astype(int)
    centroid = regionprops(greater_than_otsu_threshold, blurred)[0].centroid
    if debug_path is not None:
        _make_debug_image_centroid((frame, sobel_filter, blurred, greater_than_otsu_threshold, centroid), debug_path)
    return centroid


def _make_debug_image_centroid(input_tuple, fpath):
    """Make a 4-panel figure showing the steps in the centroid finding process."""
    original, sobel, blur, threshold, centroid = input_tuple

    _, axes = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Raw image')
    axes[0].scatter(centroid[1], centroid[0], color='tab:blue')

    axes[1].imshow(sobel, cmap='gray')
    axes[1].set_title('Sobel filter')

    axes[2].imshow(blur, cmap='gray')
    axes[2].set_title('Gaussian blur')

    axes[3].imshow(threshold, cmap='jet')
    axes[3].set_title('Otsu threshold')
    axes[3].scatter(centroid[1], centroid[0], color='tab:blue')

    plt.axis('off')
    plt.savefig(fpath.joinpath("centroid_process_debug.pdf"))
