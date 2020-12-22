#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020, Ross A. Beyer (rbeyer@seti.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This program attempts to read information from two sets of image metadata
# and provide an analysis of what kind of stereo pair they might make.
#
# It currently only works with ISIS cubes that have had spiceinit run on
# them.  However, it could be adapted to read other kinds of image metadata
# and perform its calculations.
#
# The math and quality metrics are based on Becker et al. 2015,
# Criteria for Automated Identification of Stereo Image Pairs
# (https://www.hou.usra.edu/meetings/lpsc2015/pdf/2703.pdf)


import argparse
import math
import logging
import os
import subprocess
from collections import abc
# from functools import reduce
from pathlib import Path


def quality(value: float, ideal, low: float, high: float) -> float:
    """Return a quality value based on *value*.

    The value of *ideal* would be a perfect value of *value*, but
    if *value* is between *low* and *high* it is acceptable.  *ideal*
    must be between *low* and *high*, otherwise a ValueError is raised.

    A quality value of one indicates that *value* is *ideal*.
    Quality values between zero and one indicate that *value*
    is between *low* and *high* (the closer to *ideal*, the higher
    the quality score).  Values less than zero are beyond the
    acceptable range of *low* and *high*.

    If *ideal* is a two-tuple, then this indicates that all values
    between and including those values are "ideal" values.  Again,
    these two values must be between *low* and *high*
    """

    if isinstance(ideal, abc.Sequence):
        if len(ideal) == 2:
            if not (low <= ideal[0] <= ideal[1] <= high):
                raise ValueError(
                    f"The ideal values ({ideal}) is are not between low "
                    f"({low}) and high ({high})."
                )
            if ideal[0] <= value <= ideal[1]:
                return 1

            if value < ideal[0]:
                return (value - low) / (ideal[0] - low)
            else:
                return (value - high) / (ideal[1] - high)
        else:
            raise ValueError(
                f"The provided ideal value must be either a single value or a"
                f"two-tuple.  It was neither: {ideal}"
            )
    else:
        if not (low <= ideal <= high):
            raise ValueError(
                f"The ideal value ({ideal}) is not between low ({low}) and "
                f"high ({high})."
            )

        if value == ideal:
            return 1

        if value < ideal:
            return (value - low) / (ideal - low)
        else:
            return (value - high) / (ideal - high)


def incidence_quality(incidence_angle: float) -> float:
    """Returns a quality score based on the value of *incidence_angle*,
    which is expected to be in decimal degrees, zero being normal to
    the surface.

    Becker et al. (2015) indicates:
    - Limits: Between 40° and 65° depending on smoothness
        (shadows to be avoided).
    - Recommended: Nominally 50°
    """
    return quality(incidence_angle, 50, 40, 65)


def emission_quality(emission_angle: float) -> float:
    """Returns a quality score based on the value of *emission_angle*,
    which is expected to be in decimal degrees, zero being normal to
    the surface.

    Becker et al. (2015) indicates:
    - Limits: Between 0° and the complement of the maximum slope
      (conservatively 45°, greater for smoother terrains) for optical images.
      Greater than the slope (≥15° even for smooth surfaces) for radar.
    - Recommended: No recommendation
    """
    return quality(emission_angle, 22.5, 0, 45)


def phase_quality(phase_angle: float) -> float:
    """Returns a quality score based on the value of *phase_angle*,
    which is expected to be in decimal degrees, zero being normal to
    the surface.

    Becker et al. (2015) indicates:
    - Limits: Between 5° and 120°.
    - Recommended: ≥ 30°
    """
    return quality(phase_angle, 60, 5, 120)


def gsd_quality(gsd1: float, gsd2: float) -> float:
    """Returns a quality score based on the two ground
    sample distances, *gsd1* and *gsd2*.

    Image pairs with GSD ratios larger than 2.5 can be used but are
    not optimal, as details only seen in the smaller scale image
    will be lost. If required, images with ratios greater than ~2.5
    should be resampled to the GSD of the lower scale image
    (Becker at al., 2015).
    """
    ratio = max(gsd1, gsd2) / min(gsd1, gsd2)
    return quality(ratio, 1, 1, 2.5)


def stereo_strength_quality(parallax_height_ratio: float) -> float:
    """Returns a quality score based on the Parallax/Height Ratio (*dp*).

    Becker et al. (2015) indicates:
    - Limits: Between 0.1 (5°) and 1 (~45°).
    - Recommended: 0.4 (20°) to 0.6 (30°).
    """
    return quality(parallax_height_ratio, (0.4, 0.6), 0.1, 1)


def illumination_quality(shadow_tip_distance: float) -> float:
    """Returns a quality score based on the Shadow-Tip Distance (*dsh*).

    Becker et al. (2015) indicates:
    - Limits: 0 to 2.58.
    - Recommended: 0
    """
    return quality(shadow_tip_distance, 0, 0, 2.58)


def delta_solar_az_quality(az1: float, az2: float) -> float:
    """Returns a quality score based on the two solar azimuth values,
    in degrees.

    In practice, Shadow-Tip Distance alone does not guarantee similar
    illumination.  The absolute difference in solar azimuth angle
    between stereo pairs can be optionally constrained.

    Becker et al. (2015) indicates:
    - Limits: 0° to 100°.
    - Recommended: ≤ 20°
    """
    az_diff = abs(az1 - az2)
    return quality(az_diff, (0, 20), 0, 100)


def stereo_overlap_quality(area_fraction: float) -> float:
    """Returns a quality score based on the stereo area overlap.

    Becker et al. (2015) indicates:
    - Limits: Between 30% and 100%.
    - Recommended: 50% to 100%.
    """
    return quality(area_fraction, (0.5, 1.0), 0.3, 1)


def parallax(
    emission1: float, gndaz1: float, emission2: float, gndaz2: float
) -> float:
    """Returns the parallax angle between the two look vectors described
       by the emission angles and sub-spacecraft ground azimuth angles.

       Input angles are assumed to be radians, as is the return value."""

    def x(emi, gndaz):
        return math.sin(gndaz) * math.sin(emi)

    def y(emi, gndaz):
        return math.cos(gndaz) * math.sin(emi)

    def z(emi):
        return math.cos(emi)

    one_dot_two = (
        (x(emission1, gndaz1) * x(emission2, gndaz2))
        + (y(emission1, gndaz1) * y(emission2, gndaz2))
        + (z(emission1) * z(emission2))
    )
    return math.acos(one_dot_two)


def dp(
    emission1: float,
    gndaz1: float,
    emission2: float,
    gndaz2: float,
    radar=False,
):
    """Returns the Parallax/Height Ratio (dp) as detailed in
    Becker et al.(2015).

    The input angles are assumed to be in radians.  If *radar* is true,
    then cot() is substituted for tan() in the calculations.

    Physically, dp represents the amount of parallax difference
    that would be measured between an object in the two images, for
    unit height.
    """

    def emission_trig(emi: float):
        e = math.tan(emi)
        if radar:
            # Cotangent is 1 / tan()
            return 1 / e
        else:
            return e

    def px(emi: float, scazgnd: float):
        return -1 * emission_trig(emi) * math.cos(scazgnd)

    def py(emi: float, scazgnd: float):
        return emission_trig(emi) * math.sin(scazgnd)

    px1 = px(emission1, gndaz1)
    px2 = px(emission2, gndaz2)
    py1 = py(emission1, gndaz1)
    py2 = py(emission2, gndaz2)

    return math.sqrt(math.pow(px1 - px2, 2) + math.pow(py1 - py2, 2))


def dsh(
    incidence1: float, solar_az1: float, incidence2: float, solar_az2: float
):
    """Returns the Shadow-Tip Distance (dsh) as detailed in
    Becker et al.(2015).

    The input angles are assumed to be in radians.

    This is defined as the distance between the tips of the shadows
    in the two images for a hypothetical vertical post of unit
    height. The "shadow length" describes the shadow of a hypothetical
    pole so it applies whether there are actually shadows in the
    image or not. It's a simple and consistent geometrical way to
    quantify the difference in illumination. This quantity is
    computed analogously to dp.
    """

    def shx(inc: float, sunazgnd: float):
        return -1 * math.tan(inc) * math.cos(sunazgnd)

    def shy(inc: float, sunazgnd: float):
        return math.tan(inc) * math.sin(sunazgnd)

    shx1 = shx(incidence1, solar_az1)
    shx2 = shx(incidence2, solar_az2)
    shy1 = shy(incidence1, solar_az1)
    shy2 = shy(incidence2, solar_az2)

    return math.sqrt(math.pow(shx1 - shx2, 2) + math.pow(shy1 - shy2, 2))

# BSD 3-Clause License
#
# Copyright (c) 2020, Andrew Michael Annex
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The below functions are modified version of those written by Ross Beyer according to the Apache License above

import sh
import pvl
import warnings


class ImgInfo(object):
    def __init__(self, path: os.PathLike):
        # First try it as an ISIS cube:
        try:
            cam_to = Path(path).with_suffix(".caminfo")
            sh.caminfo(f'from={path}', f'to={cam_to}', 'polygon=True')
            caminfo = pvl.load(str(cam_to))
        except subprocess.CalledProcessError:
            # Maybe it is the PVL output of caminfo?
            caminfo = pvl.load(str(path))

        camgeo = caminfo["Caminfo"]["Geometry"]

        self.path = Path(path)

        self.incidence_angle = float(camgeo["IncidenceAngle"])
        self.emission_angle = float(camgeo["EmissionAngle"])
        self.phase_angle = float(camgeo["PhaseAngle"])
        self.sc_azimuth = float(camgeo["SubSpacecraftGroundAzimuth"])
        self.solar_azimuth = float(camgeo["SubSolarGroundAzimuth"])
        self.gsd = float(camgeo["ObliquePixelResolution"])

        try:
            from shapely import wkt
            self.geometry = wkt.loads(caminfo["Caminfo"]["Polygon"]["GisFootprint"])
        except ImportError:
            self.geometry = None

    def incidence_quality(self) -> float:
        return incidence_quality(self.incidence_angle)

    def emission_quality(self) -> float:
        return emission_quality(self.emission_angle)

    def phase_quality(self) -> float:
        return phase_quality(self.phase_angle)

    def qualities(self) -> tuple:
        return (
            self.incidence_quality(),
            self.emission_quality(),
            self.phase_quality(),
        )


def get_report(file1, file2) -> str:
    """
    Generate the report of the stereo quality
    :param file1:
    :param file2:
    :return:
    """
    im1 = ImgInfo(file1)
    im2 = ImgInfo(file2)

    if im1.geometry is not None and im2.geometry is not None:
        overlap = area_overlap(im1.geometry, im2.geometry)
    else:
        warnings.warn("Shapely is not installed, assuming overlap ratio is perfect (1.0)")
        overlap = 1.0

    # Gather qualities:
    parallax_ang = parallax(
        math.radians(im1.emission_angle),
        math.radians(im1.sc_azimuth),
        math.radians(im2.emission_angle),
        math.radians(im2.sc_azimuth),
    )

    gsd_q = gsd_quality(im1.gsd, im2.gsd)
    parallax_height_ratio = dp(
        math.radians(im1.emission_angle),
        math.radians(im1.sc_azimuth),
        math.radians(im2.emission_angle),
        math.radians(im2.sc_azimuth),
    )
    stereo_q = stereo_strength_quality(parallax_height_ratio)

    stereo_tip_distance = dsh(
        math.radians(im1.incidence_angle),
        math.radians(im1.solar_azimuth),
        math.radians(im2.incidence_angle),
        math.radians(im2.solar_azimuth),
    )
    illum_q = illumination_quality(stereo_tip_distance)
    delta_sol_q = delta_solar_az_quality(im1.solar_azimuth, im2.solar_azimuth)
    overlap_q = stereo_overlap_quality(overlap)

    output_string = f"""
    Stereo Pair Quality Report:

    Image 1: {im1.path.name}
    Image 2: {im2.path.name}
                            Image 1  Image 2
           Incidence Angle: {im1.incidence_angle:>6.2f}   {im2.incidence_angle:>6.2f}
                   Quality: {im1.incidence_quality():>6.2f}   {im2.incidence_quality():>6.2f}
            Emission Angle: {im1.emission_angle:>6.2f}   {im2.emission_angle:>6.2f}
                   Quality: {im1.emission_quality():>6.2f}   {im2.emission_quality():>6.2f}
               Phase Angle: {im1.phase_angle:>6.2f}   {im2.phase_angle:>6.2f}
                   Quality: {im1.phase_quality():>6.2f}   {im2.phase_quality():>6.2f}

     Subspacecraft Azimuth: {im1.sc_azimuth:>6.2f}   {im2.sc_azimuth:>6.2f}
          Subsolar Azimuth: {im2.solar_azimuth:>6.2f}   {im2.solar_azimuth:>6.2f}
                   Quality:     {delta_sol_q:>6.2f}
    Ground Sample Distance: {im1.gsd:>6.2f}   {im2.gsd:>6.2f}
                   Quality:     {gsd_q:>6.2f}

    Stereo Overlap Fraction: {overlap:.2f}
            Overlap Quality: {overlap_q:.2f}

                Parallax Angle: {math.degrees(parallax_ang):.2f}
    Parallax/Height Ratio (dp): {parallax_height_ratio:.2f}
       Stereo Strength Quality: {stereo_q:.2f}

    Stereo Tip Distance (dsh): {stereo_tip_distance:.2f}
         Illumination Quality: {illum_q:.2f}
    """

    # Evaluate qualities:
    qualities = [gsd_q, stereo_q, illum_q, delta_sol_q, overlap_q]
    qualities.extend(im1.qualities())
    qualities.extend(im2.qualities())

    neg_q = list()
    pos_q = list()
    for q in qualities:
        if q > 0:
            pos_q.append(q)
        else:
            neg_q.append(q)

    if len(neg_q) != 0:
        output_string += f"""
    This pair of images probably won't make a good stereo pair because there were
    {len(neg_q)} qualities less than or equal to zero.
            """

    # The commented out code below wraps up all the quality factors
    # into a Drake-Equation-like overall quality metric, but I'm not
    # sure that doing so actually gets you anywhere.  I don't think
    # that there is an independent absolute measure of 'stereo quality'
    # since there are so many different ways to approach the problem.
    #
    # stereo_quality = reduce(lambda x, y: x * y, pos_q)

    # output_string += f"""
    # The overall quality factor for this stereo pair is: {stereo_quality:5.3f}
    # This value is the result of multiplying all of the positive qualities
    # together, and ranges from zero to one.
    # """

    return output_string


def area_overlap(geom1, geom2) -> float:
    """Returns a ratio of areas which represent the area of the
    intersection area of *geom1* and *geom2*, divided by the
    union of areas of of *geom1*, and *geom2*."""
    if not geom1.intersects(geom2):
        raise ValueError(f"Geometries do not overlap.")

    intersection_geom = geom1.intersection(geom2)
    union_geom = geom1.union(geom2)

    return intersection_geom.area / union_geom.area