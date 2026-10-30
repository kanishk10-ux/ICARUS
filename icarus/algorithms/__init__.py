# -*- coding: utf-8 -*-
"""Domain-Wide Scientific Algorithms Layer for ICARUS.

Modular numerical and statistical frameworks covering all 8 scientific domains,
strictly independent of deep neural networks.
"""
from icarus.algorithms.eo import EOAlgorithms
from icarus.algorithms.oceans import OceanAlgorithms
from icarus.algorithms.atmosphere import AtmosphereAlgorithms
from icarus.algorithms.solid_earth import SolidEarthAlgorithms
from icarus.algorithms.climate import ClimateAlgorithms
from icarus.algorithms.heliophysics import HeliophysicsAlgorithms
from icarus.algorithms.planetary import PlanetaryAlgorithms
from icarus.algorithms.astronomy import AstronomyAlgorithms

__all__ = [
    "EOAlgorithms",
    "OceanAlgorithms",
    "AtmosphereAlgorithms",
    "SolidEarthAlgorithms",
    "ClimateAlgorithms",
    "HeliophysicsAlgorithms",
    "PlanetaryAlgorithms",
    "AstronomyAlgorithms",
]
