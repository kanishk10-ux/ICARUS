# -*- coding: utf-8 -*-
"""Data ingestion, preprocessing, and provider registry for ICARUS."""
from icarus.data.reader import ScientificDataReader
from icarus.data.registry import DatasetRecord, ScientificDataRegistry

__all__ = ["ScientificDataReader", "DatasetRecord", "ScientificDataRegistry"]
