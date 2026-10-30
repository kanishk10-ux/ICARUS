# -*- coding: utf-8 -*-
"""Base enumerations and common models for ICARUS schemas."""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DomainEnum(str, Enum):
    """The eight scientific domains of ICARUS."""
    EARTH_OBSERVATION = "earth_observation"
    OCEANS = "oceans"
    ATMOSPHERE = "atmosphere"
    SOLID_EARTH = "solid_earth"
    CLIMATE = "climate"
    HELIOPHYSICS = "heliophysics"
    PLANETARY = "planetary"
    ASTRONOMY = "astronomy"


class CapabilityStatus(str, Enum):
    """Refined 5-tier lifecycle status for capabilities."""
    IMPLEMENTED = "implemented"      # Fully functional and mathematically verified
    EXPERIMENTAL = "experimental"    # Implemented but pending formal validation or in preview
    PLANNED = "planned"              # Architecture defined, pending model integration or adapter
    UNAVAILABLE = "unavailable"      # Hardware, license, or dependency constraint in environment
    FAILED = "failed"                # Attempted execution returned error or invalid results


class TaskStatus(str, Enum):
    """Status of worker task execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


class WorkerRuntimeStatus(str, Enum):
    """Worker availability state."""
    HEALTHY = "healthy"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class SpatialBounds(BaseModel):
    """Spatial bounding box or point specification."""
    min_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    max_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    min_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    max_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    crs: str = Field("EPSG:4326", description="Coordinate Reference System")

    def is_valid(self) -> bool:
        if None in (self.min_lat, self.max_lat, self.min_lon, self.max_lon):
            return False
        return self.min_lat <= self.max_lat and self.min_lon <= self.max_lon


class TemporalRange(BaseModel):
    """Temporal coverage specification."""
    start_date: str = Field(..., description="ISO date YYYY-MM-DD or datetime")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD or datetime")


class ArtifactRef(BaseModel):
    """Reference to an input or output scientific artifact."""
    artifact_id: str
    artifact_type: str = Field(..., description="e.g. geotiff, netcdf, csv, plot, json")
    format: str
    location: str = Field(..., description="Local filepath, URL, or object storage key")
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

