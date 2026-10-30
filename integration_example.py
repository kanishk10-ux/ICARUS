#!/usr/bin/env python3
"""
ICARUS Scientific Research Engine - Integration Example

This script demonstrates how to use ICARUS components programmatically
outside of the Jupyter notebook environment.

Usage:
    python integration_example.py
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Ensure stdout handles UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add ICARUS to path (when used as standalone)
# sys.path.insert(0, str(Path(__file__).parent))

def demo_environment_check():
    """Demonstrate environment detection."""
    print("=" * 60)
    print("ICARUS - Environment Check")
    print("=" * 60)
    
    import platform
    import shutil
    
    info = {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "disk_free_gb": shutil.disk_usage('.').free // (1024**3),
    }
    
    # Check PyTorch
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024**2)
    except ImportError:
        info["pytorch_version"] = "not installed"
        info["cuda_available"] = False
    
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    return info


def demo_registry_loading():
    """Demonstrate loading and querying registries."""
    print("\n" + "=" * 60)
    print("ICARUS - Registry Loading")
    print("=" * 60)
    
    registry_dir = Path(__file__).parent
    
    # Load model registry
    model_reg_path = registry_dir / "model_registry.json"
    if model_reg_path.exists():
        with open(model_reg_path) as f:
            model_registry = json.load(f)
        print(f"\n  Models registered: {len(model_registry.get('models', []))}")
        for model in model_registry.get("models", []):
            print(f"    - {model['id']}: {model['name']} [{model.get('status', 'unknown')}]")
    else:
        print("  Model registry not found.")
    
    # Load capability registry
    cap_reg_path = registry_dir / "capability_registry.json"
    if cap_reg_path.exists():
        with open(cap_reg_path) as f:
            cap_registry = json.load(f)
        caps = cap_registry.get("capabilities", [])
        print(f"\n  Capabilities registered: {len(caps)}")
        
        # Count by status
        from collections import Counter
        status_counts = Counter(c.get("status", "unknown") for c in caps)
        for status, count in sorted(status_counts.items()):
            print(f"    {status}: {count}")
    else:
        print("  Capability registry not found.")
    
    # Load dataset registry
    ds_reg_path = registry_dir / "dataset_registry.json"
    if ds_reg_path.exists():
        with open(ds_reg_path) as f:
            ds_registry = json.load(f)
        datasets = ds_registry.get("datasets", [])
        print(f"\n  Datasets registered: {len(datasets)}")
        for ds in datasets:
            auth = "[AUTH]" if ds.get("auth_required", False) else "[PUBLIC]"
            print(f"    {auth:8s} {ds['id']}: {ds['name']} [{ds.get('connector_status', 'unknown')}]")
    else:
        print("  Dataset registry not found.")


def demo_research_result_structure():
    """Demonstrate the standard research result structure."""
    print("\n" + "=" * 60)
    print("ICARUS - Research Result Structure")
    print("=" * 60)
    
    result = {
        "research_question": "What is the current NDVI for agricultural regions near Pune, India?",
        "domain": "earth_observation",
        "subdomain": "vegetation_analysis",
        "status": "completed",
        "summary": "NDVI analysis computed using synthetic demonstration data.",
        "findings": [
            {
                "type": "measurement",
                "description": "Mean NDVI value for the region",
                "value": 0.45,
                "unit": "dimensionless",
                "note": "SYNTHETIC - demonstration only"
            }
        ],
        "measurements": [
            {"name": "mean_ndvi", "value": 0.45, "unit": "dimensionless"},
            {"name": "min_ndvi", "value": 0.12, "unit": "dimensionless"},
            {"name": "max_ndvi", "value": 0.78, "unit": "dimensionless"},
        ],
        "artifacts": [],
        "visualization_data": {},
        "methodology": [
            "NDVI calculated as (NIR - Red) / (NIR + Red)",
            "Standard spectral index computation"
        ],
        "data_sources": [],
        "models_used": [],
        "validation": {"method": "range_check", "valid": True},
        "uncertainty": {"source": "synthetic_data", "note": "Results from demo data"},
        "limitations": [
            "Uses synthetic data - not real satellite observations",
            "No atmospheric correction applied"
        ],
        "provenance": {
            "engine": "ICARUS v0.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "local",
            "reproducible": True
        },
        "warnings": ["This result uses synthetic demonstration data"]
    }
    
    print(json.dumps(result, indent=2))
    return result


def demo_data_profiling():
    """Demonstrate the data profiler concept."""
    print("\n" + "=" * 60)
    print("ICARUS - Data Profiler Demo")
    print("=" * 60)
    
    # Simulated profiling of a GeoTIFF file
    profile = {
        "file_path": "example_sentinel2.tif",
        "format": "GeoTIFF",
        "exists": False,
        "note": "File does not exist - showing expected profile structure",
        "expected_fields": {
            "dimensions": {"width": 10980, "height": 10980},
            "bands": 13,
            "dtype": "uint16",
            "crs": "EPSG:32643",
            "bounds": {"left": 0, "bottom": 0, "right": 109800, "top": 109800},
            "nodata": 0,
            "transform": "Affine(10.0, 0.0, ...)",
            "metadata": {}
        }
    }
    
    print(json.dumps(profile, indent=2))


def main():
    """Run all demos."""
    print("+==========================================================+")
    print("|     ICARUS - Scientific Research Engine v0.1.0           |")
    print("|     Integration Example                                  |")
    print("+==========================================================+")
    
    demo_environment_check()
    demo_registry_loading()
    demo_data_profiling()
    result = demo_research_result_structure()
    
    print("\n" + "=" * 60)
    print("Integration example complete.")
    print("See ICARUS_Scientific_Research_Engine.ipynb for full usage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
