# ICARUS — Multi-Domain Scientific Research Engine

> **I**ntelligent **C**omprehensive **A**nalysis and **R**esearch **U**nified **S**ystem

## Overview

ICARUS is a modular, multi-domain scientific research engine designed to accept research questions, identify relevant scientific domains, retrieve real datasets, execute analysis using specialist models and algorithms, and return structured, reproducible research outputs.

### Key Features

- 🌍 **8 Scientific Domains** — Earth observation, oceans, atmosphere, solid earth, climate, heliophysics, planetary science, astronomy
- 🤖 **Model Registry** — Curated scientific AI models (Prithvi, Aurora, SeisBench, Zoobot, etc.)
- 📊 **Algorithm Library** — Domain-specific scientific algorithms (NDVI, spectral analysis, seismic processing, etc.)
- 🔗 **Data Connectors** — Interfaces to major scientific data providers (NASA, ESA, NOAA, USGS, etc.)
- 📋 **Capability Registry** — Machine-readable registry with honest status tracking
- 🔬 **Research Pipeline** — Structured workflow from question to reproducible results
- 📦 **Standard Output Format** — Consistent research result schema across all domains

## Quick Start

### Kaggle (Recommended)

1. Upload the notebook to Kaggle
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
3. Add required API keys via Kaggle Secrets (optional, for live data)
4. Run all cells

### Local

```bash
pip install -r requirements.txt
jupyter notebook ICARUS_Scientific_Research_Engine.ipynb
```

## Architecture

```
┌─────────────────────────────────────────────┐
│             Research Interface               │
│   run_research() / run_icarus_pipeline()     │
├─────────────────────────────────────────────┤
│            Research Orchestrator              │
│  Domain ID → Planning → Execution → Report   │
├────────┬────────┬────────┬──────────────────┤
│  Data  │ Model  │ Algo   │  Capability      │
│Connectors│Registry│Library │  Registry        │
├────────┴────────┴────────┴──────────────────┤
│           Domain Modules (×8)                │
│  EO │ Ocean │ Atm │ Earth │ Clim │ ...      │
├─────────────────────────────────────────────┤
│        Scientific Data Profiler              │
├─────────────────────────────────────────────┤
│     Environment Detection & GPU Mgmt         │
└─────────────────────────────────────────────┘
```

## Domains

| Domain | Models | Algorithm Support | Data Connectors |
|--------|--------|------------------|----------------|
| Earth Observation | Prithvi EO 2.0 | NDVI, NDWI, NBR, change detection | Copernicus, NASA Earthdata |
| Oceans | Aurora Wave | SST, salinity, anomaly detection | CMEMS, PO.DAAC, Argo |
| Atmosphere | Microsoft Aurora | Temperature, wind, rainfall analysis | ECMWF CDS, NOAA |
| Solid Earth | PhaseNet, EQTransformer | Seismic processing, phase picking | USGS, FDSN |
| Climate | Prithvi WxC | Climate trends, drought indices | ERA5, GISTEMP |
| Heliophysics | Surya | Solar activity analysis | NASA SDO, NOAA SWPC |
| Planetary | Lunar Crater FM | Crater statistics, terrain analysis | NASA PDS |
| Astronomy | Zoobot, AstroCLIP | Spectral, light-curve, transit analysis | MAST, Gaia, SDSS |

## Configuration

See `CONFIG` in the notebook for all settings. Key options:

- `use_gpu` — Enable GPU acceleration
- `enable_live_data` — Enable real data connector calls
- `download_models` — Auto-download model checkpoints
- `enable_llm` — Enable LLM-based research planner (future)

## Files

| File | Description |
|------|-------------|
| `ICARUS_Scientific_Research_Engine.ipynb` | Main notebook |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `model_registry.json` | Model definitions and metadata |
| `capability_registry.json` | Capability status tracking |
| `dataset_registry.json` | Data source catalog |
| `integration_example.py` | Programmatic usage example |
| `sample_result.json` | Example research output |

## API Keys

| Provider | Environment Variable | Required For |
|----------|---------------------|-------------|
| Copernicus | `COPERNICUS_CLIENT_ID`, `COPERNICUS_CLIENT_SECRET` | Satellite imagery |
| NASA Earthdata | `NASA_EARTHDATA_USERNAME`, `NASA_EARTHDATA_PASSWORD` | NASA datasets |
| ECMWF CDS | `CDS_API_KEY` | ERA5, atmospheric data |
| CMEMS | `COPERNICUS_MARINE_USERNAME`, `COPERNICUS_MARINE_PASSWORD` | Ocean data |

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black .

# Run integration example
python integration_example.py
```

## License

Apache 2.0

## Acknowledgments

- NASA/IBM for Prithvi foundation models
- Microsoft for Aurora weather models
- SeisBench team for seismological ML framework
- Galaxy Zoo / Zoobot for galaxy morphology models
- NASA IMPACT for Surya and Lunar Foundation Models
- All data providers: ESA/Copernicus, NASA, NOAA, USGS, ECMWF

---
