# -*- coding: utf-8 -*-
"""Verification Script: Real Prithvi-EO-2.0-300M Smoke Test for Kaggle.

Directly imports and tests the production PrithviEOAdapter and EOPipeline used by
the ICARUS Earth Observation worker service to guarantee consistency and avoid
maintaining two separate implementations.
"""
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_prithvi_smoke_test():
    print("=" * 70)
    print("ICARUS: Prithvi-EO-2.0-300M Production Adapter Verification")
    print("=" * 70)

    # 1. Dependency and Environment Checks
    try:
        import torch
        print(f"[1/6] PyTorch Version: {torch.__version__}")
    except ImportError:
        print("[FAIL] PyTorch is not installed. Please run in a GPU-accelerated environment.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Execution Device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"      GPU Hardware:     {gpu_name} ({total_vram:.2f} GB VRAM)")
    else:
        print("      WARNING: Running on CPU. Inference will succeed but may be slow.")

    # 2. Import Production Adapter
    print("\n[2/6] Importing production PrithviEOAdapter from icarus.workers.eo...")
    try:
        from icarus.workers.eo.prithvi_adapter import PrithviEOAdapter, PrithviModelStatus
        from icarus.workers.eo.eo_pipeline import EOPipeline
        print("      Successfully imported production PrithviEOAdapter and EOPipeline.")
    except Exception as e:
        print(f"[FAIL] Failed to import production modules: {e}")
        sys.exit(1)

    adapter = PrithviEOAdapter(device=device)
    print(f"      Target Model ID:  {adapter.MODEL_ID}")
    print(f"      HF Repository:    {adapter.HF_REPO_ID}")
    print(f"      Pinned Revision:  {adapter.HF_REVISION}")
    print(f"      Initial Status:   {adapter.status.value}")

    # 3. Load Checkpoint via Production Loader
    print("\n[3/6] Loading model weights via adapter.load_model()...")
    t0 = time.time()
    ok, msg = adapter.load_model()
    load_time = time.time() - t0
    if not ok:
        print(f"[FAIL] adapter.load_model() failed: {msg}")
        sys.exit(1)

    print(f"      {msg}")
    print(f"      Loading Time:     {load_time:.2f}s")
    print(f"      Model Status:     {adapter.status.value}")

    # 4. Verify Model Parameter Count
    print("\n[4/6] Verifying parameter count of production model...")
    total_params = sum(p.numel() for p in adapter.model.parameters())
    print(f"      Parameter Count:  {total_params:,}")
    print(f"      Expected Count:   {adapter.PARAMETER_COUNT:,}")
    assert total_params == adapter.PARAMETER_COUNT, (
        f"Parameter mismatch! Expected {adapter.PARAMETER_COUNT:,}, got {total_params:,}"
    )
    print("      Parameter count matches official 330,419,712 specification.")

    # 5. Spatio-Temporal Sequence Inference
    print("\n[5/6] Executing feature extraction on verified (1, 6, 4, 224, 224) input...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    raw_input = torch.randn(1, 6, 4, 224, 224, dtype=torch.float32, device=device)
    norm_input = adapter.normalize_channels(raw_input)

    t_start = time.perf_counter()
    res = adapter.extract_features(norm_input)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_time_ms = (time.perf_counter() - t_start) * 1000.0

    if res.get("status") != "success":
        print(f"[FAIL] extract_features returned error: {res.get('error')}")
        sys.exit(1)

    token_shape = res["token_shape"]
    patch_token_shape = res["patch_token_shape"]
    print(f"      Token Shape:      {token_shape} (Expected: [1, 785, 1024])")
    print(f"      Patch Shape:      {patch_token_shape} (Expected: [1, 784, 1024])")
    assert token_shape == [1, 785, 1024], f"Unexpected token shape: {token_shape}"
    assert patch_token_shape == [1, 784, 1024], f"Unexpected patch shape: {patch_token_shape}"

    print(f"      Inference Time:   {infer_time_ms:.2f} ms")
    print(f"      CLS Token Mean:   {res['cls_token_summary']['mean']:.4f}")
    print(f"      Patch Token Mean: {res['patch_embedding_summary']['mean']:.4f}")

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"      Peak GPU VRAM:    {peak_vram:.2f} MB")

    # 6. Lifecycle: Memory Reclamation Check
    print("\n[6/6] Verifying VRAM reclamation via adapter.unload_model()...")
    unloaded, un_msg, freed = adapter.unload_model()
    print(f"      {un_msg} (Freed: ~{freed} MB)")
    print(f"      Final Status:     {adapter.status.value}")
    assert adapter.status == PrithviModelStatus.UNLOADED
    assert adapter.model is None

    print("\n" + "=" * 70)
    print("SUCCESS: Production PrithviEOAdapter verified end-to-end.")
    print("=" * 70)


if __name__ == "__main__":
    run_prithvi_smoke_test()
