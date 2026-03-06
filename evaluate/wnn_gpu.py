"""
wnn_gpu.py — Person-1 (API & Alignment)

Handles:
  - Input validation for MuData with two modalities (RNA, ATAC)
  - Cell alignment across modalities (same cells, same order)
  - Reproducibility metadata (params + alignment report)
  - Seeds + optional RMM memory pool init (safe no-ops without RAPIDS)
  - Host↔device helpers for later GPU stages (cuML/cuGraph)

Note: This file is CPU-safe. If CuPy/RMM aren’t installed, GPU bits are skipped.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Iterable
import contextlib
import random
import hashlib  # <-- added
import numpy as np
from mudata import MuData


# --------------------------- GPU plumbing (safe no-ops) ---------------------------
def init_seeds_and_pool(seed: int) -> None:
    """Set CPU/CuPy RNG seeds and (optionally) initialize an RMM pool for CuPy."""
    # CPU determinism
    random.seed(seed)
    np.random.seed(seed)

    # Optional: CuPy RNG + RMM pool (only if available)
    with contextlib.suppress(ImportError):
        import cupy as cp  # type: ignore
        cp.random.seed(seed)
        with contextlib.suppress(ImportError):
            import rmm  # type: ignore
            # Initialize a simple pool allocator if not already initialized
            if not rmm.is_initialized():
                rmm.reinitialize(pool_allocator=True)
            # Tell CuPy to use RMM for allocations
            cp.cuda.set_allocator(rmm.rmm_cupy_allocator)


def to_device(x):
    """Move a NumPy array (or array-like) to GPU (CuPy) if available; otherwise return unchanged."""
    try:
        import cupy as cp  # type: ignore
        # (Extend later for sparse types if needed)
        return cp.asarray(x)
    except ImportError:
        return x


def to_host(x):
    """Bring a CuPy array back to host NumPy if needed; otherwise return unchanged."""
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return x


# ----------------------------- Validation -----------------------------
def validate_mudata(mdata: MuData, modalities: Tuple[str, str] = ("rna", "atac")) -> MuData:
    """
    Ensure the input is a MuData with the expected modalities and that
    each modality has non-empty obs/var.
    """
    if not isinstance(mdata, MuData):
        raise TypeError("Expected a mudata.MuData object")

    for key in modalities:
        if key not in mdata.mod:
            raise KeyError(f"MuData is missing modality '{key}'")
        ad = mdata.mod[key]
        if ad.n_obs == 0 or ad.n_vars == 0:
            raise ValueError(f"Modality '{key}' is empty (no cells or no features).")
    return mdata


# ----------------------------- Alignment ------------------------------
@dataclass
class AlignmentReport:
    """Summary of alignment results; stored in mdata.uns['wnn_alignment']."""
    rna_n_before: int
    atac_n_before: int
    n_intersection: int
    n_only_rna: int
    n_only_atac: int
    kept_fraction: float
    order_hash: str
    notes: str = "Cells present in both modalities were kept; others dropped."

    def to_meta(self) -> Dict[str, Any]:
        return asdict(self)


def _stable_order_hash(names) -> str:
    """Stable, cross-process/content-based hash of the final cell order."""
    h = hashlib.sha1()
    for n in names:
        h.update(str(n).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _intersection_in_reference_order(ref_names: Iterable[str], other_names: Iterable[str]):
    """Intersection of two name lists, preserving the order of the first (reference)."""
    other_set = set(other_names)
    return [c for c in ref_names if c in other_set]


def align_modalities(mdata: MuData, rna_key: str = "rna", atac_key: str = "atac") -> AlignmentReport:
    """
    Make RNA and ATAC refer to the same cells in the same order.
    Steps:
      1) Compute intersection (RNA order).
      2) Subset both modalities to that intersection.
      3) Verify identical final orders.
      4) Write aligned AnnData back into mdata.mod[...].
      5) Return a report with counts + a quick hash of final order.
    """
    ad_rna = mdata.mod[rna_key]
    ad_atac = mdata.mod[atac_key]

    rna_n_before = ad_rna.n_obs
    atac_n_before = ad_atac.n_obs

    inter = _intersection_in_reference_order(ad_rna.obs_names.tolist(),
                                             ad_atac.obs_names.tolist())

    # ---- added: empty intersection guard ----
    if len(inter) == 0:
        raise ValueError(
            "RNA and ATAC share 0 cells after matching obs_names. "
            "Check barcode prefixes/suffixes (e.g., '-1' vs no suffix) or preprocessing steps."
        )

    n_intersection = len(inter)
    n_only_rna = rna_n_before - n_intersection
    n_only_atac = atac_n_before - n_intersection
    kept_fraction = (n_intersection / max(rna_n_before, atac_n_before)) if max(rna_n_before, atac_n_before) else 0.0

    ad_rna_aligned = ad_rna[inter].copy()
    ad_atac_aligned = ad_atac[inter].copy()

    if list(ad_rna_aligned.obs_names) != list(ad_atac_aligned.obs_names):
        raise RuntimeError("Post-alignment orders differ between RNA and ATAC — this should not happen.")

    mdata.mod[rna_key] = ad_rna_aligned
    mdata.mod[atac_key] = ad_atac_aligned

    # ---- changed: use stable content hash instead of Python's randomizing hash() ----
    order_hash = _stable_order_hash(ad_rna_aligned.obs_names.tolist())

    return AlignmentReport(
        rna_n_before=rna_n_before,
        atac_n_before=atac_n_before,
        n_intersection=n_intersection,
        n_only_rna=n_only_rna,
        n_only_atac=n_only_atac,
        kept_fraction=kept_fraction,
        order_hash=order_hash,
    )


# ------------------------------- API ----------------------------------
def wnn_gpu(
    mdata: MuData,
    modalities: Tuple[str, str] = ("rna", "atac"),
    k: int = 15,
    metric: str = "euclidean",
    weight_mode: str = "local",
    resolution: float = 1.0,
    seed: int = 0,
    deterministic: bool = True,
    fp_dtype: str = "float32",
) -> MuData:
    """
    Public entrypoint for the WNN pipeline (Person-1 portion only).
    Validates input, aligns modalities, and records params + alignment report.
    Also seeds RNG and (optionally) initializes an RMM pool for GPU runs.
    """
    # 0) Seeds + (optional) RMM pool — safe to do even on CPU-only setups
    init_seeds_and_pool(seed)

    # 1) Validate input
    md = validate_mudata(mdata, modalities)

    # 2) Align modalities (your core deliverable)
    ref_key, other_key = modalities
    report = align_modalities(md, rna_key=ref_key, atac_key=other_key)

    # 3) Record parameters and alignment summary in md.uns
    params = md.uns.setdefault("wnn_params", {})
    params.update(
        dict(
            modalities=list(modalities),
            k=k,
            metric=metric,
            weight_mode=weight_mode,
            resolution=resolution,
            seed=seed,
            deterministic=deterministic,
            fp_dtype=fp_dtype,
        )
    )
    md.uns["wnn_alignment"] = report.to_meta()
    md.uns.setdefault("wnn_runtime", {})  # teammates will fill timings later

    # 4) Return aligned MuData
    return md
