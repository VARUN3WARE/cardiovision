"""Twin cohort: a similarity space and a 3D map of the 303 dataset patients ("Similar patients" card).

Writes ../frontend/src/cohort/cohort.json, a static file the UI uses to find a patient's most similar
real patients in the browser. Deterministic: re-running gives a byte-identical file. Run after train.py:

    .venv/bin/python make_cohort.py

Similarity space (the frontend places a new patient with exactly these numbers, see cohort/knn.ts):
  1. The 55 model inputs, encoded as the API encodes them (models/encoding.json), i.e. the values in
     /explain's `feature_values`. VHD is re-coded to its severity order (none < mild < moderate <
     severe) exactly as the models do (cardio_models.VHD_ORD); the alphabetical codes would make "mild"
     the farthest level from "none". BBB keeps its codes (LBBB 0, none 1, RBBB 2): both blocks sit one
     step from "none".
  2. Each input is standardised over the 303 patients: z = (x - mean) / sd.
  3. Each z is multiplied by the CAD model's mean |attribution| for that input (models/feature_info.json
     influence["cad"], log-odds units). For an input the model uses roughly linearly the attribution is
     about beta * (x - mean), so a weighted difference approximates how far apart two patients are in
     that input's share of the CAD log-odds: similarity reflects what the model uses (typical angina,
     age, wall motion, hypertension, diabetes and T-wave inversion dominate; inputs the model barely
     uses barely count; the constant "Exertional CP" gets weight 0). Derived scores (Diamond-Forrester
     pre-test probability, risk-factor count, ...) are functions of these inputs, so they are not added
     a second time.
  Distance = Euclidean distance between weighted vectors. The 10 nearest neighbours use this full
  distance; the 3D map is only a picture of it.

3D map: t-SNE (3 components, exact gradient, PCA initialisation, fixed random_state) of the weighted
vectors. PCA is used instead if t-SNE fails or preserves 10-neighbourhoods worse than PCA
(sklearn trustworthiness). Coordinates are centred and scaled so 98% of patients lie inside radius 1.

Inputs that were not entered are left out of the distance and of the identity test in the browser
(the API fills them in for the model, unrecorded findings as absent and measurements as the cohort
median, and lists them in the response's `imputed`); `schema_to_column` maps those schema names to
dataset columns.

Recognising the patient's own record (`identity`; knn.ts matchRecord applies these numbers): a dataset
patient who agrees on every compared input within SAME_TOLERANCE sd is this patient (left out of the
neighbours); one who agrees on all but MAX_CHANGED or fewer is this patient's record with those inputs
changed (also left out, so its own outcome is not counted). `bits` says how specific agreeing on each
input is: -log2 of the chance that two different cohort patients have the same value. A match counts
only when the agreeing inputs carry at least EXACT_BITS (EDITED_BITS for a changed record) and no other
patient qualifies, so a few entered inputs (say age and sex) never "identify" anyone. BMI and the
obesity flag are compared through weight and height when those are given (the API derives them).
With these values every dataset patient, sent in Quick mode or with every field, is recognised as
itself, and no simulated new patient (each field taken from a different dataset patient) was.

Closeness (`closeness`): each dataset patient's distance to its 10th most similar other patient; the UI
says "no close matches" when a patient's 10th is farther than for 95% of the dataset's own patients
(recomputed in the browser on the same inputs when some were not entered).

Map groups (`groups`): when one binary input (the one the CAD model weighs most) splits the map, i.e.
almost every patient's map neighbours share its value, the caption names it with each group's size and
CAD rate. Omitted when that is not the case.

Per patient: dataset row index, map coordinates, the 55 encoded inputs (for neighbours in the browser),
angiography outcomes (CAD = Cath; LAD/LCX/RCA = >=50% narrowing), the mean out-of-fold ("held-out")
probability per target from models/oof_predictions.npz (5 CV repeats; fits that never saw the
patient), age and sex. `check` lists Python-computed neighbours for the preset patients; the frontend
unit test asserts the browser code finds the same ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # single-threaded BLAS: bit-identical output on every run

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE, trustworthiness  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.cardio_models import VHD_ORD  # noqa: E402
from app.config import FEATURE_INFO_PATH, MODELS_DIR, TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE  # noqa: E402
from app.encoders import binarize_target, encode_dataframe, load_encoding  # noqa: E402
from app.schemas import SCHEMA_TO_COLUMN  # noqa: E402
from train import build_features, load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "frontend" / "src" / "cohort" / "cohort.json"
PRESETS_PATH = ROOT / "frontend" / "src" / "presets.json"
K = 10  # neighbours shown in the UI
SEED = 0
RECODE = {"VHD": VHD_ORD}  # encoded code -> similarity value, applied before standardising
TSNE_ARGS = dict(n_components=3, perplexity=30.0, early_exaggeration=12.0, learning_rate="auto",
                 max_iter=1500, init="pca", method="exact")
# Recognising the patient's own record (see the docstring; knn.ts reads these from the JSON).
SAME_TOLERANCE = 0.02  # sd: inputs closer than this count as the same value
EXACT_BITS = 20.0  # a given other patient would agree on the inputs by chance about once in a million
EDITED_BITS = 32.0  # stricter: a changed record may differ in any MAX_CHANGED of the compared inputs
MAX_CHANGED = 3
# Derived by the API from other inputs (schemas.PatientFeatures._consistency): compared through their
# sources when those are given. Row 153 is BMI 25.0 / obesity "Y" in the dataset, but 64 kg and 160 cm
# give 24.999... in floating point, so the API sets the flag to "N".
COMPARE_VIA = {"BMI": ["Weight", "Length"], "Obesity": ["Weight", "Length"]}
CLOSE_QUANTILE = 0.95  # "close": the 10th most similar no farther than for 95% of the dataset's patients
GROUP_MIN_SHARE = 0.95  # map neighbours sharing the binary input's value, to call the map "split" by it


def sig(v: float, digits: int = 6) -> float:
    """Round to significant digits (the JSON's mean/sd; everything downstream uses the rounded values)."""
    return float(f"{v:.{digits}g}")


def num(v: float):
    """Compact JSON number: integers without '.0', other values to 4 decimals."""
    v = round(float(v), 4)
    return int(v) if v.is_integer() else v


def scaled(raw: np.ndarray, columns: list[str], mean: np.ndarray, std: np.ndarray,
           weight: np.ndarray) -> np.ndarray:
    """Encoded inputs -> weighted standardised vectors; same operations, in the same order, as knn.ts."""
    x = raw.astype(float).copy()
    for col, codes in RECODE.items():
        j = columns.index(col)
        x[:, j] = [codes[float(c)] for c in x[:, j]]
    scale = np.where(std > 0, weight / np.where(std > 0, std, 1.0), 0.0)
    return (x - mean) * scale


def sq_dists(U: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Squared distances, summed column by column (the browser's summation order: identical bits)."""
    d2 = np.zeros(len(U))
    for j in range(U.shape[1]):
        diff = U[:, j] - u[j]
        d2 += diff * diff
    return d2


def neighbours(U: np.ndarray, i: int, k: int = K) -> list[int]:
    """k nearest other patients of row i (ties broken by row index, as in the browser)."""
    d2 = sq_dists(U, U[i])
    order = np.lexsort((np.arange(len(U)), d2))
    return [int(j) for j in order if j != i][:k]


def match_bits(x: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per input (recoded values): -log2 of the chance that two different cohort patients agree on it
    within SAME_TOLERANCE sd, i.e. how much agreeing on it singles a patient out; 0 for constant inputs."""
    n = len(x)
    bits = np.zeros(x.shape[1])
    for j in np.flatnonzero(std > 0):
        same = np.abs(x[:, j, None] - x[None, :, j]) * (1.0 / std[j]) < SAME_TOLERANCE
        p = (same.sum() - n) / (n * (n - 1))
        bits[j] = -np.log2(p) if p > 0 else np.log2(n * (n - 1))
    return bits


def kth_distances(U: np.ndarray, k: int = K) -> np.ndarray:
    """Each patient's distance to its k-th most similar other patient (the browser's summation order)."""
    out = np.empty(len(U))
    for i in range(len(U)):
        d2 = sq_dists(U, U[i])
        d2[i] = np.inf
        out[i] = np.sqrt(np.sort(d2)[k - 1])
    return out


def map_groups(raw: np.ndarray, columns: list[str], weight: np.ndarray, truth: np.ndarray,
               full_nn: list[set[int]], map_nn: list[set[int]]) -> dict | None:
    """The binary input the CAD model weighs most, if it splits the map (almost every patient's map
    neighbours share its value): size and CAD rate per group for the caption; else None."""
    j = int(np.argmax(weight))
    g = raw[:, j]
    levels = np.unique(g)
    if len(levels) != 2:
        return None
    same = lambda sets: float(np.mean([np.mean(g[sorted(s)] == g[i]) for i, s in enumerate(sets)]))  # noqa: E731
    share_full, share_map = same(full_nn), same(map_nn)
    if share_map < GROUP_MIN_SHARE:
        return None
    cad = TARGETS.index("cad")
    return {
        "column": columns[j],
        "levels": [{"value": num(v), "n": int((g == v).sum()), "cad": round(float(truth[g == v, cad].mean()), 3)}
                   for v in levels],
        "neighbours_same": round(share_full, 3),
        "map_neighbours_same": round(share_map, 3),
    }


def knn_sets(Z: np.ndarray, k: int = K) -> list[set[int]]:
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return [set(np.argsort(row, kind="stable")[:k].tolist()) for row in D]


def retention(A: list[set[int]], B: list[set[int]]) -> float:
    """Mean share of each patient's k nearest in A that are also among its k nearest in B."""
    return float(np.mean([len(a & b) / len(a) for a, b in zip(A, B)]))


def embed(U: np.ndarray) -> tuple[np.ndarray, dict]:
    pca = PCA(n_components=3, random_state=SEED).fit_transform(U)
    t_pca = trustworthiness(U, pca, n_neighbors=K)
    info: dict = {"pca_trustworthiness": round(float(t_pca), 3)}
    try:
        tsne = TSNE(random_state=SEED, **TSNE_ARGS).fit_transform(U)
        if not np.isfinite(tsne).all():
            raise ValueError("non-finite t-SNE coordinates")
        t_tsne = trustworthiness(U, tsne, n_neighbors=K)
        # Stability: local neighbourhoods should not depend much on the seed.
        alt = TSNE(random_state=SEED + 1, **TSNE_ARGS).fit_transform(U)
        info["tsne_seed_agreement"] = round(retention(knn_sets(tsne), knn_sets(alt)), 3)
        info["tsne_trustworthiness"] = round(float(t_tsne), 3)
    except Exception as e:  # noqa: BLE001 - any failure means the PCA fallback
        print(f"t-SNE failed ({e}); using PCA")
        tsne, t_tsne = None, -1.0
    if tsne is not None and t_tsne >= t_pca:
        Z = tsne
        info["method"] = (f"t-SNE (3 components, perplexity {TSNE_ARGS['perplexity']:g}, exact gradient, "
                          f"PCA initialisation, random_state {SEED})")
    else:
        Z = pca
        info["method"] = f"PCA (3 components, random_state {SEED}); t-SNE was unstable or less faithful"
    Z = Z - Z.mean(axis=0)
    Z = Z / np.quantile(np.linalg.norm(Z, axis=1), 0.98)
    return Z, info


def main():
    t0 = time.time()
    df = load_dataset()
    columns = build_features(df)
    encoding = load_encoding()
    X = encode_dataframe(df, encoding, columns)
    # Round once; mean, sd, distances and the map are all computed from exactly what the JSON holds.
    raw = np.array([[num(v) for v in row] for row in X.to_numpy()], dtype=float)

    influence = json.loads(FEATURE_INFO_PATH.read_text())["influence"]["cad"]
    recoded = scaled(raw, columns, np.zeros(len(columns)), np.ones(len(columns)), np.ones(len(columns)))
    mean = np.array([sig(v) for v in recoded.mean(axis=0)])
    std = np.array([sig(v) for v in recoded.std(axis=0)])
    weight = np.array([float(influence.get(c, 0.0)) if s > 0 else 0.0 for c, s in zip(columns, std)])
    U = scaled(raw, columns, mean, std, weight)

    Z, emb_info = embed(U)
    full_nn, map_nn = knn_sets(U), knn_sets(Z)
    emb_info["neighbour_retention"] = round(retention(full_nn, map_nn), 3)

    truth = np.array([binarize_target(df[TARGET_COLUMN_MAP[t]], TARGET_POSITIVE[TARGET_COLUMN_MAP[t]]).to_numpy()
                      for t in TARGETS]).T
    oof = np.load(MODELS_DIR / "oof_predictions.npz")
    held_out = np.array([oof[t].mean(axis=0) for t in TARGETS]).T

    # How informative are 10 similar patients? Leave-one-out share with each outcome vs the patient's own
    # outcome (reported here only; the weights come from the full-data CAD model, so this is optimistic).
    nn_all = [neighbours(U, i) for i in range(len(U))]
    share = np.array([truth[nn].mean(axis=0) for nn in nn_all])
    knn_auc = {t: round(float(roc_auc_score(truth[:, j], share[:, j])), 3) for j, t in enumerate(TARGETS)}

    bits = match_bits(recoded, std)
    kth = kth_distances(U)
    close_at = float(np.quantile(kth, CLOSE_QUANTILE))
    groups = map_groups(raw, columns, weight, truth, full_nn, map_nn)

    presets = json.loads(PRESETS_PATH.read_text()) if PRESETS_PATH.exists() else []
    check_rows = sorted({int(p["patient_index"]) for p in presets} | {0, 302})

    out = {
        "about": "Generated by backend/make_cohort.py (see its docstring). Do not edit by hand.",
        "n": int(len(df)),
        "k": K,
        "targets": list(TARGETS),
        "columns": columns,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "weight": [float(v) for v in weight],
        "recode": {c: {str(int(k)): num(v) for k, v in sorted(m.items())} for c, m in RECODE.items()},
        "weighting": "z-scored inputs x CAD model mean |attribution| (log-odds), models/feature_info.json",
        "held_out": "mean out-of-fold probability over 5 CV repeats (models/oof_predictions.npz)",
        "embedding": emb_info,
        "knn_loo_auc": knn_auc,
        "schema_to_column": {k: v for k, v in SCHEMA_TO_COLUMN.items() if v in columns},
        "identity": {
            "tolerance_sd": SAME_TOLERANCE,
            "exact_bits": EXACT_BITS,
            "edited_bits": EDITED_BITS,
            "max_changed": MAX_CHANGED,
            "via": COMPARE_VIA,
            "bits": [round(float(b), 3) for b in bits],
        },
        "closeness": {"k": K, "quantile": CLOSE_QUANTILE, "distance": round(close_at, 6)},
        "groups": groups,
        "check": [{"i": i, "neighbours": nn_all[i]} for i in check_rows],
        "patients": [
            {
                "i": int(i),
                "xyz": [round(float(v), 3) for v in Z[i]],
                "x": [num(v) for v in raw[i]],
                "angio": [int(v) for v in truth[i]],
                "p": [round(float(v), 3) for v in held_out[i]],
                "age": int(df["Age"].iloc[i]),
                "sex": "M" if df["Sex"].iloc[i] == "Male" else "F",
            }
            for i in range(len(df))
        ],
    }

    # One top-level field per line and one patient per line: compact, but diffs stay readable.
    compact = lambda v: json.dumps(v, separators=(",", ":"), ensure_ascii=False)  # noqa: E731
    lines = ["{"] + [f" {json.dumps(k)}: {compact(v)}," for k, v in out.items() if k != "patients"]
    lines += [' "patients": [', ",\n".join("  " + compact(p) for p in out["patients"]), " ]", "}"]
    text = "\n".join(lines) + "\n"
    assert json.loads(text) == out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)

    top = sorted(zip(columns, weight / np.where(std > 0, std, 1)), key=lambda kv: -kv[1])[:8]
    print("Largest per-unit weights (weight / sd):", ", ".join(f"{c} {v:.3f}" for c, v in top))
    print(f"Embedding: {emb_info}")
    print(f"Leave-one-out AUC of the outcome share among {K} similar patients (optimistic): {knn_auc}")
    print(f"Close: 10th most similar within {close_at:.3f} for {CLOSE_QUANTILE:.0%} of patients "
          f"(median {np.median(kth):.3f}, max {kth.max():.3f})")
    specific = sorted(zip(columns, bits), key=lambda t: -t[1])[:8]
    print("Most specific inputs (bits):", ", ".join(f"{c} {b:.1f}" for c, b in specific))
    print(f"Map groups: {groups}")
    for p in presets:
        i = int(p["patient_index"])
        nn = nn_all[i]
        counts = truth[nn].sum(axis=0)
        print(f"{p['id']:9} #{i:3}: {K} similar -> " + ", ".join(f"{t.upper()} {c}" for t, c in zip(TARGETS, counts))
              + f"  (patient: {', '.join(t.upper() for t, v in zip(TARGETS, truth[i]) if v) or 'none'})")
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)} ({len(text) / 1024:.0f} KB, sha256 {digest}) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
