#!/usr/bin/env python3
"""Regenerate parts.json: which BodyParts3D meshes CardioVision uses, and what each one is.

Reads the official BodyParts3D index files (downloaded into cache/ on first run) and resolves every element
file (FJ id) to its most specific FMA concept. Coronary arteries are grouped by the model output they show:
the LAD, LCX and RCA trees by FMA part-of membership, the left main on its own (not labelled in the dataset).
"""
from __future__ import annotations

import collections
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
BASE = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/"
INDEX = ["isa_element_parts.txt", "partof_element_parts.txt"]


def load(fn: str):
    path = CACHE / fn
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(BASE + fn, path)
    by_c: dict[str, set[str]] = collections.defaultdict(set)
    name: dict[str, str] = {}
    for line in path.read_text().splitlines()[1:]:
        c, n, f = line.split("\t")
        by_c[c].add(f)
        name[c] = n
    by_f: dict[str, set[str]] = collections.defaultdict(set)
    for c, fs in by_c.items():
        for f in fs:
            by_f[f].add(c)
    return by_c, by_f, name


isa, isa_by_f, isa_name = load("isa_element_parts.txt")
partof, _, partof_name = load("partof_element_parts.txt")


def concept(fma: str) -> set[str]:
    return isa.get(fma, set()) | partof.get(fma, set())


def specific(fj: str) -> tuple[str, str]:
    """The most specific IS-A concept of an element file (fewest element files)."""
    c = min(isa_by_f[fj], key=lambda c: (len(isa[c]), -len(isa_name[c])))
    return c, isa_name[c]


def fj_sort(ids):
    return sorted(ids, key=lambda s: (int("".join(ch for ch in s[2:] if ch.isdigit())), s))


lad = concept("FMA3862") | concept("FMA3892")          # anterior interventricular branch + its septal branches
lcx = concept("FMA3895")                               # circumflex branch of left coronary artery
rca = concept("FMA50039") | concept("FMA3845")         # right coronary artery incl. PDA and its septal branches
lm = concept("FMA3855")                                # trunk of left coronary artery
veins = concept("FMA12846") | concept("FMA4706")       # cardiac veins + coronary sinus
assert not (lad & lcx or lad & rca or lcx & rca or lm & (lad | lcx | rca)), "vessel trees overlap"

heart_parts = [
    dict(key="ventricles", layer="myocardium", fma="FMA13884", label="Ventricular wall", files=["FJ2428"]),
    dict(key="atria", layer="atrium", fma="FMA13883", label="Atrial walls", files=["FJ2438", "FJ2439"]),
    dict(key="lv_cavity", layer="cavity", fma="FMA9466", label="Left ventricular cavity", files=["FJ2422"]),
    dict(key="rv_cavity", layer="cavity", fma="FMA9291", label="Right ventricular cavity", files=["FJ2423"]),
    dict(key="aorta", layer="artery", fma="FMA3734", label="Aorta and arch branches",
         files=["FJ3413", "FJ3411", "FJ1931", "FJ3417", "FJ3483", "FJ3479"]),
    dict(key="pulmonary", layer="pulmonary", fma="FMA8612", label="Pulmonary trunk and arteries",
         files=["FJ2966", "FJ3019", "FJ2924"]),
    dict(key="venae_cavae", layer="vein", fma="FMA4720", label="Venae cavae", files=["FJ3645", "FJ3441"]),
    dict(key="lad", layer="coronary", vessel="lad", fma="FMA3862",
         label="Anterior interventricular branch (LAD)", files=fj_sort(lad)),
    dict(key="lcx", layer="coronary", vessel="lcx", fma="FMA3895", label="Circumflex branch (LCX)", files=fj_sort(lcx)),
    dict(key="rca", layer="coronary", vessel="rca", fma="FMA50039", label="Right coronary artery (RCA)", files=fj_sort(rca)),
    dict(key="lm", layer="coronary", vessel="lm", fma="FMA3855", label="Left main coronary artery", files=fj_sort(lm)),
    dict(key="cardiac_veins", layer="cardiac_vein", fma="FMA12846", label="Cardiac veins", files=fj_sort(veins)),
]

thorax_parts = [
    dict(key="skin", layer="skin", fma="FMA7163", label="Skin", files=fj_sort(concept("FMA7163"))),
    dict(key="ribs", layer="bone", fma="FMA7574", label="Ribs", files=fj_sort(concept("FMA7574"))),
    dict(key="costal_cartilage", layer="cartilage", fma="FMA7591", label="Costal cartilages",
         files=fj_sort(concept("FMA7591"))),
    dict(key="sternum", layer="bone", fma="FMA7485", label="Sternum", files=fj_sort(concept("FMA7485"))),
    dict(key="thoracic_spine", layer="bone", fma="FMA9139", label="Thoracic vertebrae",
         files=fj_sort(concept("FMA9139"))),
    dict(key="clavicles", layer="bone", fma="FMA13321", label="Clavicles", files=fj_sort(concept("FMA13321"))),
    dict(key="scapulae", layer="bone", fma="FMA13394", label="Scapulae", files=fj_sort(concept("FMA13394"))),
    dict(key="diaphragm", layer="diaphragm", fma="FMA13295", label="Diaphragm", files=fj_sort(concept("FMA13295"))),
    dict(key="trachea", layer="airway", fma="FMA7394", label="Trachea and left main bronchus",
         files=["FJ2541", "FJ2450"]),
]

elements = {}
for part in heart_parts + thorax_parts:
    for fj in part["files"]:
        fma, name = specific(fj)
        elements[fj] = dict(fma=fma, name=name)

out = dict(
    source=dict(
        archive=BASE + "isa_BP3D_4.0_obj_99.zip",
        credit="BodyParts3D, (c) The Database Center for Life Science licensed under CC Attribution 4.0 International",
        licence="https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html",
        coordinates="millimetres, patient LPS-like frame; glTF = (x, z, -y) * 0.001",
    ),
    groups=dict(
        heart=dict(output="bp3d_heart.glb", parts=heart_parts),
        thorax=dict(output="bp3d_thorax.glb", parts=thorax_parts),
    ),
    elements=dict(sorted(elements.items(), key=lambda kv: fj_sort([kv[0]]) and int("".join(c for c in kv[0][2:] if c.isdigit())))),
)
(HERE / "parts.json").write_text(json.dumps(out, indent=1) + "\n")
counts = {p["key"]: len(p["files"]) for p in heart_parts + thorax_parts}
print("parts.json:", counts, "elements:", len(elements))
