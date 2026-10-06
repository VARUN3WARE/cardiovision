"""Adversarial re-run of the claimed features_* winners under fresh experiment names.

usage: [HARNESS_SEED_OFFSET=100] python exp_verify_features_run.py target:config [...]
Saves results as verify_features_<config-without-prefix>__<target>[__seedNNN].
Uses the original CONFIGS factories unchanged (imported, not copied) so this tests their code.
"""
import sys
import warnings

from exp_features_run import CONFIGS
from harness import evaluate, load_xy

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    X, Y, _ = load_xy()
    for item in sys.argv[1:]:
        t, name = item.split(":")
        vname = "verify_" + name
        evaluate(vname, t, CONFIGS[name](t), X, Y)
