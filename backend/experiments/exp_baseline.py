"""Baseline: the current production model (XGBoost, fixed params, scale_pos_weight)."""
from harness import evaluate_all, load_xy
from xgboost import XGBClassifier

_, Y, _ = load_xy()


def make_model_for(target):
    pos = Y[target].sum()
    spw = (len(Y[target]) - pos) / pos

    def make():
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, n_jobs=2,
        )
    return make


if __name__ == "__main__":
    evaluate_all("baseline_xgb", make_model_for)
