"""Compare the original and augmented models on domain-disjoint URL data."""
import hashlib
import json
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from demo import (MODEL_PATH, extract, fit_features, lexical_features,
                  normalized_host, transform_features)

ROOT = Path(__file__).resolve().parent
EXTERNAL_MD5 = "ef52aea834e5057c0f77ff9065322fa4"
SEED = 42
PER_CLASS_LIMIT = 45_000
MAX_RECALL_DROP = 0.02
OFFICIAL_URLS = (
    "https://dash.cloudflare.com/", "https://github.com", "https://translate.google.com/",
    "https://chatgpt.com/", "https://www.example.org", "https://www.example.org/",
)


def valid_host(url):
    try:
        return normalized_host(url)
    except (ValueError, UnicodeError):
        return None


def with_domains(frame):
    frame = frame.copy()
    frame["host"] = frame["URL"].map(valid_host)
    frame = frame.dropna(subset=["host"]).copy()
    groups = {host: extract(host).top_domain_under_public_suffix or host
              for host in frame["host"].unique()}
    frame["group"] = frame["host"].map(groups)
    return frame


def measure(bundle, frame):
    model, scaler, vectorizer = bundle
    matrix = transform_features(frame, lexical_features(frame), scaler, vectorizer)
    scores = model.predict_proba(matrix)[:, 1]
    predictions = scores > 0.5
    tn, fp, fn, tp = confusion_matrix(frame["y"], predictions, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fp / (tn + fp)), "recall": float(tp / (tp + fn)),
        "f1": float(f1_score(frame["y"], predictions)),
        "roc_auc": float(roc_auc_score(frame["y"], scores)),
    }


def audit(bundle, official):
    model, scaler, vectorizer = bundle
    matrix = transform_features(official, lexical_features(official), scaler, vectorizer)
    return dict(zip(official["URL"], map(float, model.predict_proba(matrix)[:, 1]), strict=True))


def main():
    external_path = ROOT / "data" / "edgephish.csv"
    with external_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != EXTERNAL_MD5:
        raise ValueError("Tệp dữ liệu ngoài không khớp bản balanced_urls.csv của Zenodo.")

    data_dir = ROOT / "data"
    uci = pd.read_csv(data_dir / "PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"],
                      keep_default_na=False)
    uci["y"] = uci["label"].eq(0).astype(int)
    manifest = pd.read_csv(data_dir / "splits.csv", index_col="source_row")
    uci = with_domains(uci)
    official = with_domains(pd.DataFrame({"URL": OFFICIAL_URLS}))
    reserved = set(official["group"])
    uci_train = uci.loc[manifest.index[manifest["split"].eq("train")]]
    uci_train = uci_train.loc[~uci_train["group"].isin(reserved)]
    uci_test = uci.loc[manifest.index[manifest["split"].eq("test")]]

    external = pd.read_csv(external_path, usecols=["url", "label"], keep_default_na=False)
    external = external.rename(columns={"url": "URL"})
    external["URL"] = external["URL"].str.strip()
    external["y"] = external["label"].map({"legitimate": 0, "phishing": 1})
    assert external["y"].notna().all(), "Nhãn nguồn ngoài ngoài quy ước dự kiến"
    external = external.loc[external.groupby("URL")["y"].transform("nunique").eq(1)]
    external = with_domains(external.drop_duplicates("URL"))
    external = external.loc[~external["group"].isin(set(uci["group"]) | reserved)].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    external_train_idx, external_test_idx = next(splitter.split(external, groups=external["group"]))
    external_train = external.iloc[external_train_idx]
    external_test = external.iloc[external_test_idx]
    assert set(external_train["group"]).isdisjoint(external_test["group"])
    assert set(external_test["group"]).isdisjoint(uci["group"])
    assert set(external_test["y"]) == {0, 1}
    supplement = pd.concat([group.sample(n=min(len(group), PER_CLASS_LIMIT), random_state=SEED)
                            for _, group in external_train.groupby("y")])
    combined = pd.concat([uci_train, supplement], ignore_index=True)
    assert combined["URL"].is_unique
    assert set(combined["group"]).isdisjoint(reserved)
    assert set(combined["group"]).isdisjoint(uci_test["group"])
    assert set(combined["group"]).isdisjoint(external_test["group"])

    counts = {
        "uci_train": len(uci_train), "external_train": len(supplement),
        "external_train_legitimate": int(supplement["y"].eq(0).sum()),
        "external_train_phishing": int(supplement["y"].eq(1).sum()),
        "external_test_legitimate": int(external_test["y"].eq(0).sum()),
        "external_test_phishing": int(external_test["y"].eq(1).sum()),
        "external_test_domains": int(external_test["group"].nunique()),
        "uci_test": len(uci_test),
    }
    print(json.dumps(counts), flush=True)
    baseline = joblib.load(MODEL_PATH)
    old_external = measure(baseline, external_test)
    old_uci = measure(baseline, uci_test)
    print("Original external:", old_external, flush=True)

    matrix, scaler, vectorizer = fit_features(combined, lexical_features(combined))
    model = LGBMClassifier(objective="binary", n_estimators=90, colsample_bytree=0.40,
                           max_bin=31, random_state=SEED, n_jobs=6, verbosity=-1,
                           force_col_wise=True, num_leaves=47, min_child_samples=12,
                           learning_rate=0.12)
    model.fit(matrix, combined["y"].astype(int))
    candidate = (model, scaler, vectorizer)
    new_external = measure(candidate, external_test)
    new_uci = measure(candidate, uci_test)
    old_audit, new_audit = audit(baseline, official), audit(candidate, official)
    promote = (
        new_external["fpr"] < old_external["fpr"]
        and new_external["recall"] >= old_external["recall"] - MAX_RECALL_DROP
        and new_uci["recall"] >= old_uci["recall"] - MAX_RECALL_DROP
        and sum(score > 0.5 for score in new_audit.values())
        < sum(score > 0.5 for score in old_audit.values())
    )
    results = {
        "counts": counts, "external": {"original": old_external, "candidate": new_external},
        "uci": {"original": old_uci, "candidate": new_uci},
        "official_audit": {"original": old_audit, "candidate": new_audit},
        "max_recall_drop": MAX_RECALL_DROP, "passes_replacement_gate": bool(promote),
    }
    joblib.dump(candidate, ROOT / "models" / "lightgbm_augmented.joblib", compress=3)
    (ROOT / "evaluation_augmented.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
