import argparse
from functools import lru_cache
import ipaddress
from pathlib import Path
import string
from urllib.parse import urlsplit

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tldextract

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lightgbm.joblib"
extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def normalized_host(url):
    if not url or any(char.isspace() or ord(char) < 32 for char in url):
        raise ValueError("URL chứa khoảng trắng hoặc ký tự điều khiển.")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Nhập URL đầy đủ bắt đầu bằng http:// hoặc https://.")
    _ = parts.port
    host = parts.hostname.rstrip(".").lower()
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        ascii_host = host.encode("idna").decode("ascii")
        labels = ascii_host.split(".")
        if len(ascii_host) > 253 or any(
            not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
            or any(ch not in string.ascii_lowercase + string.digits + "-" for ch in label)
            for label in labels
        ):
            raise ValueError("Tên máy chủ không hợp lệ.")
        return ascii_host


keywords = ("login", "verify", "account", "secure", "update", "password")
punctuation = set(string.punctuation)
numeric_columns = ["url_length", "host_length", "digit_count", "punctuation_count",
                   "dot_count", "hyphen_count", "slash_count", "query_marker_count",
                   "subdomain_count", "is_ip", "keyword_count"]

def host_features(host):
    try:
        ipaddress.ip_address(host)
        return 0, 1
    except ValueError:
        subdomain = extract(host).subdomain
        return (len(subdomain.split(".")) if subdomain else 0), 0

host_metrics = lru_cache(maxsize=250_000)(host_features)

def lexical_features(frame):
    rows = []
    for url, host in zip(frame["URL"], frame["host"], strict=True):
        subdomains, is_ip = host_metrics(host)
        lower_url = url.lower()
        rows.append((len(url), len(host), sum(ch.isdigit() for ch in url),
                     sum(ch in punctuation for ch in url), url.count("."), url.count("-"),
                     url.count("/"), url.count("?"), subdomains, is_ip,
                     sum(keyword in lower_url for keyword in keywords)))
    return pd.DataFrame(rows, index=frame.index, columns=numeric_columns, dtype=np.float32)

def combine(numeric, ngrams):
    return sparse.hstack([sparse.csr_matrix(numeric), ngrams], format="csr")

def ngram_text(frame):
    return frame["URL"]

def fit_features(frame, lexical):
    scaler = StandardScaler()
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                           min_df=3, max_features=20_000, dtype=np.float32)
    matrix = combine(scaler.fit_transform(lexical), tfidf.fit_transform(ngram_text(frame)))
    assert int(scaler.n_samples_seen_) == len(frame)
    return matrix, scaler, tfidf

def transform_features(frame, lexical, scaler, tfidf):
    return combine(scaler.transform(lexical), tfidf.transform(ngram_text(frame)))



def train_and_save():
    raw = pd.read_csv(ROOT / "data/PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"],
                      keep_default_na=False, dtype={"URL": "string"})
    split = pd.read_csv(ROOT / "data/splits.csv", index_col="source_row")
    train = raw.loc[split.index[split["split"].eq("train")]].copy()
    train["host"] = train["URL"].map(normalized_host)
    matrix, scaler, tfidf = fit_features(train, lexical_features(train))
    model = LGBMClassifier(objective="binary", n_estimators=90, colsample_bytree=0.40,
                           max_bin=31, random_state=42, n_jobs=6, verbosity=-1,
                           force_col_wise=True, num_leaves=47, min_child_samples=12,
                           learning_rate=0.12)
    model.fit(matrix, train["label"].eq(0).astype(int))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".tmp")
    joblib.dump((model, scaler, tfidf), temporary_path, compress=3)
    temporary_path.replace(MODEL_PATH)
    print(f"Đã lưu mô hình từ {len(train):,} URL huấn luyện.")


def predict_url(url):
    url = url.strip()
    host = normalized_host(url)
    model, scaler, tfidf = joblib.load(MODEL_PATH)
    frame = pd.DataFrame({"URL": [url], "host": [host]})
    matrix = transform_features(frame, lexical_features(frame), scaler, tfidf)
    score = float(model.predict_proba(matrix)[0, 1])
    return score, score > 0.5


def main():
    parser = argparse.ArgumentParser(description="Minh họa phát hiện URL lừa đảo tại chỗ.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Tạo lại mô hình từ tập train.")
    mode.add_argument("--url", help="Chuỗi URL cần phân loại.")
    args = parser.parse_args()
    if args.train:
        train_and_save()
        return
    try:
        score, phishing = predict_url(args.url)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Lỗi: {error}\n")
    print("Dự đoán: Có dấu hiệu lừa đảo" if phishing else "Dự đoán: Hợp lệ")
    print(f"Điểm phishing: {score:.4f} · Ngưỡng: 0,5")


if __name__ == "__main__":
    main()
