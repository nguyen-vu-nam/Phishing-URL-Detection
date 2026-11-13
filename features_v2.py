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

def representation(url):
    parts = urlsplit(url)
    host = normalized_host(url)
    # Exclude scheme and conventional www prefix from learned text.
    host = host.removeprefix("www.")
    path = "" if parts.path in ("", "/") else parts.path
    query = ("?" + parts.query) if parts.query else ""
    return host + path + query

def lexical_features(frame):
    rows = []
    for url in frame["URL"]:
        host = normalized_host(url).removeprefix("www.")
        text = representation(url)
        subdomains, is_ip = host_metrics(host)
        rows.append((len(text), len(host), sum(ch.isdigit() for ch in text),
                     sum(ch in punctuation for ch in text), text.count("."), text.count("-"),
                     text.count("/"), text.count("?"), subdomains, is_ip,
                     sum(keyword in text.lower() for keyword in keywords)))
    return pd.DataFrame(rows, index=frame.index, columns=numeric_columns, dtype=np.float32)

def combine(numeric, ngrams):
    return sparse.hstack([sparse.csr_matrix(numeric), ngrams], format="csr")

def ngram_text(frame):
    return frame["URL"].map(representation)

def fit_features(frame, lexical):
    scaler = StandardScaler()
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                           min_df=3, max_features=20_000, dtype=np.float32)
    matrix = combine(scaler.fit_transform(lexical), tfidf.fit_transform(ngram_text(frame)))
    assert int(scaler.n_samples_seen_) == len(frame)
    return matrix, scaler, tfidf

def transform_features(frame, lexical, scaler, tfidf):
    return combine(scaler.transform(lexical), tfidf.transform(ngram_text(frame)))



