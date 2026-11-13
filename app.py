from pathlib import Path
from urllib.parse import urlsplit

from demo import predict_url


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "link.txt"
OUTPUT_FILE = ROOT / "ket_qua.txt"

# Chỉ bỏ qua model với đúng trang chủ HTTPS đã cấu hình.
# Không tự động tin subdomain, đường dẫn con, query, fragment hay port lạ.
OFFICIAL_HOMEPAGES = {
    "chatgpt.com": ("ChatGPT", "OpenAI"),
    "translate.google.com": ("Google Dịch", "Google"),
    "github.com": ("GitHub", "GitHub"),
}


def official_homepage(url):
    try:
        parts = urlsplit(url.strip())
        if parts.scheme.lower() != "https":
            return None
        if not parts.hostname or parts.username is not None or parts.password is not None:
            return None
        if parts.port not in (None, 443):
            return None
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            return None

        host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        return OFFICIAL_HOMEPAGES.get(host)
    except (ValueError, UnicodeError):
        return None


def load_links():
    if not INPUT_FILE.exists():
        INPUT_FILE.write_text(
            "# Moi dong mot URL. Xoa dong nay va dan URL can kiem tra.\n"
            "https://example.com\n"
            "https://github.com\n",
            encoding="utf-8",
        )
        print(f"Da tao file mau: {INPUT_FILE}")
        print("Hay them URL vao link.txt roi chay lai app.py.")
        return None

    links = []
    for line_number, raw_line in enumerate(
        INPUT_FILE.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        url = raw_line.strip()
        if not url or url.startswith("#"):
            continue
        links.append((line_number, url))
    return links


def check_one(line_number, url):
    official = official_homepage(url)
    if official:
        service, publisher = official
        return (
            "TRANG CHU TRONG DANH SACH XAC MINH",
            f"Hostname khop chinh xac trang chu HTTPS da cau hinh: {service} ({publisher}).",
            None,
        )

    try:
        score, phishing = predict_url(url)
        score_text = f"{score:.4f}"
        if phishing:
            return (
                "TIN HIEU PHISHING CAO",
                f"LightGBM score={score_text}; khong mo lien ket neu chua xac minh nguon gui.",
                score,
            )
        return (
            "CHUA PHAT HIEN TIN HIEU PHISHING MANH",
            f"LightGBM score={score_text}; diem thap khong dam bao website an toan.",
            score,
        )
    except ValueError as error:
        return "URL KHONG HOP LE", str(error), None
    except FileNotFoundError as error:
        return "THIEU MODEL", str(error), None
    except Exception as error:
        return "LOI KIEM TRA", f"{type(error).__name__}: {error}", None


def main():
    links = load_links()
    if links is None:
        return
    if not links:
        print("link.txt chua co URL nao de kiem tra.")
        return

    output_lines = [
        "KET QUA KIEM TRA URL",
        f"So URL: {len(links)}",
        "=" * 90,
    ]

    print(f"Bat dau kiem tra {len(links)} URL tu link.txt...\n")

    for index, (line_number, url) in enumerate(links, start=1):
        status, detail, score = check_one(line_number, url)
        block = (
            f"[{index}/{len(links)}] Dong {line_number}\n"
            f"URL: {url}\n"
            f"Ket luan: {status}\n"
            f"Chi tiet: {detail}\n"
            + "-" * 90
        )
        print(block)
        output_lines.append(block)

    OUTPUT_FILE.write_text("\n".join(output_lines) + "\n", encoding="utf-8-sig")
    print(f"\nHoan tat. Da luu ket qua vao: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
