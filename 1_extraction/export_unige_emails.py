import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
from email import policy
import argparse
import ssl
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from getpass import getpass
import re
import json
import hashlib
import mimetypes
from dateutil import parser as date_parser

IMAP_SERVER = "mail.unige.it"
IMAP_SSL_PORT = 993
IMAP_STARTTLS_PORT = 143


def decode_mime_words(s):
    if not s:
        return ""
    decoded_parts = decode_header(s)
    parts = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            try:
                parts.append(part.decode(enc or "utf-8", errors="ignore"))
            except Exception:
                parts.append(part.decode("utf-8", errors="ignore"))
        else:
            parts.append(part)
    return "".join(parts).strip()


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove repeated separators
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove very common quoted reply prefixes
    text = re.sub(r"(?m)^>.*$", "", text)

    # Remove common signature starter
    text = re.split(r"(?im)\n(--\s*\n|cordiali saluti|distinti saluti|best regards|kind regards|regards,|saluti,)", text)[0]

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def html_to_text(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return clean_text(text)


def extract_addresses(header_value):
    if not header_value:
        return ""
    return decode_mime_words(header_value)


def safe_filename(name, default="attachment"):
    name = name or default
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return name[:180] if name else default


def hash_id(value):
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def anonymize_email_addresses(text):
    if not text:
        return ""

    def _replace(match):
        return f"anon_{hash_id(match.group(0))}@redacted.local"

    return re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", _replace, text)


def anonymize_text(text):
    if not text:
        return ""
    return anonymize_email_addresses(text)


def get_text_and_attachments(msg, attachment_dir, email_uid):
    text_plain_parts = []
    text_html_parts = []
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")
            filename = part.get_filename()

            if "attachment" in content_disposition.lower() or filename:
                payload = part.get_payload(decode=True)
                decoded_name = decode_mime_words(filename) if filename else "attachment.bin"
                safe_name = safe_filename(decoded_name)
                attachment_path = attachment_dir / f"{email_uid}_{safe_name}"

                if payload:
                    with open(attachment_path, "wb") as f:
                        f.write(payload)

                attachments.append({
                    "filename": decoded_name,
                    "saved_path": str(attachment_path),
                    "content_type": content_type,
                    "size_bytes": len(payload) if payload else 0,
                })
                continue

            if content_type == "text/plain" and "attachment" not in content_disposition.lower():
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    try:
                        text_plain_parts.append(payload.decode(charset, errors="ignore"))
                    except Exception:
                        text_plain_parts.append(payload.decode("utf-8", errors="ignore"))

            elif content_type == "text/html" and "attachment" not in content_disposition.lower():
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    try:
                        text_html_parts.append(payload.decode(charset, errors="ignore"))
                    except Exception:
                        text_html_parts.append(payload.decode("utf-8", errors="ignore"))
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        if payload:
            try:
                decoded = payload.decode(charset, errors="ignore")
            except Exception:
                decoded = payload.decode("utf-8", errors="ignore")

            if content_type == "text/plain":
                text_plain_parts.append(decoded)
            elif content_type == "text/html":
                text_html_parts.append(decoded)

    body_plain = "\n\n".join([p for p in text_plain_parts if p]).strip()
    body_html = "\n\n".join([p for p in text_html_parts if p]).strip()

    if body_plain:
        body_clean = clean_text(body_plain)
    else:
        body_clean = html_to_text(body_html)

    return body_plain, body_html, body_clean, attachments


def list_mailboxes(mail):
    status, mailboxes = mail.list()
    if status != "OK":
        return []
    names = []
    for m in mailboxes:
        decoded = m.decode(errors="ignore")
        match = re.search(r'"([^"]+)"\s*$', decoded)
        if match:
            names.append(match.group(1))
        else:
            names.append(decoded)
    return names


def select_folder(mail, folder_name):
    for candidate in [folder_name, f'"{folder_name}"']:
        status, _ = mail.select(candidate)
        if status == "OK":
            return True
    return False


def _make_ctx(seclevel: int = 2, check_hostname: bool = True) -> ssl.SSLContext:
    """Build an SSL context with a given OpenSSL security level.

    The root cause of SSLV3_ALERT_HANDSHAKE_FAILURE is that Python's OpenSSL
    defaults to SECLEVEL=2, which filters out all ciphers with keys < 112 bits
    and all MD5/SHA-1 MACs.  mail.unige.it (Zimbra) requires ciphers that are
    excluded at SECLEVEL=2.  Thunderbird uses NSS instead of OpenSSL and has no
    such restriction, which is why it connects fine.

    seclevel=1  allows 80-bit equivalent security (removes the SHA-1 ban)
    seclevel=0  allows everything including RC4/DES (last-resort diagnostic)
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = check_hostname
    ctx.verify_mode = ssl.CERT_REQUIRED if check_hostname else ssl.CERT_NONE
    if seclevel < 2:
        cipher_str = f"DEFAULT@SECLEVEL={seclevel}" if seclevel > 0 else "ALL:@SECLEVEL=0"
        try:
            ctx.set_ciphers(cipher_str)
        except ssl.SSLError as exc:
            print(f"[DEBUG] set_ciphers({cipher_str!r}) failed: {exc}")
    return ctx


def connect_imap(server, timeout=30, no_verify_ssl=False):
    """Try multiple IMAP connection strategies and return the first that works.

    CONFIRMED ROOT CAUSE (tested against mail.unige.it):
      1. The server's certificate is signed by an academic CA (GEANT/GARR/Sectigo)
         that is in the Windows system trust store but NOT in Python's bundled CA
         bundle.  Python ssl therefore rejects the cert, which manifests as
         SSLV3_ALERT_HANDSHAKE_FAILURE.
      2. The server also requires ciphers filtered by OpenSSL SECLEVEL=2.

    FIX: ctx.load_default_certs(ssl.Purpose.SERVER_AUTH) loads the Windows trust
    store.  Combined with SECLEVEL=0 (allows the server's preferred ciphers), this
    is confirmed to work.

    Strategy order:
      1. SSL:993  Windows-store + SECLEVEL=0     <- CONFIRMED WORKING
      2. SSL:993  Windows-store + SECLEVEL=1
      3. SSL:993  SECLEVEL=0 (no Windows store)  <- non-Windows fallback
      4. STARTTLS:143  Windows-store + SECLEVEL=0
      5. STARTTLS:143  SECLEVEL=0
      6. SSL:993  SECLEVEL=0 + no cert verify    <- only with --no-verify-ssl
    """
    failures = []

    def _try_ssl(label, ctx):
        print(f"\n[DEBUG] {label}")
        try:
            mail = imaplib.IMAP4_SSL(server, IMAP_SSL_PORT, ssl_context=ctx)
            print(f"[DEBUG] {label} -- succeeded.")
            return mail
        except Exception as exc:
            failures.append((label, exc))
            print(f"[DEBUG] {label} -- failed: {type(exc).__name__}: {exc}")
            return None

    def _try_starttls(label, ctx):
        print(f"\n[DEBUG] {label}")
        try:
            mail = imaplib.IMAP4(server, IMAP_STARTTLS_PORT)
            typ, _ = mail.starttls(ssl_context=ctx)
            if typ != "OK":
                raise RuntimeError(f"STARTTLS response was {typ!r}, expected OK")
            print(f"[DEBUG] {label} -- succeeded.")
            return mail
        except Exception as exc:
            failures.append((label, exc))
            print(f"[DEBUG] {label} -- failed: {type(exc).__name__}: {exc}")
            return None

    def _win_ctx(seclevel=0, check_hostname=True):
        """SSL context that also loads the Windows system certificate store.

        mail.unige.it's cert is signed by an academic CA present in the Windows
        trust store but NOT in Python's bundled CA bundle.  Thunderbird uses the
        Windows store (via NSS on Windows); Python does not by default.
        load_default_certs(SERVER_AUTH) is a no-op on non-Windows.
        """
        ctx = _make_ctx(seclevel=seclevel, check_hostname=check_hostname)
        try:
            ctx.load_default_certs(ssl.Purpose.SERVER_AUTH)
        except AttributeError:
            pass
        return ctx

    # Strategy 1 -- SSL:993 Windows-store + SECLEVEL=0  (CONFIRMED WORKING)
    m = _try_ssl(
        f"Strategy 1 -- SSL:{IMAP_SSL_PORT} Windows-store + SECLEVEL=0 (recommended)",
        _win_ctx(seclevel=0),
    )
    if m: return m, f"SSL:{IMAP_SSL_PORT} Windows-store SECLEVEL=0"

    # Strategy 2 -- SSL:993 Windows-store + SECLEVEL=1
    m = _try_ssl(
        f"Strategy 2 -- SSL:{IMAP_SSL_PORT} Windows-store + SECLEVEL=1",
        _win_ctx(seclevel=1),
    )
    if m: return m, f"SSL:{IMAP_SSL_PORT} Windows-store SECLEVEL=1"

    # Strategy 3 -- SSL:993 SECLEVEL=0 (no Windows store, non-Windows fallback)
    m = _try_ssl(
        f"Strategy 3 -- SSL:{IMAP_SSL_PORT} SECLEVEL=0 (no Windows store)",
        _make_ctx(seclevel=0),
    )
    if m: return m, f"SSL:{IMAP_SSL_PORT} SECLEVEL=0"

    # Strategy 4 -- STARTTLS:143 Windows-store + SECLEVEL=0
    m = _try_starttls(
        f"Strategy 4 -- STARTTLS:{IMAP_STARTTLS_PORT} Windows-store + SECLEVEL=0",
        _win_ctx(seclevel=0),
    )
    if m: return m, f"STARTTLS:{IMAP_STARTTLS_PORT} Windows-store SECLEVEL=0"

    # Strategy 5 -- STARTTLS:143 SECLEVEL=0
    m = _try_starttls(
        f"Strategy 5 -- STARTTLS:{IMAP_STARTTLS_PORT} SECLEVEL=0",
        _make_ctx(seclevel=0),
    )
    if m: return m, f"STARTTLS:{IMAP_STARTTLS_PORT} SECLEVEL=0"

    # Strategy 6 -- SSL:993 SECLEVEL=0 + no cert verification (diagnostic only)
    if no_verify_ssl:
        print("\n[DEBUG] Strategy 6 -- SSL:993 SECLEVEL=0 + certificate verification DISABLED")
        m = _try_ssl(
            f"Strategy 6 -- SSL:{IMAP_SSL_PORT} SECLEVEL=0 no-verify",
            _make_ctx(seclevel=0, check_hostname=False),
        )
        if m: return m, f"SSL:{IMAP_SSL_PORT} SECLEVEL=0 no-verify [INSECURE]"

    # All strategies failed -- detailed diagnostic
    lines = [
        "\n" + "=" * 62,
        "ALL IMAP CONNECTION STRATEGIES FAILED",
        "=" * 62,
        "",
        "Error per strategy:",
    ]
    for label, exc in failures:
        lines.append(f"  [{label}]")
        lines.append(f"    {type(exc).__name__}: {exc}")
    lines += [
        "",
        "Diagnostic steps:",
        "  1. Check port reachability from PowerShell:",
        "       Test-NetConnection -ComputerName mail.unige.it -Port 993",
        "  2. Inspect TLS handshake (if openssl is installed):",
        "       openssl s_client -connect mail.unige.it:993",
        "  3. Verify you are NOT behind a TLS-intercepting proxy (corporate/uni Wi-Fi).",
        "  4. Run with --no-verify-ssl to disable cert verification as a diagnostic step.",
        "=" * 62,
    ]
    raise ConnectionError("\n".join(lines))


def parse_search_criteria(criteria):
    if not criteria:
        return ["ALL"]

    criteria = criteria.strip()
    match = re.match(r'(?i)^SINCE\s+"?([0-3]?\d-[A-Za-z]{3}-\d{4})"?$', criteria)
    if match:
        return ["SINCE", match.group(1)]

    if criteria.upper() in {"ALL", "UNSEEN", "SEEN", "ANSWERED", "FLAGGED", "DRAFT"}:
        return [criteria.upper()]

    return [criteria]


def choose_folder(mail, initial_folder):
    if select_folder(mail, initial_folder):
        return initial_folder

    print(f'\nCould not open folder "{initial_folder}".')
    mailboxes = list_mailboxes(mail)
    if mailboxes:
        print("\nAvailable mailboxes:")
        for idx, mb in enumerate(mailboxes, start=1):
            print(f"  {idx:>2}. {mb}")
    else:
        print("\nCould not list mailboxes.")

    while True:
        retry = input("\nType the exact folder name to use (or press Enter to cancel): ").strip()
        if not retry:
            return None
        if select_folder(mail, retry):
            return retry
        print(f'Folder "{retry}" not accessible. Please try again.')


def parse_date_safe(date_header):
    if not date_header:
        return None
    try:
        return parsedate_to_datetime(date_header).isoformat()
    except Exception:
        try:
            return date_parser.parse(date_header).isoformat()
        except Exception:
            return None


def extract_domain(from_header):
    if not from_header:
        return ""
    match = re.search(r"@([A-Za-z0-9\.-]+\.[A-Za-z]{2,})", from_header)
    return match.group(1).lower() if match else ""


def main():
    parser = argparse.ArgumentParser(description="Export UniGe emails via IMAP to JSONL and CSV")
    parser.add_argument("--test-connection", action="store_true", help="Connect, login, list folders, then exit")
    parser.add_argument("--folder", default="INBOX", help="Mailbox folder name (default: INBOX)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of emails to export (0 = no limit)")
    parser.add_argument("--search-criteria", default="ALL", help='IMAP search criteria (e.g. ALL, UNSEEN, SINCE "01-Jan-2025")')
    parser.add_argument("--server", default=IMAP_SERVER, help=f"IMAP server hostname (default: {IMAP_SERVER})")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable TLS certificate verification as a last-resort strategy"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "raw"
    attachment_dir = output_dir / "attachments"
    output_dir.mkdir(exist_ok=True)
    attachment_dir.mkdir(exist_ok=True)

    email_address = input("UniGe email address: ").strip()
    password = getpass("Password: ")
    folder_name = args.folder
    if not folder_name:
        folder_name = input("Folder to export [INBOX]: ").strip() or "INBOX"

    print(f"\nConnecting to {args.server} ...")
    if args.no_verify_ssl:
        print("[WARN] --no-verify-ssl is active: TLS certificate verification is DISABLED (diagnostic mode only).")
    try:
        mail, method = connect_imap(args.server, no_verify_ssl=args.no_verify_ssl)
    except Exception as exc:
        print("\nConnection failed before login.")
        print(str(exc))
        return

    print(f"\nConnected using {method}.")

    try:
        mail.login(email_address, password)
    except imaplib.IMAP4.error as e:
        print("\nLogin failed.")
        print("Possible causes:")
        print("- wrong email or password")
        print("- mailbox is not active, only forwarding is active")
        print("- temporary access/security restriction")
        print(f"\nError: {e}")
        try:
            mail.logout()
        except Exception:
            pass
        return
    except Exception as e:
        print("\nUnexpected error during login.")
        print(f"Error: {type(e).__name__}: {e}")
        try:
            mail.logout()
        except Exception:
            pass
        return

    print("\nConnected successfully.")

    if args.test_connection:
        print("\nConnection test mode enabled.")
        print("\nAvailable mailboxes:")
        for mb in list_mailboxes(mail):
            print(" ", mb)
        mail.logout()
        print("\nConnection test completed successfully.")
        return

    selected_folder = choose_folder(mail, folder_name)
    if not selected_folder:
        print("\nNo folder selected. Exiting.")
        mail.logout()
        return

    criteria_parts = parse_search_criteria(args.search_criteria)
    print(f"\nSearching messages with criteria: {criteria_parts}")
    status, data = mail.search(None, *criteria_parts)
    if status != "OK":
        print("Could not search messages with the provided criteria.")
        print("Try simpler criteria such as ALL or UNSEEN.")
        mail.logout()
        return

    message_ids = data[0].split()
    if args.limit and args.limit > 0:
        message_ids = message_ids[: args.limit]

    print(f"\nFound {len(message_ids)} messages in {selected_folder}.")

    rows = []
    jsonl_path = output_dir / "emails_export.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for msg_id in tqdm(message_ids, desc="Exporting emails"):
            try:
                status, msg_data = mail.fetch(msg_id, "(RFC822)")
                if status != "OK" or not msg_data or msg_data[0] is None:
                    print(f"[WARN] Skipping message {msg_id.decode(errors='ignore')}: fetch failed.")
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email, policy=policy.default)

                subject = decode_mime_words(msg.get("Subject", ""))
                from_ = extract_addresses(msg.get("From", ""))
                to_ = extract_addresses(msg.get("To", ""))
                cc_ = extract_addresses(msg.get("Cc", ""))
                date_raw = msg.get("Date", "")
                date_iso = parse_date_safe(date_raw)
                message_id_header = msg.get("Message-ID", "")
                in_reply_to = msg.get("In-Reply-To", "")
                references = msg.get("References", "")

                uid_basis = message_id_header or f"{selected_folder}_{msg_id.decode(errors='ignore')}_{subject}_{date_raw}"
                email_uid = hash_id(uid_basis)

                body_plain, body_html, body_clean, attachments = get_text_and_attachments(
                    msg, attachment_dir, email_uid
                )

                row = {
                    "id": email_uid,
                    "imap_msg_id": msg_id.decode(errors="ignore"),
                    "folder": selected_folder,
                    "message_id_header": message_id_header,
                    "in_reply_to": in_reply_to,
                    "references": references,
                    "date_raw": date_raw,
                    "date_iso": date_iso,
                    "from": from_,
                    "from_domain": extract_domain(from_),
                    "to": to_,
                    "cc": cc_,
                    "subject": subject,
                    "body_plain": body_plain,
                    "body_html": body_html,
                    "body_clean": body_clean,
                    "has_attachment": len(attachments) > 0,
                    "attachment_count": len(attachments),
                    "attachments": attachments,
                    "language": "",
                    "topic_label": "",
                    "priority_label": "",
                }

                rows.append(row)
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"[WARN] Error exporting message {msg_id.decode(errors='ignore')}: {type(exc).__name__}: {exc}")
                continue

    mail.logout()

    df = pd.DataFrame(rows)

    csv_columns = [
        "id",
        "folder",
        "date_iso",
        "from",
        "from_domain",
        "to",
        "cc",
        "subject",
        "body_plain",
        "body_html",
        "body_clean",
        "has_attachment",
        "attachment_count",
        "attachments",
        "language",
        "topic_label",
        "priority_label",
    ]

    if "attachments" in df.columns:
        df["attachments"] = df["attachments"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    df[csv_columns].to_csv(output_dir / "emails_export.csv", index=False, encoding="utf-8-sig")
    df.to_pickle(output_dir / "emails_export_full.pkl")

    print("\nDone.")
    print(f"JSONL saved to: {jsonl_path}")
    print(f"CSV saved to: {output_dir / 'emails_export.csv'}")
    print(f"Full dataframe saved to: {output_dir / 'emails_export_full.pkl'}")
    print(f"Attachments saved under: {attachment_dir}")


if __name__ == "__main__":
    main()