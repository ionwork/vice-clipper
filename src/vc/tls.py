"""Trust the OS certificate store. Antivirus HTTPS scanning (Avast, Kaspersky, ESET ...) re-signs every
connection with its own root, which Windows trusts but Python's certifi bundle does not."""
from __future__ import annotations

import os
import ssl
import sys
import tempfile
from pathlib import Path

_done = False


def setup() -> None:
    """In-process: truststore (httpx, aiohttp/aiogram, google-genai, websockets).
    Subprocesses and requests-based libs (streamlink, yt-dlp, chat-downloader): certifi + OS roots as one PEM file."""
    global _done
    if _done:
        return
    _done = True
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:  # pragma: no cover
        pass
    if sys.platform != "win32":
        return
    import certifi
    pems = [Path(certifi.where()).read_text(encoding="utf-8")]
    for store in ("ROOT", "CA"):
        for der, enc, _trust in ssl.enum_certificates(store):
            if enc == "x509_asn":
                pems.append(ssl.DER_cert_to_PEM_cert(der))
    bundle = Path(tempfile.gettempdir()) / "vc-ca-bundle.pem"
    bundle.write_text("\n".join(pems), encoding="utf-8")
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(var, str(bundle))
