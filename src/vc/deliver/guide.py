"""The operating guide, one source: Telegram /help and `vc guide [--send]` print exactly this text.
Telegram HTML subset (<b>, <code>), everything else escaped."""
from __future__ import annotations


def usage_text() -> str:
    return "\n".join([
        "<b>vice-clipper – Bedienung</b>",
        "",
        "<b>Starten am PC</b>",
        "<code>uv run vc start</code> – Bot und Pipeline laufen. Ab dann kommen alle 15 min die 2–3 Clips mit "
        "dem höchsten Viralitätswert V (Schätzung, keine Garantie). Alle anderen werden verworfen.",
        "Nur den Bot starten: <code>uv run vc start --standby</code>, dann in Telegram /start.",
        "",
        "<b>Pausieren</b>",
        "/pause – stoppt nur die Lieferung. Aufnahme und Bewertung laufen weiter.",
        "/stop – stoppt die ganze Pipeline. Der Bot bleibt erreichbar.",
        "",
        "<b>Wieder starten</b>",
        "/resume nach /pause · /start nach /stop",
        "",
        "<b>Unterwegs</b>",
        "/status – Stand · /next – sofort den nächsten fertigen Clip",
        "",
        "<b>Komplett beenden am PC</b>",
        "<code>uv run vc stop</code>",
        "",
        "<b>Streamer ändern</b>",
        "In der .env: <code>STREAMERS=twitch:login:de, twitch:login2:en</code> (de = deutsche Texte, en = englische).",
        "Greift ohne Neustart beim nächsten Live-Check (etwa 1 Minute). Prüfen: <code>uv run vc streamers</code>",
        "",
        "Hinweis: /start funktioniert nur, wenn am PC <code>vc start</code> läuft (auch mit --standby).",
        "",
        "Weitere Befehle: /hype 8 · /clip &lt;streamer&gt; · /launch (GTA-6-Modus) · /launch probe · "
        "YouTube-/Twitch-VOD-Link schicken",
    ])


def plain(text: str) -> str:
    """Terminal version of the same text (tags removed, entities decoded)."""
    import html
    import re
    return html.unescape(re.sub(r"</?(b|code)>", "", text))
