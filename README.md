# vice-clipper

Lokale Pipeline, die Twitch- und Kick-Streams automatisch zu fertigen **9:16-Clips mit Untertiteln** schneidet (TikTok, Reels, Shorts). Gebaut für GTA-6-Streams, funktioniert aber mit jedem Stream.

Die Pipeline läuft komplett auf deinem PC. Fertige Clips kommen per Telegram auf dein Handy. Hochladen musst du selbst. Es wird nichts automatisch gepostet und nichts generiert: Das Tool schneidet nur vorhandenes Material, legt Untertitel darauf und mischt den Ton ab.

```
Stream-Aufnahme ─┐
Chat-Analyse ────┼─> Kandidat ─> KI-Bewertung ─> Untertitel (Whisper) ─> Render (ffmpeg) ─> Telegram
Twitch-Clips ────┘   (Ollama lokal oder Gemini)
```

## Was es macht

- **Aufnahme:** Nimmt die eingetragenen Streamer live auf. Es bleiben immer nur die letzten 5 Minuten als Puffer gespeichert.
- **Momente erkennen:** Erkennt starke Momente am Chat (plötzliche Nachrichtenflut, Lachen, Schock, Hype, „clip that“, Emote-Wellen), an Lautstärke-Spitzen und an Clips, die Zuschauer auf Twitch erstellen.
- **KI-Bewertung:** Eine lokale KI (Ollama) bewertet jeden Kandidaten anhand von Bildern, Transkript und Lautstärke und schreibt einen Hook-Text. Statt Ollama geht auch Gemini.
- **Viralitäts-Score V:** Jeder Clip bekommt einen Wert von 0 bis 100. Alle 15 Minuten bekommst du die 2–3 besten Clips. Der Rest wird verworfen.
- **Layouts:**

  | Inhalt | Darstellung |
  |---|---|
  | Gameplay | Facecam oben, Spiel unten (50/50) |
  | Just Chatting / IRL | Bild mittig, oben und unten unscharf |

- **Untertitel:** Wort-für-Wort-Untertitel mit Hervorhebung. Die Sprache (Deutsch oder Englisch) legst du pro Streamer fest.
- **Telegram-Paket:** Mit jedem Clip kommen Titel, Caption und Hashtags, getrennt für TikTok, Instagram und YouTube.
- **Steuerung:** Per Telegram (/start, /pause, /stop, /status …) und über ein lokales Dashboard im Browser.

> **Rechtliches:** Clips von fremden Streams darfst du nur posten, wenn der Streamer das erlaubt, zum Beispiel über eine Clipping-Kampagne oder eine ausdrückliche Freigabe. Halte dich an die Regeln der Plattformen. Die Verantwortung liegt bei dir.

---

## 1. Voraussetzungen

| Was | Wofür | Installation |
|---|---|---|
| **uv** | Python 3.12 und alle Pakete | Windows: `winget install astral-sh.uv` · macOS: `brew install uv` |
| **ffmpeg** | Schneiden, Rendern, Ton | Windows: `winget install Gyan.FFmpeg` · macOS: `brew install ffmpeg` |
| **Ollama** (empfohlen) | Lokale KI-Bewertung, kostenlos | [ollama.com/download](https://ollama.com/download), danach `ollama pull gemma3:12b` |
| **Git** | Repo klonen | Windows: `winget install Git.Git` · macOS: vorinstalliert |

`streamlink`, `yt-dlp` und `faster-whisper` installiert `uv` automatisch.

**Hardware:**
- Am besten eine NVIDIA-Grafikkarte mit **12 GB VRAM oder mehr** (für `gemma3:12b` und schnelles Whisper).
- Mit weniger Grafikspeicher nimmst du ein kleineres Modell, zum Beispiel `ollama pull gemma3:4b`, und trägst es in `config.yaml` unter `judge.ollama_model` ein.
- Ohne Grafikkarte läuft alles auch auf der CPU, nur deutlich langsamer.
- Mac mit Apple Silicon funktioniert ebenfalls.

**Speicherplatz:** etwa 15 GB (KI-Modell, Whisper-Modell und Pakete). Das Whisper-Modell lädt beim ersten Clip automatisch herunter.

## 2. Installation

```bash
git clone https://github.com/<DEIN-NAME>/vice-clipper.git
cd vice-clipper
uv sync
```

Kopiere die Beispiel-Konfiguration nach `.env`:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env
# macOS / Linux
cp .env.example .env
```

## 3. Was du eintragen musst

Alles Geheime steht nur in **`.env`**. Diese Datei wird nie hochgeladen (`.gitignore`).

Am einfachsten geht es mit dem Assistenten. Er fragt alle Schlüssel ab, prüft sie live und schreibt die `.env`:

```bash
uv run vc setup
```

Oder du trägst alles von Hand in `.env` ein:

| Eintrag | Pflicht | Woher |
|---|---|---|
| `TWITCH_CLIENT_ID` | ja | [dev.twitch.tv/console/apps](https://dev.twitch.tv/console/apps): „Anwendung registrieren“, OAuth-Redirect `http://localhost`, Kategorie „Application Integration“. Die **Client-ID** steht danach in der App. |
| `TWITCH_CLIENT_SECRET` | ja | In derselben App auf „Neues Geheimnis“ klicken |
| `TELEGRAM_BOT_TOKEN` | ja | In Telegram **@BotFather** öffnen, `/newbot` schicken, Token kopieren |
| `TELEGRAM_CHAT_ID` | ja | `uv run vc setup` findet sie automatisch: Schick deinem Bot einmal `/start`. Nur Nachrichten aus diesem Chat werden beantwortet. |
| `GEMINI_API_KEY` | nur mit Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Bei `judge.provider: ollama` leer lassen. |
| `TWITCH_USER_OAUTH` | nein | Nur nötig, wenn du Aufnahmen ohne Werbung willst und Twitch Turbo oder ein Abo hast |
| `STREAMERS` | ja | Siehe unten |

### Streamer eintragen

Die Liste der Streamer steht **nur** in `.env`. Mehrere Einträge trennst du mit Komma:

```env
STREAMERS=twitch:login_eins:de, twitch:login_zwei:en, kick:login_drei:en
```

Format: `plattform:login:sprache[:erlaubnis]`
- **plattform:** `twitch` (Standard) oder `kick`
- **login:** der Kanalname aus der URL, zum Beispiel `twitch.tv/login_eins`
- **sprache:** `de` oder `en`. Danach richten sich Hook, Untertitel, Caption und Hashtags.
- **erlaubnis** (optional): `campaign`, `public_ok`, `asked` oder `none`. Clips von Streamern mit `campaign` kommen zuerst.

Änderungen greifen ohne Neustart beim nächsten Live-Check, etwa nach einer Minute. Prüfen kannst du die Liste so:

```bash
uv run vc streamers
```

**Facecam-Position (optional):** Der Cam-Rahmen wird automatisch gesucht. Wenn das nicht passt, ziehst du ihn im Dashboard auf (`http://127.0.0.1:8000`, sobald `vc start` läuft) oder trägst ihn in `streamers.yaml` ein. Dort ist ein Beispiel.

## 4. Prüfen

```bash
uv run vc doctor
```

Der Befehl prüft ffmpeg, streamlink, den Video-Encoder, Schriften, alle Schlüssel, die Streamer-Logins, Telegram und Ollama. Alles sollte ✓ zeigen.

Optional kannst du noch die Tests laufen lassen:

```bash
uv run pytest -q
```

## 5. Starten, pausieren, stoppen

| Aktion | Am PC | In Telegram |
|---|---|---|
| Alles starten | `uv run vc start` | – |
| Nur den Bot starten, Pipeline per Handy | `uv run vc start --standby` | dann `/start` |
| Lieferung pausieren (Aufnahme läuft weiter) | – | `/pause` |
| Lieferung fortsetzen | – | `/resume` |
| Pipeline stoppen (Bot bleibt an) | – | `/stop` |
| Pipeline wieder starten | – | `/start` |
| Stand anzeigen | `uv run vc status` | `/status` |
| Nächsten fertigen Clip sofort schicken | – | `/next` |
| Alles beenden | `uv run vc stop` oder Strg+C | – |

`/start` funktioniert nur, wenn am PC `vc start` läuft (auch mit `--standby`).

Die komplette Anleitung bekommst du mit `uv run vc guide` oder `/help` in Telegram.

**Weitere Befehle in Telegram:**
- `/clip <streamer>`: sofort einen Clip der letzten Sekunden schneiden
- `/hype 8`: Mindest-Hype ändern
- `/add twitch login de`: Streamer hinzufügen
- Einen YouTube- oder Twitch-VOD-Link schicken: Das Video wird nach guten Momenten durchsucht.

Mit `uv run vc --help` siehst du alle Befehle am PC.

### GTA 6 oder Probe-Modus

Standardmäßig läuft der **Probe-Modus**, der auch GTA-V-Material erlaubt, damit du alles vorher testen kannst.

Auf reines GTA 6 umschalten:

```bash
uv run vc launch          # oder /launch in Telegram
uv run vc launch --undo   # zurück in den Probe-Modus
```

## 6. Anpassen (`config.yaml`)

Alle Einstellungen ohne Geheimnisse stehen kommentiert in `config.yaml`. Die wichtigsten:

| Einstellung | Bedeutung |
|---|---|
| `judge.provider` | `ollama` (lokal, kostenlos) oder `gemini` (Cloud, kostet pro Aufruf) |
| `judge.ollama_model` | KI-Modell. Es muss Bilder verstehen können (Vision). |
| `judge.min_hype` | Wie streng die KI aussortiert (1–10) |
| `delivery.window_minutes` | Länge eines Lieferfensters (Standard 15) |
| `delivery.per_window_min` / `per_window_max` | Clips pro Fenster |
| `delivery.min_v` | Mindest-Viralitätswert, darunter wird nie geliefert |
| `render.split_cam_height` | `960` = Facecam 50 % / Gameplay 50 % |
| `render.force_layout` | Leer = automatisch, sonst `split`, `gameplay`, `fullcam` oder `irl` |
| `captions.whisper_model` | `large-v3-turbo` (genau) oder `small` (schnell, für schwache PCs) |
| `captions.highlight_color` | Farbe des hervorgehobenen Worts (Hex) |
| `virality.weights` | Gewichtung der Signale für den Score V |

Die Gewichte des Viralitäts-Scores sind begründete Startwerte. Mit `uv run vc tune` prüfst du sie gegen echte Twitch-Clips und deren Views. `--apply` übernimmt bessere Gewichte nur dann, wenn sie messbar besser sind. Einen viralen Clip garantiert kein Score.

## 7. Häufige Probleme

| Problem | Lösung |
|---|---|
| `ffmpeg nicht gefunden` | ffmpeg installieren und das Terminal neu öffnen |
| Ollama antwortet nicht | Ollama starten und `ollama pull gemma3:12b` ausführen |
| Ollama ist zu langsam oder bricht ab | Kleineres Modell nehmen, `judge.ollama_model: gemma3:4b` |
| Keine Clips kommen an | `uv run vc status`: Ist ein Streamer live? Ist `TELEGRAM_CHAT_ID` richtig? Steht etwas in `data/logs/vc.log`? |
| Der Bot reagiert nicht | `TELEGRAM_CHAT_ID` stimmt nicht. Führe `uv run vc setup` erneut aus und schick dem Bot `/start`. |
| „Keine Streamer in .env“ | `STREAMERS=` in `.env` ausfüllen (siehe Abschnitt 3) |
| Untertitel langsam | `captions.whisper_model: small` |

Alle Daten (Aufnahmen, Clips, Datenbank, Logs) landen im Ordner `data/`. Der Ordner wird nie hochgeladen.

## 8. Aufbau des Codes

```
src/vc/
  sources/   Aufnahme (streamlink), Chat (Twitch IRC, Kick), Twitch-Clips, VOD-Scout
  detect/    Chat-Statistik, Trigger, Viralitäts-Score V, Tuning
  judge/     KI-Bewertung (Ollama oder Gemini), Prompts
  render/    Layouts, Untertitel (faster-whisper -> ASS), ffmpeg
  deliver/   Telegram-Bot, 15-Minuten-Fenster, Texte
  web/       lokales Dashboard
  cli.py     alle Befehle (uv run vc --help)
config.yaml  Einstellungen
.env         Schlüssel und Streamer (nicht im Repo)
```

Status eines Clips: `NEW → JUDGED → CAPTIONED → RENDERED → DELIVERED → POSTED` (oder `REJECTED` / `FAILED`). Alles steht in SQLite, deshalb macht die Pipeline nach einem Absturz dort weiter, wo sie war.

## Lizenz

Code: MIT, siehe [LICENSE](LICENSE). Schriften: Noto Sans unter der SIL Open Font License 1.1, siehe [assets/fonts/OFL.txt](assets/fonts/OFL.txt).
