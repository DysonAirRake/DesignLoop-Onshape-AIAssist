import os
import re
import json
import base64
import secrets
import datetime
import hashlib
import csv
import uuid
import logging
import math
import traceback
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from functools import wraps
from urllib.parse import urlencode

import requests
from flask import (
    Flask, session, redirect, request,
    url_for, jsonify, abort, send_from_directory
)
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# iframe session cookie settings
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,         # MUST serve over HTTPS (ngrok, etc.)
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_NAME="designloop_session",
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=8),
)

# Paths
DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

SESSION_LOG     = DATA_DIR / "session_events.jsonl"
EMBEDDINGS_LOG  = DATA_DIR / "embeddings.jsonl"
IMAGES_DIR      = DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

VIEW_CACHE_DIR  = DATA_DIR / "view_cache"
VIEW_CACHE_DIR.mkdir(exist_ok=True)
VIEW_CACHE_INDEX = VIEW_CACHE_DIR / "index.json"

ITERATION_STORE = DATA_DIR / "iteration_store.json"
REPORTS_DIR     = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOGS_DIR / "server.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("designloop")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client  = OpenAI(api_key=OPENAI_API_KEY)

_key_preview = (OPENAI_API_KEY[:8] + "..." + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else "NOT SET"
log.info(f"OpenAI key in use: {_key_preview}")

PROMPTS_DIR = Path(__file__).parent / "prompts"

def _load_prompt(filename: str) -> str:
    """Read a prompt text file, stripping trailing whitespace."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()

def _clean_text(text: str) -> str:
    """Normalize typographic Unicode to plain ASCII equivalents."""
    return (
        text
        .replace('\u2018', "'").replace('\u2019', "'")   # ' '
        .replace('\u201a', "'")                           # ‚
        .replace('\u201c', '"').replace('\u201d', '"')   # " "
        .replace('\u201e', '"')                           # „
        .replace('\u2013', '-').replace('\u2014', '--')  # – —
        .replace('\u2026', '...')                        # …
        .replace('\u00b4', "'").replace('\u0060', "'")   # ´ `
        .encode('ascii', errors='replace').decode('ascii')
    )

try:
    import httpx._models as _hx_models
    if hasattr(_hx_models, '_normalize_header_value'):
        _orig_normalize_header = _hx_models._normalize_header_value

        def _safe_normalize_header(value, encoding=None):
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            try:
                return value.encode(encoding or 'utf-8')
            except (UnicodeEncodeError, AttributeError):
                return value.encode('utf-8', errors='replace')

        _hx_models._normalize_header_value = _safe_normalize_header
        log.info("httpx header encoding patched → UTF-8")
except Exception:
    pass

ONSHAPE_CLIENT_ID     = os.getenv("ONSHAPE_CLIENT_ID")
ONSHAPE_CLIENT_SECRET = os.getenv("ONSHAPE_CLIENT_SECRET")
ONSHAPE_API_KEY       = os.getenv("ONSHAPE_API_KEY")
ONSHAPE_API_SECRET    = os.getenv("ONSHAPE_API_SECRET")

# View cache
def _view_cache_key(did: str, wid: str, eid: str) -> str:
    return hashlib.sha1(f"{did}|{wid}|{eid}".encode()).hexdigest()[:16]

def _read_cache_index() -> dict:
    with _cache_index_lock:
        try:
            return json.loads(VIEW_CACHE_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return {}

def _write_cache_index(index: dict):
    with _cache_index_lock:
        try:
            VIEW_CACHE_INDEX.write_text(
                json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.warning(f"Could not write cache index: {exc}")

def _load_view_cache(did: str, wid: str, eid: str) -> dict | None:
    """Return cached views dict, or None if not in cache."""
    key = _view_cache_key(did, wid, eid)
    p   = VIEW_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        data  = json.loads(p.read_text(encoding="utf-8"))
        views = data.get("views")
        if views:
            log.info(f"View cache hit: {key} — {len(views)} views")
            return views
    except Exception as exc:
        log.warning(f"View cache read error ({key}): {exc}")
    return None

def _save_view_cache(did: str, wid: str, eid: str, views: dict,
                     doc_name: str = "", doc_modified_at: str = "",
                     elem_name: str = ""):
    """Persist views to disk and update the metadata index."""
    key = _view_cache_key(did, wid, eid)
    p   = VIEW_CACHE_DIR / f"{key}.json"
    ts  = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        p.write_text(
            json.dumps({
                "document_id":    did,
                "workspace_id":   wid,
                "element_id":     eid,
                "element_name":   elem_name,
                "document_name":  doc_name,
                "cached_at":      ts,
                "doc_modified_at": doc_modified_at,
                "view_count":     len(views),
                "views":          views,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        with _cache_index_lock:
            idx = _read_cache_index()
            idx[key] = {
                "document_id":    did,
                "workspace_id":   wid,
                "element_id":     eid,
                "element_name":   elem_name or "Assembly",
                "document_name":  doc_name or "Unknown",
                "cached_at":      ts,
                "doc_modified_at": doc_modified_at,
                "view_count":     len(views),
                "size_kb":        round(p.stat().st_size / 1024),
            }
            _write_cache_index(idx)
        log.info(f"View cache saved: {key} ({len(views)} views, '{doc_name}' / '{elem_name}')")
    except Exception as exc:
        log.warning(f"View cache write error ({key}): {exc}")

def _delete_view_cache(did: str, wid: str, eid: str):
    key = _view_cache_key(did, wid, eid)
    p   = VIEW_CACHE_DIR / f"{key}.json"
    if p.exists():
        p.unlink()
    with _cache_index_lock:
        idx = _read_cache_index()
        idx.pop(key, None)
        _write_cache_index(idx)
    log.info(f"View cache deleted: {key}")

def _has_view_cache(did: str, wid: str, eid: str) -> bool:
    """Lightweight existence check — does NOT load image data."""
    key = _view_cache_key(did, wid, eid)
    return (VIEW_CACHE_DIR / f"{key}.json").exists()

def _clear_all_view_cache():
    for p in VIEW_CACHE_DIR.glob("*.json"):
        p.unlink()
    log.info("View cache cleared (all entries deleted)")

# Iteration tracking
def _read_iteration_store() -> dict:
    with _iteration_store_lock:
        try:
            return json.loads(ITERATION_STORE.read_text(encoding="utf-8"))
        except Exception:
            return {}

def _write_iteration_store(store: dict):
    with _iteration_store_lock:
        try:
            ITERATION_STORE.write_text(
                json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.warning(f"Could not write iteration store: {exc}")

def _get_doc_iteration(did: str) -> int:
    """Return the current (in-progress) iteration number for a document (1-based)."""
    if not did:
        return 1
    return _read_iteration_store().get(did, {}).get("current_iteration", 1)

def _finish_doc_iteration(did: str) -> int:
    """Mark the current iteration complete and advance the counter. Returns the new iteration number."""
    with _iteration_store_lock:
        store = _read_iteration_store()
        entry = store.setdefault(did, {"current_iteration": 1, "completed": []})
        current = entry["current_iteration"]

        ctx = entry.pop("current_context", {})
        entry["completed"].append({
            "iteration":    current,
            "description":  ctx.get("description", ""),
            "analysis":     ctx.get("analysis", ""),
            "directions":   ctx.get("directions", []),
            "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        })
        entry["current_iteration"] = current + 1
        entry["input_logged"] = False
        _write_iteration_store(store)
    log.info(f"Iteration {current} completed for doc {did} → now on iteration {current + 1}")
    return current + 1


def _save_iteration_context(did: str, description: str = None,
                             analysis: str = None, direction: str = None):
    """Persist the in-progress iteration's context so it survives across requests."""
    if not did or did.startswith("{") or did.endswith("}"):
        return
    try:
        with _iteration_store_lock:
            store = _read_iteration_store()
            entry = store.setdefault(did, {"current_iteration": 1, "completed": []})
            ctx   = entry.setdefault("current_context", {})
            if description is not None:
                ctx["description"] = description
            if analysis is not None:
                ctx["analysis"] = analysis
            if direction is not None:
                ctx.setdefault("directions", []).append(direction)
            _write_iteration_store(store)
    except Exception as exc:
        log.warning(f"_save_iteration_context({did}): {exc}")


def _get_design_history_string(did: str) -> str:
    """Return a formatted plain-text block of completed iterations for use in AI prompts."""
    if not did:
        return ""
    completed = _read_iteration_store().get(did, {}).get("completed", [])
    if not completed:
        return ""

    bar = "─" * 56
    lines = [
        "DESIGN SESSION HISTORY — completed iterations",
        bar,
    ]
    for rec in completed:
        iter_n = rec.get("iteration", "?")
        desc   = (rec.get("description") or "").strip()
        dirs   = [d for d in (rec.get("directions") or []) if d.strip()]
        lines.append(f"Iteration {iter_n}")
        if desc:
            lines.append(f"  Base design: \"{desc[:300]}\"")
        if dirs:
            lines.append("  Modifications the user already visualised / implemented:")
            for d in dirs[:10]:
                lines.append(f"    • {d[:160]}")
    lines += [
        bar,
        "Do NOT re-suggest any modification already listed above.",
        "Build upon the CURRENT design state, incorporating all prior changes.",
    ]
    return "\n".join(lines)


def _iteration_has_input(did: str) -> bool:
    """Return True if the current iteration has had meaningful user input logged."""
    if not did:
        return False
    return bool(_read_iteration_store().get(did, {}).get("input_logged", False))


def _fetch_doc_modified_at(did: str) -> str:
    """Return the modifiedAt timestamp for a document from the Onshape API, or ""."""
    if not (ONSHAPE_API_KEY and ONSHAPE_API_SECRET):
        return ""
    try:
        resp = requests.get(
            f"{BASE_URL}/api/v9/documents/{did}",
            auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
            timeout=8,
        )
        if resp.ok:
            return resp.json().get("modifiedAt", "")
    except Exception as exc:
        log.warning(f"_fetch_doc_modified_at({did}): {exc}")
    return ""


def _get_or_create_report_folder(did: str, doc_name: str) -> str:
    """Return (creating if needed) the report folder name for this document session."""
    with _iteration_store_lock:
        store = _read_iteration_store()
        entry = store.get(did, {})
        if "report_folder" in entry:
            folder = entry["report_folder"]
        else:
            ts    = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M")
            safe  = re.sub(r"[^\w\-]", "_", doc_name or did[:12])[:30].strip("_")
            folder = f"{ts}_{safe}" if safe else ts
            entry  = store.setdefault(did, {"current_iteration": 1, "completed": []})
            entry["report_folder"] = folder
            _write_iteration_store(store)
    (REPORTS_DIR / folder).mkdir(parents=True, exist_ok=True)
    return folder

CONTEXT_FILE = DATA_DIR / "last_context.json"

def _load_persisted_context() -> dict:
    try:
        return json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_persisted_context(did: str, wid: str, eid: str, wvm: str = "w"):
    try:
        CONTEXT_FILE.write_text(
            json.dumps({"document_id": did, "workspace_id": wid,
                        "element_id": eid, "wvm": wvm}),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning(f"Could not persist context: {exc}")

ONSHAPE_AUTH_URL  = "https://oauth.onshape.com/oauth/authorize"
ONSHAPE_TOKEN_URL = "https://oauth.onshape.com/oauth/token"
BASE_URL          = "https://cad.onshape.com"

REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:5000/auth/callback")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Response headers
@app.after_request
def set_iframe_headers(response):
    """Set CORS and framing headers required for Onshape iframe embedding."""
    origin = request.headers.get("Origin", "")

    allowed_origins = [
        "https://cad.onshape.com",
        "https://oauth.onshape.com",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ]
    if origin in allowed_origins or origin.endswith(".ngrok-free.app") or origin.endswith(".ngrok-free.dev"):
        response.headers["Access-Control-Allow-Origin"]      = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"]     = "Content-Type"
        response.headers["Access-Control-Allow-Methods"]     = "GET, POST, OPTIONS"

    response.headers.pop("X-Frame-Options", None)

    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://cad.onshape.com https://*.onshape.com;"
    )

    return response


# Server-side cache (keyed by document_id)
_server_cache: dict[str, dict] = {}

# thread-safety locks
_iteration_store_lock = threading.RLock()
_cache_index_lock     = threading.RLock()
_log_lock             = threading.Lock()


def _doc_key(did: str) -> str:
    return did or "_no_doc"


def _entry(did: str) -> dict:
    k = _doc_key(did)
    if k not in _server_cache:
        _server_cache[k] = {}
    return _server_cache[k]


def cache_views(did: str, views: dict):
    _entry(did)["_views"] = views


def get_cached_views(did: str) -> dict | None:
    return _server_cache.get(_doc_key(did), {}).get("_views")


def cache_analysis(did: str, user_desc: str, ai_text: str):
    e = _entry(did)
    e["user_description"] = user_desc
    e["ai_analysis"]      = ai_text


def get_cached_analysis(did: str) -> tuple[str, str]:
    e = _server_cache.get(_doc_key(did), {})
    return e.get("user_description", ""), e.get("ai_analysis", "")


def cache_visual_description(did: str, desc: str):
    _entry(did)["visual_description"] = desc


def get_cached_visual_description(did: str) -> str:
    return _server_cache.get(_doc_key(did), {}).get("visual_description", "")


def cache_last_concepts(did: str, concept: dict):
    """Accumulate generated concepts within the current step (last 3 kept)."""
    e   = _entry(did)
    lst = e.setdefault("_last_concepts", [])
    lst.append(concept)
    e["_last_concepts"] = lst[-3:]


def get_last_concepts(did: str) -> list:
    return _server_cache.get(_doc_key(did), {}).get("_last_concepts", [])


def clear_last_concepts(did: str):
    _entry(did)["_last_concepts"] = []


# Research logging pipeline
def _session_id() -> str:
    """Persistent per-browser session identifier for grouping events."""
    if "research_session_id" not in session:
        session["research_session_id"] = str(uuid.uuid4())
    return session["research_session_id"]


def log_event(event_type: str, payload: dict, did: str = "") -> str:
    """Write a structured JSON event to session_events.jsonl. Returns the event_id."""
    event_id = str(uuid.uuid4())
    if not did:
        did = session.get("document_id", "")
    record = {
        "event_id":        event_id,
        "session_id":      _session_id(),
        "ts":              datetime.datetime.now(datetime.UTC).isoformat(),
        "event":           event_type,
        "document_id":     did,
        "iteration_number": _get_doc_iteration(did),
        **payload,
    }
    try:
        with _log_lock:
            with open(SESSION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.error(f"log_event failed: {exc}")

    if did and event_type in ("ANALYSIS", "NEXT_STEP_GENERATED"):
        try:
            with _iteration_store_lock:
                store = _read_iteration_store()
                store.setdefault(did, {"current_iteration": 1, "completed": []})["input_logged"] = True
                _write_iteration_store(store)
        except Exception as exc:
            log.warning(f"log_event: could not set input_logged for {did}: {exc}")

    return event_id


def log_embedding(event_id: str, field_name: str, text: str, embedding: list[float],
                  document_id: str = "", iteration_number: int = 0,
                  event_type: str = "", session_id: str = ""):
    """Write an embedding vector linked to a specific event and field to embeddings.jsonl."""
    record = {
        "event_id":        event_id,
        "session_id":      session_id,
        "document_id":     document_id,
        "iteration_number": iteration_number,
        "event_type":      event_type,
        "field":           field_name,
        "text":            text[:500],
        "vector_dim":      len(embedding),
        "vector":          embedding,
        "ts":              datetime.datetime.now(datetime.UTC).isoformat(),
    }
    try:
        with _log_lock:
            with open(EMBEDDINGS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.error(f"log_embedding failed: {exc}")


def compute_embedding(text: str) -> list[float]:
    """Get an embedding vector from OpenAI's text-embedding-3-small."""
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
    except Exception as exc:
        log.error(f"Embedding API error: {exc}")
        return []


def save_image_artifact(b64_data: str, label: str, event_id: str) -> str:
    """Save a base64-encoded image to data/images/. Returns the filename."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label[:30])
    filename = f"{ts}_{safe_label}_{event_id[:8]}.png"
    filepath = IMAGES_DIR / filename
    try:
        img_bytes = base64.b64decode(b64_data)
        filepath.write_bytes(img_bytes)
        log.info(f"Saved image artifact: {filename} ({len(img_bytes)} bytes)")
    except Exception as exc:
        log.error(f"Image save failed: {exc}")
        filename = ""
    return filename


def export_session_csv():
    """Export the JSONL event log to a CSV. Called on-demand via /export-csv."""
    csv_path = DATA_DIR / "session_export.csv"
    events = []
    try:
        with open(SESSION_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except FileNotFoundError:
        return None

    if not events:
        return None

    # Collect all unique keys across all events
    all_keys = set()
    for ev in events:
        all_keys.update(ev.keys())
    all_keys = sorted(all_keys)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            writer.writerow(ev)

    return csv_path


def _compute_similarity_html(did: str) -> str:
    """Compute pairwise cosine similarities across iterations and return a self-contained HTML section."""
    def _cos(a: list, b: list) -> float:
        dot  = sum(x * y for x, y in zip(a, b, strict=False))
        na   = math.sqrt(sum(x * x for x in a))
        nb   = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def _centroid(vecs: list[list]) -> list:
        if not vecs:
            return []
        dim = len(vecs[0])
        c   = [0.0] * dim
        for v in vecs:
            for idx_c, x in enumerate(v):
                c[idx_c] += x
        n = len(vecs)
        return [x / n for x in c]

    if not EMBEDDINGS_LOG.exists():
        return ""

    recs: list[dict] = []
    try:
        with open(EMBEDDINGS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("document_id") == did and r.get("vector"):
                    recs.append(r)
    except Exception:
        return ""

    if not recs:
        return ""

    by_iter_field: dict[tuple, list] = defaultdict(list)
    for r in recs:
        it    = int(r.get("iteration_number") or 0)
        field = r.get("field", "")
        by_iter_field[(it, field)].append(r["vector"])

    iterations = sorted(set(k[0] for k in by_iter_field.keys()))
    if len(iterations) < 1:
        return ""

    fields_cfg = [
        ("user_description", "Design Description",     "#a8e847"),
        ("ai_analysis",      "AI Analysis",            "#47b8e8"),
        ("user_direction",   "Direction / Modification","#e8a847"),
    ]

    rows_html = ""
    for field, label, color in fields_cfg:
        vecs = {it: _centroid(by_iter_field.get((it, field), [])) for it in iterations}
        avail = [it for it in iterations if vecs[it]]
        for a_idx in range(len(avail)):
            for b_idx in range(a_idx + 1, len(avail)):
                it_a, it_b = avail[a_idx], avail[b_idx]
                sim  = _cos(vecs[it_a], vecs[it_b])
                dist = 1 - sim
                hue  = int(sim * 120)
                sim_color  = f"hsl({hue},65%,55%)"
                dist_color = f"hsl({int(dist*120)},65%,55%)"
                rows_html += (
                    f'<tr>'
                    f'<td><span style="color:{color};font-weight:600">{label}</span></td>'
                    f'<td>Iter {it_a} vs {it_b}</td>'
                    f'<td style="color:{sim_color};font-weight:700;font-family:monospace">{sim:.4f}</td>'
                    f'<td style="color:{dist_color};font-weight:700;font-family:monospace">{dist:.4f}</td>'
                    f'</tr>'
                )

    if not rows_html:
        return ""

    # Direction novelty (sequential: each direction vs all prior ones)
    dir_recs = sorted(
        [r for r in recs if r.get("field") == "user_direction"],
        key=lambda r: (r.get("iteration_number", 0), r.get("ts", "")),
    )
    novelty_rows = ""
    seen_vecs: list[list] = []
    for dr in dir_recs:
        vec  = dr["vector"]
        text = dr.get("text", "")[:80]
        it_n = dr.get("iteration_number", "?")
        if seen_vecs:
            max_sim  = max(_cos(vec, prev) for prev in seen_vecs)
            novelty  = 1 - max_sim
        else:
            max_sim  = 0.0
            novelty  = 1.0
        hue_n = int(novelty * 120)
        novelty_rows += (
            f'<tr>'
            f'<td>Iter {it_n}</td>'
            f'<td style="color:var(--dim)">{text}</td>'
            f'<td style="font-family:monospace">{max_sim:.4f}</td>'
            f'<td style="color:hsl({hue_n},65%,55%);font-weight:700;font-family:monospace">{novelty:.4f}</td>'
            f'</tr>'
        )
        seen_vecs.append(vec)

    novelty_section = ""
    if novelty_rows:
        novelty_section = f"""
      <h4 style="font-size:11px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.06em;color:var(--accent2);margin:18px 0 8px">
        Direction Novelty — each direction vs all prior directions</h4>
      <p style="font-size:10px;color:var(--muted);margin-bottom:8px">
        novelty_score = 1 − max similarity to any prior direction.
        1.0 = completely new territory, 0.0 = exact repeat.</p>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr>
          <th style="{_th_css()}">Iter</th>
          <th style="{_th_css()}">Direction</th>
          <th style="{_th_css()}">Max Sim to Prior</th>
          <th style="{_th_css()}">Novelty Score</th>
        </tr></thead>
        <tbody>{novelty_rows}</tbody>
      </table>"""

    return f"""
  <section style="max-width:860px;margin:32px auto;padding:0 20px">
    <div style="border:1px solid var(--border);border-radius:8px;overflow:hidden">
      <div style="background:var(--surface);padding:12px 18px;border-bottom:1px solid var(--border)">
        <h3 style="font-size:13px;font-weight:700;color:var(--accent);
                    text-transform:uppercase;letter-spacing:0.06em;margin:0">
          Semantic Similarity Analysis</h3>
        <p style="font-size:10px;color:var(--muted);margin-top:4px">
          Cosine similarity between embedding centroids across iterations.
          Similarity → 1.0 = same semantic meaning. Semantic distance = 1 − sim.
          Low sim = substantial design concept shift between iterations.</p>
      </div>
      <div style="padding:14px 18px">
        <h4 style="font-size:11px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.06em;color:var(--accent2);margin:0 0 8px">
          Cross-Iteration Embedding Similarity</h4>
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead><tr>
            <th style="{_th_css()}">Field</th>
            <th style="{_th_css()}">Comparison</th>
            <th style="{_th_css()}">Cosine Similarity ↑</th>
            <th style="{_th_css()}">Semantic Distance ↓</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        {novelty_section}
      </div>
    </div>
  </section>"""


def _th_css() -> str:
    return ("text-align:left;padding:7px 10px;background:var(--surface);"
            "color:var(--muted);font-size:9px;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.05em")


def _build_session_report(did: str = "", doc_name: str = "") -> str:
    """Read session_events.jsonl and write a self-contained HTML report, returning the URL path."""
    events = []
    try:
        with open(SESSION_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        return

    if not events:
        return

    target_did = did
    if target_did:
        _target_eids: set[str] = {
            ev["event_id"] for ev in events
            if ev.get("document_id") == target_did and ev.get("event_id")
        }
        filtered: list = []
        for ev in events:
            d = ev.get("document_id", "")
            if d == target_did:
                filtered.append(ev)
            elif not d and ev.get("event") == "IMAGE_SAVED" and ev.get("event_ref") in _target_eids:
                filtered.append(ev)
        events = filtered
        if not events:
            return

    def _esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace("\n", "<br>"))

    def _ts(raw: str) -> str:
        return raw[:19].replace("T", " ") if raw else ""

    if did:
        _report_folder   = _get_or_create_report_folder(did, doc_name)
        _concepts_folder = REPORTS_DIR / _report_folder / "generated_concepts"
        _views_root      = REPORTS_DIR / _report_folder / "views"
        _concepts_folder.mkdir(parents=True, exist_ok=True)
        _views_root.mkdir(parents=True, exist_ok=True)
    else:
        _report_folder   = None
        _concepts_folder = None
        _views_root      = None

    view_load_iter: dict[str, int] = {}
    for ev in events:
        if ev.get("event") == "VIEW_LOAD":
            view_load_iter[ev.get("event_id", "")] = int(ev.get("iteration_number") or 1)

    img_path_lookup: dict[str, str]            = {}
    views_by_iter:   dict[int, dict[str, str]] = {}

    for ev in events:
        if ev.get("event") != "IMAGE_SAVED":
            continue
        ref       = ev.get("event_ref", "")
        filename  = ev.get("filename", "")
        img_type  = ev.get("image_type", "")
        view_name = ev.get("view_name", "")
        if not ref or not filename:
            continue
        src_path = IMAGES_DIR / filename
        if not src_path.exists():
            continue

        if img_type == "cad_view" and view_name:
            iter_n    = view_load_iter.get(ref, 1)
            safe_name = re.sub(r"[^\w]", "_", view_name.lower().strip()) + ".png"
            if _views_root is not None:
                iter_views_dir = _views_root / f"iter{iter_n}"
                iter_views_dir.mkdir(parents=True, exist_ok=True)
                dst = iter_views_dir / safe_name
                try:
                    dst.write_bytes(src_path.read_bytes())
                except Exception:
                    pass
                views_by_iter.setdefault(iter_n, {})[view_name] = (
                    f"views/iter{iter_n}/{safe_name}"
                )
            else:
                views_by_iter.setdefault(iter_n, {})[view_name] = (
                    f"../../images/{filename}"
                )
        else:
            if _concepts_folder is not None:
                dst = _concepts_folder / filename
                if not dst.exists():
                    try:
                        dst.write_bytes(src_path.read_bytes())
                    except Exception:
                        pass
                img_path_lookup[ref] = f"generated_concepts/{filename}"
            else:
                img_path_lookup[ref] = f"../../images/{filename}"

    doc_names: dict[str, str] = {}
    for entry in _read_cache_index().values():
        did  = entry.get("document_id", "")
        name = entry.get("document_name", "")
        if did and name:
            doc_names[did] = name

    docs_iters: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    docs_order: list[str] = []

    for ev in sorted(events, key=lambda e: e.get("ts", "")):
        if ev.get("event") == "IMAGE_SAVED":
            continue
        did    = ev.get("document_id") or "unknown"
        iter_n = int(ev.get("iteration_number") or 1)
        if did not in docs_order:
            docs_order.append(did)
        docs_iters[did][iter_n].append(ev)

    iter_store = _read_iteration_store()

    doc_sections_html = []
    total_events = 0

    for did in docs_order:
        doc_name = doc_names.get(did, did[:20] + "…")
        iters    = docs_iters[did]
        current_iter = iter_store.get(did, {}).get("current_iteration", 1)
        iter_blocks_html = []

        for iter_n in sorted(iters.keys()):
            evs = iters[iter_n]
            total_events += len(evs)
            in_progress  = (iter_n == current_iter)
            status_cls   = "badge-inprogress" if in_progress else "badge-complete"
            status_label = "In Progress" if in_progress else "Complete"
            event_htmls  = []

            iter_concepts = []

            for ev in evs:
                etype = ev.get("event", "")

                if etype == "ANALYSIS":
                    user_desc = _esc(ev.get("user_description", ""))
                    ai_resp   = _esc(ev.get("ai_response", ""))
                    event_htmls.append(f"""
              <div class="ev-block ev-analysis">
                <div class="ev-hdr"><span class="tag tag-analysis">Analysis</span><span class="ev-ts">{_ts(ev.get('ts',''))}</span></div>
                <div class="field"><div class="fl">Design Description</div><div class="fb user-text">{user_desc}</div></div>
                <div class="field"><div class="fl">AI Functionality Suggestions</div><div class="fb ai-text">{ai_resp}</div></div>
              </div>""")

                elif etype == "NEXT_STEP_GENERATED":
                    direction_raw = ev.get("direction", "")
                    direction     = _esc(direction_raw)
                    eid_ref       = ev.get("event_id", "")
                    rel_path      = img_path_lookup.get(eid_ref, "")
                    if rel_path:
                        iter_concepts.append({
                            "direction": direction_raw,
                            "path":      rel_path,
                            "ts":        ev.get("ts", ""),
                        })
                        img_html = (
                            f'<a href="{rel_path}" target="_blank">'
                            f'<img class="concept-img" src="{rel_path}" '
                            f'alt="Generated concept" '
                            f'style="max-width:200px;border-radius:6px;'
                            f'border:1px solid var(--border);display:block;'
                            f'cursor:zoom-in"/></a>'
                        )
                    else:
                        img_html = '<div class="no-img">Image not available</div>'
                    event_htmls.append(f"""
              <div class="ev-block ev-generation">
                <div class="ev-hdr"><span class="tag tag-generation">Generation</span><span class="ev-ts">{_ts(ev.get('ts',''))}</span></div>
                <div class="field"><div class="fl">Direction / Modification</div><div class="fb user-text">{direction}</div></div>
                <div class="field"><div class="fl">Generated Concept Image</div><div class="img-wrap">{img_html}</div></div>
              </div>""")

                elif etype == "VIEW_LOAD":
                    # Views are now shown in the top gallery — just log metadata here
                    count  = ev.get("count", 0)
                    source = ev.get("source", "")
                    event_htmls.append(f"""
              <div class="ev-block ev-viewload">
                <div class="ev-hdr">
                  <span class="tag tag-dim">CAD Views</span>
                  <span style="font-size:10px;color:var(--dim);margin-left:6px">
                    {count} view(s) loaded · {_esc(source)}</span>
                  <span class="ev-ts">{_ts(ev.get('ts',''))}</span>
                </div>
              </div>""")

                elif etype == "ITERATION_COMPLETE":
                    event_htmls.append(f"""
              <div class="iter-complete-marker">&#10003; Iteration {iter_n} marked complete &nbsp;·&nbsp; {_ts(ev.get('ts',''))}</div>""")

            events_joined = "".join(event_htmls) or '<div class="no-events">No events recorded.</div>'

            def _make_gallery(gal_id: str, slides: list, tag_label: str,
                               tag_css: str) -> str:
                """Build a slideshow gallery block. Each slide dict: {path, caption_main, caption_sub}"""
                if not slides:
                    return ""
                sl_html = ""
                th_html = ""
                for ci, s in enumerate(slides):
                    act_s  = "slide active" if ci == 0 else "slide"
                    act_t  = "gthumb active" if ci == 0 else "gthumb"
                    esc_cap = _esc(s.get("caption_main", "")[:80])
                    sl_html += (
                        f'<div class="{act_s}" id="{gal_id}_s{ci}">'
                        f'<img src="{s["path"]}" alt="Slide {ci+1}" '
                        f'onclick="openModal(this.src,\'{esc_cap.replace(chr(39), chr(39))}\')"/>'
                        f'<div class="slide-caption">'
                        f'<span class="slide-n">{ci+1}\u202f/\u202f{len(slides)}</span>'
                        f'{esc_cap}'
                        f'<span style="font-size:9px;color:var(--muted);margin-left:8px">'
                        f'{s.get("caption_sub","")}</span>'
                        f'</div></div>'
                    )
                    th_html += (
                        f'<img class="{act_t}" id="{gal_id}_t{ci}" src="{s["path"]}" '
                        f'onclick="galSelect(\'{gal_id}\',{ci},{len(slides)})"/>'
                    )
                return (
                    f'<div class="iter-gallery">'
                    f'<div class="gal-hdr">'
                    f'<span class="tag {tag_css}" style="font-size:8px">{tag_label}</span>'
                    f'<span style="font-size:10px;color:var(--dim);margin-left:8px">'
                    f'{len(slides)} image(s)</span>'
                    f'<button class="gal-nav" '
                    f'onclick="galPrev(\'{gal_id}\',{len(slides)})">&#8592;</button>'
                    f'<button class="gal-nav" '
                    f'onclick="galNext(\'{gal_id}\',{len(slides)})">&#8594;</button>'
                    f'</div>'
                    f'<div class="slides-wrap" id="{gal_id}_wrap">{sl_html}</div>'
                    f'<div class="thumbs-strip" id="{gal_id}_thumbs">{th_html}</div>'
                    f'</div>'
                )

            iter_views = views_by_iter.get(iter_n, {})
            view_order = ["Isometric", "Front", "Back", "Left", "Right", "Top", "Bottom"]
            sorted_views = (
                [(vn, iter_views[vn]) for vn in view_order if vn in iter_views]
                + [(vn, p) for vn, p in iter_views.items() if vn not in view_order]
            )
            view_slides = [
                {"path": p, "caption_main": vn, "caption_sub": ""}
                for vn, p in sorted_views
            ]
            views_gallery_html = _make_gallery(
                f"vgal_{did[:8]}_{iter_n}", view_slides,
                "CAD Views", "tag-views",
            )

            concept_slides = [
                {
                    "path":         c["path"],
                    "caption_main": c["direction"],
                    "caption_sub":  _ts(c["ts"]),
                }
                for c in iter_concepts
            ]
            concepts_gallery_html = _make_gallery(
                f"cgal_{did[:8]}_{iter_n}", concept_slides,
                "Generated Concepts", "tag-generation",
            )

            iter_blocks_html.append(f"""
        <div class="iter-block">
          <div class="iter-hdr">
            <h3>Iteration {iter_n}</h3>
            <span class="badge {status_cls}">{status_label}</span>
          </div>
          {views_gallery_html}
          <div class="iter-events">{events_joined}</div>
          {concepts_gallery_html}
        </div>""")

        completed_count = len([doc_it for doc_it in iters if doc_it < current_iter])
        sim_html = _compute_similarity_html(did) if did and did != "unknown" else ""
        doc_sections_html.append(f"""
    <section class="doc-section">
      <div class="doc-hdr">
        <div>
          <h2>{_esc(doc_name)}</h2>
          <div class="doc-meta">Document ID: {did} &nbsp;·&nbsp; {len(iters)} iteration(s) &nbsp;·&nbsp; {completed_count} completed</div>
        </div>
      </div>
      {"".join(iter_blocks_html)}
    </section>
    {sim_html}""")

    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    body_content = "".join(doc_sections_html) if doc_sections_html else '<div class="empty">No session data recorded yet.</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DesignLoop Session Report</title>
<style>
  :root {{
    --bg:#0b0b0d; --surface:#141417; --surface2:#1b1b1f; --border:#2a2a32;
    --accent:#a8e847; --accent2:#47b8e8; --red:#e85a5a;
    --muted:#55555f; --dim:#8a8a9c; --text:#e2e2ea; --text-hi:#f4f4fa;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          font-size:14px; line-height:1.6; padding:0 0 60px; }}
  a {{ color:var(--accent2); }}

  /* ─── Header ─── */
  .report-hdr {{ background:var(--surface); border-bottom:1px solid var(--border);
                 padding:20px 32px; display:flex; align-items:baseline; gap:16px; }}
  .wordmark {{ font-size:20px; font-weight:700; letter-spacing:-0.03em; }}
  .wordmark span {{ color:var(--accent); }}
  .generated {{ font-size:11px; color:var(--muted); }}
  .stats-row {{ padding:10px 32px; background:var(--surface2); border-bottom:1px solid var(--border);
                font-size:11px; color:var(--dim); display:flex; gap:24px; }}
  .stats-row strong {{ color:var(--text); }}

  /* ─── Document section ─── */
  .doc-section {{ max-width:860px; margin:32px auto; padding:0 20px; }}
  .doc-hdr {{ margin-bottom:16px; padding-bottom:12px; border-bottom:2px solid var(--border); }}
  .doc-hdr h2 {{ font-size:18px; font-weight:700; color:var(--text-hi); }}
  .doc-meta {{ font-size:11px; color:var(--muted); margin-top:4px; font-family:monospace; }}

  /* ─── Iteration block ─── */
  .iter-block {{ border:1px solid var(--border); border-radius:8px; margin-bottom:20px;
                 overflow:hidden; }}
  .iter-hdr {{ display:flex; align-items:center; gap:12px; padding:12px 18px;
               background:var(--surface); border-bottom:1px solid var(--border); }}
  .iter-hdr h3 {{ font-size:14px; font-weight:700; color:var(--text-hi); }}
  .iter-events {{ padding:14px 18px; display:flex; flex-direction:column; gap:14px; }}

  /* ─── Badges ─── */
  .badge {{ font-size:9px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
            padding:2px 8px; border-radius:20px; }}
  .badge-complete {{ background:#1a2e0a; color:var(--accent); border:1px solid var(--accent); }}
  .badge-inprogress {{ background:#1a1a0a; color:#e8c947; border:1px solid #e8c947; }}

  /* ─── Event blocks ─── */
  .ev-block {{ background:var(--surface2); border:1px solid var(--border);
               border-radius:6px; overflow:hidden; }}
  .ev-hdr {{ display:flex; align-items:center; gap:10px; padding:8px 14px;
             background:var(--surface); border-bottom:1px solid var(--border); }}
  .ev-ts {{ font-size:10px; color:var(--muted); font-family:monospace; margin-left:auto; }}

  .tag {{ font-size:8px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
          padding:2px 7px; border-radius:3px; }}
  .tag-analysis {{ background:var(--accent); color:#000; }}
  .tag-generation {{ background:var(--accent2); color:#000; }}
  .tag-views {{ background:#4a2e8a; color:#d4aaff; }}
  .tag-dim {{ background:#333; color:var(--dim); }}

  /* ─── Fields ─── */
  .field {{ padding:10px 14px; border-bottom:1px solid var(--border); }}
  .field:last-child {{ border-bottom:none; }}
  .fl {{ font-size:9px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
         color:var(--muted); margin-bottom:6px; }}
  .fb {{ font-size:13px; line-height:1.7; color:var(--text-hi); }}
  .user-text {{ color:var(--text-hi); font-style:italic; }}
  .ai-text {{ color:var(--text); white-space:pre-wrap; }}
  .ev-viewload .fb {{ padding:10px 14px; }}

  /* ─── Images ─── */
  .img-wrap {{ padding:10px 14px; }}
  .concept-img {{ max-width:200px; border-radius:6px; border:1px solid var(--border);
                  display:block; cursor:zoom-in; }}
  .no-img {{ font-size:11px; color:var(--muted); font-style:italic; padding:10px 14px; }}

  /* ─── Concept Gallery (per iteration slideshow) ─── */
  .iter-gallery {{ border-bottom:1px solid var(--border); background:#0d0d10; }}
  .gal-hdr {{ display:flex; align-items:center; gap:6px; padding:8px 14px;
               border-bottom:1px solid var(--border); }}
  .gal-nav {{ background:var(--surface2); border:1px solid var(--border); color:var(--text);
              border-radius:4px; padding:2px 9px; font-size:14px; cursor:pointer; line-height:1.4; }}
  .gal-nav:hover {{ background:var(--surface); }}
  .slides-wrap {{ position:relative; }}
  .slide {{ display:none; }}
  .slide.active {{ display:block; }}
  .slide img {{ width:100%; max-height:520px; object-fit:contain;
                background:#08080a; cursor:zoom-in; }}
  .slide-caption {{ padding:8px 14px; font-size:12px; color:var(--dim);
                    background:var(--surface); border-top:1px solid var(--border);
                    display:flex; align-items:baseline; gap:6px; }}
  .slide-n {{ font-family:monospace; font-size:10px; color:var(--muted); margin-right:4px; }}
  .thumbs-strip {{ display:flex; gap:4px; padding:8px 14px; overflow-x:auto;
                   background:var(--surface2); border-top:1px solid var(--border);
                   scrollbar-width:none; }}
  .thumbs-strip::-webkit-scrollbar {{ display:none; }}
  .gthumb {{ width:52px; height:52px; object-fit:cover; border-radius:3px;
              border:2px solid var(--border); cursor:pointer; flex-shrink:0; }}
  .gthumb.active {{ border-color:var(--accent2); }}

  /* ─── Fullscreen modal ─── */
  .modal {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.9);
            z-index:9999; align-items:center; justify-content:center;
            flex-direction:column; cursor:zoom-out; }}
  .modal.open {{ display:flex; }}
  .modal img {{ max-width:90vw; max-height:85vh; object-fit:contain;
                border-radius:6px; box-shadow:0 0 60px #000; }}
  .modal-lbl {{ margin-top:12px; color:var(--dim); font-size:12px; max-width:80vw;
                text-align:center; }}
  .modal-close {{ position:fixed; top:16px; right:20px; color:var(--dim);
                   font-size:28px; cursor:pointer; line-height:1; }}
  .modal-close:hover {{ color:var(--text); }}

  /* ─── Iteration complete marker ─── */
  .iter-complete-marker {{ background:#0d1f06; border:1px solid #2a4a10;
                            color:var(--accent); font-size:11px; border-radius:4px;
                            padding:7px 12px; letter-spacing:0.02em; }}

  /* ─── Misc ─── */
  .no-events {{ font-size:11px; color:var(--muted); font-style:italic; padding:6px 0; }}
  .empty {{ text-align:center; padding:60px; color:var(--muted); font-size:13px; }}
  .ev-viewload {{ opacity:0.65; }}
</style>
</head>
<body>

<!-- Fullscreen modal for zooming images -->
<div class="modal" id="repModal" onclick="closeModal()">
  <span class="modal-close">&times;</span>
  <img id="repModalImg" src="" alt=""/>
  <div class="modal-lbl" id="repModalLbl"></div>
</div>

<div class="report-hdr">
  <div class="wordmark">Design<span>Loop</span></div>
  <div style="font-size:13px;color:var(--dim);font-weight:600">Session Report</div>
  <div class="generated" style="margin-left:auto">Generated {generated_at}</div>
</div>
<div class="stats-row">
  <span><strong>{len(docs_order)}</strong> document(s)</span>
  <span><strong>{total_events}</strong> events logged</span>
  <span>Report file: <strong>data/session_report.html</strong></span>
</div>

{body_content}

<script>
function galSelect(galId, idx, total) {{
  for (var k = 0; k < total; k++) {{
    var s = document.getElementById(galId + '_s' + k);
    var t = document.getElementById(galId + '_t' + k);
    if (s) s.className = k === idx ? 'slide active' : 'slide';
    if (t) t.className = k === idx ? 'gthumb active' : 'gthumb';
  }}
  var th = document.getElementById(galId + '_t' + idx);
  if (th) th.scrollIntoView({{block:'nearest',inline:'center'}});
}}
function galNext(galId, total) {{
  for (var k = 0; k < total; k++) {{
    if (document.getElementById(galId + '_s' + k).className.includes('active')) {{
      galSelect(galId, (k + 1) % total, total); return;
    }}
  }}
}}
function galPrev(galId, total) {{
  for (var k = 0; k < total; k++) {{
    if (document.getElementById(galId + '_s' + k).className.includes('active')) {{
      galSelect(galId, (k - 1 + total) % total, total); return;
    }}
  }}
}}
function openModal(src, lbl) {{
  document.getElementById('repModalImg').src = src;
  document.getElementById('repModalLbl').textContent = lbl || '';
  document.getElementById('repModal').className = 'modal open';
}}
function closeModal() {{
  document.getElementById('repModal').className = 'modal';
  document.getElementById('repModalImg').src = '';
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeModal();
}});
</script>
</body>
</html>"""

    if did and _report_folder:
        report_path = REPORTS_DIR / _report_folder / "report.html"
        url_path    = f"/reports/{_report_folder}/report.html"
    else:
        report_path = REPORTS_DIR / "index.html"
        url_path    = "/reports/index.html"

    try:
        report_path.write_text(html, encoding="utf-8")
        log.info(f"Session report written → {report_path} ({total_events} events)")
    except Exception as exc:
        log.error(f"Session report write error: {exc}")

    return url_path


# Auth
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "access_token" not in session:
            return redirect(url_for("auth_login"))
        return f(*args, **kwargs)
    return decorated


def get_onshape_headers():
    return {
        "Authorization": f"Bearer {session['access_token']}",
        "Accept":        "application/json;charset=UTF-8;qs=0.09",
        "Content-Type":  "application/json",
    }


def refresh_access_token():
    rt = session.get("refresh_token")
    if not rt:
        return False
    resp = requests.post(ONSHAPE_TOKEN_URL, data={
        "grant_type":    "refresh_token",
        "refresh_token": rt,
        "client_id":     ONSHAPE_CLIENT_ID,
        "client_secret": ONSHAPE_CLIENT_SECRET,
    })
    if resp.ok:
        td = resp.json()
        session["access_token"]  = td["access_token"]
        session["refresh_token"] = td.get("refresh_token", rt)
        return True
    return False


@app.route("/auth/login")
def auth_login():
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    params = {
        "response_type": "code",
        "client_id":     ONSHAPE_CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "scope":         "OAuth2ReadPII OAuth2Read",
        "state":         state,
    }
    return redirect(f"{ONSHAPE_AUTH_URL}?{urlencode(params)}")


@app.route("/auth/callback")
def auth_callback():
    if request.args.get("state") != session.pop("oauth_state", None):
        abort(400, "State mismatch — possible CSRF attack.")
    code = request.args.get("code")
    if not code:
        abort(400, "Missing authorization code.")
    resp = requests.post(ONSHAPE_TOKEN_URL, data={
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     ONSHAPE_CLIENT_ID,
        "client_secret": ONSHAPE_CLIENT_SECRET,
    })
    resp.raise_for_status()
    td = resp.json()
    session["access_token"]  = td["access_token"]
    session["refresh_token"] = td.get("refresh_token", "")
    return redirect(url_for("home"))


@app.route("/auth/logout")
def auth_logout():
    session.clear()
    return redirect(url_for("home"))


# Onshape view fetching
VIEW_MATRICES = {
    "Front":     "Front",
    "Back":      "Back",
    "Left":      "Left",
    "Right":     "Right",
    "Top":       "Top",
    "Bottom":    "Bottom",
    "Isometric": "0.612,0.612,0,0,-0.354,0.354,0.707,0,0.707,-0.707,0.707,0",
}


def fetch_shaded_view(did, wvm, wvmid, eid, view_matrix,
                      output_height=1080, output_width=1920, pixelSize=0,
                      explodedStateID=None,
                      API_ACCESS=None, API_SECRET=None):
    API_ACCESS = API_ACCESS or ONSHAPE_API_KEY
    API_SECRET = API_SECRET or ONSHAPE_API_SECRET

    candidates = [
        f"{BASE_URL}/api/v12/assemblies/d/{did}/{wvm}/{wvmid}/e/{eid}/shadedviews",
        f"{BASE_URL}/api/v6/partstudios/d/{did}/{wvm}/{wvmid}/e/{eid}/shadedviews",
    ]
    last_response = None
    for url in candidates:
        response = requests.get(
            url,
            auth=(API_ACCESS, API_SECRET),
            headers={
                "Accept":       "application/json;charset=UTF-8; qs=0.09",
                "Content-Type": "application/json",
            },
            params={
                "did": did, "wvm": wvm, "wvmid": wvmid, "eid": eid,
                "viewMatrix":   view_matrix,
                "outputHeight": output_height,
                "outputWidth":  output_width,
                "pixelSize":    pixelSize,
                "explodedViewId": explodedStateID,
            },
        )
        if response.ok:
            log.info(f"View fetched ({url.split('/')[5]}): {view_matrix}")
            return str(response.json()["images"][0])
        last_response = response
    log.error(f"View fetch failed: {last_response.status_code} — {last_response.text[:300]}")
    raise ValueError(f"Onshape API error {last_response.status_code}: {last_response.text[:200]}")


def fetch_all_principal_views(did: str, wid: str, eid: str, wvm: str = "w",
                               output_height=1080, output_width=1920):
    if not all([did, wid, eid]):
        raise ValueError(
            "No Onshape document context provided. "
            "Open the app from inside Onshape or paste a document URL."
        )
    results = {}
    for view_name, matrix in VIEW_MATRICES.items():
        try:
            results[view_name] = fetch_shaded_view(
                did, wvm, wid, eid, matrix,
                output_height=output_height, output_width=output_width,
            )
        except Exception as e:
            log.error(f"Could not fetch view '{view_name}': {e}")
    return results


# Routes

@app.route("/")
@app.route("/home")
def home():
    user_id = request.args.get("userId", "").strip()
    if user_id:
        session["onshape_user_id"] = user_id
        session.permanent = True
    return send_from_directory("templates", "index.html")


@app.route("/resolve-context", methods=["POST"])
def resolve_context():
    """Fill in missing workspace/element IDs for a given documentId via the Onshape API."""
    if not (ONSHAPE_API_KEY and ONSHAPE_API_SECRET):
        return jsonify({"error": "Onshape API credentials not configured"}), 400

    data = request.get_json(force=True) or {}
    did  = data.get("documentId",  "").strip()
    wid  = data.get("workspaceId", "").strip()
    eid  = data.get("elementId",   "").strip()

    if not did:
        return jsonify({"error": "documentId is required"}), 400

    try:
        doc_name = ""

        if not wid or len(wid) < 8:
            doc_resp = requests.get(
                f"{BASE_URL}/api/v9/documents/{did}",
                auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                timeout=8,
            )
            if not doc_resp.ok:
                return jsonify({"error": f"Cannot look up document {did[:8]}…: "
                                         f"HTTP {doc_resp.status_code}"}), 400
            doc_data = doc_resp.json()
            wid      = (doc_data.get("defaultWorkspace") or {}).get("id", "")
            doc_name = doc_data.get("name", "")
            if not wid:
                return jsonify({"error": "Could not determine workspace for document"}), 400

        elem_name = ""
        if not eid or len(eid) < 8:
            elem_resp = requests.get(
                f"{BASE_URL}/api/v9/documents/d/{did}/w/{wid}/elements",
                auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                timeout=8,
            )
            if elem_resp.ok:
                elems = elem_resp.json()
                if isinstance(elems, list):
                    asm_elems = [e for e in elems if e.get("type") == "Assembly"]
                    if asm_elems:
                        eid       = asm_elems[0]["id"]
                        elem_name = asm_elems[0].get("name", "Assembly")
            if not eid:
                return jsonify({"error": "No Assembly element found in document"}), 404
        else:
            # eid was supplied — try to look up its name
            elem_resp = requests.get(
                f"{BASE_URL}/api/v9/documents/d/{did}/w/{wid}/elements",
                auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                timeout=8,
            )
            if elem_resp.ok:
                elems = elem_resp.json()
                if isinstance(elems, list):
                    for e in elems:
                        if e.get("id") == eid:
                            elem_name = e.get("name", "")
                            break

        log_event("CONTEXT_SET", {
            "document_id": did, "workspace_id": wid, "element_id": eid,
            "source": "resolve_context",
        }, did=did)
        log.info(f"resolve-context: doc:{did} ws:{wid} elem:{eid} ({elem_name or '?'})")
        return jsonify({
            "status":           "ok",
            "documentId":        did,
            "workspaceId":       wid,
            "elementId":         eid,
            "element_name":      elem_name,
            "doc_name":          doc_name,
            "cache_hit":         _has_view_cache(did, wid, eid),
            "current_iteration": _get_doc_iteration(did),
            "current_context":   _read_iteration_store().get(did, {}).get("current_context", {}),
        })
    except Exception as exc:
        log.error(f"/resolve-context error: {exc}\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500


@app.route("/detect-document", methods=["POST"])
def detect_document():
    """Find the most recently modified Onshape assembly document, with optional documentId hint."""
    if not ONSHAPE_API_KEY or not ONSHAPE_API_SECRET:
        return jsonify({"error": "Onshape API credentials not configured in .env"}), 400

    body         = request.get_json(force=True) or {}
    hinted_did   = body.get("documentId", "").strip()

    try:
        if hinted_did:
            doc_resp = requests.get(
                f"{BASE_URL}/api/v9/documents/{hinted_did}",
                auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                timeout=8,
            )
            if doc_resp.ok:
                doc_data = doc_resp.json()
                hinted_wid = (doc_data.get("defaultWorkspace") or {}).get("id", "")
                if hinted_wid:
                    elem_resp = requests.get(
                        f"{BASE_URL}/api/v9/documents/d/{hinted_did}/w/{hinted_wid}/elements",
                        auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                        timeout=8,
                    )
                    if elem_resp.ok:
                        elems = elem_resp.json()
                        if isinstance(elems, list):
                            asm_elems = [e for e in elems if e.get("type") == "Assembly"]
                            if asm_elems:
                                eid       = asm_elems[0]["id"]
                                elem_name = asm_elems[0].get("name", "Assembly")
                                doc_name  = doc_data.get("name", "")
                                log_event("CONTEXT_SET", {
                                    "document_id": hinted_did, "workspace_id": hinted_wid,
                                    "element_id": eid, "source": "detect_document_hinted",
                                }, did=hinted_did)
                                log.info(f"detect-document (hinted): doc:{hinted_did} elem:{eid}")
                                return jsonify({
                                    "status":           "ok",
                                    "documentId":        hinted_did,
                                    "workspaceId":       hinted_wid,
                                    "elementId":         eid,
                                    "wvm":               "w",
                                    "element_name":      elem_name,
                                    "doc_name":          doc_name,
                                    "cache_hit":         _has_view_cache(hinted_did, hinted_wid, eid),
                                    "current_iteration": _get_doc_iteration(hinted_did),
                                    "current_context":   _read_iteration_store().get(hinted_did, {}).get("current_context", {}),
                                    "assembly_count":    len(asm_elems),
                                })

        resp = requests.get(
            f"{BASE_URL}/api/v9/documents",
            auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
            params={"sortColumn": "modifiedAt", "sortOrder": "desc", "limit": 20},
            timeout=12,
        )
        if not resp.ok:
            log.error(f"detect-document: documents API {resp.status_code} — {resp.text[:200]}")
            return jsonify({"error": f"Onshape documents API returned {resp.status_code}"}), 400

        docs = resp.json().get("items", [])
        log.info(f"detect-document: checking {len(docs)} recent documents")

        for doc in docs:
            did = doc.get("id", "")
            wid = (doc.get("defaultWorkspace") or {}).get("id", "")
            if not did or not wid:
                continue

            elem_resp = requests.get(
                f"{BASE_URL}/api/v9/documents/d/{did}/w/{wid}/elements",
                auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                timeout=8,
            )
            if not elem_resp.ok:
                continue

            elems = elem_resp.json()
            if not isinstance(elems, list):
                continue
            assembly_elems = [e for e in elems if e.get("type") == "Assembly"]

            if assembly_elems:
                eid          = assembly_elems[0]["id"]
                elem_name    = assembly_elems[0].get("name", "Assembly")
                doc_name     = doc.get("name", "")
                log.info(f"detect-document: matched — doc:{did} ws:{wid} elem:{eid} ({elem_name})")
                log_event("CONTEXT_SET", {
                    "document_id": did, "workspace_id": wid, "element_id": eid,
                    "source": "api_detection",
                }, did=did)
                return jsonify({
                    "status":           "ok",
                    "documentId":        did,
                    "workspaceId":       wid,
                    "elementId":         eid,
                    "wvm":               "w",
                    "element_name":      elem_name,
                    "doc_name":          doc_name,
                    "cache_hit":         _has_view_cache(did, wid, eid),
                    "current_iteration": _get_doc_iteration(did),
                    "current_context":   _read_iteration_store().get(did, {}).get("current_context", {}),
                    "assembly_count":    len(assembly_elems),
                })

        return jsonify({"error": "No assembly found in recent documents"}), 404

    except Exception as exc:
        log.error(f"detect-document error: {exc}\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500


@app.route("/list-documents", methods=["GET"])
def list_documents():
    """Return recent Onshape documents that contain at least one Assembly element."""
    if not ONSHAPE_API_KEY or not ONSHAPE_API_SECRET:
        return jsonify({"error": "Onshape API credentials not configured"}), 400

    try:
        resp = requests.get(
            f"{BASE_URL}/api/v9/documents",
            auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
            params={"sortColumn": "modifiedAt", "sortOrder": "desc", "limit": 12},
            timeout=12,
        )
        if not resp.ok:
            return jsonify({"error": f"Onshape API {resp.status_code}"}), 400

        docs = resp.json().get("items", [])

        def _check_doc(doc):
            did      = doc.get("id", "")
            wid      = (doc.get("defaultWorkspace") or {}).get("id", "")
            name     = doc.get("name", "")
            modified = doc.get("modifiedAt", "")
            if not did or not wid:
                return None
            try:
                er = requests.get(
                    f"{BASE_URL}/api/v9/documents/d/{did}/w/{wid}/elements",
                    auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
                    timeout=8,
                )
                if not er.ok:
                    return None
                elems = er.json()
                if not isinstance(elems, list):
                    return None
                asm = [e for e in elems if e.get("type") == "Assembly"]
                if not asm:
                    return None
                return {
                    "documentId":        did,
                    "workspaceId":       wid,
                    "elementId":         asm[0]["id"],
                    "wvm":               "w",
                    "doc_name":          name,
                    "elem_name":         asm[0].get("name", "Assembly"),
                    "modifiedAt":        modified,
                    "cache_hit":         _has_view_cache(did, wid, asm[0]["id"]),
                    "current_iteration": _get_doc_iteration(did),
                }
            except Exception:
                return None

        results = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(_check_doc, d): d for d in docs}
            for fut in as_completed(futures):
                item = fut.result()
                if item:
                    results.append(item)
                    if len(results) >= 10:
                        for pending in futures:
                            pending.cancel()
                        break

        id_order = {d["id"]: i for i, d in enumerate(docs)}
        results.sort(key=lambda r: id_order.get(r["documentId"], 999))

        return jsonify({"documents": results[:10]})

    except Exception as exc:
        log.error(f"list-documents error: {exc}\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500


@app.route("/auto-context", methods=["POST"])
def auto_context():
    """Kept for backwards compatibility; always returns an error so the frontend falls through to the URL-paste flow."""
    return jsonify({"error": "No document context — paste a URL or open from Onshape"}), 400


@app.route("/set-context", methods=["POST", "OPTIONS"])
def set_context():
    """Accept document IDs from Onshape URL query params or a pasted URL."""
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(force=True)

    onshape_url = data.get("onshapeUrl", "").strip()
    if onshape_url:
        parsed = _parse_onshape_url(onshape_url)
        if parsed:
            data.update(parsed)

    did = data.get("documentId", "").strip()
    wid = data.get("workspaceId", "").strip()
    eid = data.get("elementId", "").strip()
    wvm = data.get("wvm", "w").strip() or "w"

    if not all([did, wid, eid]):
        return jsonify({
            "error": "documentId, workspaceId, and elementId are all required. "
                     "You can also paste a full Onshape document URL."
        }), 400

    for _id_val in (did, wid, eid):
        if _id_val.startswith("{") or _id_val.endswith("}"):
            return jsonify({"error": "Context IDs are still template placeholders — Onshape has not yet provided real values."}), 400

    log.info(f"Context set — doc:{did} {wvm}:{wid} elem:{eid}")
    log_event("CONTEXT_SET", {
        "document_id": did,
        "workspace_id": wid,
        "element_id": eid,
        "source": "url_paste" if onshape_url else "iframe_params",
    }, did=did)
    _store_entry = _read_iteration_store().get(did, {})
    return jsonify({
        "status":            "ok",
        "documentId":        did,
        "workspaceId":       wid,
        "elementId":         eid,
        "wvm":               wvm,
        "cache_hit":         _has_view_cache(did, wid, eid),
        "current_iteration": _get_doc_iteration(did),
        "current_context":   _store_entry.get("current_context", {}),
    })


def _parse_onshape_url(url: str) -> dict | None:
    """
    Parse a full Onshape document URL into component IDs.
    Format: https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}
    """
    m = re.search(
        r"documents/([a-f0-9]+)/(w|v|m)/([a-f0-9]+)/e/([a-f0-9]+)", url
    )
    if m:
        return {
            "documentId":  m.group(1),
            "wvm":         m.group(2),
            "workspaceId": m.group(3),
            "elementId":   m.group(4),
        }
    return None


@app.route("/get-views", methods=["POST"])
def get_views():
    try:
        body          = request.get_json(force=True) or {}
        force_refresh = bool(body.get("force_refresh", False))

        did = body.get("documentId",  "").strip()
        wid = body.get("workspaceId", "").strip()
        eid = body.get("elementId",   "").strip()
        wvm = body.get("wvm",         "w").strip() or "w"
        doc_name  = body.get("doc_name",  "")
        elem_name = body.get("elem_name", "")

        if not all([did, wid, eid]):
            return jsonify({"error": "documentId, workspaceId, and elementId are required"}), 400

        from_cache = False
        views      = None

        if not force_refresh:
            views = _load_view_cache(did, wid, eid)
            if views:
                from_cache = True

        if not views:
            views = fetch_all_principal_views(did, wid, eid, wvm,
                                              output_height=800, output_width=800)
            if not views:
                return jsonify({"error": "No views returned by Onshape."}), 500

            doc_modified_at = _fetch_doc_modified_at(did) or ""
            _save_view_cache(did, wid, eid, views, doc_name, doc_modified_at, elem_name)

            ev_id = log_event("VIEW_LOAD", {
                "view_names":   list(views.keys()),
                "count":        len(views),
                "source":       "onshape_api",
                "workspace_id": wid,
                "element_id":   eid,
            }, did=did)
            for vname, b64 in views.items():
                img_file = save_image_artifact(b64, f"view_{vname}", ev_id)
                if img_file:
                    log_event("IMAGE_SAVED", {
                        "event_ref":  ev_id,
                        "filename":   img_file,
                        "view_name":  vname,
                        "image_type": "cad_view",
                    }, did=did)

        cache_views(did, views)
        log.info(f"Views {'from disk cache' if from_cache else 'fetched from Onshape'}: {list(views.keys())}")
        return jsonify({"views": views, "from_cache": from_cache})

    except Exception as e:
        log.error(f"/get-views error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/cache-list", methods=["GET"])
def cache_list():
    """Return metadata for all cached documents (no image data)."""
    return jsonify({"cache": list(_read_cache_index().values())})


@app.route("/cache-clear", methods=["POST"])
def cache_clear():
    """Clear view cache. Body: {"all": true} wipes everything; {} clears only the current document."""
    body = request.get_json(force=True) or {}
    if body.get("all"):
        _clear_all_view_cache()
        return jsonify({"status": "ok", "message": "All cached views deleted."})

    did = body.get("documentId",  "").strip()
    wid = body.get("workspaceId", "").strip()
    eid = body.get("elementId",   "").strip()
    if not all([did, wid, eid]):
        return jsonify({"error": "documentId, workspaceId, and elementId are required."}), 400
    _delete_view_cache(did, wid, eid)
    return jsonify({"status": "ok", "message": "Cache cleared for current document."})


@app.route("/process-comment", methods=["POST", "OPTIONS"])
def process_comment():
    if request.method == "OPTIONS":
        return "", 204
    try:
        user_description     = _clean_text(request.form.get("user-description", "").strip())
        views_already_cached = request.form.get("views-cached", "").lower() == "true"

        did = request.form.get("documentId",  "").strip()
        wid = request.form.get("workspaceId", "").strip()
        eid = request.form.get("elementId",   "").strip()
        wvm = request.form.get("wvm",         "w").strip() or "w"
        if not all([did, wid, eid]):
            return jsonify({"error": "documentId, workspaceId, and elementId are required"}), 400

        cached = get_cached_views(did)
        if cached and views_already_cached:
            views = cached
            refetched = False
        else:
            try:
                views = fetch_all_principal_views(did, wid, eid, wvm,
                                                  output_height=1920, output_width=1080)
                cache_views(did, views)
                refetched = True
            except Exception as e:
                log.error(f"Onshape fetch error: {e}")
                return jsonify({"error": f"Could not retrieve CAD views: {e}"}), 500

        if not views:
            return jsonify({"error": "No views were returned by Onshape."}), 500

        dev_prompt = _load_prompt("analyse.txt")
        history_ctx = _get_design_history_string(did)
        if history_ctx:
            dev_prompt = history_ctx + "\n\n" + dev_prompt

        user_content = [
            {
                "type": "input_text",
                "text": (
                    f"Design description: {user_description}\n\n"
                    f"The following {len(views)} images show the design from "
                    f"multiple angles: {', '.join(views.keys())}."
                ),
            }
        ]
        for vname, b64 in views.items():
            user_content.append({"type": "input_text",  "text": f"[{vname} view]"})
            user_content.append({
                "type":      "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            })

        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "developer", "content": dev_prompt},
                {"role": "user",      "content": user_content},
            ],
        )
        ai_response = response.output_text
        cache_analysis(did, user_description, ai_response)
        _save_iteration_context(did, description=user_description, analysis=ai_response)

        ev_id = log_event("ANALYSIS", {
            "user_description":   user_description,
            "ai_response":        ai_response,
            "ai_response_length": len(ai_response),
            "views_refetched":    refetched,
            "model":              "gpt-4.1-mini",
        }, did=did)

        _iter = _get_doc_iteration(did)
        _sid  = _session_id()
        user_emb = compute_embedding(user_description)
        if user_emb:
            log_embedding(ev_id, "user_description", user_description, user_emb,
                          document_id=did, iteration_number=_iter,
                          event_type="ANALYSIS", session_id=_sid)

        ai_emb = compute_embedding(ai_response[:2000])
        if ai_emb:
            log_embedding(ev_id, "ai_analysis", ai_response[:2000], ai_emb,
                          document_id=did, iteration_number=_iter,
                          event_type="ANALYSIS", session_id=_sid)

        return jsonify({
            "user_description": user_description,
            "ai_response":      ai_response,
            "views":            views if refetched else {},
        })

    except Exception as e:
        log.error(f"/process-comment error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate-next-step-image", methods=["POST"])
def generate_next_step_image():
    direction = ""
    try:
        body      = request.get_json(force=True)
        direction = _clean_text(body.get("direction", "").strip())
        if not direction:
            return jsonify({"error": "'direction' field is required."}), 400

        did = body.get("documentId", "").strip()
        if not did:
            return jsonify({"error": "documentId is required"}), 400

        user_description, ai_analysis = get_cached_analysis(did)
        cached_views = get_cached_views(did) or {}

        def _clean(s: str) -> str:
            for pfx in ("data:image/png;base64,", "data:image/jpeg;base64,"):
                if s.startswith(pfx):
                    return s[len(pfx):]
            return s

        SELECTED_VIEWS = ["Isometric", "Front", "Top"]
        vision_content = []
        prev_concepts  = get_last_concepts(did)

        parts = []
        history_ctx = _get_design_history_string(did)
        if history_ctx:
            parts.append(history_ctx)
        if user_description:
            parts.append(f"Design description: {user_description}")
        if ai_analysis:
            parts.append(f"Prior analysis:\n{ai_analysis}")
        if prev_concepts:
            prev_lines = [
                f"  {i+1}. \"{pc['direction']}\" — "
                f"rendered as: {pc['prompt'][:200]}"
                for i, pc in enumerate(prev_concepts)
            ]
            parts.append(
                "PREVIOUSLY GENERATED CONCEPTS IN THIS STEP "
                "(the user may be refining one of these — use them as context):\n"
                + "\n".join(prev_lines)
            )
        parts.append(
            f"Next-step modification to visualise: {direction}\n\n"
            + _load_prompt("vision.txt")
        )
        vision_content.append({"type": "input_text", "text": "\n\n".join(parts)})

        if prev_concepts:
            last = prev_concepts[-1]
            vision_content.append({
                "type": "input_text",
                "text": f"[Most recent generated concept — direction: \"{last['direction'][:80]}\"]",
            })
            vision_content.append({
                "type":      "input_image",
                "image_url": f"data:image/png;base64,{_clean(last['image_b64'])}",
            })

        views_added = 0
        for vname in SELECTED_VIEWS:
            b64 = cached_views.get(vname)
            if b64:
                vision_content.append({"type": "input_text", "text": f"[{vname} CAD view]"})
                vision_content.append({
                    "type":      "input_image",
                    "image_url": f"data:image/png;base64,{_clean(b64)}",
                })
                views_added += 1

        vision_resp = openai_client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": vision_content}],
        )
        image_prompt = vision_resp.output_text.strip()
        # Replace smart/curly quotes and other common non-ASCII punctuation so
        # the prompt is safe for logging and the image API.
        image_prompt = (image_prompt
            .replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u2018', "'").replace('\u2019', "'")
            .replace('\u2013', '-').replace('\u2014', '-')
            .encode('ascii', errors='replace').decode('ascii'))

        result = openai_client.images.generate(
            model="gpt-image-1",
            prompt=image_prompt,
            size="1024x1024",
            n=1,
        )
        image_b64 = result.data[0].b64_json

        ev_id = log_event("NEXT_STEP_GENERATED", {
            "direction":           direction,
            "views_used":          views_added,
            "prev_concepts_used":  len(prev_concepts),
            "prompt_length":       len(image_prompt),
            "image_prompt":        image_prompt[:500],
            "model_vision":        "gpt-4.1",
            "model_image":         "gpt-image-1",
        }, did=did)

        _save_iteration_context(did, direction=direction)

        img_file = save_image_artifact(image_b64, "concept", ev_id)
        if img_file:
            log_event("IMAGE_SAVED", {"event_ref": ev_id, "filename": img_file}, did=did)

        cache_last_concepts(did, {
            "direction": direction,
            "image_b64": image_b64,
            "prompt":    image_prompt,
            "filename":  img_file,
            "ts":        datetime.datetime.now(datetime.UTC).isoformat(),
        })

        _iter = _get_doc_iteration(did)
        _sid  = _session_id()
        dir_emb = compute_embedding(direction)
        if dir_emb:
            log_embedding(ev_id, "user_direction", direction, dir_emb,
                          document_id=did, iteration_number=_iter,
                          event_type="NEXT_STEP_GENERATED", session_id=_sid)

        prompt_emb = compute_embedding(image_prompt[:2000])
        if prompt_emb:
            log_embedding(ev_id, "image_prompt", image_prompt[:2000], prompt_emb,
                          document_id=did, iteration_number=_iter,
                          event_type="NEXT_STEP_GENERATED", session_id=_sid)

        return jsonify({"image_b64": image_b64, "prompt_used": image_prompt})

    except Exception as e:
        log.error(f"/generate-next-step-image error: {e}\n{traceback.format_exc()}")
        log_event("NEXT_STEP_ERROR", {"direction": direction, "error": str(e)}, did=did)
        return jsonify({"error": str(e)}), 500


@app.route("/get-elements", methods=["GET"])
def get_elements():
    """Return all Assembly and Part Studio elements in the current document."""
    did = request.args.get("documentId",  "").strip()
    wid = request.args.get("workspaceId", "").strip()
    eid = request.args.get("elementId",   "").strip()
    if not all([did, wid]):
        return jsonify({"error": "documentId and workspaceId are required"}), 400
    if not (ONSHAPE_API_KEY and ONSHAPE_API_SECRET):
        return jsonify({"error": "Onshape API credentials not configured"}), 400

    try:
        resp = requests.get(
            f"{BASE_URL}/api/v9/documents/d/{did}/w/{wid}/elements",
            auth=(ONSHAPE_API_KEY, ONSHAPE_API_SECRET),
            timeout=8,
        )
        if not resp.ok:
            return jsonify({"error": f"Onshape API error {resp.status_code}"}), 400

        elems = resp.json()
        if not isinstance(elems, list):
            return jsonify({"error": "Unexpected response format"}), 500

        elements = [
            {
                "id":      e["id"],
                "name":    e.get("name", "Unnamed"),
                "type":    e.get("elementType", e.get("type", "")),
                "current": e["id"] == eid,
            }
            for e in elems
            if e.get("elementType") in ("ASSEMBLY", "PARTSTUDIO")
            or e.get("type") in ("Assembly", "Part Studio")
        ]
        return jsonify({"elements": elements})
    except Exception as exc:
        log.error(f"/get-elements error: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/switch-element", methods=["POST"])
def switch_element():
    """Switch the active assembly/element within the current document."""
    data = request.get_json(force=True)
    did       = data.get("documentId",  "").strip()
    wid       = data.get("workspaceId", "").strip()
    wvm       = data.get("wvm",         "w").strip() or "w"
    new_eid   = data.get("elementId",   "").strip()
    elem_name = data.get("elementName", "").strip()
    doc_name  = data.get("doc_name",    "").strip()
    if not all([did, wid, new_eid]):
        return jsonify({"error": "documentId, workspaceId, and elementId are required"}), 400

    log_event("ELEMENT_SWITCHED", {"new_element_id": new_eid, "element_name": elem_name}, did=did)
    log.info(f"Element switched → {new_eid} ({elem_name}) in doc {did}")

    return jsonify({
        "status":           "ok",
        "documentId":        did,
        "workspaceId":       wid,
        "elementId":         new_eid,
        "wvm":               wvm,
        "element_name":      elem_name,
        "doc_name":          doc_name,
        "cache_hit":         _has_view_cache(did, wid, new_eid),
        "current_iteration": _get_doc_iteration(did),
    })


@app.route("/check-model-changed", methods=["GET"])
def check_model_changed():
    """Compare the document's current modifiedAt timestamp against the cached value to detect changes."""
    did = request.args.get("documentId",  "").strip()
    wid = request.args.get("workspaceId", "").strip()
    eid = request.args.get("elementId",   "").strip()
    if not all([did, wid, eid]):
        return jsonify({"error": "documentId, workspaceId, and elementId are required"}), 400

    current_modified = _fetch_doc_modified_at(did)

    key   = _view_cache_key(did, wid, eid)
    idx   = _read_cache_index()
    entry = idx.get(key, {})
    # Prefer the stored doc_modified_at; fall back to cached_at
    stored_modified = entry.get("doc_modified_at") or entry.get("cached_at", "")
    has_cache       = bool(entry)

    changed = False
    if current_modified and stored_modified:
        changed = current_modified > stored_modified

    return jsonify({
        "changed":              changed,
        "model_modified_at":    current_modified,
        "cached_at":            stored_modified,
        "has_cache":            has_cache,
        "has_iteration_input":  _iteration_has_input(did),
    })


@app.route("/finish-iteration", methods=["POST"])
def finish_iteration():
    """Log iteration completion, increment the counter, rebuild the HTML report, and return the new iteration number."""
    body     = request.get_json(force=True) or {}
    did      = body.get("documentId", "").strip()
    doc_name = body.get("doc_name",   "").strip()
    if not did:
        return jsonify({"error": "documentId is required."}), 400

    current  = _get_doc_iteration(did)
    log_event("ITERATION_COMPLETE", {"completed_iteration": current}, did=did)
    new_iter   = _finish_doc_iteration(did)
    # Clear in-step concept memory so the new iteration starts fresh
    clear_last_concepts(did)
    report_url = _build_session_report(did, doc_name)
    log.info(f"Iteration {current} finished for doc {did}. New: {new_iter}. Report: {report_url}")
    return jsonify({
        "new_iteration":      new_iter,
        "completed_iteration": current,
        "report_url":          report_url,
    })


@app.route("/report")
def session_report():
    """Landing page listing all generated report folders with links."""
    did      = request.args.get("documentId", "").strip()
    doc_name = request.args.get("doc_name",  "").strip()
    if did:
        _build_session_report(did, doc_name)

    folders = sorted(
        [p.name for p in REPORTS_DIR.iterdir() if p.is_dir() and (p / "report.html").exists()],
        reverse=True,
    )
    if not folders:
        return ("<html><body style='font-family:sans-serif;padding:40px;background:#0b0b0d;color:#888'>"
                "<p>No reports yet. Complete at least one iteration first.</p></body></html>"), 404

    items = "".join(
        f'<li><a href="/reports/{f}/report.html" target="_blank">{f}</a></li>'
        for f in folders
    )
    return (f"<html><body style='font-family:sans-serif;padding:40px;"
            f"background:#0b0b0d;color:#d4d4dc'>"
            f"<h2 style='color:#a8e847'>DesignLoop Reports</h2>"
            f"<ul style='line-height:2;margin-top:16px'>{items}</ul>"
            f"</body></html>")


@app.route("/reports/<path:filename>")
def serve_report_file(filename):
    """Serve any file from the reports directory (HTML reports + embedded assets)."""
    return send_from_directory(str(REPORTS_DIR), filename)


@app.route("/export-csv", methods=["GET"])
def export_csv():
    """Export the session event log as a downloadable CSV."""
    path = export_session_csv()
    if not path:
        return jsonify({"error": "No events logged yet."}), 404
    return send_from_directory(str(DATA_DIR), "session_export.csv",
                               as_attachment=True)


@app.route("/export-embeddings", methods=["GET"])
def export_embeddings():
    """Download the raw embeddings JSONL file."""
    if not EMBEDDINGS_LOG.exists():
        return jsonify({"error": "No embeddings logged yet."}), 404
    return send_from_directory(str(DATA_DIR), "embeddings.jsonl",
                               as_attachment=True)


@app.route("/embedding-analysis", methods=["GET"])
def embedding_analysis():
    """Compute semantic similarity metrics across design iterations for a document."""
    def _cosine(a: list, b: list) -> float:
        dot  = sum(x * y for x, y in zip(a, b, strict=False))
        na   = math.sqrt(sum(x * x for x in a))
        nb   = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 6)

    def _centroid(vectors: list[list]) -> list:
        if not vectors:
            return []
        dim = len(vectors[0])
        c = [0.0] * dim
        for v in vectors:
            for i, x in enumerate(v):
                c[i] += x
        n = len(vectors)
        return [x / n for x in c]

    if not EMBEDDINGS_LOG.exists():
        return jsonify({"error": "No embeddings logged yet."}), 404

    doc_filter = request.args.get("document_id", "").strip()

    records = []
    try:
        with open(EMBEDDINGS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if doc_filter and rec.get("document_id", "") != doc_filter:
                    continue
                if not rec.get("vector"):
                    continue
                records.append(rec)
    except Exception as exc:
        return jsonify({"error": f"Could not read embeddings: {exc}"}), 500

    if not records:
        return jsonify({"error": "No embedding records found for this document."}), 404

    by_iter: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        it    = int(rec.get("iteration_number") or 0)
        field = rec.get("field", "unknown")
        by_iter[it][field].append(rec["vector"])

    iterations_sorted = sorted(by_iter.keys())

    per_iteration = []
    iter_desc_centroids: dict[int, list] = {}   # for drift calculation
    for it in iterations_sorted:
        fields = by_iter[it]
        desc_vecs = fields.get("user_description", [])
        dir_vecs  = fields.get("user_direction",   [])
        ai_vecs   = fields.get("ai_analysis",      [])
        all_vecs  = [v for vlist in fields.values() for v in vlist]
        centroid  = _centroid(all_vecs)
        desc_centroid = _centroid(desc_vecs) if desc_vecs else []
        iter_desc_centroids[it] = desc_centroid
        per_iteration.append({
            "iteration_number":     it,
            "n_descriptions":       len(desc_vecs),
            "n_directions":         len(dir_vecs),
            "n_ai_analyses":        len(ai_vecs),
            "n_total_embeddings":   len(all_vecs),
            "centroid_vector_dim":  len(centroid),
            "centroid":             centroid,
        })

    centroids = [entry["centroid"] for entry in per_iteration]
    n = len(centroids)
    sim_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sim = _cosine(centroids[i], centroids[j]) if (centroids[i] and centroids[j]) else None
            row.append(sim)
        sim_matrix.append(row)

    description_drift = []
    for idx in range(len(iterations_sorted) - 1):
        it_a = iterations_sorted[idx]
        it_b = iterations_sorted[idx + 1]
        ca   = iter_desc_centroids.get(it_a, [])
        cb   = iter_desc_centroids.get(it_b, [])
        if ca and cb:
            description_drift.append({
                "from_iteration": it_a,
                "to_iteration":   it_b,
                "cosine_sim":     _cosine(ca, cb),
                "semantic_distance": round(1 - _cosine(ca, cb), 6),
            })

    all_direction_records = [
        rec for rec in records if rec.get("field") == "user_direction"
    ]
    all_direction_records.sort(key=lambda r: (r.get("iteration_number", 0), r.get("ts", "")))

    direction_novelty = []
    seen_dir_vecs: list[list] = []
    for rec in all_direction_records:
        vec = rec["vector"]
        if seen_dir_vecs:
            max_sim = max(_cosine(vec, prev) for prev in seen_dir_vecs)
        else:
            max_sim = 0.0   # first direction is always novel
        direction_novelty.append({
            "iteration_number": rec.get("iteration_number"),
            "text_preview":     rec.get("text", "")[:120],
            "max_sim_to_prior": round(max_sim, 6),
            "novelty_score":    round(1 - max_sim, 6),
        })
        seen_dir_vecs.append(vec)

    analysis_alignment = []
    for it in iterations_sorted:
        desc_vecs = by_iter[it].get("user_description", [])
        ai_vecs   = by_iter[it].get("ai_analysis",      [])
        if desc_vecs and ai_vecs:
            cd = _centroid(desc_vecs)
            ca = _centroid(ai_vecs)
            analysis_alignment.append({
                "iteration_number": it,
                "cosine_sim":       _cosine(cd, ca),
            })

    field_counts = dict(Counter(r.get("field", "unknown") for r in records))

    return jsonify({
        "document_id":        doc_filter,
        "total_embeddings":   len(records),
        "iterations_found":   iterations_sorted,
        "field_counts":       field_counts,
        "per_iteration":      per_iteration,
        "similarity_matrix":  {
            "iterations": iterations_sorted,
            "matrix":     sim_matrix,
        },
        "description_drift":  description_drift,
        "direction_novelty":  direction_novelty,
        "analysis_alignment": analysis_alignment,
    })


@app.route("/report-image/<path:filename>")
def serve_report_image(filename):
    """Serve a concept or CAD-view image from data/images/ for use in the History modal."""
    safe = Path(filename).name   # strip any directory traversal
    return send_from_directory(str(IMAGES_DIR), safe)


@app.route("/get-iteration-history", methods=["GET"])
def get_iteration_history():
    """Return all completed iterations and the current in-progress context, enriched with concept image URLs."""
    did = request.args.get("documentId", "").strip()
    if not did:
        return jsonify({"error": "documentId is required"}), 400

    store = _read_iteration_store()
    entry = store.get(did, {})

    concepts_by_iter: dict[int, list] = {}
    if SESSION_LOG.exists():
        raw_events: list[dict] = []
        try:
            with open(SESSION_LOG, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_events.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

        img_by_ref: dict[str, str] = {}
        for ev in raw_events:
            if (ev.get("event") == "IMAGE_SAVED"
                    and ev.get("document_id") == did
                    and not ev.get("image_type")):   # skip cad_view entries
                ref = ev.get("event_ref", "")
                fn  = ev.get("filename", "")
                if ref and fn:
                    img_by_ref[ref] = fn

        for ev in raw_events:
            if ev.get("event") != "NEXT_STEP_GENERATED":
                continue
            if ev.get("document_id") != did:
                continue
            it  = int(ev.get("iteration_number") or 0)
            fn  = img_by_ref.get(ev.get("event_id", ""), "")
            concepts_by_iter.setdefault(it, []).append({
                "direction": ev.get("direction", ""),
                "image_url": f"/report-image/{fn}" if fn else "",
                "ts":        ev.get("ts", ""),
            })

    completed_enriched = []
    for rec in entry.get("completed", []):
        it_n = rec.get("iteration")
        completed_enriched.append({
            **rec,
            "concepts": concepts_by_iter.get(it_n, []),
        })

    current_it = entry.get("current_iteration", 1)
    current_ctx = dict(entry.get("current_context", {}))
    current_ctx["concepts"] = concepts_by_iter.get(current_it, [])

    return jsonify({
        "document_id":       did,
        "current_iteration": current_it,
        "completed":         completed_enriched,
        "current_context":   current_ctx,
    })


@app.route("/rebuild-reports", methods=["POST"])
def rebuild_reports():
    """Regenerate all report HTML files for every document that has logged events."""
    events = []
    try:
        with open(SESSION_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        return jsonify({"error": "No session log found"}), 404

    doc_map: dict[str, str] = {}
    for ev in events:
        did  = ev.get("document_id", "")
        name = ev.get("doc_name") or ev.get("document_name", "")
        if did and did not in doc_map:
            doc_map[did] = name

    rebuilt = []
    for did, doc_name in doc_map.items():
        try:
            url = _build_session_report(did, doc_name)
            rebuilt.append({"document_id": did, "report_url": url})
        except Exception as exc:
            rebuilt.append({"document_id": did, "error": str(exc)})

    return jsonify({"rebuilt": rebuilt, "count": len(rebuilt)})


@app.route("/last-concepts", methods=["GET"])
def last_concepts_status():
    """Return how many previous concepts are cached for the current step."""
    did      = request.args.get("documentId", "").strip()
    concepts = get_last_concepts(did)
    return jsonify({
        "count": len(concepts),
        "directions": [c.get("direction", "")[:80] for c in concepts],
    })


@app.route("/save-history", methods=["POST"])
def save_history():
    history  = request.get_json()
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{ts}.json"
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "saved", "file": filename})


@app.route("/health")
def health():
    """Quick health check for Onshape store verification."""
    return jsonify({
        "status":     "ok",
        "session_ok": "research_session_id" in session or True,
        "note":       "Document context is now managed client-side per tab.",
    })


def _startup_rebuild_reports():
    """On server start, regenerate all existing reports to apply the latest template."""
    if not SESSION_LOG.exists():
        return
    try:
        events = []
        with open(SESSION_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        doc_map: dict[str, str] = {}
        for ev in events:
            d   = ev.get("document_id", "")
            nm  = ev.get("document_name", "") or ""
            if d and d not in doc_map:
                doc_map[d] = nm
        for d, nm in doc_map.items():
            try:
                _build_session_report(d, nm)
            except Exception as exc:
                log.warning(f"startup rebuild failed for {d}: {exc}")
        if doc_map:
            log.info(f"Startup: rebuilt reports for {len(doc_map)} document(s)")
    except Exception as exc:
        log.warning(f"startup_rebuild_reports: {exc}")


@app.route("/debug-onshape-msg", methods=["POST"])
def debug_onshape_msg():
    """Log every raw postMessage Onshape sends to the iframe."""
    body = request.get_json(force=True) or {}
    raw  = body.get("raw", {})
    log.info(f"[onshape-postmsg] origin={body.get('origin','')} raw={json.dumps(raw)}")
    return jsonify({"status": "ok"})


@app.route("/debug-context", methods=["POST"])
def debug_context():
    """Log the URL params, referrer, and postMessage context that Onshape sent on sidebar load."""
    body = request.get_json(force=True) or {}
    entry = {
        "ts":            datetime.datetime.now(datetime.UTC).isoformat(),
        "url_params":    body.get("urlParams", {}),
        "referrer":      body.get("referrer", ""),
        "in_onshape":    body.get("inOnshape", False),
        "detected_ids":  body.get("detectedIds", {}),
        "postmsg_ctx":   body.get("postMsgCtx", None),
    }
    log.info(f"[debug-context] {json.dumps(entry)}")
    return jsonify({"status": "ok", "received": entry})


_startup_rebuild_reports()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
