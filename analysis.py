"""DesignLoop Analysis Pipeline.

Reads JSONL event log + embeddings from the app and generates:
  - Per-iteration embedding centroids and cross-iteration similarity matrix
  - Description drift, direction novelty, user↔AI coupling metrics
  - Sequential AI drift analysis and user input similarity
  - CSV + JSON exports and standalone HTML report with charts

Usage:
  python analysis.py                            # all documents
  python analysis.py --document-id <24-char-id>
  python analysis.py --export-html
  python analysis.py --data-dir /path/to/data --export-html
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math


# Data loading

def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    if not filepath.exists():
        print(f"  ⚠  File not found: {filepath}")
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠  Skipping malformed line {i} in {filepath.name}: {e}")
    return records


def load_events(data_dir: Path) -> list[dict]:
    return load_jsonl(data_dir / "session_events.jsonl")


def load_embeddings(data_dir: Path) -> list[dict]:
    return load_jsonl(data_dir / "embeddings.jsonl")


# Vector math (pure Python)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    c = [0.0] * dim
    for v in vectors:
        for idx in range(dim):
            c[idx] += v[idx]
    n = len(vectors)
    return [x / n for x in c]


# Analyzer

class DesignLoopAnalyzer:
    """Analyzes logged research data from DesignLoop sessions.

    Embedding records contain denormalized fields (document_id,
    iteration_number, event_type) so every analysis can group by
    iteration directly. Legacy join-path (event_id) is preserved.
    """

    def __init__(self, data_dir: Path, document_id: str = ""):
        self.data_dir    = data_dir
        self.doc_filter  = document_id

        self.events          = load_events(data_dir)
        self.embeddings_raw  = load_embeddings(data_dir)

        # Apply document filter
        if self.doc_filter:
            self.events = [
                e for e in self.events
                if e.get("document_id", "") == self.doc_filter
            ]
            self.embeddings_raw = [
                e for e in self.embeddings_raw
                if e.get("document_id", "") == self.doc_filter
                or not e.get("document_id")   # keep legacy records without doc_id
            ]

        # Index embeddings by event_id+field (for legacy join path)
        self._emb_by_event: dict[str, dict[str, dict]] = defaultdict(dict)
        for emb in self.embeddings_raw:
            eid   = emb.get("event_id", "")
            field = emb.get("field", "")
            self._emb_by_event[eid][field] = emb

        # Group embeddings by (iteration_number, field)
        # Uses denormalized fields when present; falls back to joining via events.
        self._emb_by_iter: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        event_iter_map = {e["event_id"]: e.get("iteration_number", 0) for e in self.events}

        for emb in self.embeddings_raw:
            vec = emb.get("vector", [])
            if not vec:
                continue
            # Prefer the field stored on the record itself
            it = emb.get("iteration_number")
            if it is None:
                it = event_iter_map.get(emb.get("event_id", ""), 0)
            it    = int(it) if it else 0
            field = emb.get("field", "unknown")
            self._emb_by_iter[it][field].append({
                "vector": vec,
                "text":   emb.get("text", ""),
                "ts":     emb.get("ts", ""),
                "event_id": emb.get("event_id", ""),
            })

        self.iterations_sorted = sorted(self._emb_by_iter.keys())

        # Categorised event lists
        self.analyses  = [e for e in self.events if e.get("event") == "ANALYSIS"]
        self.nextsteps = [e for e in self.events if e.get("event") == "NEXT_STEP_GENERATED"]
        self.view_loads = [e for e in self.events if e.get("event") == "VIEW_LOAD"]

        print(f"\n  Loaded {len(self.events)} events, {len(self.embeddings_raw)} embeddings")
        if self.doc_filter:
            print(f"  (filtered to document: {self.doc_filter})")
        print(f"  ├─ ANALYSIS events:       {len(self.analyses)}")
        print(f"  ├─ NEXT_STEP events:      {len(self.nextsteps)}")
        print(f"  ├─ VIEW_LOAD events:      {len(self.view_loads)}")
        print(f"  └─ Iterations found:      {self.iterations_sorted}")

    def iteration_similarity_matrix(self) -> dict:
        """Build centroid for each iteration and compute NxN cosine similarity matrix.

        High off-diagonal values mean semantically similar design directions.
        Low values mean substantial design shift between iterations.
        Also returns per-iteration field breakdowns.
        """
        per_iter = []
        centroids = []

        for it in self.iterations_sorted:
            fields = self._emb_by_iter[it]
            desc_vecs = [r["vector"] for r in fields.get("user_description", [])]
            dir_vecs  = [r["vector"] for r in fields.get("user_direction",   [])]
            ai_vecs   = [r["vector"] for r in fields.get("ai_analysis",      [])]
            img_vecs  = [r["vector"] for r in fields.get("image_prompt",     [])]
            all_vecs  = desc_vecs + dir_vecs + ai_vecs + img_vecs
            c         = centroid(all_vecs)
            desc_c    = centroid(desc_vecs) if desc_vecs else []
            centroids.append(c)

            per_iter.append({
                "iteration_number":   it,
                "n_descriptions":     len(desc_vecs),
                "n_directions":       len(dir_vecs),
                "n_ai_analyses":      len(ai_vecs),
                "n_image_prompts":    len(img_vecs),
                "n_total_embeddings": len(all_vecs),
                "desc_centroid":      desc_c,   # used for description_drift
            })

        # NxN similarity matrix
        n = len(centroids)
        matrix = []
        for row_idx in range(n):
            row = []
            for col_idx in range(n):
                ca = centroids[row_idx]
                cb = centroids[col_idx]
                sim = round(cosine_similarity(ca, cb), 5) if (ca and cb) else None
                row.append(sim)
            matrix.append(row)

        return {
            "iterations":      self.iterations_sorted,
            "per_iteration":   per_iter,
            "similarity_matrix": matrix,
            "note": (
                "matrix[i][j] = cosine similarity between the centroid of all "
                "embeddings in iteration i and iteration j. "
                "1.0 = identical semantic space, 0.0 = orthogonal."
            ),
        }

    def description_drift(self) -> dict:
        """Compute cosine similarity between user_description centroids for consecutive iterations.

        similarity → 1.0 means the user described the design almost identically (low cognitive shift).
        similarity → 0.0 means substantial language/concept change (high semantic drift).
        semantic_distance = 1 - cosine_sim is reported for intuitive charting.
        """
        drift = []
        for idx in range(len(self.iterations_sorted) - 1):
            it_a = self.iterations_sorted[idx]
            it_b = self.iterations_sorted[idx + 1]
            ca = self._iter_desc_centroid(it_a)
            cb = self._iter_desc_centroid(it_b)
            if ca and cb:
                sim = cosine_similarity(ca, cb)
                drift.append({
                    "from_iteration":   it_a,
                    "to_iteration":     it_b,
                    "cosine_sim":       round(sim, 5),
                    "semantic_distance": round(1 - sim, 5),
                })

        return {
            "drift": drift,
            "note": (
                "semantic_distance = 1 - cosine_sim. Higher = more conceptual "
                "shift between iterations. Values near 0 suggest design fixation."
            ),
        }

    def _iter_desc_centroid(self, it: int) -> list[float]:
        vecs = [r["vector"] for r in self._emb_by_iter[it].get("user_description", [])]
        return centroid(vecs)

    def direction_novelty(self) -> dict:
        """Compute each user_direction's novelty as 1 minus max similarity to prior directions.

        novelty_score → 1.0 means completely new direction.
        novelty_score → 0.0 means exact semantic repeat (design fixation).
        Operationalizes creative exploration breadth for AI-assisted design studies.
        """
        # Collect all direction records across iterations in chronological order
        all_dirs = []
        for it in self.iterations_sorted:
            for rec in self._emb_by_iter[it].get("user_direction", []):
                all_dirs.append({"iteration": it, **rec})
        all_dirs.sort(key=lambda r: r.get("ts", ""))

        novelty = []
        seen_vecs: list[list[float]] = []
        for rec in all_dirs:
            vec = rec["vector"]
            if seen_vecs:
                max_sim = max(cosine_similarity(vec, prev) for prev in seen_vecs)
            else:
                max_sim = 0.0   # first direction is always novel
            novelty.append({
                "iteration_number":  rec["iteration"],
                "text_preview":      rec["text"][:160],
                "ts":                rec.get("ts", ""),
                "max_sim_to_prior":  round(max_sim, 5),
                "novelty_score":     round(1 - max_sim, 5),
            })
            seen_vecs.append(vec)

        avg_novelty = (
            sum(r["novelty_score"] for r in novelty) / len(novelty) if novelty else 0
        )
        return {
            "directions":    novelty,
            "avg_novelty":   round(avg_novelty, 5),
            "note": (
                "novelty_score = 1 - max_cosine_sim to any prior direction. "
                "Average novelty across the full session is 'avg_novelty'. "
                "High average → broad exploration. Low → design fixation."
            ),
        }

    def user_ai_coupling(self) -> dict:
        """Compute cosine similarity between user input and AI output within each event.

        For ANALYSIS: cos(user_description, ai_analysis)
        For NEXT_STEP: cos(user_direction, image_prompt)
        High coupling = AI stays close to user framing.
        Low coupling = AI introduces novel semantic content.
        """
        couplings = []

        for ev in self.analyses:
            eid      = ev.get("event_id", "")
            user_emb = self._emb_by_event[eid].get("user_description", {}).get("vector", [])
            ai_emb   = self._emb_by_event[eid].get("ai_analysis",      {}).get("vector", [])
            if user_emb and ai_emb:
                couplings.append({
                    "event":            "ANALYSIS",
                    "event_id":         eid,
                    "iteration_number": ev.get("iteration_number", 0),
                    "ts":               ev.get("ts", ""),
                    "user_text":        ev.get("user_description", "")[:100],
                    "coupling":         round(cosine_similarity(user_emb, ai_emb), 5),
                })

        for ev in self.nextsteps:
            eid      = ev.get("event_id", "")
            user_emb = self._emb_by_event[eid].get("user_direction", {}).get("vector", [])
            ai_emb   = self._emb_by_event[eid].get("image_prompt",   {}).get("vector", [])
            if user_emb and ai_emb:
                couplings.append({
                    "event":            "NEXT_STEP",
                    "event_id":         eid,
                    "iteration_number": ev.get("iteration_number", 0),
                    "ts":               ev.get("ts", ""),
                    "user_text":        ev.get("direction", "")[:100],
                    "coupling":         round(cosine_similarity(user_emb, ai_emb), 5),
                })

        couplings.sort(key=lambda c: c["ts"])
        avg = sum(c["coupling"] for c in couplings) / len(couplings) if couplings else 0

        return {
            "couplings":        couplings,
            "average_coupling": round(avg, 5),
            "count":            len(couplings),
            "note": (
                "coupling = cosine_sim(user_input, ai_output) within the same event. "
                "High coupling → AI stays close to user framing. "
                "Low coupling → AI introduces novel semantic content."
            ),
        }

    def ai_output_drift(self) -> dict:
        """Track how AI outputs evolve across the session.

        Computes sequential similarity, drift from initial, and centroid spread.
        """
        ai_vecs = []
        for ev in self.analyses:
            eid = ev.get("event_id", "")
            vec = self._emb_by_event[eid].get("ai_analysis", {}).get("vector", [])
            if vec:
                ai_vecs.append({"ts": ev.get("ts", ""), "type": "analysis",
                                 "iter": ev.get("iteration_number", 0), "vector": vec})

        for ev in self.nextsteps:
            eid = ev.get("event_id", "")
            vec = self._emb_by_event[eid].get("image_prompt", {}).get("vector", [])
            if vec:
                ai_vecs.append({"ts": ev.get("ts", ""), "type": "image_prompt",
                                 "iter": ev.get("iteration_number", 0), "vector": vec})

        ai_vecs.sort(key=lambda x: x["ts"])
        if len(ai_vecs) < 2:
            return {"sequential": [], "drift_from_initial": [],
                    "centroid_spread": 0, "note": "Need ≥2 AI outputs"}

        sequential = []
        for idx in range(1, len(ai_vecs)):
            sim = cosine_similarity(ai_vecs[idx - 1]["vector"], ai_vecs[idx]["vector"])
            sequential.append({
                "step": idx,
                "from_type": ai_vecs[idx - 1]["type"],
                "to_type":   ai_vecs[idx]["type"],
                "from_iter": ai_vecs[idx - 1]["iter"],
                "to_iter":   ai_vecs[idx]["iter"],
                "cosine_sim": round(sim, 5),
            })

        first_vec = ai_vecs[0]["vector"]
        drift_from_initial = [
            {"step": k, "type": v["type"], "iter": v["iter"],
             "sim_to_first": round(cosine_similarity(first_vec, v["vector"]), 5)}
            for k, v in enumerate(ai_vecs)
        ]

        vectors = [v["vector"] for v in ai_vecs]
        c = centroid(vectors)
        distances = [euclidean_distance(v, c) for v in vectors]
        avg_dist = sum(distances) / len(distances)

        return {
            "sequential_similarity": sequential,
            "drift_from_initial":    drift_from_initial,
            "centroid_spread":       round(avg_dist, 5),
            "total_ai_outputs":      len(ai_vecs),
        }

    def user_input_similarity(self) -> dict:
        """Pairwise cosine similarity between all user inputs (descriptions + directions)."""
        all_user = []
        for ev in self.analyses:
            eid = ev.get("event_id", "")
            vec = self._emb_by_event[eid].get("user_description", {}).get("vector", [])
            if vec:
                all_user.append({
                    "type": "description", "vector": vec,
                    "text": ev.get("user_description", "")[:100],
                    "ts":   ev.get("ts", ""),
                    "iter": ev.get("iteration_number", 0),
                })
        for ev in self.nextsteps:
            eid = ev.get("event_id", "")
            vec = self._emb_by_event[eid].get("user_direction", {}).get("vector", [])
            if vec:
                all_user.append({
                    "type": "direction", "vector": vec,
                    "text": ev.get("direction", "")[:100],
                    "ts":   ev.get("ts", ""),
                    "iter": ev.get("iteration_number", 0),
                })

        if len(all_user) < 2:
            return {"pairs": [], "note": "Need ≥2 user inputs with embeddings"}

        pairs = []
        for aa in range(len(all_user)):
            for bb in range(aa + 1, len(all_user)):
                sim = cosine_similarity(all_user[aa]["vector"], all_user[bb]["vector"])
                pairs.append({
                    "a_type": all_user[aa]["type"], "a_iter": all_user[aa]["iter"],
                    "a_text": all_user[aa]["text"],  "a_ts":   all_user[aa]["ts"],
                    "b_type": all_user[bb]["type"], "b_iter": all_user[bb]["iter"],
                    "b_text": all_user[bb]["text"],  "b_ts":   all_user[bb]["ts"],
                    "cosine_similarity": round(sim, 5),
                })

        pairs.sort(key=lambda p: p["cosine_similarity"], reverse=True)
        return {"pairs": pairs, "count": len(pairs)}

    def session_timeline(self) -> list[dict]:
        """Build a chronological event timeline with relevant fields per event type."""
        timeline = []
        for ev in sorted(self.events, key=lambda e: e.get("ts", "")):
            entry = {
                "ts":               ev.get("ts", ""),
                "event":            ev.get("event", ""),
                "event_id":         ev.get("event_id", ""),
                "iteration_number": ev.get("iteration_number", 0),
                "session":          ev.get("session_id", ""),
            }
            et = ev.get("event", "")
            if et == "ANALYSIS":
                entry["user_input"] = ev.get("user_description", "")[:120]
                entry["ai_length"]  = ev.get("ai_response_length", 0)
            elif et == "NEXT_STEP_GENERATED":
                entry["direction"]     = ev.get("direction", "")[:120]
                entry["prompt_length"] = ev.get("prompt_length", 0)
            elif et == "CONTEXT_SET":
                entry["document_id"] = ev.get("document_id", "")
                entry["source"]      = ev.get("source", "")
            timeline.append(entry)
        return timeline

    def export_all(self, output_dir: Path = None) -> dict:
        """Run all analyses and export JSON, CSV, and HTML report."""
        out = output_dir or (self.data_dir / "analysis_output")
        out.mkdir(exist_ok=True)
        print("\n  Running analyses…\n")

        iter_sim   = self.iteration_similarity_matrix()
        desc_drift = self.description_drift()
        dir_nov    = self.direction_novelty()
        coupling   = self.user_ai_coupling()
        ai_drift   = self.ai_output_drift()
        user_sim   = self.user_input_similarity()
        timeline   = self.session_timeline()

        # JSON dumps
        self._write_json(out / "iteration_similarity_matrix.json", iter_sim)
        self._write_json(out / "description_drift.json",           desc_drift)
        self._write_json(out / "direction_novelty.json",           dir_nov)
        self._write_json(out / "user_ai_coupling.json",            coupling)
        self._write_json(out / "ai_output_drift.json",             ai_drift)
        self._write_json(out / "user_input_similarity.json",       user_sim)
        self._write_json(out / "session_timeline.json",            timeline)

        # CSV exports (flat tables only)
        if iter_sim.get("per_iteration"):
            # Drop centroid vectors from CSV (too wide)
            flat = [{k: v for k, v in row.items() if k != "desc_centroid"}
                    for row in iter_sim["per_iteration"]]
            self._write_csv(out / "per_iteration_summary.csv", flat)
        if desc_drift.get("drift"):
            self._write_csv(out / "description_drift.csv", desc_drift["drift"])
        if dir_nov.get("directions"):
            self._write_csv(out / "direction_novelty.csv", dir_nov["directions"])
        if coupling.get("couplings"):
            self._write_csv(out / "user_ai_coupling.csv", coupling["couplings"])
        if ai_drift.get("sequential_similarity"):
            self._write_csv(out / "ai_sequential_similarity.csv",
                            ai_drift["sequential_similarity"])
        if user_sim.get("pairs"):
            self._write_csv(out / "user_input_similarity.csv", user_sim["pairs"])
        self._write_csv(out / "session_timeline.csv", timeline)

        print(f"  ✓  Iteration similarity matrix — {len(self.iterations_sorted)} iterations")
        print(f"  ✓  Description drift          — {len(desc_drift.get('drift',[]))} transitions")
        avg_nov = dir_nov.get('avg_novelty', 'n/a')
        print(f"  ✓  Direction novelty           — avg_novelty={avg_nov}")
        avg_cpl = coupling.get('average_coupling', 'n/a')
        print(f"  ✓  User↔AI coupling            — avg={avg_cpl}")
        spread  = ai_drift.get('centroid_spread', 'n/a')
        print(f"  ✓  AI output drift             — spread={spread}")
        print(f"  ✓  Session timeline            — {len(timeline)} events")

        summary = self._build_summary(iter_sim, desc_drift, dir_nov, coupling, ai_drift)
        self._write_json(out / "summary.json", summary)
        print(f"\n  Summary")
        for k, v in summary.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
        print(f"\n  All outputs saved to: {out}/\n")
        return summary

    def _build_summary(self, iter_sim, desc_drift, dir_nov, coupling, ai_drift) -> dict:
        drift_vals = [d["semantic_distance"] for d in desc_drift.get("drift", [])]
        return {
            "total_events":          len(self.events),
            "total_embeddings":      len(self.embeddings_raw),
            "iterations_found":      self.iterations_sorted,
            "analysis_count":        len(self.analyses),
            "generation_count":      len(self.nextsteps),
            "avg_description_drift": round(sum(drift_vals) / len(drift_vals), 5) if drift_vals else None,
            "avg_direction_novelty": dir_nov.get("avg_novelty"),
            "avg_user_ai_coupling":  coupling.get("average_coupling"),
            "ai_centroid_spread":    ai_drift.get("centroid_spread"),
            "generated_at":          datetime.utcnow().isoformat() + "Z",
        }

    def export_html_report(self, output_dir: Path = None):
        """Generate standalone HTML report with charts."""
        out = output_dir or (self.data_dir / "analysis_output")
        out.mkdir(exist_ok=True)

        iter_sim   = self.iteration_similarity_matrix()
        desc_drift = self.description_drift()
        dir_nov    = self.direction_novelty()
        coupling   = self.user_ai_coupling()
        ai_drift   = self.ai_output_drift()
        timeline   = self.session_timeline()

        # Chart datasets
        # 1. Cross-iteration similarity heatmap data (flat list for Chart.js bubble)
        iters      = iter_sim["iterations"]
        matrix     = iter_sim["similarity_matrix"]
        heatmap_data = []
        for ri, row in enumerate(matrix):
            for ci, val in enumerate(row):
                if val is not None:
                    heatmap_data.append({"x": iters[ci], "y": iters[ri], "v": val})

        # 2. Description drift
        drift_labels = [f"{d['from_iteration']}→{d['to_iteration']}"
                        for d in desc_drift.get("drift", [])]
        drift_values = [d["semantic_distance"] for d in desc_drift.get("drift", [])]

        # 3. Direction novelty
        nov_labels = [f"It{r['iteration_number']} #{ri+1}" for ri, r in
                      enumerate(dir_nov.get("directions", []))]
        nov_values = [r["novelty_score"] for r in dir_nov.get("directions", [])]

        # 4. Coupling over time
        coup_labels = [f"{c['event']} It{c['iteration_number']}" for c in
                       coupling.get("couplings", [])]
        coup_values = [c["coupling"] for c in coupling.get("couplings", [])]

        # 5. Sequential AI similarity
        seq_labels = [f"Step {s['step']} (It{s['to_iter']})" for s in
                      ai_drift.get("sequential_similarity", [])]
        seq_values = [s["cosine_sim"] for s in ai_drift.get("sequential_similarity", [])]

        # Matrix HTML table
        if iters:
            mat_header = "<tr><th>It\\</th>" + "".join(f"<th>{i}</th>" for i in iters) + "</tr>"
            mat_rows   = ""
            for ri, row in enumerate(matrix):
                cells = ""
                for ci, val in enumerate(row):
                    if val is None:
                        cells += "<td>–</td>"
                    else:
                        # Colour cells: green=high, grey=low
                        g = int(val * 180)
                        bg = f"rgb({40},{g},{40})" if ri != ci else "#2a3a2a"
                        cells += f'<td style="background:{bg};color:#fff">{val:.3f}</td>'
                mat_rows += f"<tr><th>{iters[ri]}</th>{cells}</tr>"
            matrix_table = f"<table class='mat'>{mat_header}{mat_rows}</table>"
        else:
            matrix_table = "<p style='color:#666'>No iteration data yet.</p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>DesignLoop Iteration Analysis</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0e0e11;
          color: #d4d4dc; padding: 40px; max-width: 960px; margin: 0 auto; }}
  h1   {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  h2   {{ font-size: 15px; font-weight: 600; margin-top: 36px; color: #a8e847; }}
  .sub {{ color: #555; font-size: 12px; margin-bottom: 28px; }}
  .metric {{ display:inline-block; background:#1a1a1f; border:1px solid #2a2a32;
             border-radius:6px; padding:14px 20px; margin:6px 8px 6px 0; }}
  .metric .val {{ font-size:22px; font-weight:700; color:#a8e847; }}
  .metric .lbl {{ font-size:10px; color:#666; text-transform:uppercase;
                   letter-spacing:.05em; margin-top:4px; }}
  .chart-wrap {{ background:#141417; border:1px solid #2a2a32; border-radius:6px;
                  padding:20px; margin-top:10px; }}
  canvas {{ width:100% !important; height:220px !important; }}
  .note {{ font-size:11px; color:#555; margin-top:6px; font-style:italic; }}
  table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:12px; }}
  table.mat td, table.mat th {{ padding:6px 10px; border:1px solid #1f1f24;
                                  text-align:center; }}
  table.mat th {{ background:#1a1a1f; color:#888; }}
  th {{ text-align:left; padding:7px 10px; background:#1a1a1f; color:#888;
        font-size:10px; text-transform:uppercase; letter-spacing:.04em; }}
  td {{ padding:6px 10px; border-bottom:1px solid #1f1f24; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
<h1>DesignLoop — Iteration Analysis</h1>
<p class="sub">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
{' · document: ' + self.doc_filter if self.doc_filter else ''}</p>

<div>
  <div class="metric"><div class="val">{len(self.iterations_sorted)}</div><div class="lbl">Iterations</div></div>
  <div class="metric"><div class="val">{len(self.analyses)}</div><div class="lbl">Analyses</div></div>
  <div class="metric"><div class="val">{len(self.nextsteps)}</div><div class="lbl">Generations</div></div>
  <div class="metric"><div class="val">{dir_nov.get('avg_novelty', 0):.3f}</div><div class="lbl">Avg Direction Novelty</div></div>
  <div class="metric"><div class="val">{coupling.get('average_coupling', 0):.3f}</div><div class="lbl">Avg User↔AI Coupling</div></div>
</div>

<h2>Cross-Iteration Similarity Matrix</h2>
<p class="note">Each cell = cosine similarity between the centroid of all embeddings in iteration i vs j.
  Green = semantically similar. Diagonal is always 1.0 (self-similarity).</p>
{matrix_table}

<h2>Description Drift Between Iterations</h2>
<p class="note">semantic_distance = 1 − cosine_sim between consecutive iteration descriptions.
  Higher = the user's concept shifted more.</p>
<div class="chart-wrap">
  <canvas id="driftChart"></canvas>
</div>

<h2>Direction Novelty Over Session</h2>
<p class="note">novelty_score = 1 − max similarity to any prior direction.
  1.0 = entirely new territory. 0.0 = exact repeat.</p>
<div class="chart-wrap">
  <canvas id="novChart"></canvas>
</div>

<h2>User ↔ AI Coupling</h2>
<p class="note">cosine_sim between each user input and the AI's corresponding output.
  High = AI stayed close to user framing.</p>
<div class="chart-wrap">
  <canvas id="coupChart"></canvas>
</div>

<h2>Sequential AI Similarity</h2>
<p class="note">cosine_sim between consecutive AI outputs.
  Low = AI changed substantially between steps.</p>
<div class="chart-wrap">
  <canvas id="seqChart"></canvas>
</div>

<h2>Direction Novelty Detail</h2>
<table>
<tr><th>Iter</th><th>Direction</th><th>Max Sim to Prior</th><th>Novelty</th></tr>
{''.join(f'<tr><td>{r["iteration_number"]}</td><td>{r["text_preview"][:100]}</td><td>{r["max_sim_to_prior"]:.4f}</td><td>{r["novelty_score"]:.4f}</td></tr>' for r in dir_nov.get("directions", []))}
</table>

<h2>Event Timeline</h2>
<table>
<tr><th>Time</th><th>Iter</th><th>Event</th><th>Detail</th></tr>
{''.join(f'<tr><td>{ev.get("ts","")[:19]}</td><td>{ev.get("iteration_number","")}</td><td>{ev.get("event","")}</td><td>{ev.get("user_input","") or ev.get("direction","") or ev.get("document_id","")}</td></tr>' for ev in timeline[:60])}
</table>

<script>
function bar(id, labels, data, label, color) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{ labels, datasets: [{{ label, data, backgroundColor: color + 'aa',
                                   borderColor: color, borderWidth: 1 }}] }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ grid: {{ color: '#1f1f24' }}, ticks: {{ color: '#666', maxRotation: 45 }} }},
        y: {{ min: 0, max: 1, grid: {{ color: '#1f1f24' }}, ticks: {{ color: '#666' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#888' }} }} }}
    }}
  }});
}}
bar('driftChart',  {json.dumps(drift_labels)}, {json.dumps(drift_values)},  'Semantic Distance', '#e85050');
bar('novChart',    {json.dumps(nov_labels)},   {json.dumps(nov_values)},    'Novelty Score',     '#a8e847');
bar('coupChart',   {json.dumps(coup_labels)},  {json.dumps(coup_values)},   'User↔AI Coupling',  '#47b8e8');
bar('seqChart',    {json.dumps(seq_labels)},   {json.dumps(seq_values)},    'Sequential Sim',    '#e8a847');
</script>
</body>
</html>"""

        report_path = out / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  ✓  HTML report saved to: {report_path}")

    # I/O helpers

    @staticmethod
    def _write_json(path: Path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


# CLI

def main():
    parser = argparse.ArgumentParser(description="DesignLoop Analysis Pipeline")
    parser.add_argument("--data-dir",    default="./data",
                        help="Path to the data directory (default: ./data)")
    parser.add_argument("--output-dir",  default=None,
                        help="Output directory (default: <data-dir>/analysis_output)")
    parser.add_argument("--document-id", default="",
                        help="Filter to a specific Onshape document ID")
    parser.add_argument("--export-html", action="store_true",
                        help="Also generate a standalone HTML report")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not data_dir.exists():
        print(f"\n  ✗  Data directory not found: {data_dir}")
        print(f"     Run the DesignLoop app first to generate data.\n")
        sys.exit(1)

    print(f"\n  DesignLoop Analysis Pipeline")
    print(f"  {'─' * 40}")
    print(f"  Data dir: {data_dir.resolve()}")

    analyzer = DesignLoopAnalyzer(data_dir, document_id=args.document_id)
    analyzer.export_all(output_dir)

    if args.export_html:
        analyzer.export_html_report(output_dir)

    print("  Done.\n")


if __name__ == "__main__":
    main()
