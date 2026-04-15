DesignLoop
==========

A Flask web application used as an AI-assisted design tool embedded in Onshape as a sidebar app.
Built for a MIE498 research thesis studying the influence of AI on creativity in iterative CAD
design (University of Toronto, 2026).

The app gives designers a structured loop:
  1. Describe  — write a description of the current design state
  2. Analyse   — GPT-4.1-mini reads the description + live CAD screenshots and returns 5 next-step suggestions
  3. Generate  — user picks a direction; GPT-4.1 grounds it against the actual geometry; gpt-image-1 renders a concept sketch

Each session is logged to JSONL files for semantic analysis: descriptions, AI outputs, generated
images, and text-embedding-3-small vectors.


Setup
-----

    pip install -r requirements.txt
    cp .env.example .env
    # Fill in your API keys in .env
    python app.py

The app runs at http://localhost:5000. To embed it in Onshape, expose it via ngrok
(https://ngrok.com/) and register it as an Application Extension in the Onshape Developer Portal
(https://dev-portal.onshape.com).


Files
-----

  app.py                     Flask backend — Onshape API integration, OpenAI calls, session logging, report generation
  templates/index.html       Single-page frontend — three-panel UI (Model / Analyse / Generate)
  analysis.py                Offline analysis script — reads the JSONL logs and computes semantic trajectory metrics
  onshape_audit_analysis.py  Fetches Onshape document history to extract CAD action counts per iteration (Layer 1)
  glassworks_urls.py         Helper to generate Onshape Glassworks API URLs for manually fetching document history
  capture_screenshots.py     Playwright automation for taking report screenshots of the full app flow
  prompts/analyse.txt        System prompt for the GPT-4.1-mini analysis step
  prompts/vision.txt         System prompt for the GPT-4.1 vision -> image prompt generation step
  ldraw/                     LDraw -> STEP converters for importing LEGO Technic parts into Onshape


Analysis Pipeline
-----------------

After running study sessions, use analysis.py to compute semantic metrics:

    python analysis.py --document-id <24-char-onshape-doc-id> --export-html

Outputs to data/analysis_<timestamp>/:
  description_drift.csv          semantic distance between consecutive iteration descriptions
  direction_novelty.csv          novelty score per modification direction
  user_ai_coupling.csv           cosine similarity between user input and AI output per event
  iteration_similarity_matrix.json  NxN cross-iteration similarity
  report.html                    standalone HTML report with charts

For Layer 1 behavioral analysis (Onshape CAD action counts), run onshape_audit_analysis.py after
placing raw document history JSON files in data/audit_logs/.


Data Structure
--------------

  data/
    session_events.jsonl    append-only event log
    embeddings.jsonl        embedding vectors, fully denormalized
    iteration_store.json    per-document iteration state
    images/                 CAD view PNGs and generated concept images
    view_cache/             cached Onshape view renders
    reports/                per-document HTML session reports

Each embedding record in embeddings.jsonl contains: document_id, iteration_number, event_type,
field (user_description / ai_analysis / user_direction / image_prompt), text, and the 1536-dim vector.


LDraw Tools
-----------

ldraw/ldraw_to_step_direct.py converts LDraw .dat files to STEP with true analytical geometry
(cylinders, planes) rather than tessellated triangles — required for Onshape to snap mates to
cylindrical holes.

ldraw/ldraw_to_step_pipeline.py automates the full batch pipeline via LDView + FreeCAD.

Edit the CONFIG block at the top of each file to point to your local LDraw library and output
directories.
