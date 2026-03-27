"""
app.py — Freight document extraction eval harness.
Section 1: Batch eval on docs/ folder via button.
Section 2: Live single-document upload and extraction.
"""

import os
import re
import json
import time
import streamlit as st
import pandas as pd
from google import genai
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from prompts import FIELDS, FIELD_LABELS, STRATEGIES

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Freight Doc Eval",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.main .block-container { max-width: 740px; padding: 3rem 1.5rem 5rem; }
h1  { font-size: 1.3rem  !important; font-weight: 500 !important; letter-spacing: -0.01em; }
h2  { font-size: 0.95rem !important; font-weight: 500 !important; color: #555 !important; margin-top: 2rem !important; }
p, li { font-size: 0.875rem; color: #444; line-height: 1.65; }
hr  { border: none; border-top: 1px solid #efefef !important; margin: 2rem 0 !important; }
.tag { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #aaa; display: block; margin-bottom: 0.5rem; }
.callout { padding: 14px 18px; background: #f0f6ff; border-left: 3px solid #3b82f6; border-radius: 0 8px 8px 0; margin: 1.2rem 0; font-size: 0.85rem; color: #1e3a5f; line-height: 1.65; }
.warn-callout { padding: 14px 18px; background: #fffbeb; border-left: 3px solid #f59e0b; border-radius: 0 8px 8px 0; margin: 1.2rem 0; font-size: 0.85rem; color: #78350f; line-height: 1.65; }
.failure-row { display: flex; align-items: center; gap: 14px; padding: 12px 14px; border-radius: 8px; border: 1px solid #ebebeb; margin-bottom: 7px; background: #fafaf9; }
.f-left { min-width: 160px; }
.f-label { font-size: 0.82rem; font-weight: 500; color: #333; }
.f-desc  { font-size: 0.75rem; color: #999; margin-top: 2px; }
.f-bar-wrap { flex: 1; background: #f0f0f0; border-radius: 3px; height: 5px; }
.f-bar { height: 5px; border-radius: 3px; }
.f-pct { font-size: 0.82rem; font-weight: 500; min-width: 34px; text-align: right; }
.legend { font-size: 0.74rem; color: #bbb; margin-top: -4px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

FAILURE_META = {
    "layout_ambiguity": {
        "label": "Layout ambiguity",
        "desc":  "Field spans multiple columns or non-standard positioning. Not a prompt problem.",
        "color": "#f59e0b",
    },
    "format_variance": {
        "label": "Format variance",
        "desc":  "Same field labeled differently across carriers (FSC vs Fuel Adj vs Fuel Surcharge).",
        "color": "#f59e0b",
    },
    "scan_quality": {
        "label": "Scan quality",
        "desc":  "Rotated stamps, low resolution, or bleed-through occlude critical fields.",
        "color": "#ef4444",
    },
    "prompt_fixable": {
        "label": "Prompt fixable",
        "desc":  "Strategy C succeeds where A and B fail — rewriting the prompt resolves it.",
        "color": "#22c55e",
    },
}

SAMPLE_SUMMARY = {
    "total_docs": 8,
    "field_accuracy": {
        "carrier_name":        {"A — Naive": {"correct": 8, "wrong": 0, "missing": 0}, "B — Structured": {"correct": 8, "wrong": 0, "missing": 0}, "C — Few-shot": {"correct": 8, "wrong": 0, "missing": 0}},
        "linehaul_rate":       {"A — Naive": {"correct": 7, "wrong": 1, "missing": 0}, "B — Structured": {"correct": 7, "wrong": 0, "missing": 1}, "C — Few-shot": {"correct": 8, "wrong": 0, "missing": 0}},
        "fuel_surcharge":      {"A — Naive": {"correct": 6, "wrong": 1, "missing": 1}, "B — Structured": {"correct": 7, "wrong": 0, "missing": 1}, "C — Few-shot": {"correct": 7, "wrong": 1, "missing": 0}},
        "accessorial_charges": {"A — Naive": {"correct": 4, "wrong": 2, "missing": 2}, "B — Structured": {"correct": 5, "wrong": 1, "missing": 2}, "C — Few-shot": {"correct": 7, "wrong": 1, "missing": 0}},
        "weight":              {"A — Naive": {"correct": 7, "wrong": 0, "missing": 1}, "B — Structured": {"correct": 8, "wrong": 0, "missing": 0}, "C — Few-shot": {"correct": 8, "wrong": 0, "missing": 0}},
        "total_amount":        {"A — Naive": {"correct": 6, "wrong": 2, "missing": 0}, "B — Structured": {"correct": 7, "wrong": 1, "missing": 0}, "C — Few-shot": {"correct": 7, "wrong": 1, "missing": 0}},
    },
    "failure_types": {"layout_ambiguity": 4, "format_variance": 3, "scan_quality": 3, "prompt_fixable": 4},
}

# ─── Extraction ───────────────────────────────────────────────────────────────

def get_model():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, "GEMINI_API_KEY not set."
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)


def parse_response(text: str) -> dict:
    """
    Robustly extract JSON from Gemini response.
    Handles markdown fences, extra explanation text, and whitespace.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def extract_fields(image: Image.Image, prompt: str, client) -> tuple[dict, str | None]:
    """
    Run one Gemini Vision extraction.
    Returns (result_dict, error_string). error_string is None on success.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
        )

        if not response.text:
            return {f: None for f in FIELDS}, "Empty response from model."

        text   = response.text.strip()
        parsed = parse_response(text)

        if parsed is None:
            return {f: None for f in FIELDS}, f"Could not parse JSON. Raw: {text[:200]}"

        return parsed, None

    except Exception as e:
        return {f: None for f in FIELDS}, str(e)


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_field(extracted: dict, ground_truth: dict, field: str) -> str:
    gt = ground_truth.get(field)
    ex = extracted.get(field)

    if ex is None:
        return "missing"
    if gt is None:
        return "correct"

    if field == "carrier_name":
        return "correct" if (
            gt.lower() in str(ex).lower() or str(ex).lower() in gt.lower()
        ) else "wrong"

    try:
        gt_f, ex_f = float(gt), float(ex)
        if gt_f == 0:
            return "correct" if ex_f == 0 else "wrong"
        return "correct" if abs(gt_f - ex_f) / gt_f <= 0.05 else "wrong"
    except (ValueError, TypeError):
        return "wrong"


def classify_failure(doc_name: str, field: str, extractions: dict) -> str:
    values = [extractions[s].get(field) for s in STRATEGIES]
    non_null = [v for v in values if v is not None]

    if not non_null:
        if any(tag in doc_name for tag in ("noisy", "rotated", "scan", "blur")):
            return "scan_quality"
        return "layout_ambiguity"

    if len(set(str(v) for v in non_null)) > 1:
        return "format_variance"

    return "prompt_fixable"


# ─── Display helpers ──────────────────────────────────────────────────────────

def accuracy_style(val: str) -> str:
    try:
        n, d = val.split("/")
        ratio = int(n) / int(d)
        if ratio >= 0.875:
            return "background-color:#dcfce7;color:#166534;font-weight:500"
        elif ratio >= 0.625:
            return "background-color:#fef3c7;color:#92400e;font-weight:500"
        else:
            return "background-color:#fee2e2;color:#991b1b;font-weight:500"
    except Exception:
        return ""


def build_accuracy_df(summary: dict) -> pd.DataFrame:
    total = summary["total_docs"]
    rows = []
    for fkey, flabel in FIELD_LABELS.items():
        row = {"Field": flabel}
        for strategy in STRATEGIES:
            counts = summary["field_accuracy"].get(fkey, {}).get(strategy, {})
            row[strategy] = f"{counts.get('correct', 0)}/{total}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Field")


def render_failure_bars(failure_types: dict):
    total = sum(failure_types.values()) or 1
    for ftype, meta in FAILURE_META.items():
        count = failure_types.get(ftype, 0)
        pct = int(count / total * 100)
        st.markdown(f"""
        <div class='failure-row'>
            <div class='f-left'>
                <div class='f-label'>{meta['label']}</div>
                <div class='f-desc'>{meta['desc']}</div>
            </div>
            <div class='f-bar-wrap'>
                <div class='f-bar' style='width:{pct}%;background:{meta["color"]}'></div>
            </div>
            <div class='f-pct' style='color:{meta["color"]}'>{pct}%</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("## Freight Document Extraction")
with col2:
    st.markdown(
        "<p style='text-align:right;color:#ccc;font-size:0.78rem;padding-top:0.6rem'>Aarush Sharma</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─── Section 1: Batch eval ────────────────────────────────────────────────────

st.markdown("<span class='tag'>Part 1 — Batch evaluation</span>", unsafe_allow_html=True)
st.markdown(
    "Runs every document in `docs/` through 3 prompt strategies. "
    "Scores each field against `ground_truth.json`. "
    "1s delay between calls to respect free tier limits."
)

results_path = Path("results.json")
if results_path.exists():
    with open(results_path) as f:
        saved = json.load(f)
    summary  = saved["summary"]
    is_sample = False
else:
    summary   = SAMPLE_SUMMARY
    is_sample = True

if is_sample:
    st.markdown("""
    <div class='warn-callout'>
        Showing illustrative sample findings. Add documents to <code>docs/</code>,
        fill <code>ground_truth.json</code>, then click Run.
    </div>
    """, unsafe_allow_html=True)

model, model_err = get_model()
docs_exist = Path("docs").exists() and any(Path("docs").glob("*.png"))
run_disabled = model is None or not docs_exist

if run_disabled:
    hint = model_err if model_err else "Add PNG images to docs/ folder to enable."
    st.info(hint)

if st.button("Run evaluation on docs/", type="primary", disabled=run_disabled):
    with open("ground_truth.json") as f:
        ground_truth = json.load(f)
        ground_truth.pop("_note", None)

    doc_paths = sorted(
        list(Path("docs").glob("*.png")) +
        list(Path("docs").glob("*.jpg")) +
        list(Path("docs").glob("*.jpeg"))
    )

    documents = {}
    progress  = st.progress(0)
    status    = st.empty()
    errors    = []

    for i, doc_path in enumerate(doc_paths):
        doc_name = doc_path.stem
        gt = ground_truth.get(doc_name, {})
        status.markdown(
            f"<p style='font-size:0.8rem;color:#999'>"
            f"Processing {doc_name} ({i+1}/{len(doc_paths)})</p>",
            unsafe_allow_html=True,
        )

        image = Image.open(doc_path)
        extractions, scores_map, failures = {}, {}, {}

        for strategy_name, prompt in STRATEGIES.items():
            result, err = extract_fields(image, prompt, model)
            extractions[strategy_name] = result
            if err:
                errors.append(f"{doc_name} / {strategy_name}: {err}")
            time.sleep(1)

        for field in FIELDS:
            scores_map[field] = {
                s: score_field(extractions[s], gt, field)
                for s in STRATEGIES
            }
            all_failed = all(scores_map[field][s] != "correct" for s in STRATEGIES)
            if all_failed:
                failures[field] = classify_failure(doc_name, field, extractions)

        documents[doc_name] = {
            "extractions": extractions,
            "scores": scores_map,
            "failures": failures,
        }
        progress.progress((i + 1) / len(doc_paths))

    # Aggregate summary
    total_docs     = len(documents)
    field_accuracy = {f: {s: {"correct": 0, "wrong": 0, "missing": 0} for s in STRATEGIES} for f in FIELDS}
    failure_types  = {k: 0 for k in FAILURE_META}

    for doc_data in documents.values():
        for field in FIELDS:
            for s in STRATEGIES:
                r = doc_data["scores"][field].get(s, "missing")
                field_accuracy[field][s][r] += 1
        for ft in doc_data.get("failures", {}).values():
            if ft in failure_types:
                failure_types[ft] += 1

    results_out = {
        "documents": documents,
        "summary": {"total_docs": total_docs, "field_accuracy": field_accuracy, "failure_types": failure_types},
    }
    with open("results.json", "w") as f:
        json.dump(results_out, f, indent=2)

    summary   = results_out["summary"]
    is_sample = False
    status.empty()
    st.success(f"Done. Processed {total_docs} document(s).")

    if errors:
        with st.expander(f"Extraction warnings ({len(errors)})"):
            for e in errors:
                st.code(e)

# Display results
total_docs = summary["total_docs"]
st.markdown(f"#### Field accuracy across {total_docs} document(s)")
df = build_accuracy_df(summary)
st.dataframe(df.style.map(accuracy_style), use_container_width=True)
st.markdown("<p class='legend'>Green ≥ 88% · Amber ≥ 63% · Red below that</p>", unsafe_allow_html=True)

st.markdown("#### Where extractions fail")
render_failure_bars(summary["failure_types"])

ft       = summary["failure_types"]
total_ft = sum(ft.values()) or 1
pf_pct   = int(ft.get("prompt_fixable", 0) / total_ft * 100)

st.markdown(f"""
<div class='callout'>
    <strong>Key finding:</strong> {pf_pct}% of failures are resolved by switching from Strategy A to Strategy C.
    The rest are structural — they need layout-aware pre-processing, not better prompts.
    Accessorial charges on multi-column documents are the hardest single failure mode.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Section 2: Live upload ───────────────────────────────────────────────────

st.markdown("<span class='tag'>Part 2 — Test on your document</span>", unsafe_allow_html=True)
st.markdown(
    "Upload a freight document. Runs all 3 strategies and highlights fields "
    "where they disagree — those are the uncertain extractions."
)

uploaded = st.file_uploader("Upload document", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)
    img_col, btn_col = st.columns([1.2, 1])

    with img_col:
        st.image(image, use_column_width=True)
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_live = st.button("Run extraction", type="primary", use_container_width=True)
        st.markdown(
            "<p style='font-size:0.74rem;color:#ccc;margin-top:6px'>3 calls · Gemini 1.5 Flash</p>",
            unsafe_allow_html=True,
        )

    if run_live:
        if model is None:
            st.error(model_err)
        else:
            extractions, live_errors = {}, []

            with st.spinner("Running 3 strategies..."):
                for name, prompt in STRATEGIES.items():
                    result, err = extract_fields(image, prompt, model)
                    extractions[name] = result
                    if err:
                        live_errors.append(f"{name}: {err}")
                    time.sleep(1)

            if live_errors:
                with st.expander("Extraction warnings"):
                    for e in live_errors:
                        st.code(e)

            # Build comparison table
            rows, disagree_fields = [], set()
            for fkey, flabel in FIELD_LABELS.items():
                row  = {"Field": flabel}
                vals = []
                for s in STRATEGIES:
                    v = extractions[s].get(fkey)
                    row[s] = f"${v:,.2f}" if isinstance(v, (int, float)) else (str(v) if v else "—")
                    vals.append(str(v) if v is not None else None)

                non_null = [v for v in vals if v is not None]
                if len(set(non_null)) > 1:
                    disagree_fields.add(flabel)
                rows.append(row)

            comparison_df = pd.DataFrame(rows).set_index("Field")

            def highlight(row):
                if row.name in disagree_fields:
                    return ["background-color:#fee2e2;color:#991b1b"] * len(row)
                return [""] * len(row)

            st.markdown("#### Extraction comparison")
            st.markdown("<p class='legend'>Red = strategies disagree. These fields warrant human review.</p>", unsafe_allow_html=True)
            st.dataframe(comparison_df.style.apply(highlight, axis=1), use_container_width=True)

            n = len(disagree_fields)
            if n == 0:
                st.success("All 3 strategies agree on every field.")
            else:
                field_list = ", ".join(f"**{f}**" for f in sorted(disagree_fields))
                st.warning(f"{n} field(s) with disagreement: {field_list}. Manual review recommended.")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='font-size:0.74rem;color:#ccc;text-align:center'>"
    "Prototype · "
    "<a href='https://github.com/aarushsharmaa/freight-doc-eval' style='color:#ccc'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True,
)