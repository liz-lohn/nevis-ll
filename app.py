# app.py
import json
import re
import time
import random
import hashlib
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI
from openai import APITimeoutError, APIError, RateLimitError

# -----------------------------
# CONFIG
# -----------------------------
FACTFIND_PROMPT_ID = "pmpt_693d4553c83c81909f3fbc78efb33b63055f568a34e78791"  # <-- replace
CIF_PROMPT_ID = "pmpt_693d4db6a7508195b704c60032c4ea710abc5fabab892f08"              # <-- replace
MODEL_NAME = "gpt-5.1"                      # can switch to e.g. "gpt-4.1"

# -----------------------------
# OPENAI CLIENT (with longer timeout)
# -----------------------------
client = OpenAI(
    timeout=300.0,  # increased for chunked processing
    max_retries=0,  # we'll handle retries ourselves
)

# -----------------------------
# B) RETRIES WITH BACKOFF
# -----------------------------
def call_with_retries(fn, tries: int = 4, base_delay: float = 2.0):
    """
    Retry wrapper for transient OpenAI errors (timeouts, rate limits, server errors).
    Uses exponential backoff + jitter.
    """
    last_err = None
    for attempt in range(tries):
        try:
            return fn()
        except (APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            sleep_s = base_delay * (2 ** attempt) + random.random()
            time.sleep(sleep_s)
    raise last_err


def extract_output_text(resp: Any) -> str:
    """
    Robustly get text from a Responses API response across minor SDK variations.
    Prefer resp.output_text if present.
    """
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback: try to walk resp.output content
    # Response output is usually a list of messages with content parts.
    try:
        out = resp.output  # type: ignore[attr-defined]
        parts = []
        for item in out:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
        return "\n".join(parts).strip()
    except Exception:
        # As a last resort, stringify
        return str(resp)


def try_repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues:
    - Remove trailing commas before closing braces/brackets
    - Close unclosed structures
    """
    # Remove trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Try to close unclosed structures
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    if open_braces > 0:
        text += '}' * open_braces
    if open_brackets > 0:
        text += ']' * open_brackets
    
    return text


# -----------------------------
# C) STREAMLIT CACHING
# -----------------------------
@st.cache_data(show_spinner=False)
def preprocess_cached(transcript_text: str, client_names: str, prompt_id: str, model: str) -> Dict[str, Any]:
    """
    Calls the saved preprocessing prompt:
    inputs: transcript + client_names
    output: preprocessed JSON dict
    Cached so reruns don't repeat calls.
    """
    variables = {
        "client_names": client_names,
        "transcript": transcript_text,
    }

    resp = call_with_retries(lambda: client.responses.create(
        model=model,
        prompt={"id": prompt_id, "variables": variables},
        max_output_tokens=32000,  # increased significantly for chunked preprocessing
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    ))

    text = extract_output_text(resp)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to repair common JSON issues
        try:
            repaired = try_repair_json(text)
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            pass  # Fall through to original error
        
        # Check if JSON appears truncated (ends mid-string or mid-structure)
        text_stripped = text.strip()
        error_pos = getattr(e, 'pos', None) or getattr(e, 'colno', None) or 0
        
        if not text_stripped.endswith('}') and not text_stripped.endswith(']'):
            raise ValueError(
                f"JSON response appears truncated (ends at {len(text)} chars, error at pos {error_pos}). "
                f"Response may have exceeded token limit. Error: {e}. "
                f"Last 300 chars: {text[-300:]}"
            )
        
        # Show context around the error
        start = max(0, error_pos - 200)
        end = min(len(text), error_pos + 200)
        context = text[start:end]
        
        raise ValueError(
            f"Invalid JSON response from preprocessing API. Error: {e}. "
            f"Response length: {len(text)} chars. Error at position {error_pos}. "
            f"Context around error: ...{context}..."
        )


@st.cache_data(show_spinner=False)
def cif_cached(preprocessed_json: Dict[str, Any], prompt_id: str, model: str) -> Dict[str, Any]:
    """
    Calls the saved CIF extraction prompt:
    input: preprocessed JSON
    output: CIF JSON dict
    Cached so reruns don't repeat calls.
    """
    variables = {
        "preprocessed": json.dumps(preprocessed_json),
    }

    resp = call_with_retries(lambda: client.responses.create(
        model=model,
        prompt={"id": prompt_id, "variables": variables},
        max_output_tokens=16000,  # increased for full CIF structure
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    ))

    text = extract_output_text(resp)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from CIF extraction API. Error: {e}. Response length: {len(text)} chars. Response preview: {text[:500]}...")


# -----------------------------
# CHUNKING AND COMPRESSION HELPERS
# -----------------------------
def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """
    Split transcript into chunks (smaller chunks to avoid input token limits).
    Splits on blank lines to keep conversation blocks together.
    """
    parts = text.split("\n\n")
    chunks, buf, size = [], [], 0
    for p in parts:
        p_len = len(p) + 2
        if size + p_len > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += p_len
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def merge_preprocessed(outputs: list[dict]) -> dict:
    """
    Merge preprocessed chunks into one preprocessed object.
    Re-numbers segment_id to be clean.
    """
    segments = []
    normalised_parts = []
    for out in outputs:
        normalised_parts.append(out.get("normalised_transcript", "") or "")
        segments.extend(out.get("segments", []) or [])

    # re-number segment_id to be clean
    for i, seg in enumerate(segments, start=1):
        seg["segment_id"] = i

    return {
        "normalised_transcript": "\n".join([s for s in normalised_parts if s.strip()]),
        "segments": segments,
        # global_entities optional – you can drop to reduce size
        "global_entities": {}
    }


def compress_for_cif(preprocessed: dict, max_msg_chars: int = 240) -> dict:
    """
    Reduce payload for CIF extraction:
    - drop normalised_transcript (huge)
    - keep segment topic + brief_summary
    - keep messages but truncate text
    - keep entities
    - IMPORTANT: preserve facts field (contains savings numbers, income, etc.)
    """
    compressed_segments = []
    for seg in preprocessed.get("segments", []) or []:
        msgs = []
        for m in seg.get("messages", []) or []:
            text = (m.get("text") or "")
            # keep only short snippet; enough for goals/risk phrases
            text = text[:max_msg_chars]
            # OPTIONAL: keep only messages that have entities OR are from client
            entities = m.get("entities") or []
            if not entities and m.get("speaker") not in ("CLIENT_1", "CLIENT_2"):
                continue

            msgs.append({
                "speaker": m.get("speaker"),
                "text": text,
                "entities": entities
            })

        compressed_seg = {
            "segment_id": seg.get("segment_id"),
            "topic_label": seg.get("topic_label"),
            "brief_summary": seg.get("brief_summary"),
            "messages": msgs[:60]  # hard cap to avoid runaway payload
        }
        
        # Preserve facts field - this contains important data like savings numbers
        if "facts" in seg and seg["facts"]:
            compressed_seg["facts"] = seg["facts"]
        
        # Preserve other important fields that might be needed
        if "primary_clients_discussed" in seg:
            compressed_seg["primary_clients_discussed"] = seg["primary_clients_discussed"]
        
        compressed_segments.append(compressed_seg)

    return {"segments": compressed_segments}


# -----------------------------
# CIF EDITOR HELPERS
# -----------------------------
def flatten_cif(cif: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn nested CIF dict into a list:
    [{"path": "clients[0].personal.first_name", "value": "John", "confidence": 0.98}, ...]
    """
    if not cif or not isinstance(cif, dict):
        return []
    
    rows: List[Dict[str, Any]] = []
    confidences = cif.get("meta", {}).get("field_confidences", {}) if isinstance(cif.get("meta"), dict) else {}

    def walk(node: Any, path: str = ""):
        if isinstance(node, dict):
            # Skip empty dicts
            if not node:
                return
            for k, v in node.items():
                # skip the confidence map itself
                if path == "meta" and k == "field_confidences":
                    continue
                new_path = f"{path}.{k}" if path else k
                walk(v, new_path)
        elif isinstance(node, list):
            # Skip empty lists
            if not node:
                return
            for i, v in enumerate(node):
                new_path = f"{path}[{i}]"
                walk(v, new_path)
        else:
            # ignore meta primitives (but allow meta.field_confidences paths through)
            if path.startswith("meta.") and path != "meta.field_confidences":
                return
            rows.append({
                "path": path,
                "value": node,
                "confidence": confidences.get(path, None),
            })

    walk(cif)
    return rows


def set_by_path(obj: Any, path: str, new_value: Any) -> None:
    """
    Set a value in a nested dict/list structure using a JSONPath-ish path.
    Example: clients[0].personal.first_name
    """
    # Parse into tokens: strings and ints
    tokens: List[Any] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            j = path.find("]", i)
            tokens.append(int(path[i+1:j]))
            i = j + 1
        elif ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
        else:
            buf += ch
            i += 1
    if buf:
        tokens.append(buf)

    node = obj
    for t in tokens[:-1]:
        if isinstance(t, int):
            node = node[t]
        else:
            node = node[t]
    last = tokens[-1]
    if isinstance(last, int):
        node[last] = new_value
    else:
        node[last] = new_value


def coerce_value(text: str) -> Any:
    """
    Basic coercion for edited values:
    - empty => None
    - int/float if parseable
    - else string
    """
    if text is None:
        return None
    s = str(text).strip()
    if s == "":
        return None
    # try int then float
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="CIF population prototype", layout="wide")
st.title("AI-assisted CIF population prototype")

if "cif" not in st.session_state:
    st.session_state["cif"] = None
if "busy" not in st.session_state:
    st.session_state["busy"] = False

# -------- Screen 1 --------
if st.session_state["cif"] is None:
    st.subheader("Step 1: Upload transcript and enter client names")

    client_names = st.text_input("Client name(s)", placeholder="e.g. John Smith and Jane Smith")
    uploaded = st.file_uploader("Transcript file (.txt)", type=["txt"])

    populate = st.button("Populate CIF", type="primary", disabled=st.session_state["busy"])

    if populate:
        if not client_names or not uploaded:
            st.error("Please provide both client names and a transcript file.")
        else:
            st.session_state["busy"] = True
            try:
                transcript_text = uploaded.read().decode("utf-8", errors="replace")
                
                # Debug: show transcript info
                st.info(f"📄 Transcript length: {len(transcript_text):,} characters")

                with st.spinner("Pre-processing transcript in chunks..."):
                    chunks = chunk_text(transcript_text, max_chars=6000)  # Reduced chunk size to avoid input limits
                    st.info(f"📦 Split into {len(chunks)} chunks (max 6000 chars each)")
                    
                    progress = st.progress(0)
                    status_text = st.empty()
                    pre_outputs = []
                    
                    for i, chunk in enumerate(chunks, start=1):
                        status_text.text(f"Processing chunk {i}/{len(chunks)} ({len(chunk):,} chars)...")
                        pre = preprocess_cached(chunk, client_names, FACTFIND_PROMPT_ID, MODEL_NAME)
                        pre_outputs.append(pre)
                        progress.progress(i / len(chunks))
                    
                    status_text.text(f"✅ Processed all {len(chunks)} chunks. Merging results...")
                    
                    # Debug: show facts from each chunk
                    total_facts = 0
                    for i, pre in enumerate(pre_outputs, 1):
                        chunk_facts = sum(len(seg.get("facts", [])) for seg in pre.get("segments", []))
                        total_facts += chunk_facts
                        if chunk_facts > 0:
                            st.write(f"  Chunk {i}: {chunk_facts} facts found")
                    
                    merged_pre = merge_preprocessed(pre_outputs)
                    merged_facts = sum(len(seg.get("facts", [])) for seg in merged_pre.get("segments", []))
                    st.info(f"📊 Merged {len(merged_pre.get('segments', []))} segments with {merged_facts} total facts from all chunks")
                    
                    pre_for_cif = compress_for_cif(merged_pre)
                    compressed_facts = sum(len(seg.get("facts", [])) for seg in pre_for_cif.get("segments", []))
                    st.info(f"📦 Compressed to {len(pre_for_cif.get('segments', []))} segments with {compressed_facts} facts preserved")
                    status_text.empty()

                with st.spinner("Extracting CIF..."):
                    cif = cif_cached(
                        preprocessed_json=pre_for_cif,
                        prompt_id=CIF_PROMPT_ID,
                        model=MODEL_NAME,
                    )

                st.session_state["cif"] = cif
                st.success("CIF generated.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate CIF: {e}")
            finally:
                st.session_state["busy"] = False

# -------- Screen 2 --------
else:
    cif = st.session_state["cif"]
    st.subheader("Step 2: Review and edit CIF fields")

    # Debug: show CIF structure info
    with st.expander("🔍 Debug: View CIF structure", expanded=False):
        st.json(cif)
        st.write(f"CIF type: {type(cif)}")
        st.write(f"CIF keys: {list(cif.keys()) if isinstance(cif, dict) else 'Not a dict'}")

    rows = flatten_cif(cif)
    
    # Debug: show row count
    st.write(f"Found {len(rows)} total fields")

    # Optional: filter out null fields to reduce noise
    show_nulls = st.checkbox("Show empty (null) fields", value=False)
    if not show_nulls:
        rows = [r for r in rows if r["value"] is not None]
        st.write(f"Showing {len(rows)} non-null fields")

    if len(rows) == 0:
        st.warning("⚠️ No fields found in CIF. The CIF structure might be empty or in an unexpected format.")
        st.write("Please check the debug section above to see the CIF structure.")
        st.stop()

    st.caption("Edit values on the left; confidence is shown on the right. Download exports the current JSON.")

    edited_rows: List[Dict[str, Any]] = []

    for r in rows:
        c1, c2, c3 = st.columns([4, 5, 1.5])
        with c1:
            st.markdown(f"**{r['path']}**")
        with c2:
            current = "" if r["value"] is None else str(r["value"])
            new_val = st.text_input(
                label="",
                value=current,
                key=f"edit::{r['path']}",
            )
        with c3:
            conf = r["confidence"]
            conf_str = f"{conf:.2f}" if isinstance(conf, (float, int)) else "–"
            st.markdown(f"Conf: **{conf_str}**")

        edited_rows.append({"path": r["path"], "value": new_val})

    col_a, col_b, col_c = st.columns([1.2, 1.2, 2.6])

    with col_a:
        if st.button("Save edits"):
            for r in edited_rows:
                set_by_path(st.session_state["cif"], r["path"], coerce_value(r["value"]))
            st.success("Saved.")

    with col_b:
        if st.button("Start over"):
            st.session_state["cif"] = None
            st.rerun()

    with col_c:
        st.download_button(
            "Download CIF JSON",
            data=json.dumps(st.session_state["cif"], indent=2),
            file_name="cif.json",
            mime="application/json",
        )
