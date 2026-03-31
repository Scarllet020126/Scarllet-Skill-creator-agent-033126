from __future__ import annotations

import os
import io
import re
import json
import time
import math
import random
import textwrap
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import altair as alt
import yaml

# Optional providers (import guarded for HF Spaces dependency variance)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

try:
    import httpx
except Exception:
    httpx = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# ---------------------------
# Constants / UI configuration
# ---------------------------

APP_TITLE = "Agentic Medical Device Reviewer"
APP_SUBTITLE = "FDA 510(k) + TFDA Premarket • Agent Pipelines • Note Keeper • Agents/Skills Studio • Workflow Runner"

SUPPORTED_MODELS = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    # Anthropic (examples; can be extended in UI with custom IDs)
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    # Grok
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

# Provider detection is heuristic; can be overridden by model naming patterns.
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "GROK_API_KEY",
}

LANG_OPTIONS = [("English", "en"), ("繁體中文", "zh-Hant")]
THEME_OPTIONS = [("Light", "light"), ("Dark", "dark")]

# 20 painter-inspired skins (names + CSS)
PAINTER_STYLES: List[Tuple[str, str]] = [
    ("Monet Mist", "linear-gradient(135deg, rgba(164,206,255,0.55), rgba(240,252,255,0.55), rgba(191,236,210,0.55))"),
    ("Van Gogh Night", "linear-gradient(135deg, rgba(10,14,40,0.85), rgba(20,40,90,0.85), rgba(255,204,0,0.18))"),
    ("Hokusai Wave", "linear-gradient(135deg, rgba(0,64,128,0.75), rgba(210,240,255,0.55), rgba(0,120,180,0.55))"),
    ("Klimt Gold", "linear-gradient(135deg, rgba(35,25,5,0.85), rgba(180,140,30,0.45), rgba(255,220,120,0.30))"),
    ("Picasso Blue", "linear-gradient(135deg, rgba(10,25,60,0.85), rgba(0,95,175,0.55), rgba(200,235,255,0.28))"),
    ("Matisse Cutout", "linear-gradient(135deg, rgba(250,80,80,0.35), rgba(80,200,160,0.35), rgba(70,120,250,0.28))"),
    ("Rothko Fields", "linear-gradient(135deg, rgba(110,10,10,0.55), rgba(10,10,10,0.65), rgba(180,120,10,0.38))"),
    ("Da Vinci Sepia", "linear-gradient(135deg, rgba(55,40,25,0.85), rgba(180,150,110,0.45), rgba(250,245,235,0.25))"),
    ("Turner Fog", "linear-gradient(135deg, rgba(210,190,140,0.35), rgba(200,220,240,0.55), rgba(240,230,210,0.35))"),
    ("Frida Garden", "linear-gradient(135deg, rgba(10,90,40,0.55), rgba(200,40,90,0.35), rgba(240,220,120,0.22))"),
    ("Basquiat Neon", "linear-gradient(135deg, rgba(20,20,20,0.90), rgba(255,240,0,0.28), rgba(0,240,255,0.18))"),
    ("Magritte Sky", "linear-gradient(135deg, rgba(120,190,255,0.55), rgba(240,250,255,0.65), rgba(120,180,255,0.35))"),
    ("Dali Desert", "linear-gradient(135deg, rgba(200,150,80,0.55), rgba(250,230,190,0.55), rgba(120,80,50,0.35))"),
    ("Cezanne Provence", "linear-gradient(135deg, rgba(60,120,80,0.55), rgba(210,220,170,0.50), rgba(120,160,220,0.35))"),
    ("O'Keeffe Bloom", "linear-gradient(135deg, rgba(255,180,200,0.45), rgba(255,245,250,0.65), rgba(180,120,160,0.25))"),
    ("Seurat Pointillism", "linear-gradient(135deg, rgba(60,90,150,0.55), rgba(220,200,140,0.45), rgba(240,240,240,0.35))"),
    ("Caravaggio Chiaroscuro", "linear-gradient(135deg, rgba(5,5,5,0.92), rgba(80,60,30,0.55), rgba(230,210,160,0.18))"),
    ("Rembrandt Amber", "linear-gradient(135deg, rgba(30,20,10,0.92), rgba(140,90,30,0.50), rgba(255,220,160,0.18))"),
    ("Shitao Ink", "linear-gradient(135deg, rgba(20,20,20,0.85), rgba(240,240,240,0.45), rgba(160,160,160,0.25))"),
    ("Warhol Pop", "linear-gradient(135deg, rgba(255,0,120,0.35), rgba(0,180,255,0.30), rgba(255,230,0,0.25))"),
]

PAINTER_STYLE_NAMES = [n for n, _ in PAINTER_STYLES]
PAINTER_STYLE_BG = {n: bg for n, bg in PAINTER_STYLES}

DEFAULT_SETTINGS = {
    "theme": "light",
    "lang": "zh-Hant",
    "painter_style": PAINTER_STYLE_NAMES[0],
    "model": "gpt-4o-mini",
    "max_tokens": 12000,
    "temperature": 0.2,
    "allow_custom_models": True,
}

# Size guards (avoid accidental huge contexts)
MAX_DOC_CHARS_PER_FILE = 120_000
MAX_TOTAL_CONTEXT_CHARS = 300_000


# ---------------------------
# Default agents.yaml (fallback)
# ---------------------------

DEFAULT_AGENTS_YAML = """
agents:
  note_structurer_agent:
    name: "審查筆記結構化整理代理"
    description: "將零散的審查筆記整理成結構清楚的 Markdown，標出重點與待辦。"
    category: "Note Keeper"
    model: "gemini-2.5-flash"
    temperature: 0.15
    max_tokens: 12000
    system_prompt: |
      你協助 FDA/TFDA 審查員將個人審查筆記整理為結構化 Markdown。
      任務：
      1. 辨識主要主題，建立清楚的標題層級。
      2. 將重要資訊條列化，保留所有技術細節，不得刪除關鍵資訊。
      3. 針對明確的待辦或補件需求，以「待辦/補件」小節清楚列出。
      4. 請勿捏造新的事實內容。
    user_prompt_template: |
      以下是審查員原始筆記（可雜亂或片段），請整理為清晰 Markdown。
      === 原始筆記 ===
      {{input}}

  pdf_to_markdown_agent:
    name: "PDF 轉換為結構化 Markdown 代理"
    description: "將 PDF 擷取出的文字整理成條理清楚的 Markdown，保留標題、清單與表格。"
    category: "文件前處理"
    model: "gemini-2.5-flash"
    temperature: 0.15
    max_tokens: 12000
    system_prompt: |
      你負責將由 PDF 擷取出的原始文字整理為乾淨、可閱讀且結構化的 Markdown。
      規則：保留標題層級、重建表格、修正 OCR 斷行，但不得憑空新增內容。
    user_prompt_template: |
      以下為自 PDF 擷取的原始文字，請轉為結構化 Markdown。
      === PDF 擷取文字 ===
      {{input}}

  fda_510k_intel_agent:
    name: "FDA 510(k) 情資彙整代理"
    description: "針對特定 510(k) 個案彙整公開資訊與技術重點，產出審查導向摘要。"
    category: "FDA 510(k)"
    model: "gpt-4o-mini"
    temperature: 0.2
    max_tokens: 12000
    system_prompt: |
      你是一位 FDA 510(k) 審查官助理，整理產品名稱、分類、適應症、predicate 比較、
      性能測試摘要、風險與風險控制。避免捏造不可得細節，資訊不足請註明。
      請用清楚 Markdown 結構並至少 3 個表格。
    user_prompt_template: |
      下列是本案 510(k) 相關輸入資訊與上下文，請彙整與分析。
      === 使用者輸入 ===
      {{input}}
"""


# ---------------------------
# Utilities
# ---------------------------

def now_ts() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def est_tokens(text: str) -> int:
    # Rough estimate; avoids depending on provider tokenizer.
    return max(1, math.ceil(len(text) / 4))


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def safe_truncate(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n[...TRUNCATED...]", True


def as_markdown_codeblock(label: str, content: str) -> str:
    return f"### {label}\n\n```text\n{content}\n```\n"


def add_log(module: str, level: str, message: str, meta: Optional[dict] = None) -> None:
    if "live_log" not in st.session_state:
        st.session_state["live_log"] = []
    st.session_state["live_log"].append({
        "ts": now_ts(),
        "module": module,
        "level": level,
        "message": message,
        "meta": meta or {},
    })


def get_lang() -> str:
    return st.session_state["settings"]["lang"]


def t(en: str, zh: str) -> str:
    return en if get_lang() == "en" else zh


def apply_style(theme: str, painter_style: str) -> None:
    bg = PAINTER_STYLE_BG.get(painter_style, PAINTER_STYLE_BG[PAINTER_STYLE_NAMES[0]])
    is_dark = (theme == "dark")

    base_bg = "rgba(10,10,12,0.75)" if is_dark else "rgba(255,255,255,0.70)"
    card_bg = "rgba(20,20,25,0.72)" if is_dark else "rgba(255,255,255,0.75)"
    text_color = "#F3F4F6" if is_dark else "#111827"
    subtle_text = "#C7CDD8" if is_dark else "#4B5563"
    border = "rgba(255,255,255,0.12)" if is_dark else "rgba(17,24,39,0.10)"
    accent = "#60A5FA" if is_dark else "#2563EB"

    css = f"""
    <style>
      .stApp {{
        background: {bg};
        background-attachment: fixed;
        color: {text_color};
      }}
      .block-container {{
        padding-top: 1.2rem;
      }}
      .wow-header {{
        padding: 16px 18px;
        border-radius: 16px;
        background: {base_bg};
        border: 1px solid {border};
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
      }}
      .wow-card {{
        padding: 14px 14px;
        border-radius: 16px;
        background: {card_bg};
        border: 1px solid {border};
        backdrop-filter: blur(10px);
      }}
      .wow-chip {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid {border};
        background: rgba(0,0,0,0.12);
        margin-right: 6px;
        font-size: 12px;
        color: {subtle_text};
      }}
      .wow-metric {{
        font-size: 28px;
        font-weight: 700;
        color: {text_color};
        line-height: 1.0;
      }}
      .wow-subtle {{
        color: {subtle_text};
        font-size: 12px;
      }}
      .wow-accent {{
        color: {accent};
        font-weight: 650;
      }}
      .wow-divider {{
        height: 1px;
        background: {border};
        margin: 10px 0;
      }}
      .wow-status-ok {{
        background: rgba(34,197,94,0.14);
        border: 1px solid rgba(34,197,94,0.25);
        color: {text_color};
      }}
      .wow-status-warn {{
        background: rgba(245,158,11,0.14);
        border: 1px solid rgba(245,158,11,0.25);
        color: {text_color};
      }}
      .wow-status-bad {{
        background: rgba(239,68,68,0.14);
        border: 1px solid rgba(239,68,68,0.25);
        color: {text_color};
      }}
      /* Make Streamlit widgets blend better with WOW UI */
      div[data-baseweb="input"] > div {{
        background: rgba(0,0,0,0.10) !important;
      }}
      textarea {{
        background: rgba(0,0,0,0.10) !important;
      }}
      .stButton button {{
        border-radius: 12px !important;
        border: 1px solid {border} !important;
      }}
      a {{
        color: {accent} !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def provider_for_model(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("gpt-") or "openai" in m:
        return "openai"
    if m.startswith("gemini-") or "google" in m:
        return "gemini"
    if m.startswith("claude-") or "anthropic" in m:
        return "anthropic"
    if m.startswith("grok-") or "xai" in m:
        return "grok"
    # fallback
    return "openai"


def get_api_key(provider: str) -> Tuple[Optional[str], str]:
    """
    Returns (key, source) where source is: env|user|missing
    """
    env_var = PROVIDER_ENV_VARS.get(provider)
    env_key = os.getenv(env_var, "").strip() if env_var else ""
    if env_key:
        return env_key, "env"

    user_keys = st.session_state.get("api_keys", {})
    user_key = (user_keys.get(provider) or "").strip()
    if user_key:
        return user_key, "user"

    return None, "missing"


def provider_readiness() -> Dict[str, Dict[str, Any]]:
    out = {}
    for p in ["openai", "gemini", "anthropic", "grok"]:
        key, src = get_api_key(p)
        last_err = st.session_state.get("provider_last_error", {}).get(p)
        out[p] = {
            "ready": bool(key),
            "source": src,
            "last_error": last_err,
        }
    return out


def set_provider_error(provider: str, err: str) -> None:
    if "provider_last_error" not in st.session_state:
        st.session_state["provider_last_error"] = {}
    st.session_state["provider_last_error"][provider] = err


def clear_provider_error(provider: str) -> None:
    if "provider_last_error" in st.session_state:
        st.session_state["provider_last_error"].pop(provider, None)


# ---------------------------
# LLM calling layer
# ---------------------------

def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    provider = provider_for_model(model)
    api_key, source = get_api_key(provider)

    if not api_key:
        raise RuntimeError(f"Missing API key for provider: {provider}")

    add_log("LLM", "INFO", f"LLM call start: provider={provider}, model={model}, key_source={source}",
            {"provider": provider, "model": model, "max_tokens": max_tokens, "temperature": temperature})

    start = time.time()
    try:
        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not installed. Add `openai` to requirements.")
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=float(clamp(temperature, 0.0, 1.0)),
                max_tokens=int(max_tokens),
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )
            text = resp.choices[0].message.content or ""

        elif provider == "gemini":
            if genai is None:
                raise RuntimeError("Google Gemini SDK not installed. Add `google-generativeai` to requirements.")
            genai.configure(api_key=api_key)
            gm = genai.GenerativeModel(model)
            resp = gm.generate_content(
                user_prompt,
                generation_config={
                    "temperature": float(clamp(temperature, 0.0, 1.0)),
                    "max_output_tokens": int(max_tokens),
                },
                system_instruction=system_prompt or None,
            )
            text = getattr(resp, "text", "") or ""

        elif provider == "anthropic":
            if Anthropic is None:
                raise RuntimeError("Anthropic SDK not installed. Add `anthropic` to requirements.")
            client = Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=int(max_tokens),
                temperature=float(clamp(temperature, 0.0, 1.0)),
                system=system_prompt or "",
                messages=[{"role": "user", "content": user_prompt or ""}],
            )
            # Anthropics response content can be list of blocks
            text = ""
            for blk in resp.content:
                if getattr(blk, "type", "") == "text":
                    text += blk.text

        elif provider == "grok":
            if httpx is None:
                raise RuntimeError("httpx not installed. Add `httpx` to requirements.")
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "temperature": float(clamp(temperature, 0.0, 1.0)),
                "max_tokens": int(max_tokens),
                "messages": [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            }
            with httpx.Client(timeout=120) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            text = data["choices"][0]["message"]["content"] or ""

        else:
            raise RuntimeError(f"Unknown provider: {provider}")

        clear_provider_error(provider)
        duration = time.time() - start
        add_log("LLM", "INFO", f"LLM call done: provider={provider}, model={model}, seconds={duration:.2f}",
                {"provider": provider, "model": model, "duration_s": duration})
        return text.strip()

    except Exception as e:
        duration = time.time() - start
        err = f"{type(e).__name__}: {str(e)}"
        set_provider_error(provider, err)
        add_log("LLM", "ERROR", f"LLM call failed: provider={provider}, model={model}, seconds={duration:.2f} • {err}",
                {"provider": provider, "model": model, "duration_s": duration})
        raise


# ---------------------------
# Docs ingestion
# ---------------------------

def extract_pdf_text(file_bytes: bytes, page_from: Optional[int] = None, page_to: Optional[int] = None) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Add `pypdf` to requirements.")

    reader = PdfReader(io.BytesIO(file_bytes))
    n = len(reader.pages)
    pf = 1 if page_from is None else max(1, page_from)
    pt = n if page_to is None else min(n, page_to)
    pf_i = pf - 1
    pt_i = pt - 1

    add_log("DOCS", "INFO", f"PDF extract start: pages={n}, range={pf}-{pt}")
    parts = []
    for i in range(pf_i, pt_i + 1):
        try:
            parts.append(reader.pages[i].extract_text() or "")
        except Exception as e:
            parts.append(f"\n[EXTRACT_ERROR page {i+1}: {e}]\n")
    text = "\n\n".join(parts).strip()
    add_log("DOCS", "INFO", f"PDF extract done: chars={len(text)}")
    return text


def read_uploaded_file_to_text(uploaded) -> Tuple[str, str]:
    """
    Returns (label, text).
    Supports txt, md, csv, json, pdf.
    """
    name = getattr(uploaded, "name", "uploaded")
    suffix = (name.split(".")[-1] if "." in name else "").lower()
    data = uploaded.getvalue()

    if suffix == "pdf":
        text = extract_pdf_text(data)
        return name, text

    if suffix in ("txt", "md"):
        return name, data.decode("utf-8", errors="ignore")

    if suffix == "csv":
        df = pd.read_csv(io.BytesIO(data))
        # Prefer a compact representation
        text = df.to_csv(index=False)
        return name, text

    if suffix == "json":
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        return name, text

    # fallback binary -> best-effort decode
    return name, data.decode("utf-8", errors="ignore")


def assemble_context_from_inputs(
    pasted_text: str,
    uploads: List[Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Assemble a bounded context string from pasted text + uploaded docs.
    Returns (context_str, meta).
    """
    meta = {"files": [], "truncated_files": 0, "truncated_total": False}

    blocks = []
    if pasted_text.strip():
        txt, trunc = safe_truncate(pasted_text.strip(), MAX_DOC_CHARS_PER_FILE)
        if trunc:
            meta["truncated_files"] += 1
        blocks.append(f"## Pasted Context\n\n{txt}\n")

    total_chars = sum(len(b) for b in blocks)

    for up in uploads or []:
        label, text = read_uploaded_file_to_text(up)
        text, trunc = safe_truncate(text.strip(), MAX_DOC_CHARS_PER_FILE)
        if trunc:
            meta["truncated_files"] += 1
        meta["files"].append({"name": label, "chars": len(text)})
        block = f"## File: {label}\n\n{text}\n"
        blocks.append(block)
        total_chars += len(block)
        if total_chars > MAX_TOTAL_CONTEXT_CHARS:
            meta["truncated_total"] = True
            break

    context = "\n\n---\n\n".join(blocks).strip()
    if meta["truncated_total"]:
        context = safe_truncate(context, MAX_TOTAL_CONTEXT_CHARS)[0]

    add_log("DOCS", "INFO", f"Context assembled: chars={len(context)}, files={len(meta['files'])}",
            {"chars": len(context), "files": meta["files"], "truncated_files": meta["truncated_files"], "truncated_total": meta["truncated_total"]})
    return context, meta


# ---------------------------
# agents.yaml management
# ---------------------------

RE_AGENT_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def load_agents_yaml_from_disk(path: str = "agents.yaml") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            data = yaml.safe_load(raw) or {}
            add_log("YAML", "INFO", f"Loaded agents.yaml from disk: {path}")
            return data
        except Exception as e:
            add_log("YAML", "ERROR", f"Failed to parse {path}: {e}")
            return yaml.safe_load(DEFAULT_AGENTS_YAML)
    return yaml.safe_load(DEFAULT_AGENTS_YAML)


def is_standard_agents_yaml(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("agents"), dict)


def normalize_agents_yaml(
    raw_yaml_text: str,
    use_ai_if_needed: bool,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str, List[str], List[str]]:
    """
    Returns: (standard_dict, normalized_yaml_text, warnings, errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Parse
    try:
        parsed = yaml.safe_load(raw_yaml_text) or {}
    except Exception as e:
        errors.append(f"YAML parse error: {type(e).__name__}: {e}")
        add_log("YAML", "ERROR", errors[-1])
        # AI repair attempt: wrap as-is to LLM if allowed
        if use_ai_if_needed:
            try:
                repaired = call_llm(
                    model=model,
                    system_prompt="You fix YAML syntax errors. Output only valid YAML. Preserve content and intent. Do not add secrets.",
                    user_prompt=f"Fix this YAML to be valid:\n\n```yaml\n{raw_yaml_text}\n```",
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                parsed = yaml.safe_load(repaired) or {}
                warnings.append("AI repaired YAML syntax errors.")
                add_log("YAML", "WARN", "AI repaired YAML syntax errors.")
            except Exception as e2:
                errors.append(f"AI repair failed: {type(e2).__name__}: {e2}")
                add_log("YAML", "ERROR", errors[-1])
                # fallback to default
                data = yaml.safe_load(DEFAULT_AGENTS_YAML)
                return data, yaml.safe_dump(data, sort_keys=False, allow_unicode=True), warnings, errors
        else:
            data = yaml.safe_load(DEFAULT_AGENTS_YAML)
            return data, yaml.safe_dump(data, sort_keys=False, allow_unicode=True), warnings, errors

    # Standard case
    if is_standard_agents_yaml(parsed):
        std = parsed
        std, w2, e2 = validate_and_patch_agents(std)
        warnings.extend(w2)
        errors.extend(e2)
        norm_text = yaml.safe_dump(std, sort_keys=False, allow_unicode=True)
        return std, norm_text, warnings, errors

    # Non-standard structural mapping (deterministic best-effort)
    std: Dict[str, Any] = {"agents": {}}

    if isinstance(parsed, dict):
        # If looks like {agent_id: {..}}, treat as agents
        candidate_agents = parsed.get("agents")
        if isinstance(candidate_agents, dict):
            parsed_agents = candidate_agents
        else:
            parsed_agents = parsed

        if isinstance(parsed_agents, dict):
            for k, v in parsed_agents.items():
                if not isinstance(v, dict):
                    continue
                agent_id = str(k).strip()
                if agent_id.lower() in ("version", "meta", "settings", "defaults"):
                    continue
                std["agents"][agent_id] = v

            warnings.append("Detected non-standard YAML: treated top-level mapping as agents.")
        else:
            errors.append("Unsupported YAML structure: expected mapping.")
    elif isinstance(parsed, list):
        # If list of agents; create ids
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            agent_id = item.get("agent_id") or item.get("id") or f"agent_{i+1}"
            std["agents"][str(agent_id)] = item
        warnings.append("Detected non-standard YAML: treated top-level list as agents.")
    else:
        errors.append("Unsupported YAML structure.")

    # Field normalization per agent
    std, w2, e2 = normalize_agent_fields(std)
    warnings.extend(w2)
    errors.extend(e2)

    # AI-assisted rewrite into strict standardized YAML (optional)
    if use_ai_if_needed:
        try:
            ai_out = call_llm(
                model=model,
                system_prompt=(
                    "You are an agents.yaml normalization engine.\n"
                    "Rewrite the given content into standardized agents.yaml with top-level key `agents:`.\n"
                    "For each agent, require: name, description, category, model, temperature, max_tokens, system_prompt, user_prompt_template.\n"
                    "Preserve meaning. Do not remove important content. Do not invent factual details.\n"
                    "Ensure user_prompt_template contains {{input}} if no other placeholders exist.\n"
                    "Output YAML only."
                ),
                user_prompt=f"Normalize to standardized agents.yaml:\n\n```yaml\n{yaml.safe_dump(std, sort_keys=False, allow_unicode=True)}\n```",
                max_tokens=max_tokens,
                temperature=0.0,
            )
            parsed_ai = yaml.safe_load(ai_out) or {}
            if is_standard_agents_yaml(parsed_ai):
                std = parsed_ai
                warnings.append("AI produced standardized agents.yaml.")
                add_log("YAML", "INFO", "AI produced standardized agents.yaml.")
            else:
                warnings.append("AI normalization output was not recognized as standardized; kept deterministic normalization.")
                add_log("YAML", "WARN", "AI normalization output not standardized; using deterministic output.")
        except Exception as e:
            warnings.append(f"AI normalization skipped due to error: {type(e).__name__}: {e}")
            add_log("YAML", "WARN", warnings[-1])

    std, w3, e3 = validate_and_patch_agents(std)
    warnings.extend(w3)
    errors.extend(e3)

    norm_text = yaml.safe_dump(std, sort_keys=False, allow_unicode=True)
    return std, norm_text, warnings, errors


def normalize_agent_fields(std: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    agents = std.get("agents", {})
    if not isinstance(agents, dict):
        return {"agents": {}}, ["agents missing or invalid; reset to empty."], []

    out = {"agents": {}}
    for raw_id, cfg in agents.items():
        if not isinstance(cfg, dict):
            continue

        agent_id = str(raw_id).strip()
        if not RE_AGENT_ID.match(agent_id):
            # sanitize
            sanitized = re.sub(r"[^a-z0-9_]", "_", agent_id.lower()).strip("_")
            if not sanitized:
                sanitized = f"agent_{len(out['agents'])+1}"
            warnings.append(f"Agent id normalized: {agent_id} -> {sanitized}")
            agent_id = sanitized

        def pick(*keys, default=None):
            for k in keys:
                if k in cfg and cfg[k] not in (None, ""):
                    return cfg[k]
            return default

        name = pick("name", "title", default=agent_id)
        desc = pick("description", "desc", "about", default="")
        category = pick("category", "group", default="Uncategorized")
        model = pick("model", "llm", "engine", default=st.session_state["settings"]["model"])
        temperature = pick("temperature", "temp", default=st.session_state["settings"]["temperature"])
        max_tokens = pick("max_tokens", "token_limit", "max_output_tokens", default=st.session_state["settings"]["max_tokens"])
        system_prompt = pick("system_prompt", "system", "sys_prompt", "instruction", default="")
        user_tmpl = pick("user_prompt_template", "user_template", "template", "prompt_template", default="")

        # Coerce types
        try:
            temperature = float(temperature)
        except Exception:
            temperature = st.session_state["settings"]["temperature"]
            warnings.append(f"{agent_id}: temperature invalid; default applied.")
        temperature = float(clamp(temperature, 0.0, 1.0))

        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = int(st.session_state["settings"]["max_tokens"])
            warnings.append(f"{agent_id}: max_tokens invalid; default applied.")
        max_tokens = max(256, min(max_tokens, 120000))

        if not user_tmpl.strip():
            user_tmpl = "{{input}}"
            warnings.append(f"{agent_id}: user_prompt_template missing; inserted {{input}}.")

        if ("{{" not in user_tmpl) and ("}}" not in user_tmpl):
            user_tmpl = user_tmpl.strip() + "\n\n{{input}}"
            warnings.append(f"{agent_id}: user_prompt_template had no placeholders; appended {{input}}.")

        out["agents"][agent_id] = {
            "name": str(name),
            "description": str(desc),
            "category": str(category),
            "model": str(model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": str(system_prompt),
            "user_prompt_template": str(user_tmpl),
        }

    return out, warnings, errors


def validate_and_patch_agents(std: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not is_standard_agents_yaml(std):
        errors.append("Not a standardized agents.yaml structure after normalization.")
        return {"agents": {}}, warnings, errors

    for agent_id, a in list(std["agents"].items()):
        missing = []
        for k in ["name", "description", "category", "model", "temperature", "max_tokens", "system_prompt", "user_prompt_template"]:
            if k not in a:
                missing.append(k)

        if missing:
            warnings.append(f"{agent_id}: missing fields patched: {', '.join(missing)}")

        a.setdefault("name", agent_id)
        a.setdefault("description", "")
        a.setdefault("category", "Uncategorized")
        a.setdefault("model", st.session_state["settings"]["model"])
        a.setdefault("temperature", st.session_state["settings"]["temperature"])
        a.setdefault("max_tokens", st.session_state["settings"]["max_tokens"])
        a.setdefault("system_prompt", "")
        a.setdefault("user_prompt_template", "{{input}}")

        # clamp
        try:
            a["temperature"] = float(clamp(float(a["temperature"]), 0.0, 1.0))
        except Exception:
            a["temperature"] = st.session_state["settings"]["temperature"]

        try:
            a["max_tokens"] = int(a["max_tokens"])
        except Exception:
            a["max_tokens"] = st.session_state["settings"]["max_tokens"]
        a["max_tokens"] = max(256, min(int(a["max_tokens"]), 120000))

        tmpl = str(a["user_prompt_template"] or "").strip()
        if not tmpl:
            a["user_prompt_template"] = "{{input}}"
            warnings.append(f"{agent_id}: empty user_prompt_template replaced with {{input}}.")
        elif ("{{" not in tmpl) and ("}}" not in tmpl):
            a["user_prompt_template"] = tmpl + "\n\n{{input}}"
            warnings.append(f"{agent_id}: user_prompt_template had no placeholders; appended {{input}}.")

    return std, warnings, errors


def agents_yaml_quality_score(std: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Heuristic 0..100 quality score.
    """
    meta = {"agents": 0, "missing_fields": 0, "bad_templates": 0, "bad_ids": 0}
    if not is_standard_agents_yaml(std):
        return 0, meta

    meta["agents"] = len(std["agents"])
    if meta["agents"] == 0:
        return 10, meta

    score = 100
    for agent_id, a in std["agents"].items():
        if not RE_AGENT_ID.match(agent_id):
            meta["bad_ids"] += 1
            score -= 2

        req = ["name", "description", "category", "model", "temperature", "max_tokens", "system_prompt", "user_prompt_template"]
        for k in req:
            if k not in a or a[k] in (None, ""):
                meta["missing_fields"] += 1
                score -= 2

        tmpl = str(a.get("user_prompt_template", ""))
        if "{{" not in tmpl or "}}" not in tmpl:
            meta["bad_templates"] += 1
            score -= 3

    score = int(clamp(score, 0, 100))
    return score, meta


def build_normalization_report_md(warnings: List[str], errors: List[str], std: Dict[str, Any]) -> str:
    q, meta = agents_yaml_quality_score(std)
    lines = []
    lines.append("# Agents YAML Normalization Report\n")
    lines.append(f"- Time: {now_ts()}")
    lines.append(f"- Quality Score: **{q}/100**")
    lines.append(f"- Agents Detected: **{meta.get('agents', 0)}**")
    lines.append(f"- Missing Field Findings: **{meta.get('missing_fields', 0)}**")
    lines.append(f"- Template Findings: **{meta.get('bad_templates', 0)}**")
    lines.append(f"- ID Findings: **{meta.get('bad_ids', 0)}**\n")

    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    if warnings:
        lines.append("## Warnings / Notes\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Agent Inventory\n")
    if is_standard_agents_yaml(std):
        rows = []
        for aid, a in std["agents"].items():
            rows.append({
                "agent_id": aid,
                "name": a.get("name", ""),
                "category": a.get("category", ""),
                "model": a.get("model", ""),
                "max_tokens": a.get("max_tokens", ""),
                "temperature": a.get("temperature", ""),
            })
        df = pd.DataFrame(rows)
        lines.append(df.to_markdown(index=False))
    else:
        lines.append("_No standardized agents available._")

    return "\n".join(lines)


# ---------------------------
# Generic agent run UI
# ---------------------------

def log_event(tab: str, agent_id: str, model: str, input_text: str, output_text: str) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "ts": now_ts(),
        "tab": tab,
        "agent": agent_id,
        "model": model,
        "tokens_est": est_tokens((input_text or "") + "\n" + (output_text or "")),
    })


def show_status_chip(status: str) -> None:
    status = (status or "pending").lower()
    klass = "wow-status-ok" if status in ("done", "ready") else "wow-status-warn" if status in ("running", "paused") else "wow-status-bad" if status in ("error",) else ""
    st.markdown(f'<span class="wow-chip {klass}">{status.upper()}</span>', unsafe_allow_html=True)


def agent_run_panel(
    agent_id: str,
    agent_cfg: Dict[str, Any],
    tab_key: str,
    initial_input: str = "",
    initial_prompt_override: str = "",
    allow_output_edit: bool = True,
    tab_name_for_history: str = "Agents",
) -> str:
    """
    Returns the effective output (possibly edited).
    """
    status_key = f"{tab_key}_{agent_id}_status"
    out_key = f"{tab_key}_{agent_id}_output"
    in_key = f"{tab_key}_{agent_id}_input"
    prompt_key = f"{tab_key}_{agent_id}_prompt_override"
    model_key = f"{tab_key}_{agent_id}_model"
    temp_key = f"{tab_key}_{agent_id}_temp"
    mtok_key = f"{tab_key}_{agent_id}_max_tokens"
    view_key = f"{tab_key}_{agent_id}_view"

    if status_key not in st.session_state:
        st.session_state[status_key] = "pending"
    st.session_state.setdefault(in_key, initial_input)
    st.session_state.setdefault(prompt_key, initial_prompt_override)
    st.session_state.setdefault(model_key, agent_cfg.get("model", st.session_state["settings"]["model"]))
    st.session_state.setdefault(temp_key, agent_cfg.get("temperature", st.session_state["settings"]["temperature"]))
    st.session_state.setdefault(mtok_key, agent_cfg.get("max_tokens", st.session_state["settings"]["max_tokens"]))
    st.session_state.setdefault(view_key, "Markdown")

    with st.container():
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        cols = st.columns([0.72, 0.28])
        with cols[0]:
            st.markdown(f"### {agent_cfg.get('name', agent_id)}")
            st.caption(agent_cfg.get("description", ""))
            st.markdown(
                f'<span class="wow-chip">{agent_id}</span>'
                f'<span class="wow-chip">{agent_cfg.get("category","")}</span>',
                unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown("**Status**")
            show_status_chip(st.session_state[status_key])

        st.markdown('<div class="wow-divider"></div>', unsafe_allow_html=True)

        # Model and decoding controls
        c1, c2, c3 = st.columns([0.45, 0.25, 0.30])
        with c1:
            model = st.selectbox(
                t("Model", "模型"),
                options=SUPPORTED_MODELS + (["(custom)"] if st.session_state["settings"].get("allow_custom_models") else []),
                index=(SUPPORTED_MODELS.index(st.session_state[model_key]) if st.session_state[model_key] in SUPPORTED_MODELS else len(SUPPORTED_MODELS)),
                key=f"{model_key}_select",
            )
            if model == "(custom)":
                model = st.text_input(t("Custom model id", "自訂模型 ID"), value=st.session_state[model_key], key=f"{model_key}_custom").strip()
            st.session_state[model_key] = model

        with c2:
            st.session_state[temp_key] = st.slider(
                t("Temperature", "溫度"),
                min_value=0.0, max_value=1.0,
                value=float(st.session_state[temp_key]),
                step=0.05,
                key=f"{temp_key}_slider"
            )
        with c3:
            st.session_state[mtok_key] = st.number_input(
                t("Max tokens", "最大 tokens"),
                min_value=256, max_value=120000,
                value=int(st.session_state[mtok_key]),
                step=256,
                key=f"{mtok_key}_num"
            )

        # Prompt override
        with st.expander(t("Prompt controls", "提示詞控制"), expanded=False):
            st.markdown("**System prompt (read-only)**")
            st.code(agent_cfg.get("system_prompt", ""), language="markdown")
            st.markdown("**User prompt template (session override)**")
            st.session_state[prompt_key] = st.text_area(
                t("Override user prompt template", "覆寫 user_prompt_template（此步驟專用）"),
                value=st.session_state[prompt_key] or agent_cfg.get("user_prompt_template", "{{input}}"),
                height=160,
                key=f"{prompt_key}_ta"
            )

        # Input + Run
        st.session_state[in_key] = st.text_area(
            t("Input", "輸入"),
            value=st.session_state[in_key],
            height=180,
            key=f"{in_key}_ta"
        )

        run_cols = st.columns([0.22, 0.22, 0.56])
        with run_cols[0]:
            run = st.button(t("Run agent", "執行代理"), key=f"{tab_key}_{agent_id}_run")
        with run_cols[1]:
            st.session_state[view_key] = st.selectbox(
                t("Output view", "輸出檢視"),
                ["Markdown", "Text"],
                index=0 if st.session_state[view_key] == "Markdown" else 1,
                key=f"{view_key}_sel"
            )

        if run:
            st.session_state[status_key] = "running"
            add_log("AGENT", "INFO", f"Agent run start: {agent_id}", {"agent_id": agent_id, "tab": tab_name_for_history})
            try:
                tmpl = (st.session_state[prompt_key] or agent_cfg.get("user_prompt_template", "{{input}}"))
                user_prompt = tmpl.replace("{{input}}", st.session_state[in_key] or "")
                # Support other placeholders left as-is; user can manage multi-input agents manually here.
                sys_prompt = agent_cfg.get("system_prompt", "")

                out = call_llm(
                    model=st.session_state[model_key],
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(st.session_state[mtok_key]),
                    temperature=float(st.session_state[temp_key]),
                )
                st.session_state[out_key] = out
                st.session_state[status_key] = "done"
                log_event(tab_name_for_history, agent_id, st.session_state[model_key], st.session_state[in_key], out)
                add_log("AGENT", "INFO", f"Agent run done: {agent_id}", {"agent_id": agent_id})
            except Exception as e:
                st.session_state[out_key] = f"[ERROR] {type(e).__name__}: {e}"
                st.session_state[status_key] = "error"
                add_log("AGENT", "ERROR", f"Agent run failed: {agent_id} • {type(e).__name__}: {e}", {"agent_id": agent_id})

        # Output
        out = st.session_state.get(out_key, "")
        if out:
            st.markdown(t("#### Output", "#### 輸出"))
            if st.session_state[view_key] == "Markdown":
                st.markdown(out, unsafe_allow_html=True)
            else:
                st.code(out, language="text")

            if allow_output_edit:
                st.markdown(t("**Editable output (used as next-step input when chained)**",
                              "**可編修輸出（用於串接下一步輸入）**"))
                edited = st.text_area(
                    t("Edit output", "編修輸出"),
                    value=out,
                    height=220,
                    key=f"{out_key}_edit"
                )
                st.session_state[out_key] = edited
                out = edited

            copy_cols = st.columns([0.5, 0.5])
            with copy_cols[0]:
                st.download_button(
                    t("Download output (.md)", "下載輸出 (.md)"),
                    data=out,
                    file_name=f"{agent_id}_output.md",
                    mime="text/markdown",
                    key=f"{tab_key}_{agent_id}_dl"
                )
            with copy_cols[1]:
                st.caption(t("Copy via download button or manual selection.", "可用下載按鈕或手動選取複製。"))

        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.get(out_key, "")


# ---------------------------
# Skills: standardize + convert to agents + execute
# ---------------------------

def standardize_skill_md_with_llm(skill_md: str, model: str, max_tokens: int, temperature: float) -> str:
    system = (
        "You are the Skill Creator.\n"
        "Transform the given skill description into a standardized SKILL.md.\n"
        "Requirements:\n"
        "- Output must be Markdown.\n"
        "- Include YAML frontmatter with: name, description, compatibility (optional).\n"
        "- Provide sections: Purpose & Triggering, Inputs, Outputs, Procedure, Edge Cases, Examples.\n"
        "- Preserve intent; do not invent capabilities or facts.\n"
    )
    return call_llm(
        model=model,
        system_prompt=system,
        user_prompt=f"Skill input:\n\n```markdown\n{skill_md}\n```",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def skill_md_to_agents_yaml_with_llm(skill_md: str, model: str, max_tokens: int, temperature: float) -> str:
    system = (
        "Convert a standardized SKILL.md into standardized agents.yaml.\n"
        "Output YAML only.\n"
        "Schema:\n"
        "agents:\n"
        "  agent_id:\n"
        "    name: string\n"
        "    description: string\n"
        "    category: string\n"
        "    model: string\n"
        "    temperature: number\n"
        "    max_tokens: integer\n"
        "    system_prompt: string\n"
        "    user_prompt_template: string\n"
        "Rules:\n"
        "- Preserve skill intent.\n"
        "- user_prompt_template must contain {{input}}.\n"
        "- Use a fast default model (gpt-4o-mini or gemini-2.5-flash) if not specified.\n"
    )
    return call_llm(
        model=model,
        system_prompt=system,
        user_prompt=f"Convert this SKILL.md:\n\n```markdown\n{skill_md}\n```",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def execute_task_using_skill(
    skill_md: str,
    task_prompt: str,
    related_context: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    system = (
        "You are an assistant that MUST follow the provided SKILL.md instructions.\n"
        "Treat SKILL.md as your operating procedure and output contract.\n"
        "Do not claim tool execution that you cannot perform.\n"
        "Do not invent facts not supported by user input or related context.\n\n"
        "=== SKILL.md ===\n"
        f"{skill_md}\n"
        "\n=== END SKILL.md ===\n"
    )

    user = (
        "Execute the task using the skill.\n\n"
        f"## Task\n{task_prompt}\n\n"
        "## Related Information (optional)\n"
        f"{related_context if related_context.strip() else '[none]'}\n"
    )

    return call_llm(
        model=model,
        system_prompt=system,
        user_prompt=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------
# Dashboard
# ---------------------------

def render_dashboard() -> None:
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.markdown(f"## {t('Dashboard', '儀表板')}")
    st.caption(t("Session-level metrics, provider readiness, YAML quality, workflow status, and live logs.",
                 "顯示本次工作階段的指標、供應商連線狀態、YAML 品質、工作流程狀態與即時日誌。"))
    st.markdown("</div>", unsafe_allow_html=True)

    history = st.session_state.get("history", [])
    df = pd.DataFrame(history) if history else pd.DataFrame(columns=["ts", "tab", "agent", "model", "tokens_est"])

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-metric">{len(df)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-subtle">{t("Total Runs", "總執行次數")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-metric">{df["model"].nunique() if len(df) else 0}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-subtle">{t("Models Used", "使用模型數")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        tok = int(df["tokens_est"].sum()) if len(df) else 0
        st.markdown(f'<div class="wow-metric">{tok:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-subtle">{t("Estimated Tokens", "估計 Tokens")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="wow-card">', unsafe_allow_html=True)
        last = df["ts"].iloc[-1] if len(df) else "-"
        st.markdown(f'<div class="wow-metric">{"✓" if len(df) else "—"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wow-subtle">{t("Last Run", "最近一次")}: {last}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Provider readiness
    st.markdown("### " + t("Provider Readiness", "供應商狀態"))
    pr = provider_readiness()
    cols = st.columns(4)
    for i, p in enumerate(["openai", "gemini", "anthropic", "grok"]):
        with cols[i]:
            info = pr[p]
            ready = info["ready"]
            src = info["source"]
            last_err = info["last_error"]
            klass = "wow-status-ok" if ready else "wow-status-bad"
            st.markdown('<div class="wow-card">', unsafe_allow_html=True)
            st.markdown(f"**{p.upper()}**")
            st.markdown(f'<span class="wow-chip {klass}">{("READY" if ready else "MISSING")}</span>', unsafe_allow_html=True)
            st.caption(t("Key source", "金鑰來源") + f": {src}")
            if last_err:
                st.caption(t("Last error", "最近錯誤") + f": {last_err}")
            st.markdown("</div>", unsafe_allow_html=True)

    # YAML quality
    st.markdown("### " + t("Active agents.yaml quality", "目前 agents.yaml 品質"))
    active_agents = st.session_state.get("agents_cfg", {"agents": {}})
    score, meta = agents_yaml_quality_score(active_agents)
    klass = "wow-status-ok" if score >= 80 else "wow-status-warn" if score >= 50 else "wow-status-bad"
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.markdown(f'<span class="wow-chip {klass}">{t("Quality Score", "品質分數")}: {score}/100</span>', unsafe_allow_html=True)
    st.caption(f"{t('Agents', '代理')}: {meta.get('agents',0)} • "
               f"{t('Missing findings', '缺漏')}: {meta.get('missing_fields',0)} • "
               f"{t('Template findings', '模板問題')}: {meta.get('bad_templates',0)}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    if len(df):
        st.markdown("### " + t("Usage Charts", "使用概況圖"))
        df2 = df.copy()
        df2["ts_dt"] = pd.to_datetime(df2["ts"], errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            tab_counts = df2.groupby("tab").size().reset_index(name="count")
            chart = alt.Chart(tab_counts).mark_bar().encode(
                x=alt.X("count:Q", title=t("Runs", "次數")),
                y=alt.Y("tab:N", sort="-x", title=t("Tab", "分頁")),
                tooltip=["tab", "count"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        with c2:
            model_counts = df2.groupby("model").size().reset_index(name="count")
            chart = alt.Chart(model_counts).mark_bar().encode(
                x=alt.X("count:Q", title=t("Runs", "次數")),
                y=alt.Y("model:N", sort="-x", title=t("Model", "模型")),
                tooltip=["model", "count"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        # Tokens over time
        tok_ts = df2.dropna(subset=["ts_dt"]).sort_values("ts_dt")
        if len(tok_ts):
            chart = alt.Chart(tok_ts).mark_line(point=True).encode(
                x=alt.X("ts_dt:T", title=t("Time", "時間")),
                y=alt.Y("tokens_est:Q", title=t("Estimated tokens", "估計 tokens")),
                color=alt.Color("tab:N", title=t("Tab", "分頁")),
                tooltip=["ts", "tab", "agent", "model", "tokens_est"]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

    # Live log panel
    st.markdown("### " + t("Live Log", "即時日誌"))
    logs = st.session_state.get("live_log", [])
    if not logs:
        st.caption(t("No logs yet.", "尚無日誌。"))
        return

    ldf = pd.DataFrame(logs)
    f1, f2, f3 = st.columns([0.34, 0.33, 0.33])
    with f1:
        mod = st.selectbox(t("Module filter", "模組篩選"), ["(all)"] + sorted(ldf["module"].unique().tolist()))
    with f2:
        lvl = st.selectbox(t("Level filter", "層級篩選"), ["(all)"] + ["INFO", "WARN", "ERROR"])
    with f3:
        limit = st.number_input(t("Rows", "列數"), min_value=20, max_value=1000, value=200, step=20)

    view = ldf.copy()
    if mod != "(all)":
        view = view[view["module"] == mod]
    if lvl != "(all)":
        view = view[view["level"] == lvl]
    view = view.tail(int(limit))

    st.dataframe(view[["ts", "module", "level", "message"]], use_container_width=True, height=280)
    st.download_button(
        t("Download logs (json)", "下載日誌 (json)"),
        data=json.dumps(logs, ensure_ascii=False, indent=2),
        file_name="live_log.json",
        mime="application/json"
    )


# ---------------------------
# Agents & Skills Studio
# ---------------------------

def render_agents_skills_studio() -> None:
    st.markdown("## " + t("Agents & Skills Studio", "Agents & Skills Studio（代理/技能工作室）"))
    st.caption(t("Upload/paste agents.yaml for normalization and editing, or paste/upload skills to generate agents.",
                 "可上傳/貼上 agents.yaml 進行標準化與編修，或貼上/上傳 skill.md 產生 agents。"))

    tabs = st.tabs([
        t("Agents YAML", "Agents YAML"),
        t("Skills (skill.md)", "Skills（skill.md）"),
        t("Skill → Task Executor", "Skill → 任務執行器"),
    ])

    # ---- Agents YAML tab ----
    with tabs[0]:
        st.markdown("### " + t("Upload/Paste agents.yaml", "上傳/貼上 agents.yaml"))
        left, right = st.columns([0.52, 0.48])
        with left:
            up = st.file_uploader(t("Upload agents.yaml", "上傳 agents.yaml"), type=["yaml", "yml"], key="studio_agents_upload")
            pasted = st.text_area(t("Or paste YAML here", "或在此貼上 YAML"), value="", height=200, key="studio_agents_paste")
        with right:
            st.markdown("### " + t("Normalization controls", "標準化控制"))
            use_ai = st.checkbox(t("Use AI to normalize when needed", "需要時使用 AI 進行標準化"), value=True, key="studio_yaml_use_ai")
            model = st.selectbox(t("Normalizer model", "標準化模型"), SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(st.session_state["settings"]["model"]) if st.session_state["settings"]["model"] in SUPPORTED_MODELS else 0, key="studio_yaml_model")
            max_tokens = st.number_input(t("Max tokens", "最大 tokens"), 256, 120000, 12000, 256, key="studio_yaml_mtok")
            temp = st.slider(t("Temperature", "溫度"), 0.0, 1.0, 0.0, 0.05, key="studio_yaml_temp")
            run_norm = st.button(t("Normalize now", "立即標準化"), key="studio_yaml_norm_btn")

        if run_norm:
            raw_text = ""
            if up is not None:
                raw_text = up.getvalue().decode("utf-8", errors="ignore")
            elif pasted.strip():
                raw_text = pasted
            else:
                raw_text = yaml.safe_dump(st.session_state.get("agents_cfg", {}), sort_keys=False, allow_unicode=True)

            std, norm_text, warns, errs = normalize_agents_yaml(
                raw_yaml_text=raw_text,
                use_ai_if_needed=bool(use_ai),
                model=model,
                max_tokens=int(max_tokens),
                temperature=float(temp),
            )

            st.session_state["studio_norm_yaml_text"] = norm_text
            st.session_state["studio_norm_report"] = build_normalization_report_md(warns, errs, std)
            st.session_state["studio_norm_std"] = std

            # Apply to session as active catalog
            st.session_state["agents_cfg"] = std
            st.session_state["agents_source"] = "uploaded/normalized"
            add_log("YAML", "INFO", "Applied normalized agents.yaml to session as active agents_cfg.")

        # editor + download
        norm_text = st.session_state.get("studio_norm_yaml_text")
        report_md = st.session_state.get("studio_norm_report")
        if norm_text:
            st.markdown("### " + t("Standardized YAML editor", "標準化 YAML 編輯器"))
            edited = st.text_area(t("Edit standardized agents.yaml", "編修標準化 agents.yaml"), value=norm_text, height=320, key="studio_yaml_editor")
            apply = st.button(t("Apply edited YAML to session", "套用編修 YAML 至本次工作階段"), key="studio_apply_yaml")
            if apply:
                try:
                    parsed = yaml.safe_load(edited) or {}
                    if not is_standard_agents_yaml(parsed):
                        st.error(t("Edited YAML is not standardized (must have top-level `agents:` mapping).",
                                   "編修後 YAML 非標準格式（需有最外層 agents:）。"))
                    else:
                        st.session_state["agents_cfg"] = parsed
                        st.session_state["agents_source"] = "edited"
                        st.session_state["studio_norm_yaml_text"] = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
                        add_log("YAML", "INFO", "Applied edited standardized YAML to session.")
                        st.success(t("Applied.", "已套用。"))
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

            st.download_button(
                t("Download standardized agents.yaml", "下載標準化 agents.yaml"),
                data=edited,
                file_name="agents.standardized.yaml",
                mime="text/yaml",
                key="studio_yaml_download"
            )

        if report_md:
            with st.expander(t("Normalization report", "標準化報告"), expanded=False):
                st.markdown(report_md)
            st.download_button(
                t("Download report (md)", "下載報告 (md)"),
                data=report_md,
                file_name="agents_yaml_normalization_report.md",
                mime="text/markdown",
                key="studio_report_download"
            )

        # show active agents list
        st.markdown("### " + t("Active agents catalog", "目前使用中的 agents 目錄"))
        agents_cfg = st.session_state.get("agents_cfg", {"agents": {}})
        if not is_standard_agents_yaml(agents_cfg) or not agents_cfg["agents"]:
            st.info(t("No agents loaded.", "尚未載入代理。"))
        else:
            rows = []
            for aid, a in agents_cfg["agents"].items():
                rows.append({
                    "agent_id": aid,
                    "name": a.get("name", ""),
                    "category": a.get("category", ""),
                    "model": a.get("model", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

    # ---- Skills tab ----
    with tabs[1]:
        st.markdown("### " + t("Paste/upload skill.md and standardize", "貼上/上傳 skill.md 並標準化"))
        st.caption(t("This uses the Skill Creator behavior via LLM prompts.",
                     "此功能透過 LLM 提示詞模擬 Skill Creator 行為。"))

        colA, colB = st.columns([0.56, 0.44])
        with colA:
            skill_uploads = st.file_uploader(
                t("Upload skill.md files (multiple)", "上傳 skill.md（可多檔）"),
                type=["md", "markdown", "txt"],
                accept_multiple_files=True,
                key="skills_uploads"
            )
            skill_paste = st.text_area(
                t("Or paste a skill.md", "或貼上一份 skill.md"),
                value="",
                height=220,
                key="skills_paste"
            )

        with colB:
            model = st.selectbox(t("Skill model", "技能模型"), SUPPORTED_MODELS, index=SUPPORTED_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in SUPPORTED_MODELS else 0, key="skills_model")
            max_tokens = st.number_input(t("Max tokens", "最大 tokens"), 256, 120000, 16000, 256, key="skills_mtok")
            temp = st.slider(t("Temperature", "溫度"), 0.0, 1.0, 0.2, 0.05, key="skills_temp")
            do_std = st.button(t("Standardize skill.md", "標準化 skill.md"), key="skills_std_btn")

        if do_std:
            sources: List[Tuple[str, str]] = []
            if skill_uploads:
                for f in skill_uploads:
                    txt = f.getvalue().decode("utf-8", errors="ignore")
                    sources.append((f.name, txt))
            if skill_paste.strip():
                sources.append(("pasted_skill.md", skill_paste))

            if not sources:
                st.warning(t("Provide at least one skill input.", "請至少提供一份 skill 內容。"))
            else:
                standardized = []
                for name, txt in sources:
                    add_log("SKILL", "INFO", f"Standardizing skill: {name}")
                    try:
                        out = standardize_skill_md_with_llm(txt, model, int(max_tokens), float(temp))
                    except Exception as e:
                        out = f"[ERROR] {type(e).__name__}: {e}"
                    standardized.append({"name": name, "standardized": out})

                st.session_state["skills_standardized"] = standardized
                add_log("SKILL", "INFO", f"Skill standardization complete: {len(standardized)} items")

        standardized = st.session_state.get("skills_standardized", [])
        if standardized:
            st.markdown("### " + t("Standardized skills", "標準化結果"))
            for i, item in enumerate(standardized):
                with st.expander(f"{i+1}. {item['name']}", expanded=(i == 0)):
                    edited = st.text_area(
                        t("Edit standardized skill.md", "編修標準化 skill.md"),
                        value=item["standardized"],
                        height=320,
                        key=f"skill_std_edit_{i}"
                    )
                    item["standardized"] = edited

                    dcol = st.columns([0.5, 0.5])
                    with dcol[0]:
                        st.download_button(
                            t("Download skill.md", "下載 skill.md"),
                            data=edited,
                            file_name=f"{item['name'].replace('.','_')}.standardized.md",
                            mime="text/markdown",
                            key=f"skill_dl_{i}"
                        )
                    with dcol[1]:
                        conv = st.button(t("Convert this skill → agents.yaml", "將此 skill 轉為 agents.yaml"), key=f"skill_to_agents_{i}")
                        if conv:
                            try:
                                agents_yaml_text = skill_md_to_agents_yaml_with_llm(
                                    skill_md=edited,
                                    model=model,
                                    max_tokens=int(max_tokens),
                                    temperature=float(temp),
                                )
                                # Normalize the resulting YAML to ensure standard schema
                                std, norm_text, warns, errs = normalize_agents_yaml(
                                    raw_yaml_text=agents_yaml_text,
                                    use_ai_if_needed=True,
                                    model=model,
                                    max_tokens=int(max_tokens),
                                    temperature=0.0,
                                )
                                st.session_state["agents_cfg"] = std
                                st.session_state["agents_source"] = "generated_from_skill"
                                st.session_state["studio_norm_yaml_text"] = norm_text
                                st.session_state["studio_norm_report"] = build_normalization_report_md(warns, errs, std)
                                add_log("SKILL", "INFO", "Converted skill.md to agents.yaml and applied to session.")
                                st.success(t("Converted and applied to active agents.yaml.", "已轉換並套用至目前 agents.yaml。"))
                            except Exception as e:
                                st.error(f"{type(e).__name__}: {e}")

    # ---- Skill → Task Executor tab (NEW requested feature) ----
    with tabs[2]:
        st.markdown("### " + t("Run tasks using a pasted skill.md", "使用貼上的 skill.md 執行任務"))
        st.caption(t("Paste skill.md, provide a task, and optionally paste/upload related information (txt/md/csv/json).",
                     "貼上 skill.md、輸入任務，並可選擇貼上/上傳相關資訊（txt/md/csv/json）。"))

        top = st.columns([0.52, 0.48])
        with top[0]:
            skill_md = st.text_area(t("Paste skill.md", "貼上 skill.md"), value=st.session_state.get("skill_exec_skill_md", ""), height=240, key="skill_exec_skill_md")
            task = st.text_area(t("Task prompt", "任務提示"), value=st.session_state.get("skill_exec_task", ""), height=140, key="skill_exec_task")
        with top[1]:
            related_paste = st.text_area(t("Paste related info (optional)", "貼上相關資訊（可選）"), value="", height=150, key="skill_exec_related_paste")
            related_uploads = st.file_uploader(
                t("Upload related files (optional)", "上傳相關檔案（可選）"),
                type=["txt", "md", "markdown", "csv", "json", "pdf"],
                accept_multiple_files=True,
                key="skill_exec_uploads"
            )
            model = st.selectbox(t("Model", "模型"), SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(st.session_state["settings"]["model"]) if st.session_state["settings"]["model"] in SUPPORTED_MODELS else 0, key="skill_exec_model")
            max_tokens = st.number_input(t("Max tokens", "最大 tokens"), 256, 120000, 16000, 256, key="skill_exec_mtok")
            temp = st.slider(t("Temperature", "溫度"), 0.0, 1.0, 0.2, 0.05, key="skill_exec_temp")
            run = st.button(t("Run using skill", "用 skill 執行"), key="skill_exec_run")

        context, ctx_meta = assemble_context_from_inputs(related_paste, related_uploads or [])
        with st.expander(t("Context preview", "Context 預覽"), expanded=False):
            st.caption(f"{t('Files', '檔案')}: {len(ctx_meta.get('files', []))} • "
                       f"{t('Chars', '字元')}: {len(context)} • "
                       f"{t('Truncated files', '截斷檔案數')}: {ctx_meta.get('truncated_files', 0)}")
            st.code(context[:8000] + ("\n\n[...]" if len(context) > 8000 else ""), language="markdown")

        if run:
            if not skill_md.strip() or not task.strip():
                st.error(t("skill.md and task prompt are required.", "skill.md 與任務提示為必填。"))
            else:
                add_log("SKILL_EXEC", "INFO", "Skill execution started.")
                try:
                    out = execute_task_using_skill(
                        skill_md=skill_md,
                        task_prompt=task,
                        related_context=context,
                        model=model,
                        max_tokens=int(max_tokens),
                        temperature=float(temp),
                    )
                    st.session_state["skill_exec_output"] = out
                    log_event("Skill Executor", "skill_executor", model, task + "\n" + context, out)
                    add_log("SKILL_EXEC", "INFO", "Skill execution completed.")
                except Exception as e:
                    st.session_state["skill_exec_output"] = f"[ERROR] {type(e).__name__}: {e}"
                    add_log("SKILL_EXEC", "ERROR", f"Skill execution failed: {type(e).__name__}: {e}")

        out = st.session_state.get("skill_exec_output", "")
        if out:
            st.markdown("### " + t("Output", "輸出"))
            view = st.radio(t("View", "檢視"), ["Markdown", "Text"], horizontal=True, key="skill_exec_view")
            if view == "Markdown":
                st.markdown(out, unsafe_allow_html=True)
            else:
                st.code(out, language="text")

            edited = st.text_area(t("Edit output", "編修輸出"), value=out, height=240, key="skill_exec_output_edit")
            st.session_state["skill_exec_output"] = edited

            d1, d2 = st.columns([0.5, 0.5])
            with d1:
                st.download_button(
                    t("Download output (.md)", "下載輸出 (.md)"),
                    data=edited,
                    file_name="skill_executor_output.md",
                    mime="text/markdown"
                )
            with d2:
                if st.button(t("Send output to Note Keeper", "將輸出送至 Note Keeper"), key="skill_exec_to_note"):
                    st.session_state["note_input"] = edited
                    add_log("NOTE", "INFO", "Skill Executor output sent to Note Keeper input.")


# ---------------------------
# Workflow Runner
# ---------------------------

def render_workflow_runner() -> None:
    st.markdown("## " + t("Workflow Runner", "工作流程執行器"))
    st.caption(t("Select agents from the active agents.yaml and run them step-by-step with editable chaining.",
                 "從目前 agents.yaml 選擇代理並逐步執行，支援可編修輸出串接下一步。"))

    agents_cfg = st.session_state.get("agents_cfg", {"agents": {}})
    if not is_standard_agents_yaml(agents_cfg) or not agents_cfg["agents"]:
        st.warning(t("No agents loaded. Load/normalize agents.yaml in Agents & Skills Studio first.",
                     "尚未載入代理，請先到 Agents & Skills Studio 載入/標準化 agents.yaml。"))
        return

    agents = agents_cfg["agents"]
    agent_ids = list(agents.keys())

    # Wizard state
    st.session_state.setdefault("wf_selected", [])
    st.session_state.setdefault("wf_ordered", [])
    st.session_state.setdefault("wf_context_paste", "")
    st.session_state.setdefault("wf_context_uploads_meta", {})
    st.session_state.setdefault("wf_context_text", "")
    st.session_state.setdefault("wf_global_prompt", "")
    st.session_state.setdefault("wf_step_index", 0)
    st.session_state.setdefault("wf_outputs", {})      # agent_id -> output
    st.session_state.setdefault("wf_inputs", {})       # agent_id -> input
    st.session_state.setdefault("wf_step_status", {})  # agent_id -> status
    st.session_state.setdefault("wf_timeline", [])     # records start/end per step

    st.markdown("### " + t("1) Select agents and order", "1) 選擇代理與順序"))
    sel = st.multiselect(
        t("Choose agents", "選擇代理"),
        options=agent_ids,
        default=st.session_state["wf_selected"],
        key="wf_selected_ms"
    )
    st.session_state["wf_selected"] = sel

    if sel:
        # Order controls
        ordered = [a for a in st.session_state.get("wf_ordered", []) if a in sel]
        for a in sel:
            if a not in ordered:
                ordered.append(a)
        st.session_state["wf_ordered"] = ordered

        st.markdown(t("**Execution order**", "**執行順序**"))
        for idx, aid in enumerate(st.session_state["wf_ordered"]):
            c = st.columns([0.06, 0.64, 0.15, 0.15])
            with c[0]:
                st.markdown(f"**{idx+1}.**")
            with c[1]:
                st.markdown(f"{agents[aid].get('name', aid)}  \n`{aid}`")
            with c[2]:
                if st.button("↑", key=f"wf_up_{aid}", disabled=(idx == 0)):
                    st.session_state["wf_ordered"][idx-1], st.session_state["wf_ordered"][idx] = st.session_state["wf_ordered"][idx], st.session_state["wf_ordered"][idx-1]
                    st.rerun()
            with c[3]:
                if st.button("↓", key=f"wf_down_{aid}", disabled=(idx == len(st.session_state['wf_ordered']) - 1)):
                    st.session_state["wf_ordered"][idx+1], st.session_state["wf_ordered"][idx] = st.session_state["wf_ordered"][idx], st.session_state["wf_ordered"][idx+1]
                    st.rerun()

    st.markdown("### " + t("2) Provide task and optional documents", "2) 輸入任務與可選文件"))
    st.session_state["wf_global_prompt"] = st.text_area(
        t("Overall task prompt", "整體任務提示"),
        value=st.session_state["wf_global_prompt"],
        height=140,
        key="wf_global_prompt_ta"
    )
    st.session_state["wf_context_paste"] = st.text_area(
        t("Paste context (optional)", "貼上上下文（可選）"),
        value=st.session_state["wf_context_paste"],
        height=120,
        key="wf_context_paste_ta"
    )
    uploads = st.file_uploader(
        t("Upload docs (optional: txt/md/pdf/csv/json)", "上傳文件（可選：txt/md/pdf/csv/json）"),
        type=["txt", "md", "markdown", "pdf", "csv", "json"],
        accept_multiple_files=True,
        key="wf_docs_upload"
    )

    context_text, meta = assemble_context_from_inputs(st.session_state["wf_context_paste"], uploads or [])
    st.session_state["wf_context_text"] = context_text
    st.session_state["wf_context_uploads_meta"] = meta

    with st.expander(t("Context preview", "Context 預覽"), expanded=False):
        st.caption(f"{t('Chars', '字元')}: {len(context_text)} • {t('Files', '檔案')}: {len(meta.get('files', []))}")
        st.code(context_text[:8000] + ("\n\n[...]" if len(context_text) > 8000 else ""), language="markdown")

    st.markdown("### " + t("3) Execute step-by-step", "3) 逐步執行"))
    ordered = st.session_state.get("wf_ordered", [])
    if not ordered:
        st.info(t("Select at least one agent.", "請至少選擇一個代理。"))
        return

    # Initialize statuses
    for aid in ordered:
        st.session_state["wf_step_status"].setdefault(aid, "pending")

    # Step selector
    step_idx = st.number_input(
        t("Step index (1..n)", "步驟索引 (1..n)"),
        min_value=1, max_value=len(ordered),
        value=int(clamp(st.session_state["wf_step_index"] + 1, 1, len(ordered))),
        step=1,
        key="wf_step_idx_num"
    ) - 1
    st.session_state["wf_step_index"] = step_idx

    current_aid = ordered[step_idx]
    prev_output = ""
    if step_idx == 0:
        base_input = st.session_state["wf_global_prompt"].strip()
        if context_text.strip():
            base_input += "\n\n---\n\n" + context_text
        prev_output = base_input.strip()
    else:
        prev_aid = ordered[step_idx - 1]
        prev_output = st.session_state["wf_outputs"].get(prev_aid, "")

    st.session_state["wf_inputs"][current_aid] = prev_output

    # Step controls
    st.markdown('<div class="wow-card">', unsafe_allow_html=True)
    st.markdown(f"### {t('Current step', '目前步驟')}: {step_idx+1}/{len(ordered)}")
    st.markdown(f"**{agents[current_aid].get('name', current_aid)}**  \n`{current_aid}`")
    show_status_chip(st.session_state["wf_step_status"].get(current_aid, "pending"))
    st.markdown("</div>", unsafe_allow_html=True)

    # Run the current step using agent panel, but force input from workflow and store output back
    out = agent_run_panel(
        agent_id=current_aid,
        agent_cfg=agents[current_aid],
        tab_key="workflow",
        initial_input=st.session_state["wf_inputs"][current_aid],
        initial_prompt_override="",
        allow_output_edit=True,
        tab_name_for_history="Workflow Runner",
    )
    st.session_state["wf_outputs"][current_aid] = out
    st.session_state["wf_step_status"][current_aid] = "done" if out and not out.startswith("[ERROR]") else ("error" if out.startswith("[ERROR]") else st.session_state["wf_step_status"][current_aid])

    # Export run bundle
    st.markdown("### " + t("Export", "匯出"))
    if st.button(t("Generate run_report.md", "產生 run_report.md"), key="wf_gen_report"):
        report = ["# Workflow Run Report\n", f"- Time: {now_ts()}\n"]
        report.append("## Agents (ordered)\n")
        for i, aid in enumerate(ordered, 1):
            a = agents[aid]
            report.append(f"### {i}. {a.get('name','')} (`{aid}`)\n")
            report.append(f"- Category: {a.get('category','')}\n")
            report.append(f"- Default model: {a.get('model','')}\n")
            report.append("\n#### Output\n")
            report.append(st.session_state["wf_outputs"].get(aid, "") or "_(no output)_")
            report.append("\n\n---\n")
        st.session_state["wf_run_report_md"] = "\n".join(report)

    if st.session_state.get("wf_run_report_md"):
        st.download_button(
            t("Download run_report.md", "下載 run_report.md"),
            data=st.session_state["wf_run_report_md"],
            file_name="run_report.md",
            mime="text/markdown",
            key="wf_dl_report"
        )


# ---------------------------
# PDF → Markdown tab
# ---------------------------

def render_pdf_to_md() -> None:
    st.markdown("## " + t("PDF → Markdown", "PDF → Markdown"))
    st.caption(t("Extract PDF text and convert it into structured Markdown via the PDF agent.",
                 "擷取 PDF 文字並透過 PDF 代理轉為結構化 Markdown。"))

    up = st.file_uploader(t("Upload PDF", "上傳 PDF"), type=["pdf"], key="pdf_up")
    if not up:
        return

    page_from = st.number_input(t("Page from", "起始頁"), min_value=1, max_value=9999, value=1, step=1)
    page_to = st.number_input(t("Page to", "結束頁"), min_value=1, max_value=9999, value=9999, step=1)

    if st.button(t("Extract text", "擷取文字"), key="pdf_extract_btn"):
        try:
            txt = extract_pdf_text(up.getvalue(), int(page_from), int(page_to))
            st.session_state["pdf_extracted_text"] = txt
        except Exception as e:
            st.session_state["pdf_extracted_text"] = f"[ERROR] {type(e).__name__}: {e}"

    raw = st.session_state.get("pdf_extracted_text", "")
    if raw:
        with st.expander(t("Extracted text preview", "擷取文字預覽"), expanded=False):
            st.code(raw[:8000] + ("\n\n[...]" if len(raw) > 8000 else ""), language="text")

        agents_cfg = st.session_state.get("agents_cfg", {"agents": {}})
        a = agents_cfg.get("agents", {}).get("pdf_to_markdown_agent")
        if not a:
            st.warning(t("pdf_to_markdown_agent not found in active agents.yaml.", "目前 agents.yaml 找不到 pdf_to_markdown_agent。"))
            return

        out = agent_run_panel(
            agent_id="pdf_to_markdown_agent",
            agent_cfg=a,
            tab_key="pdf2md",
            initial_input=raw,
            tab_name_for_history="PDF → Markdown",
        )
        if out and st.button(t("Send output to Workflow context paste", "將輸出送入 Workflow 上下文"), key="pdf_to_wf"):
            st.session_state["wf_context_paste"] = (st.session_state.get("wf_context_paste", "") + "\n\n---\n\n" + out).strip()
            add_log("WORKFLOW", "INFO", "PDF→MD output appended to Workflow context paste.")


# ---------------------------
# 510(k) Intelligence tab
# ---------------------------

def render_510k_intel() -> None:
    st.markdown("## " + t("510(k) Intelligence", "510(k) 情資彙整"))
    st.caption(t("Run the 510(k) intelligence agent with your case context.", "輸入案件上下文並執行 510(k) 情資彙整代理。"))

    agents_cfg = st.session_state.get("agents_cfg", {"agents": {}})
    a = agents_cfg.get("agents", {}).get("fda_510k_intel_agent")
    if not a:
        st.warning(t("fda_510k_intel_agent not found in active agents.yaml.", "目前 agents.yaml 找不到 fda_510k_intel_agent。"))
        return

    inp = st.text_area(t("Case input", "案件輸入"), value=st.session_state.get("k510_input", ""), height=180, key="k510_input")
    out = agent_run_panel(
        agent_id="fda_510k_intel_agent",
        agent_cfg=a,
        tab_key="k510",
        initial_input=inp,
        tab_name_for_history="510(k) Intelligence",
    )
    if out and st.button(t("Send output to Note Keeper", "將輸出送至 Note Keeper"), key="k510_to_note"):
        st.session_state["note_input"] = out
        add_log("NOTE", "INFO", "510(k) intel output sent to Note Keeper input.")


# ---------------------------
# TFDA Premarket tab (lightweight retained; focused on chaining)
# ---------------------------

def render_twda_premarket() -> None:
    st.markdown("## " + t("TW Premarket (TFDA)", "TW Premarket（TFDA）"))
    st.caption(t("A lightweight TFDA workspace for drafting/notes + agent execution. (Full form systems can be layered on.)",
                 "精簡版 TFDA 工作區：草稿/筆記 + 代理執行（可再疊加完整表單系統）。"))

    st.markdown("### " + t("Application draft / notes", "申請書草稿 / 筆記"))
    app_md = st.text_area(t("Paste draft markdown", "貼上草稿 Markdown"), value=st.session_state.get("tw_app_md", ""), height=220, key="tw_app_md")

    st.markdown("### " + t("Run Note Structurer (optional)", "執行筆記結構化（可選）"))
    agents_cfg = st.session_state.get("agents_cfg", {"agents": {}})
    a = agents_cfg.get("agents", {}).get("note_structurer_agent")
    if not a:
        st.warning(t("note_structurer_agent not found in active agents.yaml.", "目前 agents.yaml 找不到 note_structurer_agent。"))
        return
    out = agent_run_panel(
        agent_id="note_structurer_agent",
        agent_cfg=a,
        tab_key="tw_note",
        initial_input=app_md,
        tab_name_for_history="TW Premarket",
    )
    if out and st.button(t("Send to Workflow global prompt", "送至 Workflow 整體任務"), key="tw_to_wf"):
        st.session_state["wf_global_prompt"] = out
        add_log("WORKFLOW", "INFO", "TW output sent to Workflow global prompt.")


# ---------------------------
# Note Keeper & Magics
# ---------------------------

def highlight_keywords_md(md: str, keywords: List[str], color: str) -> str:
    if not keywords:
        return md
    # simple HTML span injection (works in st.markdown with unsafe_allow_html)
    def repl(m):
        word = m.group(0)
        return f"<span style='background:{color}; padding:0 4px; border-radius:6px'>{word}</span>"
    out = md
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        out = re.sub(re.escape(kw), repl, out, flags=re.IGNORECASE)
    return out


def render_note_keeper() -> None:
    st.markdown("## " + t("AI Note Keeper", "AI Note Keeper（筆記管家）"))
    st.caption(t("Paste text → Markdown → formatting/keywords/entities/chat/summary + extra magics.",
                 "貼上文字 → 轉 Markdown → 排版/關鍵字/實體/聊天/摘要 + 額外魔法。"))

    st.session_state.setdefault("note_input", "")
    st.session_state.setdefault("note_md", "")
    st.session_state.setdefault("note_effective", "")

    note_input = st.text_area(t("Paste note text", "貼上筆記文字"), value=st.session_state["note_input"], height=180, key="note_input_ta")
    st.session_state["note_input"] = note_input

    # Transform to markdown
    model = st.selectbox(t("Model", "模型"), SUPPORTED_MODELS, index=SUPPORTED_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in SUPPORTED_MODELS else 0, key="note_model")
    max_tokens = st.number_input(t("Max tokens", "最大 tokens"), 256, 120000, 12000, 256, key="note_mtok")
    temp = st.slider(t("Temperature", "溫度"), 0.0, 1.0, 0.15, 0.05, key="note_temp")
    prompt = st.text_area(
        t("Transform prompt", "轉換提示詞"),
        value=st.session_state.get("note_transform_prompt", "Transform the following text into clean, structured Markdown with headings, bullets, and tables when appropriate. Do not invent facts."),
        height=100,
        key="note_transform_prompt_ta"
    )
    st.session_state["note_transform_prompt"] = prompt

    if st.button(t("Transform to Markdown", "轉為 Markdown"), key="note_to_md_btn"):
        try:
            out = call_llm(
                model=model,
                system_prompt="You are a note structuring assistant. Output Markdown only.",
                user_prompt=f"{prompt}\n\n=== INPUT ===\n{note_input}",
                max_tokens=int(max_tokens),
                temperature=float(temp),
            )
            st.session_state["note_md"] = out
            st.session_state["note_effective"] = out
            log_event("Note Keeper", "note_transform", model, note_input, out)
            add_log("NOTE", "INFO", "Note transformed to Markdown.")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    if st.session_state["note_md"]:
        view = st.radio(t("View", "檢視"), ["Markdown", "Text"], horizontal=True, key="note_view")
        if view == "Markdown":
            st.markdown(st.session_state["note_effective"], unsafe_allow_html=True)
        else:
            st.code(st.session_state["note_effective"], language="text")

        st.session_state["note_effective"] = st.text_area(
            t("Edit note (effective)", "編修筆記（有效版本）"),
            value=st.session_state["note_effective"],
            height=240,
            key="note_effective_edit"
        )

        st.download_button(
            t("Download note (.md)", "下載筆記 (.md)"),
            data=st.session_state["note_effective"],
            file_name="note.md",
            mime="text/markdown",
            key="note_dl"
        )

    st.markdown("### " + t("Magics", "魔法功能"))
    magic_tabs = st.tabs([
        t("AI Formatting", "AI Formatting"),
        t("AI Keywords", "AI Keywords"),
        t("AI Entities", "AI Entities"),
        t("AI Chat", "AI Chat"),
        t("AI Summary", "AI Summary"),
        t("AI Consistency Checker", "一致性檢查"),
        t("AI Citation & Traceability", "引用與可追溯性"),
    ])

    base_note = st.session_state.get("note_effective", "")

    # AI Formatting
    with magic_tabs[0]:
        if st.button(t("Run formatting", "執行排版優化"), key="note_fmt_btn"):
            try:
                out = call_llm(
                    model=st.session_state["settings"]["model"],
                    system_prompt="You only improve formatting/structure. Do not change facts. Output Markdown only.",
                    user_prompt=f"Format this Markdown for readability:\n\n{base_note}",
                    max_tokens=8000,
                    temperature=0.1,
                )
                st.session_state["note_effective"] = out
                log_event("Note Keeper", "note_format", st.session_state["settings"]["model"], base_note, out)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

    # AI Keywords (user input)
    with magic_tabs[1]:
        kw = st.text_input(t("Keywords (comma-separated)", "關鍵字（逗號分隔）"), value="", key="note_kw_input")
        color = st.color_picker(t("Highlight color", "高亮顏色"), value="#FDE047", key="note_kw_color")
        if st.button(t("Apply highlights (render only)", "套用高亮（僅渲染）"), key="note_kw_apply"):
            kws = [x.strip() for x in kw.split(",") if x.strip()]
            rendered = highlight_keywords_md(base_note, kws, color)
            st.markdown(rendered, unsafe_allow_html=True)

    # AI Entities (20+)
    with magic_tabs[2]:
        if st.button(t("Generate 20+ entities table", "產生 20+ 實體表"), key="note_ent_btn"):
            try:
                out = call_llm(
                    model="gemini-2.5-flash",
                    system_prompt="Extract at least 20 key entities. Output a Markdown table only.",
                    user_prompt=f"From this note, extract 20+ entities with context and regulatory relevance:\n\n{base_note}",
                    max_tokens=8000,
                    temperature=0.2,
                )
                st.session_state["note_entities"] = out
                log_event("Note Keeper", "note_entities", "gemini-2.5-flash", base_note, out)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if st.session_state.get("note_entities"):
            st.markdown(st.session_state["note_entities"], unsafe_allow_html=True)

    # AI Chat
    with magic_tabs[3]:
        q = st.text_area(t("Question / prompt", "問題 / 提示"), value="", height=100, key="note_chat_q")
        chat_model = st.selectbox(t("Model", "模型"), SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(st.session_state["settings"]["model"]) if st.session_state["settings"]["model"] in SUPPORTED_MODELS else 0, key="note_chat_model")
        if st.button(t("Ask", "詢問"), key="note_chat_btn"):
            try:
                out = call_llm(
                    model=chat_model,
                    system_prompt="Answer based only on the note context. If missing, say so.",
                    user_prompt=f"NOTE CONTEXT:\n{base_note}\n\nQUESTION:\n{q}",
                    max_tokens=8000,
                    temperature=0.2,
                )
                st.session_state["note_chat_out"] = out
                log_event("Note Keeper", "note_chat", chat_model, q, out)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if st.session_state.get("note_chat_out"):
            st.markdown(st.session_state["note_chat_out"], unsafe_allow_html=True)

    # AI Summary
    with magic_tabs[4]:
        sp = st.text_area(t("Summary prompt", "摘要提示詞"), value=st.session_state.get("note_sum_prompt", "Summarize this note for a reviewer. Provide bullets + short paragraph."), height=90, key="note_sum_prompt_ta")
        st.session_state["note_sum_prompt"] = sp
        sum_model = st.selectbox(t("Model", "模型"), SUPPORTED_MODELS, index=SUPPORTED_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in SUPPORTED_MODELS else 0, key="note_sum_model")
        if st.button(t("Generate summary", "產生摘要"), key="note_sum_btn"):
            try:
                out = call_llm(
                    model=sum_model,
                    system_prompt="Output Markdown only.",
                    user_prompt=f"{sp}\n\nNOTE:\n{base_note}",
                    max_tokens=8000,
                    temperature=0.2,
                )
                st.session_state["note_summary"] = out
                log_event("Note Keeper", "note_summary", sum_model, base_note, out)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if st.session_state.get("note_summary"):
            st.markdown(st.session_state["note_summary"], unsafe_allow_html=True)

    # AI Consistency Checker (new)
    with magic_tabs[5]:
        if st.button(t("Run consistency check", "執行一致性檢查"), key="note_cons_btn"):
            try:
                out = call_llm(
                    model="gpt-4o-mini",
                    system_prompt="Detect internal inconsistencies. Output a Markdown table with: Issue | Evidence | Risk | Suggested fix.",
                    user_prompt=f"Find inconsistencies in this note:\n\n{base_note}",
                    max_tokens=9000,
                    temperature=0.25,
                )
                st.session_state["note_consistency"] = out
                log_event("Note Keeper", "note_consistency", "gpt-4o-mini", base_note, out)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if st.session_state.get("note_consistency"):
            st.markdown(st.session_state["note_consistency"], unsafe_allow_html=True)

    # AI Citation & Traceability (new)
    with magic_tabs[6]:
        if st.button(t("Build traceability table", "建立追溯表"), key="note_trace_btn"):
            try:
                out = call_llm(
                    model="gemini-2.5-flash",
                    system_prompt="Build a traceability table. Output Markdown table with: Claim | Source excerpt | Location/Section | Confidence | Follow-up needed.",
                    user_prompt=f"Build claim→evidence traceability for this note:\n\n{base_note}",
                    max_tokens=10000,
                    temperature=0.2,
                )
                st.session_state["note_traceability"] = out
                log_event("Note Keeper", "note_traceability", "gemini-2.5-flash", base_note, out)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        if st.session_state.get("note_traceability"):
            st.markdown(st.session_state["note_traceability"], unsafe_allow_html=True)


# ---------------------------
# Sidebar (themes, language, painter styles, API keys)
# ---------------------------

def render_sidebar() -> None:
    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.caption(APP_SUBTITLE)

    # Theme / language / painter
    st.sidebar.markdown("### " + t("Appearance", "外觀"))
    theme_label_to_val = {k: v for k, v in THEME_OPTIONS}
    lang_label_to_val = {k: v for k, v in LANG_OPTIONS}

    theme_label = st.sidebar.selectbox(
        t("Theme", "主題"),
        options=[k for k, _ in THEME_OPTIONS],
        index=0 if st.session_state["settings"]["theme"] == "light" else 1
    )
    st.session_state["settings"]["theme"] = theme_label_to_val[theme_label]

    lang_label = st.sidebar.selectbox(
        t("Language", "語言"),
        options=[k for k, _ in LANG_OPTIONS],
        index=0 if st.session_state["settings"]["lang"] == "en" else 1
    )
    st.session_state["settings"]["lang"] = lang_label_to_val[lang_label]

    style = st.sidebar.selectbox(
        t("Painter style", "畫家風格"),
        options=PAINTER_STYLE_NAMES,
        index=PAINTER_STYLE_NAMES.index(st.session_state["settings"]["painter_style"]),
    )
    st.session_state["settings"]["painter_style"] = style

    if st.sidebar.button("Jackpot!", use_container_width=True):
        st.session_state["settings"]["painter_style"] = random.choice(PAINTER_STYLE_NAMES)
        add_log("UI", "INFO", f"Jackpot style chosen: {st.session_state['settings']['painter_style']}")
        st.rerun()

    # Global defaults
    st.sidebar.markdown("### " + t("Defaults", "預設"))
    st.session_state["settings"]["model"] = st.sidebar.selectbox(
        t("Default model", "預設模型"),
        SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(st.session_state["settings"]["model"]) if st.session_state["settings"]["model"] in SUPPORTED_MODELS else 0
    )
    st.session_state["settings"]["temperature"] = st.sidebar.slider(
        t("Default temperature", "預設溫度"),
        0.0, 1.0, float(st.session_state["settings"]["temperature"]), 0.05
    )
    st.session_state["settings"]["max_tokens"] = st.sidebar.number_input(
        t("Default max tokens", "預設最大 tokens"),
        min_value=256, max_value=120000,
        value=int(st.session_state["settings"]["max_tokens"]), step=256
    )

    # API Keys (env keys hidden; prompt only if env missing)
    st.sidebar.markdown("### " + t("API Keys", "API 金鑰"))
    st.session_state.setdefault("api_keys", {})

    for provider in ["openai", "gemini", "anthropic", "grok"]:
        env_var = PROVIDER_ENV_VARS[provider]
        env_key = os.getenv(env_var, "").strip()
        if env_key:
            st.sidebar.caption(f"{provider.upper()}: " + t("Configured from environment", "已由環境變數設定"))
            continue
        st.session_state["api_keys"][provider] = st.sidebar.text_input(
            f"{provider.upper()} " + t("API Key", "API 金鑰"),
            value=st.session_state["api_keys"].get(provider, ""),
            type="password",
            help=t("Stored only in session state; not written to disk.", "僅保存於 session state，不會寫入磁碟。"),
            key=f"api_key_{provider}"
        )

    # Apply style after settings chosen
    apply_style(st.session_state["settings"]["theme"], st.session_state["settings"]["painter_style"])

    st.sidebar.markdown("### " + t("Quick actions", "快速操作"))
    if st.sidebar.button(t("Clear live log", "清空即時日誌"), use_container_width=True):
        st.session_state["live_log"] = []
        add_log("UI", "INFO", "Live log cleared.")
    if st.sidebar.button(t("Clear history", "清空歷史紀錄"), use_container_width=True):
        st.session_state["history"] = []
        add_log("UI", "INFO", "History cleared.")


# ---------------------------
# App initialization
# ---------------------------

def init_state() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.session_state.setdefault("settings", DEFAULT_SETTINGS.copy())
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("live_log", [])
    st.session_state.setdefault("provider_last_error", {})
    st.session_state.setdefault("agents_source", "default")

    if "agents_cfg" not in st.session_state:
        cfg = load_agents_yaml_from_disk()
        # Ensure standardized shape
        if not is_standard_agents_yaml(cfg):
            cfg, _, _, _ = normalize_agents_yaml(
                raw_yaml_text=yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                use_ai_if_needed=False,
                model=st.session_state["settings"]["model"],
                max_tokens=12000,
                temperature=0.0,
            )
        st.session_state["agents_cfg"] = cfg
        add_log("YAML", "INFO", "Initialized agents_cfg from disk/default.")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    init_state()
    render_sidebar()

    st.markdown(f'<div class="wow-header"><h1 style="margin:0">{APP_TITLE}</h1>'
                f'<div class="wow-subtle">{APP_SUBTITLE}</div></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        t("Dashboard", "Dashboard"),
        t("TW Premarket", "TW Premarket"),
        t("510(k) Intel", "510(k) Intel"),
        t("PDF → MD", "PDF → MD"),
        t("Note Keeper", "Note Keeper"),
        t("Agents & Skills Studio", "Agents & Skills Studio"),
        t("Workflow Runner", "Workflow Runner"),
    ])

    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_twda_premarket()
    with tabs[2]:
        render_510k_intel()
    with tabs[3]:
        render_pdf_to_md()
    with tabs[4]:
        render_note_keeper()
    with tabs[5]:
        render_agents_skills_studio()
    with tabs[6]:
        render_workflow_runner()


if __name__ == "__main__":
    main()
