import os
import io
import re
import time
import json
import math
import uuid
import random
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional deps (keep app running even if missing)
try:
    import yaml
except Exception:
    yaml = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import altair as alt
except Exception:
    alt = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="WOW Regulatory Workbench", layout="wide")


# ---------------------------
# Constants / registries
# ---------------------------
LANGS = {
    "English": "en",
    "繁體中文": "zh",
}

DEFAULT_LANGUAGE = "English"
DEFAULT_THEME = "Dark"

PAINTER_STYLES = [
    "Van Gogh", "Monet", "Picasso", "Da Vinci", "Michelangelo",
    "Rembrandt", "Vermeer", "Klimt", "Matisse", "Dali",
    "Hokusai", "Frida Kahlo", "Edward Hopper", "Jackson Pollock", "Andy Warhol",
    "Georgia O'Keeffe", "Turner", "Gauguin", "Cézanne", "Caravaggio",
]

# Model catalog (can be extended by agents.yaml parsing; these are UI defaults)
MODEL_CATALOG = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    # Anthropic (examples; require anthropic package)
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    # Grok (xAI)
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

# Provider routing by model prefix/name
def infer_provider(model_id: str) -> str:
    m = (model_id or "").lower().strip()
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("grok-"):
        return "grok"
    # fallback heuristics
    if "gpt" in m:
        return "openai"
    if "gemini" in m:
        return "gemini"
    if "claude" in m:
        return "anthropic"
    if "grok" in m or "xai" in m:
        return "grok"
    return "unknown"


PROVIDER_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVEAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
}

# Grok OpenAI-compatible endpoint
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

# ---------------------------
# Localization (minimal baseline; extend as needed)
# ---------------------------
I18N = {
    "app_title": {"en": "WOW Regulatory Workbench", "zh": "WOW 法規工作台"},
    "global_settings": {"en": "Global Settings", "zh": "全域設定"},
    "theme": {"en": "Theme", "zh": "主題"},
    "language": {"en": "Language", "zh": "語言"},
    "painter_style": {"en": "Painter Style", "zh": "畫家風格"},
    "jackpot": {"en": "Jackpot (Random)", "zh": "Jackpot（隨機）"},
    "provider_keys": {"en": "Provider API Keys", "zh": "供應商 API 金鑰"},
    "ready_env": {"en": "Ready (env)", "zh": "就緒（環境變數）"},
    "ready_session": {"en": "Ready (session)", "zh": "就緒（本次工作階段）"},
    "missing": {"en": "Missing", "zh": "缺少"},
    "model": {"en": "Model", "zh": "模型"},
    "system_prompt": {"en": "System Prompt", "zh": "系統提示詞"},
    "user_prompt": {"en": "User Prompt", "zh": "使用者提示詞"},
    "max_tokens": {"en": "Max Tokens", "zh": "最大 Tokens"},
    "temperature": {"en": "Temperature", "zh": "溫度"},
    "run": {"en": "Run", "zh": "執行"},
    "rerun": {"en": "Re-run", "zh": "重新執行"},
    "reset_to_generated": {"en": "Reset to Generated Output", "zh": "重設為生成結果"},
    "output": {"en": "Output", "zh": "輸出"},
    "preview": {"en": "Preview", "zh": "預覽"},
    "download_md": {"en": "Download .md", "zh": "下載 .md"},
    "download_txt": {"en": "Download .txt", "zh": "下載 .txt"},
    "live_log": {"en": "Live Log", "zh": "即時日誌"},
    "status_wall": {"en": "Status Wall", "zh": "狀態牆"},
    "dashboard": {"en": "Dashboard", "zh": "儀表板"},
    "tw_premarket": {"en": "TW Premarket", "zh": "台灣上市前"},
    "intel_510k": {"en": "510(k) Intelligence", "zh": "510(k) 情資"},
    "pdf2md": {"en": "PDF → Markdown", "zh": "PDF → Markdown"},
    "pipeline_510k": {"en": "510(k) Review Pipeline", "zh": "510(k) 審查流程"},
    "report_gen": {"en": "510(k) Report Generator", "zh": "510(k) 審查報告產生器"},
    "note_keeper": {"en": "AI Note Keeper", "zh": "AI 筆記管理"},
    "agents_studio": {"en": "Agents Config Studio", "zh": "Agents 設定工作室"},
}

def t(key: str) -> str:
    lang = st.session_state.get("ui_lang_code", "en")
    return I18N.get(key, {}).get(lang, key)


# ---------------------------
# Session state init
# ---------------------------
def ss_init():
    st.session_state.setdefault("ui_language", DEFAULT_LANGUAGE)
    st.session_state.setdefault("ui_lang_code", LANGS.get(DEFAULT_LANGUAGE, "en"))
    st.session_state.setdefault("ui_theme", DEFAULT_THEME)
    st.session_state.setdefault("ui_style", PAINTER_STYLES[0])

    st.session_state.setdefault("run_log", [])  # list of dict events
    st.session_state.setdefault("run_history", [])  # list of run metadata dicts

    # Provider keys (session)
    st.session_state.setdefault("keys", {"openai": "", "gemini": "", "anthropic": "", "grok": ""})

    # agents.yaml (session)
    st.session_state.setdefault("agents_yaml_text", "")
    st.session_state.setdefault("agents", {})  # parsed agents dict

    # Common defaults
    st.session_state.setdefault("global_temperature", 0.2)
    st.session_state.setdefault("global_max_tokens", 2048)
    st.session_state.setdefault("global_default_model", "gpt-4o-mini")

    # Workspaces state
    st.session_state.setdefault("pipeline", {})
    st.session_state.setdefault("note_keeper", {"versions": [], "active_idx": None})

ss_init()


# ---------------------------
# WOW CSS (theme + painter styles)
# ---------------------------
STYLE_CSS_MAP = {
    "Van Gogh": {"accent": "#3E7CB1", "bg1": "#0b1220", "bg2": "#1b2a4a"},
    "Monet": {"accent": "#7DA87B", "bg1": "#0b1612", "bg2": "#123b2a"},
    "Picasso": {"accent": "#E76F51", "bg1": "#160b0b", "bg2": "#3b1510"},
    "Da Vinci": {"accent": "#C2A83E", "bg1": "#12100b", "bg2": "#2b2511"},
    "Michelangelo": {"accent": "#8D99AE", "bg1": "#0d1117", "bg2": "#222a35"},
    "Rembrandt": {"accent": "#B5651D", "bg1": "#140c08", "bg2": "#2a160e"},
    "Vermeer": {"accent": "#2A9D8F", "bg1": "#07161a", "bg2": "#0d2b31"},
    "Klimt": {"accent": "#D4AF37", "bg1": "#100d08", "bg2": "#2a1d10"},
    "Matisse": {"accent": "#F72585", "bg1": "#150612", "bg2": "#3a0f2a"},
    "Dali": {"accent": "#FFD166", "bg1": "#0a0f14", "bg2": "#172533"},
    "Hokusai": {"accent": "#118AB2", "bg1": "#061018", "bg2": "#0b2333"},
    "Frida Kahlo": {"accent": "#06D6A0", "bg1": "#05120e", "bg2": "#0c2e23"},
    "Edward Hopper": {"accent": "#8E9AAF", "bg1": "#0b0f17", "bg2": "#1b2233"},
    "Jackson Pollock": {"accent": "#EF476F", "bg1": "#12060b", "bg2": "#2b0f18"},
    "Andy Warhol": {"accent": "#8338EC", "bg1": "#0c0716", "bg2": "#1f1240"},
    "Georgia O'Keeffe": {"accent": "#FF6B6B", "bg1": "#160a0a", "bg2": "#3a1111"},
    "Turner": {"accent": "#F4A261", "bg1": "#140d07", "bg2": "#2d1b0c"},
    "Gauguin": {"accent": "#43AA8B", "bg1": "#06130f", "bg2": "#0f2e24"},
    "Cézanne": {"accent": "#577590", "bg1": "#071018", "bg2": "#112739"},
    "Caravaggio": {"accent": "#E9C46A", "bg1": "#120c07", "bg2": "#2a1a10"},
}

def inject_css(theme: str, painter: str):
    palette = STYLE_CSS_MAP.get(painter, STYLE_CSS_MAP["Van Gogh"])
    accent = palette["accent"]
    bg1 = palette["bg1"]
    bg2 = palette["bg2"]

    # Light theme overrides
    if theme.lower().startswith("light"):
        bg1 = "#f7f7fb"
        bg2 = "#ffffff"

    css = f"""
    <style>
      .wow-shell {{
        background: radial-gradient(1200px 700px at 10% 10%, {bg2} 0%, {bg1} 60%, {bg1} 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px 18px;
      }}
      .wow-card {{
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 14px 14px;
      }}
      .wow-accent {{
        color: {accent};
        font-weight: 700;
      }}
      .wow-pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        margin-right: 8px;
        font-size: 12px;
      }}
      .wow-log {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        white-space: pre-wrap;
        line-height: 1.35;
      }}
      .wow-banner {{
        border-left: 4px solid {accent};
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(255,255,255,0.05);
        border-top: 1px solid rgba(255,255,255,0.08);
        border-right: 1px solid rgba(255,255,255,0.08);
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }}
      /* Make textareas slightly nicer */
      textarea {{
        border-radius: 10px !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css(st.session_state["ui_theme"], st.session_state["ui_style"])


# ---------------------------
# Utilities
# ---------------------------
def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def approx_tokens(text: str) -> int:
    if not text:
        return 0
    # rough heuristic: ~4 chars/token English; CJK differs but good enough for indicators
    return max(1, math.ceil(len(text) / 4))

def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name.strip())
    name = re.sub(r"\s+", "_", name)
    return name[:120] if name else "download"

def log_event(workspace: str, step: str, status: str, model: str = "", provider: str = "", message: str = "", meta: Optional[dict] = None):
    evt = {
        "ts": now_utc_iso(),
        "workspace": workspace,
        "step": step,
        "status": status,  # pending/running/done/error/info
        "model": model,
        "provider": provider,
        "message": message,
        "meta": meta or {},
    }
    st.session_state["run_log"].append(evt)

def provider_env_key(provider: str) -> Optional[str]:
    for envname in PROVIDER_ENV_VARS.get(provider, []):
        val = os.getenv(envname)
        if val:
            return val
    return None

def provider_key(provider: str) -> Optional[str]:
    # env has priority; must not be displayed
    env = provider_env_key(provider)
    if env:
        return env
    v = (st.session_state.get("keys", {}) or {}).get(provider, "")
    return v.strip() or None

def provider_readiness(provider: str) -> Tuple[str, bool]:
    env = provider_env_key(provider)
    if env:
        return (t("ready_env"), True)
    sess = (st.session_state.get("keys", {}) or {}).get(provider, "").strip()
    if sess:
        return (t("ready_session"), True)
    return (t("missing"), False)

def normalize_text_output(x: Any) -> str:
    """Best-effort conversion to displayable text without crashing."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # Sometimes SDK returns rich objects; try json
    try:
        return json.dumps(x, ensure_ascii=False, indent=2)
    except Exception:
        return str(x)

def merge_system_user_for_gemini(system_prompt: str, user_prompt: str) -> str:
    sys = (system_prompt or "").strip()
    usr = (user_prompt or "").strip()
    if sys and usr:
        # Explicit boundaries reduce prompt confusion; avoids "system_instruction" entirely.
        return f"### SYSTEM\n{sys}\n\n### USER\n{usr}"
    if sys:
        return f"### SYSTEM\n{sys}"
    return usr


# ---------------------------
# LLM Clients (robust, no system_instruction for Gemini)
# ---------------------------
@dataclass
class LLMResult:
    text: str
    usage: Dict[str, Any]
    raw: Any
    provider: str
    model: str
    elapsed_s: float
    error: Optional[str] = None

def call_openai_like(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float,
                     api_key: str, base_url: Optional[str] = None) -> LLMResult:
    started = time.time()
    provider = "openai" if base_url is None else "grok"
    try:
        from openai import OpenAI
    except Exception as e:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider=provider,
            model=model,
            elapsed_s=time.time() - started,
            error=f"Missing dependency 'openai': {e}",
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        messages = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            content = normalize_text_output(resp)

        usage = {}
        try:
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", None),
                "output_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        except Exception:
            usage = {}

        return LLMResult(
            text=content,
            usage=usage,
            raw=resp,
            provider=provider,
            model=model,
            elapsed_s=time.time() - started,
        )
    except Exception as e:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider=provider,
            model=model,
            elapsed_s=time.time() - started,
            error=f"{type(e).__name__}: {e}",
        )

def call_gemini(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, api_key: str) -> LLMResult:
    started = time.time()
    try:
        import google.generativeai as genai
    except Exception as e:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider="gemini",
            model=model,
            elapsed_s=time.time() - started,
            error=f"Missing dependency 'google-generativeai': {e}",
        )

    try:
        # Configure key (global). Do not display key.
        genai.configure(api_key=api_key)

        prompt = merge_system_user_for_gemini(system_prompt, user_prompt)

        generation_config = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        }

        gm = genai.GenerativeModel(model_name=model)

        # IMPORTANT: Do NOT pass system_instruction (per request + compatibility)
        resp = gm.generate_content(
            prompt,
            generation_config=generation_config,
        )

        # Robust text extraction
        text = ""
        if hasattr(resp, "text") and resp.text:
            text = resp.text
        else:
            # Fallback: try candidates/parts
            try:
                parts = []
                for cand in (getattr(resp, "candidates", None) or []):
                    content = getattr(cand, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            if hasattr(p, "text") and p.text:
                                parts.append(p.text)
                text = "\n".join(parts).strip()
            except Exception:
                text = ""

        usage = {}
        try:
            um = getattr(resp, "usage_metadata", None)
            if um:
                usage = {
                    "input_tokens": getattr(um, "prompt_token_count", None),
                    "output_tokens": getattr(um, "candidates_token_count", None),
                    "total_tokens": getattr(um, "total_token_count", None),
                }
        except Exception:
            usage = {}

        return LLMResult(
            text=text or "",
            usage=usage,
            raw=resp,
            provider="gemini",
            model=model,
            elapsed_s=time.time() - started,
        )
    except Exception as e:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider="gemini",
            model=model,
            elapsed_s=time.time() - started,
            error=f"{type(e).__name__}: {e}",
        )

def call_anthropic(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, api_key: str) -> LLMResult:
    started = time.time()
    try:
        import anthropic
    except Exception as e:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider="anthropic",
            model=model,
            elapsed_s=time.time() - started,
            error=f"Missing dependency 'anthropic': {e}",
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Anthropic supports system as top-level field; keep it.
        # (This is not Gemini system_instruction.)
        msg = client.messages.create(
            model=model,
            system=(system_prompt or "").strip() or None,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = ""
        try:
            # msg.content is list of blocks
            blocks = getattr(msg, "content", []) or []
            parts = []
            for b in blocks:
                if getattr(b, "type", None) == "text":
                    parts.append(getattr(b, "text", "") or "")
            text = "\n".join([p for p in parts if p]).strip()
        except Exception:
            text = normalize_text_output(msg)

        usage = {}
        try:
            usage = {
                "input_tokens": getattr(msg.usage, "input_tokens", None),
                "output_tokens": getattr(msg.usage, "output_tokens", None),
                "total_tokens": None,
            }
        except Exception:
            usage = {}

        return LLMResult(
            text=text,
            usage=usage,
            raw=msg,
            provider="anthropic",
            model=model,
            elapsed_s=time.time() - started,
        )
    except Exception as e:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider="anthropic",
            model=model,
            elapsed_s=time.time() - started,
            error=f"{type(e).__name__}: {e}",
        )

def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResult:
    provider = infer_provider(model)
    key = provider_key(provider)

    if provider == "unknown":
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider=provider,
            model=model,
            elapsed_s=0.0,
            error=f"Unknown provider for model '{model}'.",
        )

    if not key:
        return LLMResult(
            text="",
            usage={},
            raw=None,
            provider=provider,
            model=model,
            elapsed_s=0.0,
            error=f"Missing API key for provider '{provider}'. Add it in the sidebar or set environment variables.",
        )

    if provider == "openai":
        return call_openai_like(model, system_prompt, user_prompt, max_tokens, temperature, api_key=key, base_url=None)
    if provider == "grok":
        return call_openai_like(model, system_prompt, user_prompt, max_tokens, temperature, api_key=key, base_url=GROK_BASE_URL)
    if provider == "gemini":
        return call_gemini(model, system_prompt, user_prompt, max_tokens, temperature, api_key=key)
    if provider == "anthropic":
        return call_anthropic(model, system_prompt, user_prompt, max_tokens, temperature, api_key=key)

    return LLMResult(
        text="",
        usage={},
        raw=None,
        provider=provider,
        model=model,
        elapsed_s=0.0,
        error=f"Provider '{provider}' not implemented.",
    )


# ---------------------------
# agents.yaml load / parse
# ---------------------------
DEFAULT_AGENTS_YAML = """# agents.yaml (defaults)
agents:
  pdf_to_markdown:
    name: PDF → Markdown Transformer
    model: gemini-2.5-flash
    max_tokens: 2048
    system_prompt: |
      You are a careful document transformer. Convert extracted PDF text into clean Markdown.
      Do not invent content. Preserve headings, lists, tables when clearly present.
  note_to_markdown:
    name: Note → Markdown
    model: gpt-4o-mini
    max_tokens: 2048
    system_prompt: |
      Transform user notes into clear Markdown with headings and bullet lists.
      Do not invent facts; mark unknowns explicitly.
  note_entities:
    name: Note Entities (20)
    model: gpt-4o-mini
    max_tokens: 1200
    system_prompt: |
      Extract exactly 20 entities from the note and present as a Markdown table with columns:
      Entity | Type | Context | Evidence pointer. Do not invent.
  note_summary:
    name: Note Summary
    model: gpt-4o-mini
    max_tokens: 1200
    system_prompt: |
      Summarize the note faithfully. Do not invent.
  pipeline_structurer:
    name: 510(k) Submission Structurer
    model: gemini-2.5-flash
    max_tokens: 2000
    system_prompt: |
      Structure the submission text into a clean Markdown outline with sections.
      Do not add new facts. If something is missing, note it under "Gaps".
  pipeline_checklist_cleaner:
    name: Checklist Cleaner
    model: gpt-4o-mini
    max_tokens: 1200
    system_prompt: |
      Clean and normalize the checklist into Markdown with clear numbering.
      Do not invent requirements.
  pipeline_memo_builder:
    name: Review Memo Builder
    model: gpt-4.1-mini
    max_tokens: 2400
    system_prompt: |
      Write a review memo in Markdown using only the provided structured submission and checklist.
      Do not invent facts. Include "Assumptions/Gaps".
  report_outline:
    name: 510(k) Report Outline Agent
    model: gpt-4o-mini
    max_tokens: 1400
    system_prompt: |
      Create a detailed report plan/outline mapped to the provided template headings.
      Include a missing-info checklist. Do not invent.
  report_writer:
    name: 510(k) Report Writer Agent
    model: gpt-4.1-mini
    max_tokens: 3500
    system_prompt: |
      Write a comprehensive 2000–3000 word Markdown 510(k) review report.
      Must include at least 5 Markdown tables and an entities table with exactly 20 entities.
      Do not invent facts; mark unknowns in "Assumptions/Gaps".
  skill_creator:
    name: Skill.md Generator
    model: gpt-4o-mini
    max_tokens: 1800
    system_prompt: |
      Create a reusable skill.md that instructs an AI to produce similar 510(k) review reports.
      Include triggering language, steps, constraints (2k–3k words, 5+ tables, 20 entities), safety rules, and 2–3 test prompts.
"""

def load_agents_from_text(yaml_text: str) -> Dict[str, Any]:
    if not yaml:
        return {}
    try:
        data = yaml.safe_load(yaml_text) or {}
        agents = data.get("agents", {}) or {}
        # Normalize minimal fields
        norm = {}
        for agent_id, cfg in agents.items():
            if not isinstance(cfg, dict):
                continue
            norm[agent_id] = {
                "name": cfg.get("name", agent_id),
                "model": cfg.get("model", st.session_state.get("global_default_model", "gpt-4o-mini")),
                "max_tokens": int(cfg.get("max_tokens", st.session_state.get("global_max_tokens", 2048))),
                "system_prompt": cfg.get("system_prompt", ""),
                "category": cfg.get("category", ""),
            }
        return norm
    except Exception:
        return {}

def ensure_agents_loaded():
    if not st.session_state["agents_yaml_text"].strip():
        st.session_state["agents_yaml_text"] = DEFAULT_AGENTS_YAML
    st.session_state["agents"] = load_agents_from_text(st.session_state["agents_yaml_text"])

ensure_agents_loaded()


# ---------------------------
# UI components
# ---------------------------
def ui_provider_keys():
    st.sidebar.subheader(t("provider_keys"))

    for provider in ["openai", "gemini", "anthropic", "grok"]:
        label, ready = provider_readiness(provider)
        st.sidebar.markdown(f"**{provider.upper()}**: `{label}`")

        if provider_env_key(provider):
            # Do not show env key input
            continue

        st.session_state["keys"][provider] = st.sidebar.text_input(
            f"{provider.upper()} API Key",
            value=st.session_state["keys"].get(provider, ""),
            type="password",
            help="Stored in session only (not persisted).",
        )

def ui_global_settings():
    st.sidebar.header(t("global_settings"))

    # Language
    ui_language = st.sidebar.selectbox(
        t("language"),
        options=list(LANGS.keys()),
        index=list(LANGS.keys()).index(st.session_state["ui_language"]) if st.session_state["ui_language"] in LANGS else 0,
    )
    st.session_state["ui_language"] = ui_language
    st.session_state["ui_lang_code"] = LANGS[ui_language]

    # Theme
    theme = st.sidebar.selectbox(
        t("theme"),
        options=["Dark", "Light"],
        index=0 if st.session_state["ui_theme"].lower().startswith("dark") else 1,
    )
    st.session_state["ui_theme"] = theme

    # Painter style + jackpot
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        style = st.selectbox(
            t("painter_style"),
            options=PAINTER_STYLES,
            index=PAINTER_STYLES.index(st.session_state["ui_style"]) if st.session_state["ui_style"] in PAINTER_STYLES else 0,
        )
        st.session_state["ui_style"] = style
    with col2:
        if st.button(t("jackpot")):
            st.session_state["ui_style"] = random.choice(PAINTER_STYLES)
            st.rerun()

    # Global generation defaults
    st.sidebar.markdown("---")
    st.sidebar.markdown("**LLM Defaults**")
    st.session_state["global_default_model"] = st.sidebar.selectbox(
        t("model"),
        options=MODEL_CATALOG,
        index=MODEL_CATALOG.index(st.session_state["global_default_model"]) if st.session_state["global_default_model"] in MODEL_CATALOG else 0,
    )
    st.session_state["global_max_tokens"] = st.sidebar.slider(
        t("max_tokens"), min_value=256, max_value=8192, value=int(st.session_state["global_max_tokens"]), step=128
    )
    st.session_state["global_temperature"] = st.sidebar.slider(
        t("temperature"), min_value=0.0, max_value=1.0, value=float(st.session_state["global_temperature"]), step=0.05
    )

    st.sidebar.markdown("---")
    ui_provider_keys()

    st.sidebar.markdown("---")
    st.sidebar.caption("Security: API keys are never written to logs or downloads. Env keys are never displayed.")
    st.sidebar.caption("Reminder: Do not paste PHI/PII unless authorized.")

def status_pill(label: str, status: str):
    color_map = {"pending": "#999", "running": "#f6c945", "done": "#36c98a", "error": "#ff4d4f", "info": "#6aa6ff"}
    c = color_map.get(status, "#999")
    st.markdown(
        f"<span class='wow-pill' style='border-color: rgba(255,255,255,0.18);'>"
        f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};margin-right:8px;'></span>"
        f"{label}</span>",
        unsafe_allow_html=True
    )

def render_live_log(workspace_filter: Optional[str] = None, limit: int = 200):
    evts = st.session_state.get("run_log", [])
    if workspace_filter:
        evts = [e for e in evts if e.get("workspace") == workspace_filter]
    evts = evts[-limit:]
    lines = []
    for e in evts:
        model = e.get("model", "")
        provider = e.get("provider", "")
        msg = e.get("message", "")
        lines.append(f"[{e.get('ts')}] {e.get('workspace')}/{e.get('step')} | {e.get('status')} | {provider}:{model} | {msg}")
    st.markdown(f"<div class='wow-log'>{normalize_text_output('\\n'.join(lines))}</div>", unsafe_allow_html=True)

def record_run(meta: Dict[str, Any]):
    st.session_state["run_history"].append(meta)

def run_agent(workspace: str, step_name: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResult:
    provider = infer_provider(model)
    prompt_tok = approx_tokens(system_prompt) + approx_tokens(user_prompt)
    log_event(workspace, step_name, "running", model=model, provider=provider, message=f"Starting (est_prompt_tokens={prompt_tok})")

    started = time.time()
    res = call_llm(model, system_prompt, user_prompt, max_tokens, temperature)
    elapsed = time.time() - started

    if res.error:
        log_event(workspace, step_name, "error", model=model, provider=provider, message=res.error)
    else:
        out_tok = approx_tokens(res.text)
        total_est = prompt_tok + out_tok
        log_event(workspace, step_name, "done", model=model, provider=provider, message=f"Done (elapsed={elapsed:.2f}s, est_total_tokens={total_est})")

    record_run({
        "run_id": str(uuid.uuid4()),
        "ts": now_utc_iso(),
        "workspace": workspace,
        "step": step_name,
        "provider": provider,
        "model": model,
        "status": "error" if res.error else "done",
        "error": res.error,
        "elapsed_s": res.elapsed_s,
        "prompt_tokens_est": prompt_tok,
        "output_tokens_est": approx_tokens(res.text),
        "usage": res.usage,
        "input_len": len(system_prompt or "") + len(user_prompt or ""),
        "output_len": len(res.text or ""),
    })
    return res


def agent_runner_ui(
    workspace: str,
    agent_id: str,
    default_title: str,
    input_text: str,
    prompt_help: str = "",
    default_system: str = "",
    default_model: Optional[str] = None,
    default_max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    output_key: Optional[str] = None,
    allow_edit_output: bool = True,
    height: int = 220,
    uid: str = "",
) -> Tuple[str, Optional[LLMResult]]:
    """
    Returns (effective_output_text, last_result)
    effective_output_text = edited if exists else generated
    """
    agents = st.session_state.get("agents", {}) or {}
    cfg = agents.get(agent_id, {})
    title = cfg.get("name", default_title)

    key_prefix = f"{workspace}:{agent_id}" + (f":{uid}" if uid else "")


    ########
    # ------------------ REPLACE FROM HERE ------------------
    # State key for the text_area widget
    widget_key = f"{key_prefix}:widget_out"

    # Run
    cols = st.columns([1, 1, 2])
    with cols[0]:
        run_clicked = st.button(t("run"), key=f"{key_prefix}:run_btn")
    with cols[1]:
        reset_clicked = st.button(t("reset_to_generated"), key=f"{key_prefix}:reset_btn")
    with cols[2]:
        st.caption("Handoff rule: the next step uses the *effective output* (edited) if present; otherwise uses generated output.")

    last_res = None
    if run_clicked:
        user_prompt = input_text or ""
        if not user_prompt.strip():
            log_event(workspace, title, "error", model=st.session_state[model_key], provider=infer_provider(st.session_state[model_key]), message="Empty input.")
            st.error("Input is empty.")
        else:
            last_res = run_agent(
                workspace=workspace,
                step_name=title,
                model=st.session_state[model_key],
                system_prompt=st.session_state[sys_key],
                user_prompt=user_prompt,
                max_tokens=int(st.session_state[mt_key]),
                temperature=float(st.session_state[temp_key]),
            )
            if last_res.error:
                st.error(last_res.error)
            else:
                st.session_state[out_gen_key] = last_res.text or ""
                # FIX: Always overwrite the UI with the fresh generated text
                st.session_state[out_eff_key] = st.session_state[out_gen_key]
                st.session_state[widget_key] = st.session_state[out_gen_key] # Forces the UI widget to refresh!

    if reset_clicked:
        st.session_state[out_eff_key] = st.session_state.get(out_gen_key, "")
        st.session_state[widget_key] = st.session_state.get(out_gen_key, "") # Forces the UI widget to refresh!

    # Output editor + preview
    st.markdown(f"#### {t('output')}")
    c1, c2 = st.columns([1, 1])
    with c1:
        if allow_edit_output:
            # FIX: Ensure widget key is initialized
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state[out_eff_key]
                
            # FIX: Remove 'value=' parameter. Let Streamlit manage it strictly via 'key'
            st.text_area(
                "Editable Output (effective handoff)",
                height=height,
                key=widget_key,
            )
            # Sync any manual edits back to our main variable
            st.session_state[out_eff_key] = st.session_state[widget_key]
        else:
            st.text_area(
                "Output",
                value=st.session_state.get(out_gen_key, ""),
                height=height,
                key=f"{out_gen_key}:readonly",
                disabled=True,
            )
    with c2:
        st.markdown(t("preview"))
        st.markdown(st.session_state[out_eff_key] or "")

    # Downloads
    dl1, dl2 = st.columns(2)
    dl_file_name = safe_filename(agent_id + ('_' + uid if uid else ''))
    with dl1:
        st.download_button(
            t("download_md"),
            data=(st.session_state[out_eff_key] or "").encode("utf-8"),
            file_name=f"{dl_file_name}.md",
            mime="text/markdown",
            key=f"{key_prefix}:dl_md",
        )
    with dl2:
        st.download_button(
            t("download_txt"),
            data=(st.session_state[out_eff_key] or "").encode("utf-8"),
            file_name=f"{dl_file_name}.txt",
            mime="text/plain",
            key=f"{key_prefix}:dl_txt",
        )

    return st.session_state[out_eff_key], last_res
    # ------------------ END OF REPLACEMENT ------------------
    

# ---------------------------
# Dashboard
# ---------------------------
def render_dashboard():
    st.markdown(f"<div class='wow-shell'><h2>{t('dashboard')}</h2></div>", unsafe_allow_html=True)

    # Status wall
    st.markdown("### " + t("status_wall"))
    pcols = st.columns(4)
    for i, p in enumerate(["openai", "gemini", "anthropic", "grok"]):
        label, ready = provider_readiness(p)
        with pcols[i]:
            st.markdown(f"<div class='wow-card'><div class='wow-accent'>{p.upper()}</div><div>{label}</div></div>", unsafe_allow_html=True)

    # Quick stats
    runs = st.session_state.get("run_history", [])
    total_runs = len(runs)
    errs = sum(1 for r in runs if r.get("status") == "error")
    st.metric("Total Runs", total_runs)
    st.metric("Errors", errs)

    # Charts if pandas/altair available
    if pd is None or alt is None:
        st.info("Install pandas + altair for dashboard charts.")
    else:
        if runs:
            df = pd.DataFrame(runs)
            # Normalize
            for c in ["workspace", "model", "provider", "status"]:
                if c not in df.columns:
                    df[c] = ""
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df["date"] = df["ts"].dt.date

            st.markdown("### Usage (by workspace)")
            chart1 = alt.Chart(df).mark_bar().encode(
                x=alt.X("workspace:N", sort="-y"),
                y="count():Q",
                color="status:N",
                tooltip=["workspace:N", "count():Q", "status:N"],
            ).properties(height=240)
            st.altair_chart(chart1, use_container_width=True)

            st.markdown("### Usage (by model)")
            chart2 = alt.Chart(df).mark_bar().encode(
                x=alt.X("model:N", sort="-y"),
                y="count():Q",
                color="provider:N",
                tooltip=["model:N", "count():Q", "provider:N"],
            ).properties(height=260)
            st.altair_chart(chart2, use_container_width=True)

            st.markdown("### Tokens (estimated) over time")
            df2 = df.groupby("date", as_index=False).agg({"prompt_tokens_est": "sum", "output_tokens_est": "sum"})
            df2["total_tokens_est"] = df2["prompt_tokens_est"] + df2["output_tokens_est"]
            chart3 = alt.Chart(df2).mark_line(point=True).encode(
                x="date:T",
                y="total_tokens_est:Q",
                tooltip=["date:T", "total_tokens_est:Q"],
            ).properties(height=220)
            st.altair_chart(chart3, use_container_width=True)

            st.markdown("### Recent Activity")
            st.dataframe(df.sort_values("ts", ascending=False).head(30), use_container_width=True, hide_index=True)
        else:
            st.caption("No runs yet.")

    st.markdown("### " + t("live_log"))
    render_live_log(workspace_filter=None, limit=120)

    st.download_button(
        "Download run log JSON",
        data=json.dumps(st.session_state.get("run_history", []), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="run_history.json",
        mime="application/json",
    )


# ---------------------------
# TW Premarket (placeholder runner surfaces; preserve tab)
# ---------------------------
def render_tw_premarket():
    workspace = "TW Premarket"
    st.markdown(f"<div class='wow-shell'><h2>{t('tw_premarket')}</h2></div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-banner'>This tab preserves the workspace structure and adds unified model/prompt controls and logging. "
                "You can wire your existing form/import logic here and route generation through the same agent runner.</div>", unsafe_allow_html=True)

    st.markdown("#### Input (Application info / context)")
    ctx = st.text_area("Paste application context (or your normalized JSON/text)", height=200, key=f"{workspace}:ctx")

    # Example: Draft agent
    out1, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_to_markdown",
        default_title="Application Draft Generator",
        input_text=ctx,
        default_system="Draft a TFDA premarket application section in Markdown from the provided info.",
        height=240,
        uid="premarket_draft"
    )

    # Example: Review agent (uses edited output)
    _out2, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_summary",
        default_title="Screen Review / Improvement",
        input_text=out1,
        default_system="Review the drafted document, identify gaps, and propose improvements in Markdown. Do not invent facts.",
        height=240,
        uid="premarket_review"
    )

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=120)


# ---------------------------
# 510(k) Intelligence (simple)
# ---------------------------
def render_510k_intelligence():
    workspace = "510(k) Intelligence"
    st.markdown(f"<div class='wow-shell'><h2>{t('intel_510k')}</h2></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        device = st.text_input("Device Name", key=f"{workspace}:device")
    with c2:
        knum = st.text_input("K-number", key=f"{workspace}:knum")
    with c3:
        sponsor = st.text_input("Sponsor", key=f"{workspace}:sponsor")

    extra = st.text_area("Extra context / pasted excerpts", height=160, key=f"{workspace}:extra")
    citations_mode = st.toggle("Citations mode (placeholders only; no browsing)", value=True, key=f"{workspace}:citations")
    out_lang = st.selectbox("Output language", options=["Follow UI", "English", "繁體中文"], index=0, key=f"{workspace}:out_lang")

    prompt = f"""Summarize and structure 510(k) intelligence for:
- Device: {device}
- K-number: {knum}
- Sponsor: {sponsor}

Context:
{extra}

Requirements:
- Use only provided information (no browsing).
- If info is missing, state it under "Gaps/Unknowns".
- If citations_mode is on, add placeholders like [Source: user-provided] near claims.
Output language: {out_lang}.
Citations mode: {citations_mode}.
"""

    out, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_summary",
        default_title="510(k) Intelligence Summarizer",
        input_text=prompt,
        default_system="You are a regulatory intelligence assistant. Produce a clear Markdown deliverable.",
        height=260,
    )

    # Optional entities post-process
    st.markdown("---")
    st.markdown("### Optional: Entities (20)")
    _entities, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_entities",
        default_title="Entities Extractor",
        input_text=out,
        height=220,
    )

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=120)


# ---------------------------
# PDF -> Markdown
# ---------------------------
def extract_pdf_text(file_bytes: bytes, page_start: int, page_end: int) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is not installed.")
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    n = len(reader.pages)
    s = max(1, page_start)
    e = min(n, page_end)
    if s > e:
        raise ValueError("Invalid page range.")
    chunks = []
    for i in range(s - 1, e):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception:
            txt = ""
        chunks.append(f"\n\n--- Page {i+1} ---\n{txt}".strip())
    return "\n".join(chunks).strip()

def render_pdf_to_md():
    workspace = "PDF→Markdown"
    st.markdown(f"<div class='wow-shell'><h2>{t('pdf2md')}</h2></div>", unsafe_allow_html=True)

    up = st.file_uploader("Upload PDF", type=["pdf"], key=f"{workspace}:pdf")
    table_mode = st.selectbox(
        "Table fidelity mode",
        options=["Fast (keep as text)", "Structured (attempt markdown tables)", "Conservative (avoid inventing table cells)"],
        index=1,
        key=f"{workspace}:table_mode",
    )

    extracted = ""
    if up is not None:
        file_bytes = up.read()
        # Page range
        if PyPDF2 is not None:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                n_pages = len(reader.pages)
            except Exception:
                n_pages = 1
        else:
            n_pages = 1

        c1, c2 = st.columns(2)
        with c1:
            ps = st.number_input("Start page", min_value=1, max_value=max(1, n_pages), value=1, step=1, key=f"{workspace}:ps")
        with c2:
            pe = st.number_input("End page", min_value=1, max_value=max(1, n_pages), value=min(5, n_pages), step=1, key=f"{workspace}:pe")

        if st.button("Extract text", key=f"{workspace}:extract"):
            try:
                log_event(workspace, "PDF Extraction", "running", message=f"pages={ps}-{pe}")
                extracted = extract_pdf_text(file_bytes, int(ps), int(pe))
                st.session_state[f"{workspace}:extracted"] = extracted
                log_event(workspace, "PDF Extraction", "done", message=f"chars={len(extracted)}")
            except Exception as e:
                log_event(workspace, "PDF Extraction", "error", message=f"{type(e).__name__}: {e}")
                st.error(f"{type(e).__name__}: {e}")

    extracted = st.session_state.get(f"{workspace}:extracted", "")
    st.markdown("#### Extracted Text")
    st.text_area("Extracted", value=extracted, height=220, key=f"{workspace}:extracted_view")

    # Transform
    prompt = f"""Convert the following extracted PDF text into clean Markdown.

Mode: {table_mode}

Rules:
- Do NOT invent missing text.
- Preserve headings/lists.
- If the PDF text indicates a table, present it as a Markdown table when feasible in the chosen mode.
- If uncertain, keep the original lines and mark with a note.

Extracted text:
{extracted}
"""
    _md, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="pdf_to_markdown",
        default_title="PDF → Markdown Transformer",
        input_text=prompt,
        height=260,
    )

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=120)


# ---------------------------
# 510(k) Review Pipeline (step-by-step, editable handoff)
# ---------------------------
def render_510k_pipeline():
    workspace = "510(k) Review Pipeline"
    st.markdown(f"<div class='wow-shell'><h2>{t('pipeline_510k')}</h2></div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-banner'>Step-by-step pipeline with editable handoff: "
                "each next step consumes the *effective output* (edited) of the prior step.</div>", unsafe_allow_html=True)

    st.markdown("### Inputs")
    sub_raw = st.text_area("Paste submission material (raw)", height=180, key=f"{workspace}:submission_raw")
    checklist_raw = st.text_area("Paste checklist", height=160, key=f"{workspace}:checklist_raw")
    out_lang = st.selectbox("Memo output language", options=["Follow UI", "English", "繁體中文"], index=0, key=f"{workspace}:memo_lang")

    # Step A: Structurer
    st.markdown("---")
    status_pill("Step A", "info")
    structured, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="pipeline_structurer",
        default_title="Submission Structurer",
        input_text=sub_raw,
        height=240,
    )

    # Step B: Checklist cleaner (optional)
    st.markdown("---")
    status_pill("Step B", "info")
    checklist_clean, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="pipeline_checklist_cleaner",
        default_title="Checklist Cleaner",
        input_text=checklist_raw,
        height=200,
    )

    # Step C: Memo builder
    st.markdown("---")
    status_pill("Step C", "info")
    memo_input = f"""Output language: {out_lang}

Structured submission:
{structured}

Checklist (cleaned):
{checklist_clean}

Task:
Write a review memo/report in Markdown. Use only these inputs. Do not invent facts.
Include "Assumptions/Gaps". Add tables where appropriate.
"""
    _memo, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="pipeline_memo_builder",
        default_title="Review Memo Builder",
        input_text=memo_input,
        height=300,
    )

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=140)


# ---------------------------
# 510(k) Report Generator Workspace (notes+template -> long report + skill.md)
# ---------------------------
def count_markdown_tables(md: str) -> int:
    # Heuristic: count header separator lines like |---|---|
    if not md:
        return 0
    return len(re.findall(r"^\s*\|.*\|\s*\n\s*\|[\s:\-|]+\|\s*$", md, flags=re.MULTILINE))

def extract_entities_table_row_count(md: str) -> Optional[int]:
    # Try to locate an "Entities" table: count table rows excluding header+separator
    if not md:
        return None
    # Find first large markdown table after a heading containing "Entities"
    m = re.search(r"(#+\s*Entities.*?\n)(\|.*\|\n\|[\s:\-|]+\|\n(?:\|.*\|\n)+)", md, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    table = m.group(2)
    rows = [r for r in table.strip().splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return 0
    return max(0, len(rows) - 2)

def approx_word_count(text: str) -> int:
    if not text:
        return 0
    # Word count heuristic; for CJK this undercounts but still a gate signal.
    words = re.findall(r"\b\w+\b", text)
    return len(words)

DEFAULT_REPORT_TEMPLATE = """# 510(k) Review Report Template (Default)

## 1. Executive Summary
## 2. Device Description
## 3. Intended Use / Indications for Use
## 4. Predicate Comparison (if applicable)
## 5. Substantial Equivalence Discussion
## 6. Performance Testing
## 7. Software / Cybersecurity (if applicable)
## 8. Biocompatibility / Sterilization (if applicable)
## 9. Labeling Review
## 10. Risk Management
## 11. Clinical Evidence (if applicable)
## 12. Manufacturing / Quality (if available)
## 13. Assumptions / Gaps
## 14. Conclusion / Recommendation
## Appendix A: Tables
## Appendix B: Entities (20)
"""

def render_report_generator():
    workspace = "510(k) Report Generator"
    st.markdown(f"<div class='wow-shell'><h2>{t('report_gen')}</h2></div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-banner'>Paste reviewer notes and a template (or use default). "
                "The agent drafts a 2000–3000 word Markdown report with ≥5 tables and exactly 20 entities, "
                "then generates a reusable skill.md.</div>", unsafe_allow_html=True)

    st.markdown("### Step 1 — Inputs")
    notes = st.text_area("Paste 510(k) review notes (required)", height=200, key=f"{workspace}:notes")
    template_mode = st.radio("Template source", ["Use default", "Paste template", "Paste prior report (infer template)"], horizontal=True, key=f"{workspace}:tmpl_mode")
    if template_mode == "Use default":
        template = st.text_area("Active template (editable)", value=st.session_state.get(f"{workspace}:template", DEFAULT_REPORT_TEMPLATE), height=220, key=f"{workspace}:template")
    else:
        template = st.text_area("Paste template / prior report", height=220, key=f"{workspace}:template")

    out_lang = st.selectbox("Output language", options=["English", "繁體中文"], index=0, key=f"{workspace}:lang")

    # Step 2 — Outline
    st.markdown("---")
    st.markdown("### Step 2 — Normalize & Outline")
    outline_input = f"""Output language: {out_lang}

Template:
{template}

Reviewer notes:
{notes}

Task:
Create a detailed report plan/outline mapped to the template headings.
Also produce a missing-info checklist. Do not invent facts; only use provided notes.
"""
    outline, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="report_outline",
        default_title="Normalize & Outline Agent",
        input_text=outline_input,
        height=220,
    )

    # Step 3 — Draft report
    st.markdown("---")
    st.markdown("### Step 3 — Draft Full Report")
    report_input = f"""Output language: {out_lang}

Template:
{template}

Report plan/outline:
{outline}

Reviewer notes:
{notes}

Hard requirements:
- 2000–3000 words target (allow slight variance if language differs).
- At least 5 Markdown tables.
- Include an "Entities" section with a Markdown table of exactly 20 entities with context.
- Do NOT invent facts; mark unknowns explicitly in "Assumptions / Gaps".
"""
    report_md, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="report_writer",
        default_title="Draft Full Report Agent",
        input_text=report_input,
        height=360,
    )

    # Step 4 — Quality gate (non-LLM heuristic checker)
    st.markdown("---")
    st.markdown("### Step 4 — Quality Gate (Heuristic)")
    wc = approx_word_count(report_md)
    tables = count_markdown_tables(report_md)
    ent_rows = extract_entities_table_row_count(report_md)

    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("Word Count (approx)", wc)
        st.caption("Target: 2000–3000 (English heuristic; CJK may differ).")
    with q2:
        st.metric("Markdown Table Count (heuristic)", tables)
        st.caption("Must be ≥ 5.")
    with q3:
        st.metric("Entities Rows (detected)", ent_rows if ent_rows is not None else "Not detected")
        st.caption("Must be exactly 20 rows (if detected).")

    ok_wc = (2000 <= wc <= 3000) if out_lang == "English" else (wc >= 1200)  # softer gate for CJK
    ok_tables = tables >= 5
    ok_entities = (ent_rows == 20) if ent_rows is not None else False

    st.markdown("**Gate Results**")
    status_pill("Word count", "done" if ok_wc else "error")
    status_pill("Tables", "done" if ok_tables else "error")
    status_pill("Entities(20)", "done" if ok_entities else "error")

    if not (ok_wc and ok_tables and ok_entities):
        st.warning("Quality gate indicates unmet constraints. Consider re-running Step 3 with adjusted prompt/model/max tokens, or manually editing.")

    # Step 5 — Skill.md generator
    st.markdown("---")
    st.markdown("### Step 5 — Generate skill.md")
    skill_input = f"""Output language: {out_lang}

Goal:
Generate a reusable skill.md that enables producing similar 510(k) review reports for related devices.

Inputs expected by the skill:
- Reviewer notes
- Report template

Constraints:
- Output report in Markdown
- 2000–3000 words (target)
- At least 5 tables
- Entities table with exactly 20 entities
- Do not invent facts; mark gaps

Use the following report as reference for structure and constraints:
{report_md}
"""
    _skill_md, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="skill_creator",
        default_title="skill.md Generator",
        input_text=skill_input,
        height=320,
    )

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=160)


# ---------------------------
# AI Note Keeper
# ---------------------------
def highlight_keywords_md(md: str, keywords: List[str], color: str) -> str:
    if not md or not keywords:
        return md or ""
    # Basic markdown-safe highlight using HTML <mark>
    # Streamlit markdown allows unsafe HTML if enabled; we will show preview separately.
    out = md
    for kw in sorted(set([k.strip() for k in keywords if k.strip()]), key=len, reverse=True):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        out = pattern.sub(lambda m: f"<span style='background:{color}; padding:0 2px; border-radius:4px;'><b>{m.group(0)}</b></span>", out)
    return out

def render_note_keeper():
    workspace = "AI Note Keeper"
    st.markdown(f"<div class='wow-shell'><h2>{t('note_keeper')}</h2></div>", unsafe_allow_html=True)

    st.markdown("<div class='wow-banner'>Paste text → transform to Markdown → run Magics (formatting, keywords, entities, chat, summary, action items, glossary).</div>", unsafe_allow_html=True)

    # Versions
    v = st.session_state["note_keeper"]
    st.markdown("### Notebook Versions (session-local)")
    vcols = st.columns([1, 2, 2])
    with vcols[0]:
        if st.button("Save snapshot", key=f"{workspace}:save_snapshot"):
            snap = {
                "ts": now_utc_iso(),
                "raw": st.session_state.get(f"{workspace}:raw", ""),
                "md": st.session_state.get(f"{workspace}:md", ""),
            }
            v["versions"].append(snap)
            v["active_idx"] = len(v["versions"]) - 1
    with vcols[1]:
        if v["versions"]:
            labels = [f"{i}: {s['ts']}" for i, s in enumerate(v["versions"])]
            idx = st.selectbox("Select snapshot", options=list(range(len(labels))), format_func=lambda i: labels[i], index=v["active_idx"] if v["active_idx"] is not None else 0, key=f"{workspace}:snap_idx")
            if st.button("Load snapshot", key=f"{workspace}:load_snapshot"):
                snap = v["versions"][idx]
                st.session_state[f"{workspace}:raw"] = snap.get("raw", "")
                st.session_state[f"{workspace}:md"] = snap.get("md", "")
                v["active_idx"] = idx
                st.rerun()
        else:
            st.caption("No snapshots saved yet.")
    with vcols[2]:
        safety = st.toggle("Safety mode (conservative rewriting)", value=True, key=f"{workspace}:safety")

    st.markdown("### Step 1 — Paste Notes")
    raw = st.text_area("Paste text", height=200, key=f"{workspace}:raw")

    st.markdown("---")
    st.markdown("### Step 2 — Transform into Markdown")
    transform_prompt = f"""Transform the following into Markdown.
Safety mode: {safety}

Rules:
- Preserve meaning.
- Do not invent facts.
- Use headings, bullets, and tables when appropriate.
- If unclear, keep original wording.

Text:
{raw}
"""
    md_out, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_to_markdown",
        default_title="Note → Markdown Transformer",
        input_text=transform_prompt,
        height=260,
        output_key=f"{workspace}:md",
        uid="step2",
    )

    # Magics
    st.markdown("---")
    st.markdown("## AI Magics")

    # AI Formatting (reformat md)
    st.markdown("### Magic 1 — AI Formatting")
    fmt_input = f"""Reformat the following Markdown to be more consistent and readable.
Safety mode: {safety}
Do not invent facts.

Markdown:
{md_out}
"""
    _fmt, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_to_markdown",
        default_title="AI Formatting",
        input_text=fmt_input,
        height=220,
        uid="fmt",
    )

    # AI Keywords (highlight)
    st.markdown("---")
    st.markdown("### Magic 2 — AI Keywords (highlight)")
    kw = st.text_input("Keywords (comma-separated)", key=f"{workspace}:kw")
    color = st.color_picker("Highlight color", value="#FFD166", key=f"{workspace}:kw_color")
    kws = [k.strip() for k in (kw or "").split(",") if k.strip()]
    highlighted = highlight_keywords_md(md_out, kws, color)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.text_area("Highlighted (raw HTML-in-markdown)", value=highlighted, height=220, key=f"{workspace}:highlighted")
    with c2:
        st.markdown("Preview (unsafe HTML enabled)")
        st.markdown(highlighted, unsafe_allow_html=True)

    st.download_button(
        "Download highlighted.md",
        data=(highlighted or "").encode("utf-8"),
        file_name="note_highlighted.md",
        mime="text/markdown",
    )

    # AI Entities
    st.markdown("---")
    st.markdown("### Magic 3 — AI Entities (20)")
    ent_input = f"""Extract exactly 20 entities with context from the following note.

Note (Markdown):
{md_out}

Output:
A Markdown table with columns: Entity | Type | Context | Evidence pointer
Exactly 20 rows (excluding header). Do not invent.
"""
    _ents, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_entities",
        default_title="AI Entities",
        input_text=ent_input,
        height=260,
        uid="ents",
    )

    # AI Chat
    st.markdown("---")
    st.markdown("### Magic 4 — AI Chat")
    chat_q = st.text_area("Ask a question about the note", height=100, key=f"{workspace}:chat_q")
    chat_input = f"""Use the note below to answer the user's question. If the answer is not supported, say so.

Note:
{md_out}

Question:
{chat_q}
"""
    _chat, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_summary",
        default_title="AI Chat",
        input_text=chat_input,
        height=220,
        uid="chat",
    )

    # AI Summary
    st.markdown("---")
    st.markdown("### Magic 5 — AI Summary")
    sum_prompt = st.text_area("Custom summary prompt (optional)", height=80, key=f"{workspace}:sum_prompt")
    sum_input = f"""Summarize the note. {("Custom instructions: " + sum_prompt) if sum_prompt.strip() else ""}
Safety mode: {safety}
Do not invent.

Note:
{md_out}
"""
    _sum, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_summary",
        default_title="AI Summary",
        input_text=sum_input,
        height=220,
        uid="sum",
    )

    # Two additional Magics (decided here): Action Items + Glossary
    st.markdown("---")
    st.markdown("### Magic 6 — Action Items Extractor")
    act_input = f"""From the note below, extract action items as a Markdown checklist with:
- Owner (if known, else 'TBD')
- Due date (if known, else 'TBD')
- Action
- Evidence pointer (quote or section)

Do not invent.

Note:
{md_out}
"""
    _act, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_summary",
        default_title="Action Items",
        input_text=act_input,
        height=220,
        uid="act",
    )

    st.markdown("---")
    st.markdown("### Magic 7 — Glossary Builder")
    glo_input = f"""Build a glossary from the note:
- 15–30 terms
- Each term: definition + where it appears (evidence pointer)
Do not invent. Prefer exact wording when possible.

Note:
{md_out}
"""
    _glo, _ = agent_runner_ui(
        workspace=workspace,
        agent_id="note_summary",
        default_title="Glossary",
        input_text=glo_input,
        height=240,
        uid="glo",
    )

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=160)


# ---------------------------
# Agents Config Studio
# ---------------------------
def validate_agents(agents: Dict[str, Any]) -> Dict[str, List[str]]:
    issues = {"unknown_models": [], "missing_system_prompts": [], "token_anomalies": []}
    for agent_id, cfg in (agents or {}).items():
        model = (cfg.get("model") or "").strip()
        if model and model not in MODEL_CATALOG:
            issues["unknown_models"].append(f"{agent_id}: {model}")
        if not (cfg.get("system_prompt") or "").strip():
            issues["missing_system_prompts"].append(agent_id)
        mt = cfg.get("max_tokens", None)
        try:
            mt_i = int(mt)
            if mt_i < 128 or mt_i > 8192:
                issues["token_anomalies"].append(f"{agent_id}: {mt_i}")
        except Exception:
            issues["token_anomalies"].append(f"{agent_id}: {mt}")
    return issues

def render_agents_studio():
    workspace = "Agents Config Studio"
    st.markdown(f"<div class='wow-shell'><h2>{t('agents_studio')}</h2></div>", unsafe_allow_html=True)

    if yaml is None:
        st.error("Missing dependency: pyyaml. Install it to use agents.yaml editor.")
        return

    st.markdown("### agents.yaml")
    up = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"], key=f"{workspace}:upload")
    if up is not None:
        st.session_state["agents_yaml_text"] = up.read().decode("utf-8", errors="replace")
        st.session_state["agents"] = load_agents_from_text(st.session_state["agents_yaml_text"])
        log_event(workspace, "agents.yaml", "done", message="Uploaded and parsed")

    st.session_state["agents_yaml_text"] = st.text_area(
        "Edit YAML",
        value=st.session_state["agents_yaml_text"],
        height=320,
        key=f"{workspace}:yaml_editor",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Parse / Apply", key=f"{workspace}:apply"):
            st.session_state["agents"] = load_agents_from_text(st.session_state["agents_yaml_text"])
            log_event(workspace, "agents.yaml", "done", message="Parsed and applied")
            st.success("Applied.")
    with c2:
        st.download_button(
            "Download agents.yaml",
            data=(st.session_state["agents_yaml_text"] or "").encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
            key=f"{workspace}:dl_agents",
        )
    with c3:
        st.caption("Tip: Add/remove agent defaults here. Runtime overrides do not modify the YAML unless you paste them back.")

    st.markdown("---")
    st.markdown("### Agents Overview")
    agents = st.session_state.get("agents", {}) or {}
    if pd is not None and agents:
        df = pd.DataFrame([
            {"agent_id": aid, "name": cfg.get("name"), "model": cfg.get("model"), "max_tokens": cfg.get("max_tokens"), "category": cfg.get("category")}
            for aid, cfg in agents.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.json(agents)

    st.markdown("### Validation")
    issues = validate_agents(agents)
    if any(issues.values()):
        if issues["unknown_models"]:
            st.warning("Unknown models:\n- " + "\n- ".join(issues["unknown_models"]))
        if issues["missing_system_prompts"]:
            st.warning("Missing system prompts:\n- " + "\n- ".join(issues["missing_system_prompts"]))
        if issues["token_anomalies"]:
            st.warning("Token anomalies:\n- " + "\n- ".join(issues["token_anomalies"]))
    else:
        st.success("No issues detected.")

    st.markdown("---")
    st.markdown("### Test Agent (mini-runner)")
    agent_ids = list(agents.keys()) if agents else []
    if agent_ids:
        aid = st.selectbox("Agent ID", options=agent_ids, key=f"{workspace}:test_aid")
        test_input = st.text_area("Test input", height=120, key=f"{workspace}:test_input")
        _out, _ = agent_runner_ui(
            workspace=workspace,
            agent_id=aid,
            default_title=f"Test: {aid}",
            input_text=test_input,
            height=220,
        )
    else:
        st.info("No agents loaded.")

    with st.expander(t("live_log"), expanded=False):
        render_live_log(workspace_filter=workspace, limit=120)


# ---------------------------
# Main UI
# ---------------------------
ui_global_settings()
inject_css(st.session_state["ui_theme"], st.session_state["ui_style"])

st.markdown(f"<div class='wow-shell'><h1>{t('app_title')}</h1>"
            f"<div class='wow-pill'>Theme: {st.session_state['ui_theme']}</div>"
            f"<div class='wow-pill'>Lang: {st.session_state['ui_language']}</div>"
            f"<div class='wow-pill'>Style: {st.session_state['ui_style']}</div>"
            f"</div>", unsafe_allow_html=True)

tabs = st.tabs([
    t("dashboard"),
    t("tw_premarket"),
    t("intel_510k"),
    t("pdf2md"),
    t("pipeline_510k"),
    t("report_gen"),
    t("note_keeper"),
    t("agents_studio"),
])

with tabs[0]:
    render_dashboard()
with tabs[1]:
    render_tw_premarket()
with tabs[2]:
    render_510k_intelligence()
with tabs[3]:
    render_pdf_to_md()
with tabs[4]:
    render_510k_pipeline()
with tabs[5]:
    render_report_generator()
with tabs[6]:
    render_note_keeper()
with tabs[7]:
    render_agents_studio()
