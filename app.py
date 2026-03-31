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
st.set_page_config(page_title="WOW Regulatory Workbench", layout="wide", initial_sidebar_state="expanded")


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

# Model catalog
MODEL_CATALOG = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4o",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-1.5-pro",
    # Anthropic
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    # Grok (xAI)
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

def infer_provider(model_id: str) -> str:
    m = (model_id or "").lower().strip()
    if m.startswith("gpt-"): return "openai"
    if m.startswith("gemini-"): return "gemini"
    if m.startswith("claude-"): return "anthropic"
    if m.startswith("grok-"): return "grok"
    if "gpt" in m: return "openai"
    if "gemini" in m: return "gemini"
    if "claude" in m: return "anthropic"
    if "grok" in m or "xai" in m: return "grok"
    return "unknown"


PROVIDER_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVEAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
}

GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

# ---------------------------
# Localization
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
    "run": {"en": "▶ Run Agent", "zh": "▶ 執行 Agent"},
    "rerun": {"en": "Re-run", "zh": "重新執行"},
    "reset_to_generated": {"en": "🔄 Reset to Generated", "zh": "🔄 重設為生成結果"},
    "output": {"en": "Output", "zh": "輸出"},
    "preview": {"en": "Preview", "zh": "預覽"},
    "download_md": {"en": "⬇️ Download .md", "zh": "⬇️ 下載 .md"},
    "download_txt": {"en": "⬇️ Download .txt", "zh": "⬇️ 下載 .txt"},
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

    st.session_state.setdefault("run_log", [])  
    st.session_state.setdefault("run_history", [])  

    st.session_state.setdefault("keys", {"openai": "", "gemini": "", "anthropic": "", "grok": ""})
    st.session_state.setdefault("agents_yaml_text", "")
    st.session_state.setdefault("agents", {})

    # Defaults updated for modern models
    st.session_state.setdefault("global_temperature", 0.2)
    st.session_state.setdefault("global_max_tokens", 12000)
    st.session_state.setdefault("global_default_model", "gpt-4o-mini")

    st.session_state.setdefault("pipeline", {})
    st.session_state.setdefault("note_keeper", {"versions": [], "active_idx": None})

ss_init()


# ---------------------------
# WOW CSS (NEAT UI)
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

    if theme.lower().startswith("light"):
        bg1 = "#f7f7fb"
        bg2 = "#ffffff"

    css = f"""
    <style>
      .wow-shell {{
        background: radial-gradient(1200px 700px at 10% 10%, {bg2} 0%, {bg1} 60%, {bg1} 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      }}
      .wow-card {{
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 16px 16px;
        transition: transform 0.2s ease;
      }}
      .wow-card:hover {{
        transform: translateY(-2px);
      }}
      .wow-accent {{
        color: {accent};
        font-weight: 700;
        font-size: 1.1em;
      }}
      .wow-pill {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        margin-right: 8px;
        font-size: 12px;
        font-weight: 600;
      }}
      .wow-log {{
        font-family: "Fira Code", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        white-space: pre-wrap;
        line-height: 1.45;
        background: rgba(0,0,0,0.2);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
      }}
      .wow-banner {{
        border-left: 4px solid {accent};
        padding: 14px 18px;
        border-radius: 10px;
        background: rgba(255,255,255,0.05);
        border-top: 1px solid rgba(255,255,255,0.08);
        border-right: 1px solid rgba(255,255,255,0.08);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 20px;
        font-size: 14px;
      }}
      /* Smooth Inputs */
      textarea, input, select {{
        border-radius: 10px !important;
      }}
      .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
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
    if not text: return 0
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
        "status": status, 
        "model": model,
        "provider": provider,
        "message": message,
        "meta": meta or {},
    }
    st.session_state["run_log"].append(evt)

def provider_env_key(provider: str) -> Optional[str]:
    for envname in PROVIDER_ENV_VARS.get(provider, []):
        val = os.getenv(envname)
        if val: return val
    return None

def provider_key(provider: str) -> Optional[str]:
    env = provider_env_key(provider)
    if env: return env
    v = (st.session_state.get("keys", {}) or {}).get(provider, "")
    return v.strip() or None

def provider_readiness(provider: str) -> Tuple[str, bool]:
    env = provider_env_key(provider)
    if env: return (t("ready_env"), True)
    sess = (st.session_state.get("keys", {}) or {}).get(provider, "").strip()
    if sess: return (t("ready_session"), True)
    return (t("missing"), False)

def normalize_text_output(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    try: return json.dumps(x, ensure_ascii=False, indent=2)
    except Exception: return str(x)

def merge_system_user_for_gemini(system_prompt: str, user_prompt: str) -> str:
    sys = (system_prompt or "").strip()
    usr = (user_prompt or "").strip()
    if sys and usr: return f"### SYSTEM\n{sys}\n\n### USER\n{usr}"
    if sys: return f"### SYSTEM\n{sys}"
    return usr


# ---------------------------
# LLM Clients
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

def call_openai_like(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, api_key: str, base_url: Optional[str] = None) -> LLMResult:
    started = time.time()
    provider = "openai" if base_url is None else "grok"
    try:
        from openai import OpenAI
    except Exception as e:
        return LLMResult("", {}, None, provider, model, time.time() - started, f"Missing dependency 'openai': {e}")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        messages = []
        if (system_prompt or "").strip(): messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        content = resp.choices[0].message.content or "" if hasattr(resp.choices[0].message, 'content') else normalize_text_output(resp)
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", None), "output_tokens": getattr(resp.usage, "completion_tokens", None)} if hasattr(resp, 'usage') else {}

        return LLMResult(content, usage, resp, provider, model, time.time() - started)
    except Exception as e:
        return LLMResult("", {}, None, provider, model, time.time() - started, f"{type(e).__name__}: {e}")

def call_gemini(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, api_key: str) -> LLMResult:
    started = time.time()
    try:
        import google.generativeai as genai
    except Exception as e:
        return LLMResult("", {}, None, "gemini", model, time.time() - started, f"Missing dependency 'google-generativeai': {e}")

    try:
        genai.configure(api_key=api_key)
        prompt = merge_system_user_for_gemini(system_prompt, user_prompt)
        gm = genai.GenerativeModel(model_name=model)
        resp = gm.generate_content(prompt, generation_config={"temperature": float(temperature), "max_output_tokens": int(max_tokens)})

        text = resp.text if hasattr(resp, "text") and resp.text else ""
        if not text:
            try:
                text = "\n".join([p.text for cand in getattr(resp, "candidates", []) for p in getattr(getattr(cand, "content", None), "parts", []) if hasattr(p, "text")])
            except Exception: pass

        usage = {}
        um = getattr(resp, "usage_metadata", None)
        if um: usage = {"input_tokens": getattr(um, "prompt_token_count", None), "output_tokens": getattr(um, "candidates_token_count", None)}

        return LLMResult(text, usage, resp, "gemini", model, time.time() - started)
    except Exception as e:
        return LLMResult("", {}, None, "gemini", model, time.time() - started, f"{type(e).__name__}: {e}")

def call_anthropic(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float, api_key: str) -> LLMResult:
    started = time.time()
    try:
        import anthropic
    except Exception as e:
        return LLMResult("", {}, None, "anthropic", model, time.time() - started, f"Missing dependency 'anthropic': {e}")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            system=(system_prompt or "").strip() or None,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "\n".join([b.text for b in getattr(msg, "content", []) if getattr(b, "type", None) == "text"])
        usage = {"input_tokens": getattr(msg.usage, "input_tokens", None), "output_tokens": getattr(msg.usage, "output_tokens", None)} if hasattr(msg, 'usage') else {}

        return LLMResult(text, usage, msg, "anthropic", model, time.time() - started)
    except Exception as e:
        return LLMResult("", {}, None, "anthropic", model, time.time() - started, f"{type(e).__name__}: {e}")

def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResult:
    provider = infer_provider(model)
    key = provider_key(provider)

    if provider == "unknown":
        return LLMResult("", {}, None, provider, model, 0.0, f"Unknown provider for model '{model}'.")
    if not key:
        return LLMResult("", {}, None, provider, model, 0.0, f"Missing API key for provider '{provider}'.")

    if provider == "openai": return call_openai_like(model, system_prompt, user_prompt, max_tokens, temperature, key)
    if provider == "grok": return call_openai_like(model, system_prompt, user_prompt, max_tokens, temperature, key, GROK_BASE_URL)
    if provider == "gemini": return call_gemini(model, system_prompt, user_prompt, max_tokens, temperature, key)
    if provider == "anthropic": return call_anthropic(model, system_prompt, user_prompt, max_tokens, temperature, key)

    return LLMResult("", {}, None, provider, model, 0.0, f"Provider '{provider}' not implemented.")


# ---------------------------
# agents.yaml
# ---------------------------
DEFAULT_AGENTS_YAML = """# agents.yaml (defaults)
agents:
  pdf_to_markdown:
    name: PDF → Markdown Transformer
    model: gemini-2.5-flash
    max_tokens: 12000
    system_prompt: |
      You are a careful document transformer. Convert extracted PDF text into clean Markdown.
      Do not invent content. Preserve headings, lists, tables when clearly present.
  note_to_markdown:
    name: Note → Markdown
    model: gpt-4o-mini
    max_tokens: 12000
    system_prompt: |
      Transform user notes into clear Markdown with headings and bullet lists.
      Do not invent facts; mark unknowns explicitly.
  note_entities:
    name: Note Entities (20)
    model: gpt-4o-mini
    max_tokens: 4000
    system_prompt: |
      Extract exactly 20 entities from the note and present as a Markdown table with columns:
      Entity | Type | Context | Evidence pointer. Do not invent.
  note_summary:
    name: Note Summary
    model: gpt-4o-mini
    max_tokens: 8000
    system_prompt: |
      Summarize the note faithfully. Do not invent.
  pipeline_structurer:
    name: 510(k) Submission Structurer
    model: gemini-2.5-flash
    max_tokens: 12000
    system_prompt: |
      Structure the submission text into a clean Markdown outline with sections.
      Do not add new facts. If something is missing, note it under "Gaps".
  pipeline_checklist_cleaner:
    name: Checklist Cleaner
    model: gpt-4o-mini
    max_tokens: 8000
    system_prompt: |
      Clean and normalize the checklist into Markdown with clear numbering.
      Do not invent requirements.
  pipeline_memo_builder:
    name: Review Memo Builder
    model: gpt-4o
    max_tokens: 12000
    system_prompt: |
      Write a review memo in Markdown using only the provided structured submission and checklist.
      Do not invent facts. Include "Assumptions/Gaps".
  report_outline:
    name: 510(k) Report Outline Agent
    model: gpt-4o-mini
    max_tokens: 8000
    system_prompt: |
      Create a detailed report plan/outline mapped to the provided template headings.
      Include a missing-info checklist. Do not invent.
  report_writer:
    name: 510(k) Report Writer Agent
    model: claude-3-5-sonnet-2024-10
    max_tokens: 12000
    system_prompt: |
      Write a comprehensive 2000–3000 word Markdown 510(k) review report.
      Must include at least 5 Markdown tables and an entities table with exactly 20 entities.
      Do not invent facts; mark unknowns in "Assumptions/Gaps".
  skill_creator:
    name: Skill.md Generator
    model: gpt-4o-mini
    max_tokens: 12000
    system_prompt: |
      Create a reusable skill.md that instructs an AI to produce similar 510(k) review reports.
      Include triggering language, steps, constraints (2k–3k words, 5+ tables, 20 entities), safety rules, and 2–3 test prompts.
"""

def load_agents_from_text(yaml_text: str) -> Dict[str, Any]:
    if not yaml: return {}
    try:
        data = yaml.safe_load(yaml_text) or {}
        agents = data.get("agents", {}) or {}
        norm = {}
        for agent_id, cfg in agents.items():
            if not isinstance(cfg, dict): continue
            norm[agent_id] = {
                "name": cfg.get("name", agent_id),
                "model": cfg.get("model", st.session_state.get("global_default_model", "gpt-4o-mini")),
                "max_tokens": int(cfg.get("max_tokens", st.session_state.get("global_max_tokens", 12000))),
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
# UI Sidebar Components
# ---------------------------
def ui_provider_keys():
    st.sidebar.subheader(t("provider_keys"))
    for provider in ["openai", "gemini", "anthropic", "grok"]:
        label, ready = provider_readiness(provider)
        st.sidebar.markdown(f"**{provider.upper()}**: `{label}`")
        if provider_env_key(provider): continue
        st.session_state["keys"][provider] = st.sidebar.text_input(
            f"{provider.upper()} API Key",
            value=st.session_state["keys"].get(provider, ""),
            type="password",
            help="Stored in session only (not persisted).",
        )

def ui_global_settings():
    st.sidebar.header(t("global_settings"))
    ui_language = st.sidebar.selectbox(t("language"), options=list(LANGS.keys()), index=list(LANGS.keys()).index(st.session_state["ui_language"]) if st.session_state["ui_language"] in LANGS else 0)
    st.session_state["ui_language"] = ui_language
    st.session_state["ui_lang_code"] = LANGS[ui_language]

    theme = st.sidebar.selectbox(t("theme"), options=["Dark", "Light"], index=0 if st.session_state["ui_theme"].lower().startswith("dark") else 1)
    st.session_state["ui_theme"] = theme

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        style = st.selectbox(t("painter_style"), options=PAINTER_STYLES, index=PAINTER_STYLES.index(st.session_state["ui_style"]) if st.session_state["ui_style"] in PAINTER_STYLES else 0)
        st.session_state["ui_style"] = style
    with col2:
        if st.button(t("jackpot")):
            st.session_state["ui_style"] = random.choice(PAINTER_STYLES)
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ LLM Defaults")
    st.session_state["global_default_model"] = st.sidebar.selectbox(t("model"), options=MODEL_CATALOG, index=MODEL_CATALOG.index(st.session_state["global_default_model"]) if st.session_state["global_default_model"] in MODEL_CATALOG else 0)
    st.session_state["global_max_tokens"] = st.sidebar.slider(t("max_tokens"), min_value=256, max_value=32768, value=int(st.session_state["global_max_tokens"]), step=512)
    st.session_state["global_temperature"] = st.sidebar.slider(t("temperature"), min_value=0.0, max_value=1.0, value=float(st.session_state["global_temperature"]), step=0.05)

    st.sidebar.markdown("---")
    ui_provider_keys()
    st.sidebar.markdown("---")
    st.sidebar.caption("🔒 Security: API keys are never written to logs or downloads. Env keys are never displayed.")


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
    if workspace_filter: evts = [e for e in evts if e.get("workspace") == workspace_filter]
    evts = evts[-limit:]
    lines = [f"[{e.get('ts')}] {e.get('workspace')}/{e.get('step')} | {e.get('status')} | {e.get('provider')}:{e.get('model')} | {e.get('message')}" for e in evts]
    st.markdown(f"<div class='wow-log'>{normalize_text_output('\\n'.join(lines))}</div>", unsafe_allow_html=True)

def record_run(meta: Dict[str, Any]):
    st.session_state["run_history"].append(meta)

def run_agent(workspace: str, step_name: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResult:
    provider = infer_provider(model)
    prompt_tok = approx_tokens(system_prompt) + approx_tokens(user_prompt)
    log_event(workspace, step_name, "running", model=model, provider=provider, message=f"Starting (est_prompt_tokens={prompt_tok})")

    res = call_llm(model, system_prompt, user_prompt, max_tokens, temperature)

    if res.error:
        log_event(workspace, step_name, "error", model=model, provider=provider, message=res.error)
    else:
        out_tok = approx_tokens(res.text)
        total_est = prompt_tok + out_tok
        log_event(workspace, step_name, "done", model=model, provider=provider, message=f"Done (elapsed={res.elapsed_s:.2f}s, est_total_tokens={total_est})")

    record_run({
        "run_id": str(uuid.uuid4()), "ts": now_utc_iso(), "workspace": workspace, "step": step_name,
        "provider": provider, "model": model, "status": "error" if res.error else "done",
        "error": res.error, "elapsed_s": res.elapsed_s, "prompt_tokens_est": prompt_tok,
        "output_tokens_est": approx_tokens(res.text), "usage": res.usage,
    })
    return res


def agent_runner_ui(
    workspace: str, agent_id: str, default_title: str, input_text: str,
    prompt_help: str = "", default_system: str = "", default_model: Optional[str] = None,
    default_max_tokens: Optional[int] = None, temperature: Optional[float] = None,
    output_key: Optional[str] = None, allow_edit_output: bool = True, height: int = 220, uid: str = "",
) -> Tuple[str, Optional[LLMResult]]:
    """ Returns (effective_output_text, last_result) """
    agents = st.session_state.get("agents", {}) or {}
    cfg = agents.get(agent_id, {})
    title = cfg.get("name", default_title)
    key_prefix = f"{workspace}:{agent_id}" + (f":{uid}" if uid else "")

    sys_key = f"{key_prefix}:system"
    model_key = f"{key_prefix}:model"
    mt_key = f"{key_prefix}:max_tokens"
    out_gen_key = f"{key_prefix}:out_generated"
    out_eff_key = output_key or f"{key_prefix}:out_effective"
    temp_key = f"{key_prefix}:temperature"
    widget_key = f"{key_prefix}:widget_out"

    st.session_state.setdefault(sys_key, cfg.get("system_prompt", default_system))
    st.session_state.setdefault(model_key, cfg.get("model", default_model or st.session_state["global_default_model"]))
    st.session_state.setdefault(mt_key, int(cfg.get("max_tokens", default_max_tokens or st.session_state["global_max_tokens"])))
    st.session_state.setdefault(out_gen_key, "")
    st.session_state.setdefault(out_eff_key, "")
    st.session_state.setdefault(temp_key, float(temperature) if temperature is not None else float(st.session_state["global_temperature"]))

    st.markdown(f"### 🤖 {title}")

    with st.expander("⚙️ Prompt & Model Controls", expanded=False):
        st.session_state[model_key] = st.selectbox(t("model"), options=MODEL_CATALOG, index=MODEL_CATALOG.index(st.session_state[model_key]) if st.session_state[model_key] in MODEL_CATALOG else 0, key=f"{model_key}:widget")
        st.session_state[mt_key] = st.slider(t("max_tokens"), min_value=256, max_value=32768, value=int(st.session_state[mt_key]), step=512, key=f"{mt_key}:widget")
        st.session_state[temp_key] = st.slider(t("temperature"), min_value=0.0, max_value=1.0, value=float(st.session_state[temp_key]), step=0.05, key=f"{temp_key}:widget")
        st.session_state[sys_key] = st.text_area(t("system_prompt"), value=st.session_state[sys_key], height=160, help=prompt_help, key=f"{sys_key}:widget")

    # Run / Reset Buttons
    cols = st.columns([1, 1, 2])
    with cols[0]:
        run_clicked = st.button(t("run"), type="primary", key=f"{key_prefix}:run_btn", use_container_width=True)
    with cols[1]:
        reset_clicked = st.button(t("reset_to_generated"), key=f"{key_prefix}:reset_btn", use_container_width=True)
    with cols[2]:
        st.caption("✨ *Next step uses edited output if present; otherwise generated.*")

    last_res = None
    if run_clicked:
        user_prompt = input_text or ""
        if not user_prompt.strip():
            log_event(workspace, title, "error", model=st.session_state[model_key], provider=infer_provider(st.session_state[model_key]), message="Empty input.")
            st.error("⚠️ Input is empty.")
        else:
            # Interactive Progress Indicator / Live Log
            with st.status(f"🚀 **Running Agent:** {title}...", expanded=True) as status:
                st.write(f"🧩 **Model:** `{st.session_state[model_key]}`")
                st.write(f"📝 **Est. Tokens:** ~{approx_tokens(st.session_state[sys_key]) + approx_tokens(user_prompt)}")
                st.info("💡 **Tip:** To stop execution immediately, click the **Stop** button in the top right corner of the screen.")
                
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
                    status.update(label="❌ Execution Failed", state="error", expanded=True)
                    st.error(last_res.error)
                else:
                    status.update(label=f"✅ Done in {last_res.elapsed_s:.1f}s", state="complete", expanded=False)
                    st.session_state[out_gen_key] = last_res.text or ""
                    st.session_state[out_eff_key] = st.session_state[out_gen_key]
                    st.session_state[widget_key] = st.session_state[out_gen_key]  # Sync the UI

    if reset_clicked:
        st.session_state[out_eff_key] = st.session_state.get(out_gen_key, "")
        st.session_state[widget_key] = st.session_state.get(out_gen_key, "")

    # Output editor
    st.markdown(f"#### 📝 {t('output')}")
    c1, c2 = st.columns([1, 1])
    with c1:
        if allow_edit_output:
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state[out_eff_key]
            
            st.text_area("Editable Output (effective handoff)", height=height, key=widget_key)
            st.session_state[out_eff_key] = st.session_state[widget_key]
        else:
            st.text_area("Output", value=st.session_state.get(out_gen_key, ""), height=height, key=f"{out_gen_key}:readonly", disabled=True)
    with c2:
        st.markdown(t("preview"))
        st.markdown(st.session_state[out_eff_key] or "")

    # Downloads
    dl1, dl2 = st.columns(2)
    dl_file_name = safe_filename(agent_id + ('_' + uid if uid else ''))
    with dl1:
        st.download_button(t("download_md"), data=(st.session_state[out_eff_key] or "").encode("utf-8"), file_name=f"{dl_file_name}.md", mime="text/markdown", key=f"{key_prefix}:dl_md")
    with dl2:
        st.download_button(t("download_txt"), data=(st.session_state[out_eff_key] or "").encode("utf-8"), file_name=f"{dl_file_name}.txt", mime="text/plain", key=f"{key_prefix}:dl_txt")

    return st.session_state[out_eff_key], last_res


# ---------------------------
# Workspace Tabs
# ---------------------------
def render_dashboard():
    st.markdown(f"<div class='wow-shell'><h2>📊 {t('dashboard')}</h2></div>", unsafe_allow_html=True)
    st.markdown("### " + t("status_wall"))
    pcols = st.columns(4)
    for i, p in enumerate(["openai", "gemini", "anthropic", "grok"]):
        label, ready = provider_readiness(p)
        with pcols[i]:
            st.markdown(f"<div class='wow-card'><div class='wow-accent'>{p.upper()}</div><div>{label}</div></div>", unsafe_allow_html=True)

    runs = st.session_state.get("run_history", [])
    st.metric("Total Runs", len(runs))
    st.metric("Errors", sum(1 for r in runs if r.get("status") == "error"))

    if pd is None or alt is None:
        st.info("Install pandas + altair for dashboard charts.")
    elif runs:
        df = pd.DataFrame(runs)
        for c in ["workspace", "model", "provider", "status"]:
            if c not in df.columns: df[c] = ""
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        
        st.markdown("### Recent Activity")
        st.dataframe(df.sort_values("ts", ascending=False).head(30), use_container_width=True, hide_index=True)

    st.markdown("### " + t("live_log"))
    render_live_log(workspace_filter=None, limit=120)

def render_tw_premarket():
    workspace = "TW Premarket"
    st.markdown(f"<div class='wow-shell'><h2>🇹🇼 {t('tw_premarket')}</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='wow-banner'>Preserved workspace structure. Wire your specific context below.</div>", unsafe_allow_html=True)

    ctx = st.text_area("Paste application context", height=200, key=f"{workspace}:ctx")

    out1, _ = agent_runner_ui(workspace, "note_to_markdown", "Application Draft Generator", ctx, "Draft a TFDA premarket application section in Markdown.", uid="tw_draft")
    _out2, _ = agent_runner_ui(workspace, "note_summary", "Screen Review / Improvement", out1, "Review the drafted document, identify gaps.", uid="tw_review")

def render_510k_intelligence():
    workspace = "510(k) Intelligence"
    st.markdown(f"<div class='wow-shell'><h2>🧠 {t('intel_510k')}</h2></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    device = c1.text_input("Device Name", key=f"{workspace}:device")
    knum = c2.text_input("K-number", key=f"{workspace}:knum")
    sponsor = c3.text_input("Sponsor", key=f"{workspace}:sponsor")

    extra = st.text_area("Extra context", height=160, key=f"{workspace}:extra")
    out_lang = st.selectbox("Output language", options=["Follow UI", "English", "繁體中文"], key=f"{workspace}:out_lang")
    
    prompt = f"Summarize 510(k) intelligence:\n- Device: {device}\n- K-number: {knum}\n- Sponsor: {sponsor}\nContext:\n{extra}\nOutput language: {out_lang}."
    
    out, _ = agent_runner_ui(workspace, "note_summary", "510(k) Summarizer", prompt, uid="intel_sum")
    agent_runner_ui(workspace, "note_entities", "Entities Extractor", out, uid="intel_ents")

def render_pdf_to_md():
    workspace = "PDF→Markdown"
    st.markdown(f"<div class='wow-shell'><h2>📄 {t('pdf2md')}</h2></div>", unsafe_allow_html=True)

    up = st.file_uploader("Upload PDF", type=["pdf"], key=f"{workspace}:pdf")
    extracted = ""
    
    if up is not None:
        file_bytes = up.read()
        n_pages = len(PyPDF2.PdfReader(io.BytesIO(file_bytes)).pages) if PyPDF2 else 1
        c1, c2 = st.columns(2)
        ps = c1.number_input("Start page", 1, max(1, n_pages), 1, key=f"{workspace}:ps")
        pe = c2.number_input("End page", 1, max(1, n_pages), min(5, n_pages), key=f"{workspace}:pe")

        if st.button("Extract text", key=f"{workspace}:extract"):
            try:
                extracted = extract_pdf_text(file_bytes, int(ps), int(pe)) if PyPDF2 else "PyPDF2 missing"
                st.session_state[f"{workspace}:extracted"] = extracted
                st.success("Extraction Complete.")
            except Exception as e:
                st.error(str(e))

    extracted = st.session_state.get(f"{workspace}:extracted", "")
    st.text_area("Extracted", value=extracted, height=220, key=f"{workspace}:extracted_view")

    agent_runner_ui(workspace, "pdf_to_markdown", "PDF → Markdown Transformer", extracted, uid="pdf_to_md")

def render_510k_pipeline():
    workspace = "510(k) Review Pipeline"
    st.markdown(f"<div class='wow-shell'><h2>⚙️ {t('pipeline_510k')}</h2></div>", unsafe_allow_html=True)

    sub_raw = st.text_area("Paste submission material (raw)", height=180, key=f"{workspace}:submission_raw")
    checklist_raw = st.text_area("Paste checklist", height=160, key=f"{workspace}:checklist_raw")
    
    status_pill("Step A", "info")
    structured, _ = agent_runner_ui(workspace, "pipeline_structurer", "Submission Structurer", sub_raw, uid="pipe_struct")

    st.markdown("---")
    status_pill("Step B", "info")
    checklist_clean, _ = agent_runner_ui(workspace, "pipeline_checklist_cleaner", "Checklist Cleaner", checklist_raw, uid="pipe_clean")

    st.markdown("---")
    status_pill("Step C", "info")
    memo_input = f"Structured submission:\n{structured}\n\nChecklist (cleaned):\n{checklist_clean}\n\nTask: Write a review memo."
    agent_runner_ui(workspace, "pipeline_memo_builder", "Review Memo Builder", memo_input, uid="pipe_memo")

def render_report_generator():
    workspace = "510(k) Report Generator"
    st.markdown(f"<div class='wow-shell'><h2>📑 {t('report_gen')}</h2></div>", unsafe_allow_html=True)

    notes = st.text_area("Paste 510(k) review notes", height=200, key=f"{workspace}:notes")
    template = st.text_area("Template", value="[Default Template here...]", height=120, key=f"{workspace}:template")

    st.markdown("---")
    outline_input = f"Template:\n{template}\n\nReviewer notes:\n{notes}\n\nTask: Create a detailed outline."
    outline, _ = agent_runner_ui(workspace, "report_outline", "Normalize & Outline Agent", outline_input, uid="rep_outline")

    st.markdown("---")
    report_input = f"Template:\n{template}\n\nOutline:\n{outline}\n\nNotes:\n{notes}\n\nRequirements: 2000-3000 words, >= 5 tables, exactly 20 entities."
    report_md, _ = agent_runner_ui(workspace, "report_writer", "Draft Full Report Agent", report_input, uid="rep_write")

    st.markdown("---")
    skill_input = f"Goal: Generate a reusable skill.md.\nReference report:\n{report_md}"
    agent_runner_ui(workspace, "skill_creator", "skill.md Generator", skill_input, uid="rep_skill")

def render_note_keeper():
    workspace = "AI Note Keeper"
    st.markdown(f"<div class='wow-shell'><h2>📔 {t('note_keeper')}</h2></div>", unsafe_allow_html=True)

    raw = st.text_area("Paste text", height=200, key=f"{workspace}:raw")
    md_out, _ = agent_runner_ui(workspace, "note_to_markdown", "Note → Markdown Transformer", f"Transform to Markdown:\n{raw}", uid="nk_step2")

    st.markdown("---")
    st.markdown("## ✨ AI Magics")
    agent_runner_ui(workspace, "note_to_markdown", "AI Formatting", f"Reformat:\n{md_out}", uid="nk_fmt")
    st.markdown("---")
    agent_runner_ui(workspace, "note_entities", "AI Entities (20)", f"Extract 20 entities:\n{md_out}", uid="nk_ents")
    st.markdown("---")
    agent_runner_ui(workspace, "note_summary", "AI Summary", f"Summarize:\n{md_out}", uid="nk_sum")
    st.markdown("---")
    agent_runner_ui(workspace, "note_summary", "Action Items", f"Extract action items:\n{md_out}", uid="nk_act")

def render_agents_studio():
    workspace = "Agents Config Studio"
    st.markdown(f"<div class='wow-shell'><h2>🛠️ {t('agents_studio')}</h2></div>", unsafe_allow_html=True)
    st.session_state["agents_yaml_text"] = st.text_area("Edit agents.yaml", value=st.session_state["agents_yaml_text"], height=320, key=f"{workspace}:yaml")

    if st.button("Apply YAML", key=f"{workspace}:apply"):
        st.session_state["agents"] = load_agents_from_text(st.session_state["agents_yaml_text"])
        st.success("Applied.")


# ---------------------------
# Main Execution
# ---------------------------
ui_global_settings()

st.markdown(f"<div class='wow-shell' style='padding-top:10px; margin-bottom:30px;'><h1>{t('app_title')}</h1>"
            f"<div class='wow-pill'>Theme: {st.session_state['ui_theme']}</div>"
            f"<div class='wow-pill'>Lang: {st.session_state['ui_language']}</div>"
            f"<div class='wow-pill'>Style: {st.session_state['ui_style']}</div>"
            f"</div>", unsafe_allow_html=True)

tabs = st.tabs(["📊 Dashboard", "🇹🇼 TW Premarket", "🧠 510(k) Intel", "📄 PDF2MD", "⚙️ Pipeline", "📑 Report Gen", "📔 Note Keeper", "🛠️ Agents Studio"])

with tabs[0]: render_dashboard()
with tabs[1]: render_tw_premarket()
with tabs[2]: render_510k_intelligence()
with tabs[3]: render_pdf_to_md()
with tabs[4]: render_510k_pipeline()
with tabs[5]: render_report_generator()
with tabs[6]: render_note_keeper()
with tabs[7]: render_agents_studio()
