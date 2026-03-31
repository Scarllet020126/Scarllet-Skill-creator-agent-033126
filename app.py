import os
import json
import base64
from datetime import datetime
from io import BytesIO

import streamlit as st
import yaml
import pandas as pd
import altair as alt
from pypdf import PdfReader

try:
    from docx import Document  # python-docx
except ImportError:
    Document = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:
    canvas = None
    letter = None

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx

# =========================
# Constants & configuration
# =========================

ALL_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

OPENAI_MODELS = {"gpt-4o-mini", "gpt-4.1-mini"}
GEMINI_MODELS = {"gemini-2.5-flash", "gemini-2.5-flash-lite"}
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
}
GROK_MODELS = {"grok-4-fast-reasoning", "grok-3-mini"}

PAINTER_STYLES = [
    "Van Gogh", "Monet", "Picasso", "Da Vinci", "Rembrandt",
    "Matisse", "Kandinsky", "Hokusai", "Yayoi Kusama", "Frida Kahlo",
    "Salvador Dali", "Rothko", "Pollock", "Chagall", "Basquiat",
    "Haring", "Georgia O'Keeffe", "Turner", "Seurat", "Escher",
]

LABELS = {
    "Dashboard": {"English": "Dashboard", "繁體中文": "儀表板"},
    "TW Premarket": {
        "English": "TW Premarket Application",
        "繁體中文": "第二、三等級醫療器材查驗登記",
    },
    "510k_tab": {"English": "510(k) Intelligence", "繁體中文": "510(k) 智能分析"},
    "PDF → Markdown": {"English": "PDF → Markdown", "繁體中文": "PDF → Markdown"},
    "Checklist & Report": {
        "English": "510(k) Review Pipeline",
        "繁體中文": "510(k) 審查全流程",
    },
    "Note Keeper & Magics": {
        "English": "Note Keeper & Magics",
        "繁體中文": "筆記助手與魔法",
    },
    "Agents Config": {
        "English": "Agents Config Studio",
        "繁體中文": "代理設定工作室",
    },
}

STYLE_CSS = {
    "Van Gogh": "body { background: radial-gradient(circle at top left, #243B55, #141E30); }",
    "Monet": "body { background: linear-gradient(120deg, #a1c4fd, #c2e9fb); }",
    "Picasso": "body { background: linear-gradient(135deg, #ff9a9e, #fecfef); }",
    "Da Vinci": "body { background: radial-gradient(circle, #f9f1c6, #c9a66b); }",
    "Rembrandt": "body { background: radial-gradient(circle, #2c1810, #0b090a); }",
    "Matisse": "body { background: linear-gradient(135deg, #ffecd2, #fcb69f); }",
    "Kandinsky": "body { background: linear-gradient(135deg, #00c6ff, #0072ff); }",
    "Hokusai": "body { background: linear-gradient(135deg, #2b5876, #4e4376); }",
    "Yayoi Kusama": "body { background: radial-gradient(circle, #ffdd00, #ff6a00); }",
    "Frida Kahlo": "body { background: linear-gradient(135deg, #f8b195, #f67280, #c06c84); }",
    "Salvador Dali": "body { background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); }",
    "Rothko": "body { background: linear-gradient(135deg, #141E30, #243B55); }",
    "Pollock": "body { background: repeating-linear-gradient(45deg,#222,#222 10px,#333 10px,#333 20px); }",
    "Chagall": "body { background: linear-gradient(135deg, #a18cd1, #fbc2eb); }",
    "Basquiat": "body { background: linear-gradient(135deg, #f7971e, #ffd200); }",
    "Haring": "body { background: linear-gradient(135deg, #ff512f, #dd2476); }",
    "Georgia O'Keeffe": "body { background: linear-gradient(135deg, #ffefba, #ffffff); }",
    "Turner": "body { background: linear-gradient(135deg, #f8ffae, #43c6ac); }",
    "Seurat": "body { background: radial-gradient(circle, #e0eafc, #cfdef3); }",
    "Escher": "body { background: linear-gradient(135deg, #232526, #414345); }",
}

# =========================
# Helper: localization & style
# =========================

def t(key: str) -> str:
    lang = st.session_state.settings.get("language", "English")
    return LABELS.get(key, {}).get(lang, key)


def apply_style(theme: str, painter_style: str):
    css = STYLE_CSS.get(painter_style, "")
    if theme == "Dark":
        css += """
        body { color: #e0e0e0; }
        .stButton>button {
            background-color: #1f2933; color: white; border-radius: 999px;
        }
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stDateInput>div>div>input {
            background-color: #111827; color: #e5e7eb; border-radius: 0.5rem;
        }
        """
    else:
        css += """
        body { color: #111827; }
        .stButton>button {
            background-color: #2563eb; color: white; border-radius: 999px;
        }
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stDateInput>div>div>input {
            background-color: #ffffff; color: #111827; border-radius: 0.5rem;
        }
        """
    # WOW status indicator extra CSS
    css += """
    .wow-card {
        border-radius: 18px;
        padding: 14px 18px;
        margin-bottom: 0.75rem;
        box-shadow: 0 14px 35px rgba(15,23,42,0.45);
        color: #f9fafb;
    }
    .wow-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.85;
    }
    .wow-card-main {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .wow-badge {
        display:inline-flex;
        align-items:center;
        padding:2px 10px;
        border-radius:999px;
        font-size:0.75rem;
        font-weight:600;
        background:rgba(15,23,42,0.2);
        border:1px solid rgba(148,163,184,0.6);
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# =========================
# LLM routing
# =========================

def get_provider(model: str) -> str:
    if model in OPENAI_MODELS:
        return "openai"
    if model in GEMINI_MODELS:
        return "gemini"
    if model in ANTHROPIC_MODELS:
        return "anthropic"
    if model in GROK_MODELS:
        return "grok"
    raise ValueError(f"Unknown model: {model}")


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    api_keys: dict | None = None,
) -> str:
    provider = get_provider(model)
    api_keys = api_keys or {}

    def get_key(name: str, env_var: str) -> str:
        return api_keys.get(name) or os.getenv(env_var) or ""

    if provider == "openai":
        key = get_key("openai", "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OpenAI API key.")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    if provider == "gemini":
        key = get_key("gemini", "GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing Gemini API key.")
        genai.configure(api_key=key)
        llm = genai.GenerativeModel(model)
        resp = llm.generate_content(
            system_prompt + "\n\n" + user_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return resp.text

    if provider == "anthropic":
        key = get_key("anthropic", "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Missing Anthropic API key.")
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text

    if provider == "grok":
        key = get_key("grok", "GROK_API_KEY")
        if not key:
            raise RuntimeError("Missing Grok (xAI) API key.")
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=60) as client:
            resp = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unsupported provider for model {model}")

# =========================
# Generic helpers
# =========================

def show_status(step_name: str, status: str):
    color = {
        "pending": "gray",
        "running": "#f59e0b",
        "done": "#10b981",
        "error": "#ef4444",
    }.get(status, "gray")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
          <div style="width:10px;height:10px;border-radius:50%;background:{color};
                      margin-right:6px;"></div>
          <span style="font-size:0.9rem;">{step_name} – <b>{status}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def log_event(tab: str, agent: str, model: str, tokens_est: int):
    st.session_state["history"].append(
        {
            "tab": tab,
            "agent": agent,
            "model": model,
            "tokens_est": tokens_est,
            "ts": datetime.utcnow().isoformat(),
        }
    )


def extract_pdf_pages_to_text(file, start_page: int, end_page: int) -> str:
    reader = PdfReader(file)
    n = len(reader.pages)
    start = max(0, start_page - 1)
    end = min(n, end_page)
    texts = []
    for i in range(start, end):
        try:
            texts.append(reader.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def extract_docx_to_text(file) -> str:
    if Document is None:
        return ""
    try:
        bio = BytesIO(file.read())
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def create_pdf_from_text(text: str) -> bytes:
    if canvas is None or letter is None:
        raise RuntimeError(
            "PDF generation library 'reportlab' is not installed. "
            "Please add 'reportlab' to requirements.txt."
        )
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 72
    line_height = 14
    y = height - margin
    for line in text.splitlines():
        if y < margin:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:2000])
        y -= line_height
    c.save()
    buf.seek(0)
    return buf.getvalue()


def show_pdf(pdf_bytes: bytes, height: int = 600):
    if not pdf_bytes:
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_html = f"""
    <iframe src="data:application/pdf;base64,{b64}"
            width="100%" height="{height}" type="application/pdf"></iframe>
    """
    st.markdown(pdf_html, unsafe_allow_html=True)

# =========================
# Agent UI runner
# =========================

def agent_run_ui(
    agent_id: str,
    tab_key: str,
    default_prompt: str,
    default_input_text: str = "",
    allow_model_override: bool = True,
    tab_label_for_history: str | None = None,
):
    agents_cfg = st.session_state.get("agents_cfg", {})
    agents_dict = agents_cfg.get("agents", {})

    if agent_id in agents_dict:
        agent_cfg = agents_dict[agent_id]
    else:
        agent_cfg = {
            "name": agent_id,
            "model": st.session_state.settings["model"],
            "system_prompt": "",
            "max_tokens": st.session_state.settings["max_tokens"],
        }

    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "pending"

    show_status(agent_cfg.get("name", agent_id), st.session_state[status_key])

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_prompt = st.text_area(
            "Prompt",
            value=st.session_state.get(f"{tab_key}_prompt", default_prompt),
            height=160,
            key=f"{tab_key}_prompt",
        )
    with col2:
        base_model = agent_cfg.get("model", st.session_state.settings["model"])
        model_index = ALL_MODELS.index(base_model) if base_model in ALL_MODELS else 0
        model = st.selectbox(
            "Model",
            ALL_MODELS,
            index=model_index,
            disabled=not allow_model_override,
            key=f"{tab_key}_model",
        )
    with col3:
        max_tokens = st.number_input(
            "max_tokens",
            min_value=1000,
            max_value=120000,
            value=int(agent_cfg.get("max_tokens", st.session_state.settings["max_tokens"])),
            step=1000,
            key=f"{tab_key}_max_tokens",
        )

    input_text = st.text_area(
        "Input Text / Markdown",
        value=st.session_state.get(f"{tab_key}_input", default_input_text),
        height=260,
        key=f"{tab_key}_input",
    )

    run = st.button("Run Agent", key=f"{tab_key}_run")

    if run:
        st.session_state[status_key] = "running"
        show_status(agent_cfg.get("name", agent_id), "running")
        api_keys = st.session_state.get("api_keys", {})
        system_prompt = agent_cfg.get("system_prompt", "")
        user_full = f"{user_prompt}\n\n---\n\n{input_text}"

        with st.spinner("Running agent..."):
            try:
                out = call_llm(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_full,
                    max_tokens=max_tokens,
                    temperature=st.session_state.settings["temperature"],
                    api_keys=api_keys,
                )
                st.session_state[f"{tab_key}_output"] = out
                st.session_state[status_key] = "done"
                token_est = int(len(user_full + out) / 4)
                log_event(
                    tab_label_for_history or tab_key,
                    agent_cfg.get("name", agent_id),
                    model,
                    token_est,
                )
            except Exception as e:
                st.session_state[status_key] = "error"
                st.error(f"Agent error: {e}")

    output = st.session_state.get(f"{tab_key}_output", "")
    view_mode = st.radio(
        "View mode", ["Markdown", "Plain text"],
        horizontal=True, key=f"{tab_key}_viewmode",
    )
    if view_mode == "Markdown":
        edited = st.text_area(
            "Output (Markdown, editable)",
            value=output,
            height=320,
            key=f"{tab_key}_output_md",
        )
    else:
        edited = st.text_area(
            "Output (Plain text, editable)",
            value=output,
            height=320,
            key=f"{tab_key}_output_txt",
        )

    st.session_state[f"{tab_key}_output_edited"] = edited

# =========================
# Sidebar
# =========================

def render_sidebar():
    with st.sidebar:
        st.markdown("### Global Settings")

        st.session_state.settings["theme"] = st.radio(
            "Theme", ["Light", "Dark"],
            index=0 if st.session_state.settings["theme"] == "Light" else 1,
        )

        st.session_state.settings["language"] = st.radio(
            "Language", ["English", "繁體中文"],
            index=0 if st.session_state.settings["language"] == "English" else 1,
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            style = st.selectbox(
                "Painter Style",
                PAINTER_STYLES,
                index=PAINTER_STYLES.index(st.session_state.settings["painter_style"]),
            )
        with col2:
            if st.button("Jackpot!"):
                import random
                style = random.choice(PAINTER_STYLES)
        st.session_state.settings["painter_style"] = style

        st.session_state.settings["model"] = st.selectbox(
            "Default Model",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
        )
        st.session_state.settings["max_tokens"] = st.number_input(
            "Default max_tokens",
            min_value=1000,
            max_value=120000,
            value=st.session_state.settings["max_tokens"],
            step=1000,
        )
        st.session_state.settings["temperature"] = st.slider(
            "Temperature",
            0.0,
            1.0,
            st.session_state.settings["temperature"],
            0.05,
        )

        st.markdown("---")
        st.markdown("### API Keys")

        keys = {}

        if os.getenv("OPENAI_API_KEY"):
            keys["openai"] = os.getenv("OPENAI_API_KEY")
            st.caption("OpenAI key from environment.")
        else:
            keys["openai"] = st.text_input("OpenAI API Key", type="password")

        if os.getenv("GEMINI_API_KEY"):
            keys["gemini"] = os.getenv("GEMINI_API_KEY")
            st.caption("Gemini key from environment.")
        else:
            keys["gemini"] = st.text_input("Gemini API Key", type="password")

        if os.getenv("ANTHROPIC_API_KEY"):
            keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
            st.caption("Anthropic key from environment.")
        else:
            keys["anthropic"] = st.text_input("Anthropic API Key", type="password")

        if os.getenv("GROK_API_KEY"):
            keys["grok"] = os.getenv("GROK_API_KEY")
            st.caption("Grok key from environment.")
        else:
            keys["grok"] = st.text_input("Grok API Key", type="password")

        st.session_state["api_keys"] = keys

        st.markdown("---")
        st.markdown("### Agents Catalog (agents.yaml)")
        uploaded_agents = st.file_uploader(
            "Upload custom agents.yaml",
            type=["yaml", "yml"],
            key="sidebar_agents_yaml",
        )
        if uploaded_agents is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents.read())
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Custom agents.yaml loaded for this session.")
                else:
                    st.warning("Uploaded YAML has no top-level 'agents' key. Using previous config.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")

# =========================
# Awesome Dashboard
# =========================

def render_dashboard():
    st.title(t("Dashboard"))
    hist = st.session_state["history"]
    if not hist:
        st.info("No runs yet.")
        return

    df = pd.DataFrame(hist)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        st.metric("Unique Tabs", df["tab"].nunique())
    with col3:
        st.metric("Approx Tokens Processed", int(df["tokens_est"].sum()))

    # WOW Status Wall (最近一次呼叫)
    st.markdown("### WOW Status Wall – Latest Activity")
    last = df.sort_values("ts", ascending=False).iloc[0]
    wow_color = "linear-gradient(135deg,#22c55e,#16a34a)"  # 綠
    if last["tokens_est"] > 40000:
        wow_color = "linear-gradient(135deg,#f97316,#ea580c)"  # 橘
    if last["tokens_est"] > 80000:
        wow_color = "linear-gradient(135deg,#ef4444,#b91c1c)"  # 紅

    st.markdown(
        f"""
        <div class="wow-card" style="background:{wow_color};">
          <div class="wow-card-title">LATEST RUN SNAPSHOT</div>
          <div class="wow-card-main">
            {last['tab']} · {last['agent']}
          </div>
          <div style="margin-top:6px;font-size:0.9rem;">
            Model: <b>{last['model']}</b> · Tokens ≈ <b>{last['tokens_est']}</b><br>
            Time (UTC): {last['ts']}
          </div>
          <div style="margin-top:8px;">
            <span class="wow-badge">Status: active</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Runs by Tab")
    chart_tab = alt.Chart(df).mark_bar().encode(
        x="tab:N",
        y="count():Q",
        color="tab:N",
        tooltip=["tab", "count()"],
    )
    st.altair_chart(chart_tab, use_container_width=True)

    st.markdown("### Runs by Model")
    chart_model = alt.Chart(df).mark_bar().encode(
        x="model:N",
        y="count():Q",
        color="model:N",
        tooltip=["model", "count()"],
    )
    st.altair_chart(chart_model, use_container_width=True)

    # Awesome heatmap: Tab × Model usage
    st.markdown("### Model × Tab Usage Heatmap")
    heat_df = df.groupby(["tab", "model"]).size().reset_index(name="count")
    heatmap = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("tab:N", title="Tab"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Runs"),
            tooltip=["tab", "model", "count"],
        )
        .properties(height=260)
    )
    st.altair_chart(heatmap, use_container_width=True)

    st.markdown("### Token Usage Over Time")
    df_time = df.copy()
    df_time["ts"] = pd.to_datetime(df_time["ts"])
    chart_time = alt.Chart(df_time).mark_line(point=True).encode(
        x="ts:T",
        y="tokens_est:Q",
        color="tab:N",
        tooltip=["ts", "tab", "agent", "model", "tokens_est"],
    )
    st.altair_chart(chart_time, use_container_width=True)

    st.markdown("### Recent Activity")
    st.dataframe(df.sort_values("ts", ascending=False).head(25), use_container_width=True)

# =========================
# Helper for TW application schema
# =========================

TW_APP_FIELDS = [
    "doc_no", "e_no", "apply_date", "case_type", "device_category", "case_kind",
    "origin", "product_class", "similar", "replace_flag", "prior_app_no",
    "name_zh", "name_en", "indications", "spec_comp",
    "main_cat", "item_code", "item_name",
    "uniform_id", "firm_name", "firm_addr",
    "resp_name", "contact_name", "contact_tel", "contact_fax", "contact_email",
    "confirm_match", "cert_raps", "cert_ahwp", "cert_other",
    "manu_type", "manu_name", "manu_country", "manu_addr", "manu_note",
    "auth_applicable", "auth_desc",
    "cfs_applicable", "cfs_desc",
    "qms_applicable", "qms_desc",
    "similar_info", "labeling_info", "tech_file_info",
    "preclinical_info", "preclinical_replace",
    "clinical_just", "clinical_info",
]

def build_tw_app_dict_from_session() -> dict:
    s = st.session_state
    apply_date = s.get("tw_apply_date")
    apply_date_str = apply_date.strftime("%Y-%m-%d") if apply_date else ""
    return {
        "doc_no": s.get("tw_doc_no", ""),
        "e_no": s.get("tw_e_no", ""),
        "apply_date": apply_date_str,
        "case_type": s.get("tw_case_type", ""),
        "device_category": s.get("tw_device_category", ""),
        "case_kind": s.get("tw_case_kind", ""),
        "origin": s.get("tw_origin", ""),
        "product_class": s.get("tw_product_class", ""),
        "similar": s.get("tw_similar", ""),
        "replace_flag": s.get("tw_replace_flag", ""),
        "prior_app_no": s.get("tw_prior_app_no", ""),
        "name_zh": s.get("tw_dev_name_zh", ""),
        "name_en": s.get("tw_dev_name_en", ""),
        "indications": s.get("tw_indications", ""),
        "spec_comp": s.get("tw_spec_comp", ""),
        "main_cat": s.get("tw_main_cat", ""),
        "item_code": s.get("tw_item_code", ""),
        "item_name": s.get("tw_item_name", ""),
        "uniform_id": s.get("tw_uniform_id", ""),
        "firm_name": s.get("tw_firm_name", ""),
        "firm_addr": s.get("tw_firm_addr", ""),
        "resp_name": s.get("tw_resp_name", ""),
        "contact_name": s.get("tw_contact_name", ""),
        "contact_tel": s.get("tw_contact_tel", ""),
        "contact_fax": s.get("tw_contact_fax", ""),
        "contact_email": s.get("tw_contact_email", ""),
        "confirm_match": bool(s.get("tw_confirm_match", False)),
        "cert_raps": bool(s.get("tw_cert_raps", False)),
        "cert_ahwp": bool(s.get("tw_cert_ahwp", False)),
        "cert_other": s.get("tw_cert_other", ""),
        "manu_type": s.get("tw_manu_type", ""),
        "manu_name": s.get("tw_manu_name", ""),
        "manu_country": s.get("tw_manu_country", ""),
        "manu_addr": s.get("tw_manu_addr", ""),
        "manu_note": s.get("tw_manu_note", ""),
        "auth_applicable": s.get("tw_auth_app", ""),
        "auth_desc": s.get("tw_auth_desc", ""),
        "cfs_applicable": s.get("tw_cfs_app", ""),
        "cfs_desc": s.get("tw_cfs_desc", ""),
        "qms_applicable": s.get("tw_qms_app", ""),
        "qms_desc": s.get("tw_qms_desc", ""),
        "similar_info": s.get("tw_similar_info", ""),
        "labeling_info": s.get("tw_labeling_info", ""),
        "tech_file_info": s.get("tw_tech_file_info", ""),
        "preclinical_info": s.get("tw_preclinical_info", ""),
        "preclinical_replace": s.get("tw_preclinical_replace", ""),
        "clinical_just": s.get("tw_clinical_app", ""),
        "clinical_info": s.get("tw_clinical_info", ""),
    }

def apply_tw_app_dict_to_session(data: dict):
    # 將標準化 dict 寫入 session_state，以更新表單
    s = st.session_state
    s["tw_doc_no"] = data.get("doc_no", "")
    s["tw_e_no"] = data.get("e_no", "")
    # 日期
    from datetime import date
    try:
        if data.get("apply_date"):
            y, m, d = map(int, str(data["apply_date"]).split("-"))
            s["tw_apply_date"] = date(y, m, d)
    except Exception:
        pass
    s["tw_case_type"] = data.get("case_type", "")
    s["tw_device_category"] = data.get("device_category", "")
    s["tw_case_kind"] = data.get("case_kind", "")
    s["tw_origin"] = data.get("origin", "")
    s["tw_product_class"] = data.get("product_class", "")
    s["tw_similar"] = data.get("similar", "")
    s["tw_replace_flag"] = data.get("replace_flag", "")
    s["tw_prior_app_no"] = data.get("prior_app_no", "")
    s["tw_dev_name_zh"] = data.get("name_zh", "")
    s["tw_dev_name_en"] = data.get("name_en", "")
    s["tw_indications"] = data.get("indications", "")
    s["tw_spec_comp"] = data.get("spec_comp", "")
    s["tw_main_cat"] = data.get("main_cat", "")
    s["tw_item_code"] = data.get("item_code", "")
    s["tw_item_name"] = data.get("item_name", "")
    s["tw_uniform_id"] = data.get("uniform_id", "")
    s["tw_firm_name"] = data.get("firm_name", "")
    s["tw_firm_addr"] = data.get("firm_addr", "")
    s["tw_resp_name"] = data.get("resp_name", "")
    s["tw_contact_name"] = data.get("contact_name", "")
    s["tw_contact_tel"] = data.get("contact_tel", "")
    s["tw_contact_fax"] = data.get("contact_fax", "")
    s["tw_contact_email"] = data.get("contact_email", "")
    s["tw_confirm_match"] = bool(data.get("confirm_match", False))
    s["tw_cert_raps"] = bool(data.get("cert_raps", False))
    s["tw_cert_ahwp"] = bool(data.get("cert_ahwp", False))
    s["tw_cert_other"] = data.get("cert_other", "")
    s["tw_manu_type"] = data.get("manu_type", "")
    s["tw_manu_name"] = data.get("manu_name", "")
    s["tw_manu_country"] = data.get("manu_country", "")
    s["tw_manu_addr"] = data.get("manu_addr", "")
    s["tw_manu_note"] = data.get("manu_note", "")
    s["tw_auth_app"] = data.get("auth_applicable", "")
    s["tw_auth_desc"] = data.get("auth_desc", "")
    s["tw_cfs_app"] = data.get("cfs_applicable", "")
    s["tw_cfs_desc"] = data.get("cfs_desc", "")
    s["tw_qms_app"] = data.get("qms_applicable", "")
    s["tw_qms_desc"] = data.get("qms_desc", "")
    s["tw_similar_info"] = data.get("similar_info", "")
    s["tw_labeling_info"] = data.get("labeling_info", "")
    s["tw_tech_file_info"] = data.get("tech_file_info", "")
    s["tw_preclinical_info"] = data.get("preclinical_info", "")
    s["tw_preclinical_replace"] = data.get("preclinical_replace", "")
    s["tw_clinical_app"] = data.get("clinical_just", "")
    s["tw_clinical_info"] = data.get("clinical_info", "")

def standardize_tw_app_info_with_llm(raw_obj) -> dict:
    """
    使用 LLM 將任意 JSON/欄位對映成標準 TFDA 申請書 schema。
    需要 Gemini API (預設使用 gemini-2.5-flash)。
    """
    api_keys = st.session_state.get("api_keys", {})
    model = "gemini-2.5-flash"
    if "gemini" not in api_keys and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("No Gemini API key available for standardizing application info.")

    raw_json = json.dumps(raw_obj, ensure_ascii=False, indent=2)
    fields_str = ", ".join(TW_APP_FIELDS)

    system_prompt = f"""
You are a data normalization assistant for a Taiwanese TFDA medical device premarket application.

Goal:
Map arbitrary JSON or CSV-like key/value structures into a STANDARD JSON object
that uses EXACTLY the following top-level keys (all strings except where noted):

{fields_str}

Rules:
- Output MUST be a single JSON object (no markdown, no comments).
- Every key above MUST appear in the JSON.
- If information for a field is clearly not present, set it to an empty string,
  or for boolean-like fields use `false`.
- Map semantically similar keys (e.g. 'device_name_zh', 'cn_name') to 'name_zh', etc.
- `apply_date` should be string like 'YYYY-MM-DD' if you can infer; otherwise empty string.
- Do NOT invent new facts; just reorganize/rename what exists.
"""

    user_prompt = f"Here is the raw data to normalize:\n\n{raw_json}"

    out = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=4000,
        temperature=0.1,
        api_keys=api_keys,
    )

    # 嘗試 parse JSON
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        # 嘗試截斷到第一個/最後一個大括號
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(out[start:end + 1])
        else:
            raise RuntimeError("LLM did not return valid JSON for application info.")
    if not isinstance(data, dict):
        raise RuntimeError("Standardized application info is not a JSON object.")
    # 確保所有欄位存在
    for k in TW_APP_FIELDS:
        if k not in data:
            data[k] = "" if k not in ("confirm_match", "cert_raps", "cert_ahwp") else False
    return data

def compute_tw_app_completeness() -> float:
    """
    計算 TFDA 申請書基本必填欄位的完成度 (0~1)
    """
    s = st.session_state
    required_keys = [
        "tw_e_no", "tw_case_type", "tw_device_category",
        "tw_origin", "tw_product_class",
        "tw_dev_name_zh", "tw_dev_name_en",
        "tw_uniform_id", "tw_firm_name", "tw_firm_addr",
        "tw_resp_name", "tw_contact_name", "tw_contact_tel",
        "tw_contact_email",
        "tw_manu_name", "tw_manu_addr",
    ]
    filled = 0
    for k in required_keys:
        v = s.get(k, "")
        if isinstance(v, str):
            if v.strip():
                filled += 1
        else:
            if v:
                filled += 1
    return filled / len(required_keys) if required_keys else 0.0

# =========================
# TW Premarket Tab
# =========================

def render_tw_premarket_tab():
    """臺灣第二、三等級醫療器材查驗登記 – 預審/形式審查 Tab"""
    st.title(t("TW Premarket"))

    st.markdown(
        """
        <div style="background:#eef2ff;border-radius:12px;padding:10px 14px;
                    border:1px solid #c7d2fe;margin-bottom:0.75rem;">
          <b>Step 1.</b> 線上填寫或由 JSON/CSV 匯入「第二、三等級醫療器材查驗登記申請」主要欄位。<br>
          <b>Step 2.</b> 貼上或上傳「預審/形式審查指引」供 AI 進行完整性檢核。<br>
          <b>Step 3.</b> 產出預審摘要報告 (Markdown)，可在頁面上修改。<br>
          <b>Step 4.</b> 以 AI 協助編修申請書內容，或把輸出串接到下一個 agent。
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Import / Export Application Info (CSV / JSON)
    # -----------------------------
    st.markdown("### Application Info 匯入 / 匯出 (JSON / CSV)")

    col_ie1, col_ie2 = st.columns(2)
    with col_ie1:
        st.markdown("**上傳 Application Info**")
        app_file = st.file_uploader(
            "Upload Application Info (JSON / CSV)",
            type=["json", "csv"],
            key="tw_app_upload",
        )
        if app_file is not None:
            try:
                if app_file.name.lower().endswith(".json"):
                    raw_data = json.load(app_file)
                else:
                    df = pd.read_csv(app_file)
                    if len(df) == 0:
                        st.error("CSV 檔案為空。")
                        raw_data = None
                    else:
                        raw_data = df.to_dict(orient="records")[0]
                if raw_data is not None:
                    # 檢查是否已是標準格式
                    if isinstance(raw_data, dict) and all(k in raw_data for k in TW_APP_FIELDS):
                        standardized = raw_data
                    else:
                        with st.spinner("使用 LLM 將欄位轉為標準 TFDA 申請書格式..."):
                            standardized = standardize_tw_app_info_with_llm(raw_data)
                    apply_tw_app_dict_to_session(standardized)
                    st.success("已將上傳資料轉換並套用至申請表單。")
                    st.session_state["tw_app_last_loaded"] = standardized
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"上傳或標準化失敗：{e}")

    with col_ie2:
        st.markdown("**下載 Application Info**")
        app_dict = build_tw_app_dict_from_session()
        # JSON
        json_bytes = json.dumps(app_dict, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="tw_premarket_application.json",
            mime="application/json",
            key="tw_app_download_json",
        )
        # CSV
        df_app = pd.DataFrame([app_dict])
        csv_bytes = df_app.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="tw_premarket_application.csv",
            mime="text/csv",
            key="tw_app_download_csv",
        )

    # JSON 預覽
    if "tw_app_last_loaded" in st.session_state:
        st.markdown("**最近載入/標準化之 Application JSON 預覽**")
        st.json(st.session_state["tw_app_last_loaded"], expanded=False)

    st.markdown("---")

    # -----------------------------
    # WOW Application Status Indicator
    # -----------------------------
    completeness = compute_tw_app_completeness()
    pct = int(completeness * 100)
    if pct >= 80:
        card_grad = "linear-gradient(135deg,#22c55e,#16a34a)"
        txt = "申請基本欄位完成度高，適合進行預審。"
    elif pct >= 50:
        card_grad = "linear-gradient(135deg,#f97316,#ea580c)"
        txt = "部分關鍵欄位仍待補齊，建議補足後再送預審。"
    else:
        card_grad = "linear-gradient(135deg,#ef4444,#b91c1c)"
        txt = "多數基本欄位尚未填寫，請先充實申請資訊。"

    st.markdown(
        f"""
        <div class="wow-card" style="background:{card_grad};margin-top:0;">
          <div class="wow-card-title">APPLICATION COMPLETENESS</div>
          <div class="wow-card-main">{pct}%</div>
          <div style="margin-top:6px;font-size:0.9rem;">{txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(completeness)

    # -----------------------------
    # Step 1 – 線上申請書草稿
    # -----------------------------
    st.markdown("### Step 1 – 線上填寫申請書（草稿）")

    if "tw_app_status" not in st.session_state:
        st.session_state["tw_app_status"] = "pending"
    show_status("申請書填寫", st.session_state["tw_app_status"])

    # 一、案件基本資料
    st.markdown("#### 一、案件基本資料")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        doc_no = st.text_input("公文文號", key="tw_doc_no")
        e_no = st.text_input("電子流水號", value=st.session_state.get("tw_e_no", "MDE"), key="tw_e_no")
    with col_a2:
        apply_date = st.date_input("申請日", key="tw_apply_date")
        case_type = st.selectbox(
            "案件類型*",
            ["一般申請案", "同一產品不同品名", "專供外銷", "許可證有效期限屆至後六個月內重新申請"],
            key="tw_case_type",
        )
    with col_a3:
        device_category = st.selectbox(
            "醫療器材類型*",
            ["一般醫材", "體外診斷器材(IVD)"],
            key="tw_device_category",
        )
        case_kind = st.selectbox("案件種類*", ["新案", "變更案", "展延案"], index=0, key="tw_case_kind")

    col_a4, col_a5, col_a6 = st.columns(3)
    with col_a4:
        origin = st.selectbox("產地*", ["國產", "輸入", "陸輸"], key="tw_origin")
    with col_a5:
        product_class = st.selectbox("產品等級*", ["第二等級", "第三等級"], key="tw_product_class")
    with col_a6:
        similar = st.selectbox("有無類似品*", ["有", "無", "全球首創"], key="tw_similar")

    col_a7, col_a8 = st.columns(2)
    with col_a7:
        replace_flag = st.radio(
            "是否勾選「替代臨床前測試及原廠品質管制資料」？*",
            ["否", "是"],
            index=0 if st.session_state.get("tw_replace_flag", "否") == "否" else 1,
            key="tw_replace_flag",
        )
    with col_a8:
        prior_app_no = st.text_input("（非首次申請）前次申請案號", key="tw_prior_app_no")

    # 二、醫療器材基本資訊
    st.markdown("#### 二、醫療器材基本資訊")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        name_zh = st.text_input("醫療器材中文名稱*", key="tw_dev_name_zh")
        name_en = st.text_input("醫療器材英文名稱*", key="tw_dev_name_en")
    with col_b2:
        indications = st.text_area("效能、用途或適應症說明", value=st.session_state.get("tw_indications", "詳如核定之中文說明書"), key="tw_indications")
        spec_comp = st.text_area("型號、規格或主要成分說明", value=st.session_state.get("tw_spec_comp", "詳如核定之中文說明書"), key="tw_spec_comp")

    st.markdown("**分類分級品項（依《醫療器材分類分級管理辦法》附表填列）**")
    col_b3, col_b4, col_b5 = st.columns(3)
    with col_b3:
        main_cat = st.selectbox(
            "主類別",
            [
                "",
                "A.臨床化學及臨床毒理學",
                "B.血液學及病理學",
                "C.免疫學及微生物學",
                "D.麻醉學",
                "E.心臟血管醫學",
                "F.牙科學",
                "G.耳鼻喉科學",
                "H.胃腸病科學及泌尿科學",
                "I.一般及整形外科手術",
                "J.一般醫院及個人使用裝置",
                "K.神經科學",
                "L.婦產科學",
                "M.眼科學",
                "N.骨科學",
                "O.物理醫學科學",
                "P.放射學科學",
            ],
            key="tw_main_cat",
        )
    with col_b4:
        item_code = st.text_input("分級品項代碼（例：A.1225）", key="tw_item_code")
    with col_b5:
        item_name = st.text_input("分級品項名稱（例：肌氨酸酐試驗系統）", key="tw_item_name")

    # 三、醫療器材商資料
    st.markdown("#### 三、醫療器材商資料")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        uniform_id = st.text_input("統一編號*", key="tw_uniform_id")
        firm_name = st.text_input("醫療器材商名稱*", key="tw_firm_name")
        firm_addr = st.text_area("醫療器材商地址*", height=80, key="tw_firm_addr")
    with col_c2:
        resp_name = st.text_input("負責人姓名*", key="tw_resp_name")
        contact_name = st.text_input("聯絡人姓名*", key="tw_contact_name")
        contact_tel = st.text_input("電話*", key="tw_contact_tel")
        contact_fax = st.text_input("聯絡人傳真", key="tw_contact_fax")
        contact_email = st.text_input("電子郵件*", key="tw_contact_email")

    confirm_match = st.checkbox(
        "我已確認上述資料與最新版醫療器材商證照資訊(名稱、地址、負責人)相符",
        key="tw_confirm_match",
    )

    st.markdown("**其它佐證（承辦人訓練證明等）**")
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        cert_raps = st.checkbox("RAPS", key="tw_cert_raps")
        cert_ahwp = st.checkbox("AHWP", key="tw_cert_ahwp")
    with col_c4:
        cert_other = st.text_input("其它，請敘明", key="tw_cert_other")

    # 四、製造廠資訊
    st.markdown("#### 四、製造廠資訊（含委託製造）")
    manu_type = st.radio(
        "製造方式",
        ["單一製造廠", "全部製程委託製造", "委託非全部製程之製造/包裝/貼標/滅菌及最終驗放"],
        index=0,
        key="tw_manu_type",
    )
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        manu_name = st.text_input("製造廠名稱*", key="tw_manu_name")
        manu_country = st.selectbox(
            "製造國別*",
            [
                "TAIWAN， ROC",
                "UNITED STATES",
                "EU (Member State)",
                "JAPAN",
                "CHINA",
                "KOREA， REPUBLIC OF",
                "OTHER",
            ],
            key="tw_manu_country",
        )
    with col_d2:
        manu_addr = st.text_area("製造廠地址*", height=80, key="tw_manu_addr")
        manu_note = st.text_area("製造廠相關說明（如(O)/(P)製造、委託範圍）", height=80, key="tw_manu_note")

    with st.expander("附件摘要：原廠授權、出產國製售證明、QMS/QSD、技術檔案、臨床資料等", expanded=False):
        auth_applicable = st.selectbox("原廠授權登記書", ["不適用", "適用"], key="tw_auth_app")
        auth_desc = st.text_area("原廠授權登記書資料說明", height=80, key="tw_auth_desc")

        cfs_applicable = st.selectbox("出產國製售證明", ["不適用", "適用"], key="tw_cfs_app")
        cfs_desc = st.text_area("出產國製售證明資料說明", height=80, key="tw_cfs_desc")

        qms_applicable = st.selectbox("QMS/QSD", ["不適用", "適用"], key="tw_qms_app")
        qms_desc = st.text_area("QMS/QSD 資料說明（含案號、登錄狀態）", height=80, key="tw_qms_desc")

        similar_info = st.text_area(
            "類似品與比較表摘要（如無類似品則說明理由）",
            height=80,
            key="tw_similar_info",
        )
        labeling_info = st.text_area(
            "標籤、說明書或包裝擬稿重點",
            height=100,
            key="tw_labeling_info",
        )
        tech_file_info = st.text_area(
            "產品結構、材料、規格、性能、用途、圖樣等技術檔案摘要",
            height=120,
            key="tw_tech_file_info",
        )
        preclinical_info = st.text_area(
            "臨床前測試 & 原廠品質管制檢驗摘要（生物相容性、電氣安全、EMC、滅菌、安定性、功能測試、軟體確效等）",
            height=140,
            key="tw_preclinical_info",
        )
        preclinical_replace = st.text_area(
            "如本案適用「替代臨床前測試及原廠品質管制資料」之說明",
            height=100,
            key="tw_preclinical_replace",
        )
        clinical_just = st.selectbox("臨床證據是否適用？", ["不適用", "適用"], key="tw_clinical_app")
        clinical_info = st.text_area(
            "臨床證據摘要（研究報告、臨床評估、臨床試驗、FDA/歐盟核定資料等）",
            height=140,
            key="tw_clinical_info",
        )

    # 產生申請書 Markdown
    if st.button("生成申請書 Markdown 草稿", key="tw_generate_md_btn"):
        missing = []
        if not e_no.strip():
            missing.append("電子流水號")
        if not case_type.strip():
            missing.append("案件類型")
        if not device_category.strip():
            missing.append("醫療器材類型")
        if not origin.strip():
            missing.append("產地")
        if not product_class.strip():
            missing.append("產品等級")
        if not name_zh.strip():
            missing.append("醫療器材中文名稱")
        if not name_en.strip():
            missing.append("醫療器材英文名稱")
        if not uniform_id.strip():
            missing.append("統一編號")
        if not firm_name.strip():
            missing.append("醫療器材商名稱")
        if not firm_addr.strip():
            missing.append("醫療器材商地址")
        if not resp_name.strip():
            missing.append("負責人姓名")
        if not contact_name.strip():
            missing.append("聯絡人姓名")
        if not contact_tel.strip():
            missing.append("電話")
        if not contact_email.strip():
            missing.append("電子郵件")
        if not manu_name.strip():
            missing.append("製造廠名稱")
        if not manu_addr.strip():
            missing.append("製造廠地址")

        if missing:
            st.warning("以下基本欄位尚未填寫完整（形式檢查）：\n- " + "\n- ".join(missing))
            st.session_state["tw_app_status"] = "error"
        else:
            st.session_state["tw_app_status"] = "done"

        apply_date_str = apply_date.strftime("%Y-%m-%d") if apply_date else ""

        app_md = f"""# 第二、三等級醫療器材查驗登記申請書（線上草稿）

## 一、案件基本資料
- 公文文號：{doc_no or "（未填）"}
- 電子流水號：{e_no or "（未填）"}
- 申請日：{apply_date_str or "（未填）"}
- 案件類型：{case_type}
- 醫療器材類型：{device_category}
- 案件種類：{case_kind}
- 產地：{origin}
- 產品等級：{product_class}
- 有無類似品：{similar}
- 是否勾選「替代臨床前測試及原廠品質管制資料」：{replace_flag}
- 前次申請案號（如適用）：{prior_app_no or "不適用"}

## 二、醫療器材基本資訊
- 中文名稱：{name_zh}
- 英文名稱：{name_en}
- 效能、用途或適應症說明：{indications}
- 型號、規格或主要成分：{spec_comp}

### 分類分級品項
- 主類別：{main_cat or "（未填）"}
- 分級品項代碼：{item_code or "（未填）"}
- 分級品項名稱：{item_name or "（未填）"}

## 三、醫療器材商資料
- 統一編號：{uniform_id}
- 醫療器材商名稱：{firm_name}
- 地址：{firm_addr}
- 負責人姓名：{resp_name}
- 聯絡人姓名：{contact_name}
- 電話：{contact_tel}
- 傳真：{contact_fax or "（未填）"}
- 電子郵件：{contact_email}
- 已確認與最新醫療器材商證照資訊相符：{"是" if confirm_match else "否"}

### 其它佐證
- RAPS：{"有" if cert_raps else "無"}
- AHWP：{"有" if cert_ahwp else "無"}
- 其它訓練/證書：{cert_other or "無"}

## 四、製造廠資訊
- 製造方式：{manu_type}
- 製造廠名稱：{manu_name}
- 製造國別：{manu_country}
- 製造廠地址：{manu_addr}
- 製造相關說明：{manu_note or "（未填）"}

## 五～七、原廠授權、出產國製售證明、QMS/QSD
- 原廠授權登記書適用性：{auth_applicable}
- 原廠授權登記書資料說明：{auth_desc or "（未填）"}
- 出產國製售證明適用性：{cfs_applicable}
- 出產國製售證明資料說明：{cfs_desc or "（未填）"}
- QMS/QSD 適用性：{qms_applicable}
- QMS/QSD 資料說明：{qms_desc or "（未填）"}

## 十～十二、類似品、標籤/說明書擬稿、產品技術檔案摘要
### 類似品相關資訊
{similar_info or "（未填）"}

### 標籤／說明書／包裝擬稿重點
{labeling_info or "（未填）"}

### 產品結構、材料、規格、性能、用途、圖樣等技術檔案摘要
{tech_file_info or "（未填）"}

## 十三～十七、特定安全性要求與臨床前測試及品質管制資料
### 臨床前測試與原廠品質管制資料摘要
{preclinical_info or "（未填）"}

### 替代「臨床前測試及原廠品質管制資料」之說明
{preclinical_replace or "（未填）"}

## 十八、臨床證據資料
- 臨床證據適用性：{clinical_just}
- 臨床證據摘要：
{clinical_info or "（未填）"}
"""
        st.session_state["tw_app_markdown"] = app_md

    st.markdown("##### 申請書 Markdown（可編輯）")
    app_md_current = st.session_state.get("tw_app_markdown", "")
    app_view_mode = st.radio(
        "申請書檢視模式", ["Markdown", "純文字"],
        horizontal=True, key="tw_app_viewmode",
    )
    if app_view_mode == "Markdown":
        app_md_edited = st.text_area(
            "申請書內容",
            value=app_md_current,
            height=320,
            key="tw_app_md_edited",
        )
    else:
        app_md_edited = st.text_area(
            "申請書內容（純文字）",
            value=app_md_current,
            height=320,
            key="tw_app_txt_edited",
        )
    st.session_state["tw_app_effective_md"] = app_md_edited

    st.markdown("---")

    # Step 2 – 預審指引
    st.markdown("### Step 2 – 輸入預審/形式審查指引（Screen Review Guidance）")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        guidance_file = st.file_uploader(
            "上傳預審指引 (PDF / TXT / MD)",
            type=["pdf", "txt", "md"],
            key="tw_guidance_file",
        )
        guidance_text_from_file = ""
        if guidance_file is not None:
            suffix = guidance_file.name.lower().rsplit(".", 1)[-1]
            if suffix == "pdf":
                guidance_text_from_file = extract_pdf_pages_to_text(guidance_file, 1, 9999)
            else:
                guidance_text_from_file = guidance_file.read().decode("utf-8", errors="ignore")
    with col_g2:
        guidance_text_manual = st.text_area(
            "或直接貼上預審/形式審查指引文字或 Markdown",
            height=200,
            key="tw_guidance_manual",
        )

    guidance_text = guidance_text_from_file or guidance_text_manual
    st.session_state["tw_guidance_text"] = guidance_text

    if guidance_text:
        st.success("已載入預審/形式審查指引文字。")
    else:
        st.info("尚未提供預審指引。可先填寫申請書草稿，稍後再補。")

    st.markdown("---")

    # Step 3 – 形式審查 / 完整性檢核
    st.markdown("### Step 3 – 形式審查 / 完整性檢核（Agent）")

    if "tw_app_effective_md" not in st.session_state or not st.session_state["tw_app_effective_md"].strip():
        st.warning("尚未產生申請書 Markdown。請先於 Step 1 填寫並點擊「生成申請書 Markdown 草稿」。")
        return

    base_app_md = st.session_state.get("tw_app_effective_md", "")
    base_guidance = st.session_state.get("tw_guidance_text", "")

    combined_input = f"""=== 申請書草稿（Markdown） ===
{base_app_md}

=== 預審 / 形式審查指引（文字/Markdown） ===
{base_guidance or "（尚未提供指引，請依一般法規常規進行形式檢核）"}
"""

    default_screen_prompt = """你是一位熟悉臺灣「第二、三等級醫療器材查驗登記」的形式審查(預審)審查員。

請根據：
1. 上述「申請書草稿（Markdown）」內容
2. 上述「預審 / 形式審查指引」(如有)

執行下列任務，並以 **繁體中文 Markdown** 輸出預審報告：

1. 形式完整性檢核
   - 建立一個表格，逐一列出本案應檢附的主要文件類別（例如：申請書、醫療器材商許可執照、原廠授權登記書、出產國製售證明、QMS/QSD、標籤/說明書擬稿、產品技術檔案、臨床前測試資料、臨床證據資料等）。
   - 對每一項，標示：
     - 「預期應附？」（是/否/不明）
     - 「申請書中是否有提及？」（有/疑似有/未見）
     - 「整體判定」（足夠/可能不足/明顯缺漏）
     - 「備註說明」（請具體指出缺漏內容或需補充重點）。

2. 重要欄位檢核
   - 針對案件基本資料（案件類型、產地、產品等級、有無類似品、替代條款勾選與否）、醫療器材名稱與分類分級品項、醫療器材商與製造廠資訊等，檢查是否有明顯未填或矛盾之處。
   - 若有，請以條列方式說明「問題項目」、「疑慮說明」、「建議申請人補充之資料」。

3. 預審評語摘要
   - 撰寫一段約 300–600 字的預審評語摘要，說明：
     - 本案送件資料整體完整性評估（例如：資料大致齊全 / 某些關鍵附件可能不足 / 明顯缺少核心技術與臨床資料…）。
     - 建議申請人下一步應補充或澄清之項目（可分成「必須補件」與「建議補充」）。

4. 請盡量避免臆測未提及之資料；若無從判斷，請明確註記「依現有輸入無法判斷」。
"""

    st.info(
        "此處預設使用 agents.yaml 中的 `tw_screen_review_agent`。"
        "若 agents.yaml 中尚未定義，可先用 fallback system_prompt（即上方 Prompt 文字）。"
    )

    agent_run_ui(
        agent_id="tw_screen_review_agent",
        tab_key="tw_screen",
        default_prompt=default_screen_prompt,
        default_input_text=combined_input,
        allow_model_override=True,
        tab_label_for_history="TW Premarket Screen Review",
    )

    st.markdown("---")

    # Step 4 – AI 協助編修申請書內容
    st.markdown("### Step 4 – AI 協助編修申請書內容")

    helper_default_prompt = """你是一位協助臺灣醫療器材查驗登記申請人的文件撰寫助手。

請在 **不改變實際技術與法規內容** 的前提下，針對以下「申請書草稿（Markdown）」：

1. 優化段落結構與標題層級，使其更符合主管機關常見文件格式。
2. 修正文句中的明顯語病或不通順處，但不得自行新增未出現在原文的重要技術/臨床資訊。
3. 如有明顯資訊不足之處，可以在文件中以「※待補：...」標註提醒，供申請人後續補充。
4. 保持輸出為 Markdown。
"""

    agent_run_ui(
        agent_id="tw_app_doc_helper",
        tab_key="tw_app_helper",
        default_prompt=helper_default_prompt,
        default_input_text=st.session_state.get("tw_app_effective_md", ""),
        allow_model_override=True,
        tab_label_for_history="TW Application Doc Helper",
    )

# =========================
# 510(k) Tab
# =========================

def render_510k_tab():
    st.title(t("510k_tab"))
    col1, col2 = st.columns(2)
    with col1:
        device_name = st.text_input("Device Name")
        k_number = st.text_input("510(k) Number (e.g., K123456)")
    with col2:
        sponsor = st.text_input("Sponsor / Manufacturer (optional)")
        product_code = st.text_input("Product Code (optional)")
    extra_info = st.text_area("Additional context (indications, technology, etc.)")

    default_prompt = f"""
You are assisting an FDA 510(k) reviewer.

Task:
1. Summarize the publicly available information (or emulate such) for:
   - Device: {device_name}
   - 510(k) number: {k_number}
   - Sponsor: {sponsor}
   - Product code: {product_code}
2. Produce a detailed, review-oriented summary (about 2000–3000 words).
3. Provide several markdown tables (e.g., device overview, indications, performance testing, risks).

Language: {st.session_state.settings["language"]}.
"""
    combined_input = f"""
=== Device Inputs ===
Device name: {device_name}
510(k) number: {k_number}
Sponsor: {sponsor}
Product code: {product_code}

Additional context:
{extra_info}
"""
    agent_run_ui(
        agent_id="fda_510k_intel_agent",
        tab_key="510k",
        default_prompt=default_prompt,
        default_input_text=combined_input,
        tab_label_for_history="510(k) Intelligence",
    )

# =========================
# PDF → Markdown Tab
# =========================

def render_pdf_to_md_tab():
    st.title(t("PDF → Markdown"))

    uploaded = st.file_uploader(
        "Upload PDF to convert selected pages to Markdown",
        type=["pdf"],
        key="pdf_to_md_uploader",
    )
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            num_start = st.number_input("From page", min_value=1, value=1, key="pdf_to_md_from")
        with col2:
            num_end = st.number_input("To page", min_value=1, value=5, key="pdf_to_md_to")

        if st.button("Extract Text", key="pdf_to_md_extract_btn"):
            text = extract_pdf_pages_to_text(uploaded, int(num_start), int(num_end))
            st.session_state["pdf_raw_text"] = text

    raw_text = st.session_state.get("pdf_raw_text", "")
    if raw_text:
        default_prompt = f"""
You are converting part of a regulatory PDF into markdown.

- Goal: produce clean, structured markdown preserving headings, lists,
  and tables (as markdown tables) as much as reasonably possible.
- Do not hallucinate content that is not in the text.

Language: {st.session_state.settings["language"]}.
"""
        agent_run_ui(
            agent_id="pdf_to_markdown_agent",
            tab_key="pdf_to_md",
            default_prompt=default_prompt,
            default_input_text=raw_text,
            tab_label_for_history="PDF → Markdown",
        )
    else:
        st.info("Upload a PDF and click 'Extract Text' to begin.")

# =========================
# 510(k) Review Pipeline Tab
# =========================

def render_510k_review_pipeline_tab():
    st.title(t("Checklist & Report"))

    st.markdown("### Step 1 – 提交資料 → 結構化 Markdown")
    raw_subm = st.text_area(
        "Paste 510(k) submission material (text/markdown)",
        height=200,
        key="subm_paste",
    )
    default_subm_prompt = """You are a 510(k) submission organizer.

Restructure the following content into organized markdown with sections such as:
- Device & submitter information
- Device description and technology
- Indications for use
- Predicate/comparator information
- Performance testing
- Risks and risk controls

Do not invent new facts; only reorganize and clarify.
"""
    if st.button("Structure Submission", key="subm_run_btn"):
        if not raw_subm.strip():
            st.warning("Please paste submission material first.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            try:
                out = call_llm(
                    model=st.session_state.settings["model"],
                    system_prompt="You structure a 510(k) submission.",
                    user_prompt=default_subm_prompt + "\n\n=== SUBMISSION ===\n" + raw_subm,
                    max_tokens=st.session_state.settings["max_tokens"],
                    temperature=0.15,
                    api_keys=api_keys,
                )
                st.session_state["subm_struct_md"] = out
                token_est = int(len(raw_subm + out) / 4)
                log_event("510(k) Review Pipeline", "Submission Structurer", st.session_state.settings["model"], token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    subm_md = st.session_state.get("subm_struct_md", "")
    if subm_md:
        st.markdown("#### Structured Submission (editable)")
        st.text_area("Submission (Markdown)", value=subm_md, height=220, key="subm_struct_md_edited")
    else:
        st.info("Structured submission will appear here.")

    st.markdown("---")
    st.markdown("### Step 2 – Checklist（貼上或由其它 Agent 產生） & Step 3 – Review Report")

    chk_md = st.text_area(
        "Paste checklist (markdown or text)",
        height=200,
        key="chk_md",
    )

    if st.button("Build Review Report", key="rep_run_btn"):
        if not subm_md or not chk_md.strip():
            st.warning("Need both structured submission and checklist.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            rep_prompt = """You are drafting an internal FDA 510(k) review memo.

Using the checklist and structured submission, write a concise review report with:
- Introduction & scope
- Device and submission overview
- Summary of key differences vs. predicate(s)
- Checklist-based assessment (use headings or tables)
- Overall conclusion and recommendations.
"""
            user_prompt = (
                rep_prompt
                + "\n\n=== CHECKLIST ===\n"
                + chk_md
                + "\n\n=== STRUCTURED SUBMISSION ===\n"
                + subm_md
            )
            try:
                out = call_llm(
                    model=st.session_state.settings["model"],
                    system_prompt="You are an FDA 510(k) reviewer.",
                    user_prompt=user_prompt,
                    max_tokens=st.session_state.settings["max_tokens"],
                    temperature=0.18,
                    api_keys=api_keys,
                )
                st.session_state["rep_md"] = out
                token_est = int(len(user_prompt + out) / 4)
                log_event("510(k) Review Pipeline", "Review Memo Builder", st.session_state.settings["model"], token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    rep_md = st.session_state.get("rep_md", "")
    if rep_md:
        st.markdown("#### Review Report (editable)")
        st.text_area("Review Report (Markdown)", value=rep_md, height=260, key="rep_md_edited")

# =========================
# Note Keeper & Magics
# =========================

def highlight_keywords(text: str, keywords: list[str], color: str) -> str:
    if not text or not keywords:
        return text
    out = text
    for kw in sorted(set([k for k in keywords if k.strip()]), key=len, reverse=True):
        safe_kw = kw.strip()
        if not safe_kw:
            continue
        span = f'<span style="color:{color};font-weight:600;">{safe_kw}</span>'
        out = out.replace(safe_kw, span)
    return out


def render_note_keeper_tab():
    st.title(t("Note Keeper & Magics"))

    st.markdown("### Step 1 – Paste Notes & Transform to Structured Markdown")
    raw_notes = st.text_area("Paste your notes (text or markdown)", height=220, key="notes_raw")

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        note_model = st.selectbox(
            "Model for Note → Markdown",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
            key="note_model",
        )
    with col_n2:
        note_max_tokens = st.number_input(
            "max_tokens",
            min_value=2000,
            max_value=120000,
            value=12000,
            step=1000,
            key="note_max_tokens",
        )

    default_note_prompt = """你是一位協助醫療器材/510(k)/TFDA 審查員整理個人筆記的助手。

請將下列雜亂或半結構化的筆記，整理成：

1. 清晰的 Markdown 結構（標題、子標題、條列）。
2. 保留所有技術與法規重點，不要憑空新增內容。
3. 顯示出：
   - 關鍵技術要點
   - 主要風險與疑問
   - 待釐清/追問事項
"""
    note_struct_prompt = st.text_area(
        "Prompt for Note → Markdown",
        value=default_note_prompt,
        height=180,
        key="note_struct_prompt",
    )

    if st.button("Transform notes to structured Markdown", key="note_run_btn"):
        if not raw_notes.strip():
            st.warning("Please paste notes first.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            user_prompt = note_struct_prompt + "\n\n=== RAW NOTES ===\n" + raw_notes
            try:
                out = call_llm(
                    model=note_model,
                    system_prompt="You organize reviewer's notes into clean markdown.",
                    user_prompt=user_prompt,
                    max_tokens=note_max_tokens,
                    temperature=0.15,
                    api_keys=api_keys,
                )
                st.session_state["note_md"] = out
                token_est = int(len(user_prompt + out) / 4)
                log_event("Note Keeper", "Note Structurer", note_model, token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    note_md = st.session_state.get("note_md", raw_notes)
    st.markdown("#### Structured Note (editable)")
    note_view = st.radio(
        "View mode for base note", ["Markdown", "Plain text"],
        horizontal=True, key="note_view_mode",
    )
    if note_view == "Markdown":
        note_md_edited = st.text_area(
            "Note (Markdown)",
            value=note_md,
            height=260,
            key="note_md_edited",
        )
    else:
        note_md_edited = st.text_area(
            "Note (Plain text)",
            value=note_md,
            height=260,
            key="note_txt_edited",
        )
    st.session_state["note_effective"] = note_md_edited

    base_note = st.session_state.get("note_effective", "")

    # Magic 1 – AI Formatting
    st.markdown("---")
    st.markdown("### Magic 1 – AI Formatting")

    fmt_model = st.selectbox(
        "Model (Formatting)",
        ALL_MODELS,
        index=ALL_MODELS.index(st.session_state.settings["model"]),
        key="fmt_model",
    )
    fmt_prompt = st.text_area(
        "Prompt for AI Formatting",
        value="請在不改變內容的前提下，統一標題層級與條列格式，讓此筆記更易讀（輸出 Markdown）。",
        height=120,
        key="fmt_prompt",
    )

    if st.button("Run AI Formatting", key="fmt_run_btn"):
        if not base_note.strip():
            st.warning("No base note available.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            user_prompt = fmt_prompt + "\n\n=== NOTE ===\n" + base_note
            try:
                out = call_llm(
                    model=fmt_model,
                    system_prompt="You are a formatting-only assistant for markdown notes.",
                    user_prompt=user_prompt,
                    max_tokens=12000,
                    temperature=0.1,
                    api_keys=api_keys,
                )
                st.session_state["fmt_note"] = out
                token_est = int(len(user_prompt + out) / 4)
                log_event("Note Keeper", "AI Formatting", fmt_model, token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    fmt_note = st.session_state.get("fmt_note", "")
    if fmt_note:
        st.text_area(
            "Formatted Note (Markdown)",
            value=fmt_note,
            height=220,
            key="fmt_note_edited",
        )

    # Magic 2 – AI Keywords (Manual highlight)
    st.markdown("---")
    st.markdown("### Magic 2 – AI Keywords (Manual highlight)")

    kw_input = st.text_input(
        "Keywords (comma-separated)",
        key="kw_input",
        value="510(k), TFDA, QMS, biocompatibility",
    )
    kw_color = st.color_picker("Color for keywords", "#ff7f50", key="kw_color")

    if st.button("Apply Keyword Highlighting", key="kw_run_btn"):
        if not base_note.strip():
            st.warning("No base note available.")
        else:
            keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
            highlighted = highlight_keywords(base_note, keywords, kw_color)
            st.session_state["kw_note"] = highlighted

    kw_note = st.session_state.get("kw_note", "")
    if kw_note:
        st.markdown("#### Note with Highlighted Keywords (Markdown rendering)")
        st.markdown(kw_note, unsafe_allow_html=True)

    # Magic 3 – AI Summary
    st.markdown("---")
    st.markdown("### Magic 3 – AI Summary")

    sum_model = st.selectbox(
        "Model (Summary)",
        ALL_MODELS,
        index=ALL_MODELS.index("gpt-4o-mini") if "gpt-4o-mini" in ALL_MODELS else 0,
        key="note_sum_model",
    )
    sum_prompt = st.text_area(
        "Prompt for Summary",
        value="請將以下審查筆記摘要為 5–10 個重點 bullet，並附上一段 3–5 句的整體摘要（使用繁體中文）。",
        height=150,
        key="note_sum_prompt",
    )
    if st.button("Run AI Summary", key="note_sum_run_btn"):
        if not base_note.strip():
            st.warning("No base note available.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            user_prompt = sum_prompt + "\n\n=== NOTE ===\n" + base_note
            try:
                out = call_llm(
                    model=sum_model,
                    system_prompt="You write executive-style regulatory summaries.",
                    user_prompt=user_prompt,
                    max_tokens=12000,
                    temperature=0.2,
                    api_keys=api_keys,
                )
                st.session_state["note_summary"] = out
                token_est = int(len(user_prompt + out) / 4)
                log_event("Note Keeper", "AI Summary", sum_model, token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    note_summary = st.session_state.get("note_summary", "")
    if note_summary:
        st.text_area(
            "Summary",
            value=note_summary,
            height=200,
            key="note_summary_edited",
        )

    # Magic 4 – AI Action Items
    st.markdown("---")
    st.markdown("### Magic 4 – AI Action Items")

    act_model = st.selectbox(
        "Model (Action Items)",
        ALL_MODELS,
        index=ALL_MODELS.index(st.session_state.settings["model"]),
        key="note_act_model",
    )
    act_prompt = st.text_area(
        "Prompt for Action Items",
        value="請從以下筆記中萃取需要後續行動的事項（補件、澄清、內部會議等），並以 Markdown 表格輸出：項目、負責人(可留空)、優先順序、說明。",
        height=150,
        key="note_act_prompt",
    )
    if st.button("Run AI Action Items", key="note_act_run_btn"):
        if not base_note.strip():
            st.warning("No base note available.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            user_prompt = act_prompt + "\n\n=== NOTE ===\n" + base_note
            try:
                out = call_llm(
                    model=act_model,
                    system_prompt="You extract action items from regulatory review notes.",
                    user_prompt=user_prompt,
                    max_tokens=12000,
                    temperature=0.2,
                    api_keys=api_keys,
                )
                st.session_state["note_actions"] = out
                token_est = int(len(user_prompt + out) / 4)
                log_event("Note Keeper", "AI Action Items", act_model, token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    note_actions = st.session_state.get("note_actions", "")
    if note_actions:
        st.text_area(
            "Action Items",
            value=note_actions,
            height=220,
            key="note_actions_edited",
        )

    # Magic 5 – AI Glossary
    st.markdown("---")
    st.markdown("### Magic 5 – AI Glossary (術語表)")

    glo_model = st.selectbox(
        "Model (Glossary)",
        ALL_MODELS,
        index=ALL_MODELS.index("gemini-2.5-flash") if "gemini-2.5-flash" in ALL_MODELS else 0,
        key="note_glo_model",
    )
    glo_prompt = st.text_area(
        "Prompt for Glossary",
        value="請從以下筆記中找出重要專有名詞 (英文縮寫、標準、指引文件名稱、專業術語)，製作 Markdown 表格：Term, Full Name/Chinese, Explanation。",
        height=150,
        key="note_glo_prompt",
    )
    if st.button("Run AI Glossary", key="note_glo_run_btn"):
        if not base_note.strip():
            st.warning("No base note available.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            user_prompt = glo_prompt + "\n\n=== NOTE ===\n" + base_note
            try:
                out = call_llm(
                    model=glo_model,
                    system_prompt="You build glossaries for regulatory/technical notes.",
                    user_prompt=user_prompt,
                    max_tokens=12000,
                    temperature=0.2,
                    api_keys=api_keys,
                )
                st.session_state["note_glossary"] = out
                token_est = int(len(user_prompt + out) / 4)
                log_event("Note Keeper", "AI Glossary", glo_model, token_est)
            except Exception as e:
                st.error(f"Error: {e}")

    note_glossary = st.session_state.get("note_glossary", "")
    if note_glossary:
        st.text_area(
            "Glossary",
            value=note_glossary,
            height=220,
            key="note_glossary_edited",
        )

# =========================
# Agents Config Tab
# =========================

def render_agents_config_tab():
    st.title(t("Agents Config"))

    agents_cfg = st.session_state["agents_cfg"]
    agents_dict = agents_cfg.get("agents", {})

    st.subheader("1. Current Agents Overview")
    if not agents_dict:
        st.warning("No agents found in current agents.yaml.")
    else:
        df = pd.DataFrame(
            [
                {
                    "agent_id": aid,
                    "name": acfg.get("name", ""),
                    "model": acfg.get("model", ""),
                    "category": acfg.get("category", ""),
                }
                for aid, acfg in agents_dict.items()
            ]
        )
        st.dataframe(df, use_container_width=True, height=260)

    st.markdown("---")
    st.subheader("2. Edit Full agents.yaml (raw text)")

    yaml_str_current = yaml.dump(
        st.session_state["agents_cfg"],
        allow_unicode=True,
        sort_keys=False,
    )
    edited_yaml_text = st.text_area(
        "agents.yaml (editable)",
        value=yaml_str_current,
        height=320,
        key="agents_yaml_text_editor",
    )

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        if st.button("Apply edited YAML to session", key="apply_edited_yaml"):
            try:
                cfg = yaml.safe_load(edited_yaml_text)
                if not isinstance(cfg, dict) or "agents" not in cfg:
                    st.error("Parsed YAML does not contain top-level key 'agents'. No changes applied.")
                else:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Updated agents.yaml in current session.")
            except Exception as e:
                st.error(f"Failed to parse edited YAML: {e}")

    with col_a2:
        uploaded_agents_tab = st.file_uploader(
            "Upload agents.yaml file",
            type=["yaml", "yml"],
            key="agents_yaml_tab_uploader",
        )
        if uploaded_agents_tab is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents_tab.read())
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Uploaded agents.yaml applied to this session.")
                else:
                    st.warning("Uploaded file has no top-level 'agents' key. Ignoring.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")

    with col_a3:
        st.download_button(
            "Download current agents.yaml",
            data=yaml_str_current.encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
            key="download_agents_yaml_current",
        )

# =========================
# Main
# =========================

st.set_page_config(page_title="Agentic Medical Device Reviewer", layout="wide")

if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "theme": "Light",
        "language": "繁體中文",
        "painter_style": "Van Gogh",
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
    }
if "history" not in st.session_state:
    st.session_state["history"] = []

# Load agents.yaml or default minimal
if "agents_cfg" not in st.session_state:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            st.session_state["agents_cfg"] = yaml.safe_load(f)
    except Exception:
        st.session_state["agents_cfg"] = {
            "agents": {
                "fda_510k_intel_agent": {
                    "name": "510(k) Intelligence Agent",
                    "model": "gpt-4o-mini",
                    "system_prompt": "You are an FDA 510(k) analyst.",
                    "max_tokens": 12000,
                    "category": "FDA 510(k)",
                },
                "pdf_to_markdown_agent": {
                    "name": "PDF to Markdown Agent",
                    "model": "gemini-2.5-flash",
                    "system_prompt": "You convert PDF-extracted text into clean markdown.",
                    "max_tokens": 12000,
                    "category": "文件前處理",
                },
                "tw_screen_review_agent": {
                    "name": "TFDA 預審形式審查代理",
                    "model": "gemini-2.5-flash",
                    "system_prompt": "You are a TFDA premarket screen reviewer.",
                    "max_tokens": 12000,
                    "category": "TFDA Premarket",
                },
                "tw_app_doc_helper": {
                    "name": "TFDA 申請書撰寫助手",
                    "model": "gpt-4o-mini",
                    "system_prompt": "You help improve TFDA application documents.",
                    "max_tokens": 12000,
                    "category": "TFDA Premarket",
                },
            }
        }

render_sidebar()
apply_style(st.session_state.settings["theme"], st.session_state.settings["painter_style"])

tab_labels = [
    t("Dashboard"),
    t("TW Premarket"),
    t("510k_tab"),
    t("PDF → Markdown"),
    t("Checklist & Report"),
    t("Note Keeper & Magics"),
    t("Agents Config"),
]
tabs = st.tabs(tab_labels)

with tabs[0]:
    render_dashboard()
with tabs[1]:
    render_tw_premarket_tab()
with tabs[2]:
    render_510k_tab()
with tabs[3]:
    render_pdf_to_md_tab()
with tabs[4]:
    render_510k_review_pipeline_tab()
with tabs[5]:
    render_note_keeper_tab()
with tabs[6]:
    render_agents_config_tab()
