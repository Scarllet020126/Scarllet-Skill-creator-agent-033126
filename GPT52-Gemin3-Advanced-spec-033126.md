Technical Specification: WOW Regulatory Workbench
Document Version: 1.0
Application: WOW Regulatory Workbench (app.py)
Architecture: Python / Streamlit (Reactive Single-Page Application)
Domain: Regulatory Affairs, Document Automation, Large Language Model (LLM) Orchestration
1. Executive Summary & Product Vision
The WOW Regulatory Workbench is an advanced, AI-powered internal tooling application built on the Streamlit framework. It is designed specifically for Regulatory Affairs (RA) professionals to accelerate the processing, drafting, structuring, and reviewing of complex regulatory submissions, such as the FDA 510(k) and Taiwanese FDA (TFDA) premarket applications.
The core vision of the application is to provide a modular, multi-agent orchestration platform where users are not just consumers of AI output, but active "human-in-the-loop" editors. The system achieves this through a concept called "Editable Handoff" (or Effective Output), where the output of one AI agent can be manually adjusted by the human expert before being fed as the input to the next AI agent in a pipeline.
To support varying levels of complexity and vendor lock-in avoidance, the application is entirely model-agnostic. It features a bespoke LLM routing layer capable of seamlessly translating and transmitting prompts to OpenAI, Google (Gemini), Anthropic (Claude), and xAI (Grok). The entire application configuration—from agent prompts to generation parameters—is managed via a hot-reloadable declarative YAML architecture, empowering prompt engineers and power users to alter the system's behavior dynamically without touching the underlying Python codebase.
2. System Architecture & High-Level Design
The application follows a monolithic, reactive architecture inherent to the Streamlit execution model. Because Streamlit executes the Python script from top to bottom upon every user interaction (button click, text input, slider change), the architecture relies heavily on a robust State Management Layer (st.session_state) to preserve data, logs, and user inputs across render cycles.
2.1 Layered Architecture Overview
Presentation & UI Layer: Dynamically injected CSS ("NEAT UI"), localization (i18n) dictionaries, and interactive status indicators.
State Management Layer: Initializes and synchronizes UI widgets, API keys, global parameters, run histories, and workspace-specific variables.
Application Logic & Workspaces Layer: The business logic divided into eight distinct "Tabs" or Workspaces, each instantiating multiple isolated agent UI components.
Agent Definition Layer (agents.yaml): The declarative configuration storing agent personas, specific model assignments, and token limits.
LLM Routing & Adapter Layer: The integration layer that normalizes requests and responses across disparate third-party AI provider SDKs.
2.2 The Reactive Execution Paradigm
Every user interaction triggers a full top-down execution. To prevent expensive recalculations (like re-parsing PDFs or re-running LLMs), the application utilizes conditional execution blocks bound to Streamlit button states (if st.button(...)) and strictly isolates persistent data inside st.session_state.
3. Core Components & Modules
3.1 LLM Routing & Integration Adapter
One of the most technically complex modules in the application is the universal LLM routing system. Different API providers have fundamentally different SDKs, prompt schemas, and response objects. The application abstractions away this complexity via a unified LLMResult dataclass and provider-specific adapter functions.
3.1.1 The LLMResult Dataclass
Every LLM call, regardless of the provider, returns an LLMResult object. This ensures downstream components (like the UI logging system and telemetry) have a predictable contract.
text (str): The final, normalized string output from the LLM.
usage (Dict): Token consumption statistics (input_tokens, output_tokens).
raw (Any): The raw SDK response object, retained for debugging or deep extraction.
provider (str): The resolved provider name (e.g., "openai", "gemini").
model (str): The specific model ID used.
elapsed_s (float): Execution time in seconds.
error (Optional[str]): Human-readable error message if the call failed.
3.1.2 Provider Heuristics (infer_provider)
To allow users to type in generic model names or experiment with new models not explicitly hardcoded into the MODEL_CATALOG, the application uses string-matching heuristics. It checks prefixes (e.g., gpt-, claude-, gemini-, grok-) and internal substrings to automatically route the model to the correct adapter.
3.1.3 Provider Adapters
OpenAI/Grok Adapter (call_openai_like): Uses the official openai Python package. It constructs the standard [{"role": "system"}, {"role": "user"}] message array. Because xAI's Grok utilizes an OpenAI-compatible endpoint, Grok calls are seamlessly routed through this same adapter by dynamically overriding the base_url parameter to https://api.x.ai/v1.
Gemini Adapter (call_gemini): Utilizes google.generativeai. Crucial Design Decision: Because older or certain tiered versions of the Gemini API possess inconsistent support for the system_instruction parameter, this adapter implements a custom merge_system_user_for_gemini function. It explicitly injects ### SYSTEM and ### USER boundary markers into a single user text prompt. This guarantees compatibility and reduces prompt confusion. It also features a robust text extraction fallback loop to traverse complex candidates and parts arrays if the standard .text property fails.
Anthropic Adapter (call_anthropic): Utilizes the anthropic SDK. It respects Anthropic's strict separation of system prompts (passed as a top-level kwarg) and user messages (passed as the messages array). It robustly handles Anthropic's block-array content structure by joining text blocks.
3.2 Agent Configuration Framework (YAML)
To prevent hardcoding prompt strings deep within UI logic, the application uses an agents.yaml paradigm.
Default State: A DEFAULT_AGENTS_YAML multi-line string provides the baseline system configurations (e.g., pdf_to_markdown, note_entities, report_writer).
Parsing: The load_agents_from_text function uses pyyaml (if available) to parse the configuration. It normalizes missing fields by falling back to st.session_state global defaults (global_default_model, global_max_tokens).
Hot-Reloading: Users can upload a new .yaml file or edit the raw YAML directly in the "Agents Config Studio" tab. Clicking "Apply" instantly overwrites the st.session_state["agents"] dictionary, instantly altering the behavior of every tool in the workbench without a server restart.
3.3 The Universal Agent UI (agent_runner_ui)
The architectural crown jewel of the application is the agent_runner_ui function. Rather than writing custom UI code for the dozens of specific LLM tasks, this single function is called repeatedly to render standardized execution blocks.
3.3.1 Parameters and Configuration
The function accepts arguments like workspace, agent_id, input_text, and a crucial uniqueness identifier, uid. Because Streamlit requires absolutely unique keys for every UI widget, combining workspace + agent_id + uid guarantees that multiple instances of the same agent (e.g., multiple "Summary" agents in the Note Keeper) do not crash the application with a StreamlitDuplicateElementKey exception.
3.3.2 Tri-State Output Synchronization (The "Lost Edit" Fix)
In reactive frameworks, managing text areas where both the AI (programmatically) and the User (manually) edit the text is notoriously difficult. If not handled correctly, a user's manual edit will be erased when the app re-renders, or an AI's new generation will fail to populate the box because Streamlit is prioritizing the user's previous input state.
The agent_runner_ui solves this via a sophisticated Tri-State Synchronization architecture:
out_gen_key (Generated State): Stores the pure, unaltered output directly from the LLM.
out_eff_key (Effective State): The canonical "truth" of the output. This is what downstream agents will consume.
widget_key (Widget State): The explicit key bound to the st.text_area Streamlit widget.
Execution Flow:
On Agent Run: The LLM completes. The text is saved to out_gen_key. The application then forces a UI update by programmatically overwriting both out_eff_key and widget_key with the new generated text.
On User Edit: The user types in the text area. Streamlit automatically updates widget_key. The code immediately syncs this back to the canonical state: st.session_state[out_eff_key] = st.session_state[widget_key].
On Reset: If the user makes a mistake editing, clicking "Reset to Generated Output" overwrites the widget_key and out_eff_key back to the pristine out_gen_key state.
This paradigm explicitly separates what the AI said from what the user finalized, ensuring data integrity and a frustration-free UX.
3.3.3 Interactive Execution & Interruption (st.status)
LLM calls can take anywhere from 2 to 60 seconds. To maintain user trust, agent_runner_ui wraps the execution in a st.status() container. This provides a dynamic, expanding UI element that displays:
The exact model being invoked.
The estimated token count of the prompt payload.
A clear warning instructing the user how to gracefully abort the call using Streamlit's native "Stop" button at the top right of the screen.
Upon completion or failure, the status container updates its label (e.g., "✅ Done in 4.2s" or "❌ Execution Failed") and collapses to keep the UI clean.
3.4 State Management & Telemetry
3.4.1 Session Initialization (ss_init)
At the very top of the script execution, ss_init() ensures all required dictionaries and arrays exist in st.session_state using .setdefault(). This guarantees that dictionary lookups never throw KeyError exceptions, even on the very first page load.
3.4.2 Logging Engine (log_event & record_run)
The workbench acts as an enterprise-grade application by maintaining strict audit trails of all activities in memory.
Live Log (run_log): A lightweight array of dictionaries capturing granular state changes (e.g., "pending", "running", "done", "error"). This is surfaced in the UI via a specialized CSS terminal-like view.
Run History (run_history): A heavier analytics table that stores deep metadata for every completed LLM call. It records run_id (UUID), elapsed time, token estimates (prompt and output), the target workspace, and the provider.
This dual-logging strategy allows the app to provide real-time debug information (Live Log) while simultaneously powering rich aggregated analytics (Dashboard) without slowing down the main execution thread.
4. Workspace Implementations (Feature Specifications)
The application partitions its features into eight distinct, tab-based workspaces. Each workspace addresses a specific phase of the regulatory affairs workflow.
4.1 Dashboard Workspace
The Dashboard acts as the telemetry command center. It reads the run_history array and utilizes pandas to aggregate data and altair to render interactive charts.
Status Wall: Displays the API key readiness for all four supported providers (OpenAI, Gemini, Anthropic, Grok).
Metrics: Shows total runs and error counts.
Visualizations: Renders bar charts for "Usage by Workspace" and "Usage by Model", alongside a time-series line chart tracking "Estimated Tokens over Time".
Data Export: Allows exporting the entire JSON telemetry log for external auditing.
4.2 TW Premarket Workspace
Targeting Taiwanese FDA applications, this is an orchestrator workspace. It accepts raw application context and chains two agents:
Draft Generator: Parses raw JSON/text into a formatted Markdown draft.
Screen Review: Acts as a critic agent, taking the output of the Draft Generator to identify gaps and propose improvements.
Technical note: This workspace serves as a conceptual template for how developers can wire up specific domain contexts into the generalized agent_runner_ui.
4.3 510(k) Intelligence Workspace
Designed for competitive analysis and predicate device research.
Inputs: Device Name, K-number, Sponsor, and a large text area for pasting excerpts or scraped context.
Citations Mode: A toggle that alters the system prompt to force the LLM to insert placeholder brackets (e.g., [Source: user-provided]) to ensure traceability of claims—a critical requirement in regulatory documentation.
Execution: Summarizes the intelligence and offers an optional secondary agent to extract exactly 20 named entities from the summary.
4.4 PDF → Markdown Workspace
A specialized ETL (Extract, Transform, Load) pipeline for unstructured regulatory PDFs.
Extraction: Uses PyPDF2 to read a byte-stream from the st.file_uploader. It implements a safe chunking mechanism, appending --- Page X --- boundaries to help the LLM maintain spatial awareness.
Transformation Agent: Feeds the raw, often messy OCR/PDF text to a high-context LLM (defaulting to Gemini 2.5 Flash due to its massive context window).
Table Fidelity Modes: Allows the user to inject specific prompt constraints ("Fast", "Structured", "Conservative") to instruct the LLM on how aggressively it should attempt to reconstruct broken PDF tables into Markdown formatting.
4.5 510(k) Review Pipeline Workspace
A rigorous, multi-step chain demonstrating the true power of the "Editable Handoff" architecture.
Step A (Structurer): Converts raw, unstructured submission text into a standardized Markdown outline. The user can manually fix any hallucinations here.
Step B (Checklist Cleaner): Normalizes an administrative checklist, ensuring standard numbering.
Step C (Memo Builder): The final synthesis. It constructs a dynamic prompt containing both the edited output of Step A and the edited output of Step B, commanding the LLM to write a comprehensive review memo based solely on those two structured inputs.
4.6 510(k) Report Generator Workspace
The most advanced and complex workspace in the application, simulating a highly structured report drafting process with built-in heuristic Quality Assurance (QA) gates.
Step 1 (Inputs): Accepts reviewer notes and a structural template (offering a built-in 14-section 510(k) template).
Step 2 (Outline): Normalizes the notes against the template.
Step 3 (Draft Report): Generates a 2000-3000 word Markdown report.
Step 4 (Quality Gate): A non-LLM, deterministic validation layer. This step does not use an AI. Instead, it uses Python heuristics to validate the LLM's adherence to strict constraints:
Word Count: Uses Regex (\b\w+\b) to approximate length. Implements a conditional gate allowing lower word counts if the output language is Traditional Chinese (CJK characters calculate differently than English words).
Table Count: Uses a complex multi-line Regex (^\s*\|.*\|\s*\n\s*\|[\s:\-|]+\|\s*$) to count valid Markdown tables, enforcing a strict >= 5 rule.
Entities Rows: Parses the document specifically looking for the "Entities" section, extracts the table, and counts the rows to ensure exactly 20 entities were extracted.
UI Feedback: Renders color-coded pills (Green/Red) indicating pass/fail status for the gates.
Step 5 (Skill Generator): Acts as a meta-agent. It reviews the successfully generated report and writes a skill.md file—a reusable instructional prompt that users can save to reproduce this exact style of report in the future.
4.7 AI Note Keeper Workspace
A flexible, notebook-style interface for ad-hoc regulatory analysis.
Versioning System: Implements a custom session-local state array (versions). Users can save snapshots of their raw text and current Markdown state, allowing them to roll back changes via a dropdown menu.
Safety Mode: A global toggle injected into prompts instructing the LLM to use "conservative rewriting" to strictly prevent the hallucination of medical or regulatory facts.
Fan-Out AI Magics: Once the base notes are converted to Markdown (Step 2), the workspace offers a fan-out architecture of independent mini-agents ("Magics"):
Magic 1 (Formatting): Cleans up messy markdown.
Magic 2 (Keywords): A hybrid Python/UI feature. Accepts comma-separated strings, uses Python re.sub to wrap matched terms in HTML <mark> tags with user-selected colors, and safely renders the HTML alongside the Markdown text.
Magic 3 (Entities): Table extraction.
Magic 4 (Chat): Q&A over the specific context of the note.
Magic 5 (Summary): High-level synthesis.
Magic 6 (Action Items): Checklists with Owners and Due Dates.
Magic 7 (Glossary): Definition builder with evidence pointers.
4.8 Agents Config Studio Workspace
A developer/power-user administration panel.
YAML Editor: A large text area bound directly to the agents_yaml_text session state.
Validation Engine (validate_agents): A deterministic Python function that audits the parsed YAML dictionary. It checks for:
Unknown Models: Flags models not listed in MODEL_CATALOG.
Missing Prompts: Flags agents lacking system instructions.
Token Anomalies: Flags max_tokens values that fall outside sensible bounds (e.g., < 128 or > 8192).
Mini-Runner: A sandboxed instance of agent_runner_ui allowing the user to immediately test YAML configuration changes without navigating away from the studio tab.
5. User Interface (UI) & Styling System (NEAT UI)
The application moves away from Streamlit's default, somewhat clinical appearance by implementing a custom CSS injection framework termed "NEAT UI."
5.1 Dynamic Theming & Painter Styles
The application defines a STYLE_CSS_MAP containing 20 curated color palettes inspired by famous painters (e.g., Van Gogh, Monet, Picasso). Each palette defines three hex codes:
accent: Used for bold text, active borders, and visual highlights.
bg1 & bg2: Used to construct a subtle, radial-gradient background for the main application shell.
If the user selects the "Light" theme from the global settings, the system overrides bg1 and bg2 to soft off-whites (#f7f7fb, #ffffff) to maintain legibility, while preserving the painter's specific accent color. The "Jackpot" button allows randomized style switching via random.choice.
5.2 CSS Component Classes
The injected CSS utilizes the unsafe_allow_html=True parameter in st.markdown to override native Streamlit DOM elements and introduce custom divs:
.wow-shell: Wraps headers and major sections in a gradient container with a soft box-shadow and 16px border radii.
.wow-card: A hover-animated (transform: translateY(-2px)) container used in the Dashboard for metrics.
.wow-pill: Inline, rounded badge components used for displaying metadata tags (Theme, Lang, Status).
.wow-banner: Alert boxes featuring a thick left-border utilizing the dynamic accent color.
.wow-log: A specialized container for raw text logs utilizing monospaced system fonts ("Fira Code", ui-monospace) and a semi-transparent dark background.
Additionally, standard Streamlit inputs (textarea, input, select) and primary buttons are targeted with global CSS rules to round their borders to 10px and apply smooth CSS transitions, elevating the aesthetic quality of the workbench.
6. Data Flow & Handoff Mechanisms
A defining feature of the WOW Regulatory Workbench is how data flows sequentially between AI agents without requiring server-side databases.
6.1 The "Effective Output" Paradigm
In a standard LLM UI, an agent generates text, and the user copies it. In this workbench, data flows via keys in st.session_state.
When Agent A finishes generating, its text populates an editable Streamlit text_area. If the human user modifies this text (e.g., correcting a misunderstood predicate device specification), the Streamlit session state variable (out_eff_key) is instantly updated with the human's changes.
When Agent B is subsequently executed, its prompt string is constructed using f-string interpolation pointing directly to Agent A's out_eff_key.
Example from Pipeline Workspace:
code
Python
memo_input = f"Structured submission:\n{structured}\n\nChecklist (cleaned):\n{checklist_clean}\n\nTask: Write a review memo."
Here, {structured} and {checklist_clean} represent the human-reviewed, "Effective Outputs" of the prior two agents. This creates a highly resilient chain-of-thought process that is inherently immune to compounding AI hallucinations, because the human acts as a verification firewall between every discrete pipeline step.
7. Error Handling, Resilience & Security
7.1 Defensive Programming
The application is built to run continuously in volatile environments (e.g., missing dependencies or network drops).
Optional Dependencies: Heavy libraries like pandas, altair, yaml, and PyPDF2 are imported inside try/except blocks. If they fail to load, the modules are set to None. Downstream UI logic checks if PyPDF2 is None: and gracefully disables file extraction features or charts, preventing catastrophic app crashes.
Robust Text Normalization: The normalize_text_output function guarantees that no matter what obscure object a third-party SDK returns (e.g., an unexpected JSON schema update from Google or Anthropic), it will be cast to a string or formatted JSON, preventing UI rendering errors.
7.2 Security & Privacy Posture
Regulatory documents often contain highly sensitive proprietary or patient data (PHI/PII).
Ephemeral Keys: API keys entered in the sidebar are saved to st.session_state["keys"]. They are never written to disk, logged in the console, or included in any downloadable telemetry files. Refreshing the browser destroys the keys.
Environment Variable Priority: The system checks environment variables (e.g., os.getenv("OPENAI_API_KEY")) before relying on session UI keys. This allows enterprise deployments to inject service accounts securely via Docker/Kubernetes secrets without exposing them in the UI.
Local Execution Logs: Telemetry and live logs are strictly contained within the active browser session state. No backend database (like SQLite or Postgres) is configured to store user prompts, ensuring that sensitive IP never leaves the volatile memory of the immediate hosting environment.
8. Internationalization (i18n) Architecture
The application implements a lightweight but highly effective internationalization framework to support both English and Traditional Chinese (繁體中文).
The I18N Dictionary: A nested dictionary mapping key identifiers (e.g., download_md, run, system_prompt) to their respective language strings.
The t() Function: A helper function that reads the currently active ui_lang_code from the session state and performs a safe .get() lookup on the dictionary. If a translation is missing, it elegantly falls back to the raw key string.
Dynamic Language Injection: In advanced workspaces (like 510(k) Intelligence and Report Generator), the selected UI language is passed directly into the LLM prompt strings (e.g., Output language: {out_lang}). This ensures the AI's generation language structurally matches the user's interface language preferences.
9. Future Extensibility
The WOW Regulatory Workbench is architecturally positioned for rapid scaling:
Adding New LLM Providers: Implementing a new provider (e.g., Mistral, Cohere) requires only creating a new call_provider() function returning an LLMResult, and adding a single line to the infer_provider heuristic map and the call_llm router.
Adding New Workspaces: Developers can add new tabs to the bottom of the script simply by defining a new render_feature() function and wiring together existing agent_runner_ui blocks. No CSS or state-management boilerplate is required.
Vector Database Integration (RAG): The current application relies on direct text pasting. The architecture natively supports the injection of a Retrieval-Augmented Generation (RAG) layer. A theoretical VectorStore class could be instantiated in ss_init(), allowing agents to query embeddings before constructing their final user_prompt payload inside the run_agent function.
10. Appendix
10.1 Key Dependencies
To achieve full functionality, the host environment should have the following Python packages installed:
streamlit (Core framework)
openai (API adapter for OpenAI and Grok)
google-generativeai (API adapter for Gemini)
anthropic (API adapter for Claude)
pyyaml (Agent studio parsing)
pandas (Dashboard analytics)
altair (Dashboard visualizations)
PyPDF2 (Document extraction)
10.2 Recommended Environment Variables
For secure, zero-touch deployments, administrators should provision the following environment variables:
OPENAI_API_KEY
GEMINI_API_KEY or GOOGLE_API_KEY
ANTHROPIC_API_KEY
XAI_API_KEY
GROK_BASE_URL (Defaults to https://api.x.ai/v1)
