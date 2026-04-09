import os
import json
import traceback
from typing import Optional
from dotenv import load_dotenv
import re

# API Clients
from groq import Groq

# UI and Utils
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Header, Footer, Input, TextArea, Button, RichLog, Label
from textual import work
from rich.text import Text

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------

GROQ_SYSTEM_PROMPT = """Analyze the provided URL, headline, and paragraph to extract key claims and entities. Assess the credibility of the source URL. Generate targeted search queries to fact-check these claims. Output strictly valid JSON. No markdown formatting, no code blocks, no preamble.

Credibility Assessment & Trust Score:
- Evaluate the domain of the provided URL. If it belongs to a highly credible news organization, set "is_url_credible" to true.
- Evaluate the text's tone, objectivity, verifiable facts, and sensationalism to assign an "arbitrary_trust_score" between 0 and 100.
- Categorize it in "information_category" as either: "misinformation", "disinformation", "malinformation", or "real news".

Query Constraints:
- Use highly specific keywords, entities, and exact match quotes (""). Exclude natural language questions.

Schema:
{
  "is_url_credible": false,
  "arbitrary_trust_score": 45,
  "information_category": "misinformation",
  "claims": ["Statement 1", "Statement 2"],
  "entities": ["Entity 1", "Entity 2"],
  "context_tags": ["Category A", "Category B"],
  "search_queries": ["\"Entity 1\" keyword", "\"Exact phrase\""],
  "metadata": {"timestamp": "2026-04-08", "domain": "domain.com"}
}"""

LLAMA_GPT_PROMPT_SYSTEM = """You are helping prepare a search request for a fact-checking AI that has access to live web search.

You will receive a structured JSON schema extracted from a news article. Your job is to write a single, natural, conversational prompt — as if a curious and skeptical journalist typed it — that asks the AI to search for and verify the key claims.

Rules:
- Write in first person, as a human asking a question
- Keep it concise (3-5 sentences max)
- Naturally weave in the most important entities, claims, and search angles
- Do NOT mention JSON, schemas, or structured data
- Do NOT ask for a structured output format
- Output only the prompt text, nothing else
- Keep it below 2 sentences to minimize token usage"""

GPT_OSS_SYSTEM_PROMPT = """You are a fact-checking AI with access to live web search. Research the claims in the user's query thoroughly.

Return ONLY a valid JSON object — no markdown, no code blocks, no preamble. Schema:
{
  "verdict_preview": "One sentence gut-check verdict before deep analysis",
  "search_summary": "2-3 sentence summary of what you searched for and found",
  "key_findings": [
    {"finding": "Specific verifiable claim or fact discovered", "supports_article": true}
  ],
  "sources": [
    {"title": "Source name", "url": "https://...", "relevance": "Why this source matters"}
  ],
  "contradictions": ["Any claim in the article contradicted by sources"],
  "reasoning": "Full step-by-step reasoning (3-5 sentences)"
}"""

QWEN_SUMMARY_SYSTEM_PROMPT = """You are an expert media analyst and fact-checking specialist with deep reasoning capabilities. You will receive:
1. The original article data (URL, headline, paragraph)
2. An initial credibility assessment from Llama (trust score, category, claims, entities)
3. A structured fact-check report from GPT-OSS-120b with sources and findings

Synthesize everything into a final authoritative verdict. Think step by step before concluding.

Return ONLY a valid JSON object — no markdown, no code blocks, no preamble. Schema:
{
  "verdict": "Real News | Misinformation | Disinformation | Malinformation",
  "confidence": 85,
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "summary": "2-3 paragraph plain-language explanation for a general audience",
  "key_sources": [
    {"title": "Source name", "url": "https://...", "note": "Why it was decisive"}
  ],
  "caveats": ["Limitation or unknown that could affect the verdict"]
}"""


# ---------------------------------------------------------
# Rich markup helpers — avoids broken [/tag] issues
# ---------------------------------------------------------


def markup(text: str, style: str) -> Text:
    """Return a Rich Text object with the given style applied safely."""
    t = Text(text)
    t.stylize(style)
    return t


def bold(text: str) -> Text:
    return markup(text, "bold")


def dim(text: str) -> Text:
    return markup(text, "dim")


def colored(text: str, color: str) -> Text:
    return markup(text, color)


def bold_colored(text: str, color: str) -> Text:
    return markup(text, f"bold {color}")


# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------


class PreprocessorApp(App):
    CSS = """
    Screen { layout: vertical; background: #000000; color: #f0f0f0; }
    #app-grid { height: 1fr; padding: 0 1; layout: vertical; }
    Header, Footer { background: transparent; color: #61afef; text-style: bold; }

    #panels { layout: horizontal; height: 1fr; }

    .panel { width: 1fr; layout: vertical; border: solid #444444; margin: 1 0; padding: 0; }
    .panel-label {
        height: 1;
        content-align: center middle;
        text-style: bold;
        background: #1a1a2e;
        color: #61afef;
        padding: 0 1;
    }
    .panel-log { height: 1fr; padding: 0 1; }

    #llama-label { background: #1a1a2e; color: #e06c75; }
    #gpt-label   { background: #1a1a2e; color: #61afef; }
    #qwen-label  { background: #1a1a2e; color: #98c379; }

    #input-area { height: auto; padding: 1 0; border-top: solid #444444; }
    Input, TextArea { background: transparent; color: #f0f0f0; border: round #61afef; padding: 0 1; margin-bottom: 1; }
    Input:focus, TextArea:focus { border: round #98c379; }
    Button { background: transparent; color: #ffffff; border: round #61afef; padding: 0 1; width: 100%; }
    #url, #headline { height: 3; }
    #paragraph { height: 8; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="app-grid"):
            with Horizontal(id="panels"):
                with Vertical(classes="panel"):
                    yield Label(
                        "🦙  LLAMA — Schema Extraction",
                        id="llama-label",
                        classes="panel-label",
                    )
                    yield RichLog(
                        id="llama-log", wrap=True, auto_scroll=True, classes="panel-log"
                    )
                with Vertical(classes="panel"):
                    yield Label(
                        "🤖  GPT-OSS-120b — Live Fact-Check",
                        id="gpt-label",
                        classes="panel-label",
                    )
                    yield RichLog(
                        id="gpt-log", wrap=True, auto_scroll=True, classes="panel-log"
                    )
                with Vertical(classes="panel"):
                    yield Label(
                        "🧠  QWEN — Final Verdict",
                        id="qwen-label",
                        classes="panel-label",
                    )
                    yield RichLog(
                        id="qwen-log", wrap=True, auto_scroll=True, classes="panel-log"
                    )
            with Vertical(id="input-area"):
                yield Input(placeholder="Article URL (Optional)", id="url")
                yield Input(placeholder="Article Headline", id="headline")
                yield TextArea(
                    placeholder="Paste Article Paragraph Here", id="paragraph"
                )
                yield Button(
                    "Run Pipeline: Llama → GPT-OSS-120b → Qwen",
                    id="submit",
                    variant="primary",
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            url = self.query_one("#url", Input).value
            headline = self.query_one("#headline", Input).value
            paragraph = self.query_one("#paragraph", TextArea).text
            self.process_article(url, headline, paragraph)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def extract_json(self, raw_text: str) -> str:
        # Strip reasoning blocks
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # Strip markdown fences
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence) :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Enforce JSON boundary extraction
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned.strip()

    def stream_collect(self, stream) -> str:
        return "".join(chunk.choices[0].delta.content or "" for chunk in stream).strip()

    # ------------------------------------------------------------------
    # Renderers — write Rich Text objects to avoid broken markup tags
    # ------------------------------------------------------------------

    def render_llama_schema(self, data: dict, write_fn):
        write_fn(
            bold_colored(
                "┌─ LLAMA EXTRACTED SCHEMA ─────────────────────────────────┐", "cyan"
            )
        )

        credible = data.get("is_url_credible", False)
        score = data.get("arbitrary_trust_score", "?")
        category = data.get("information_category", "unknown").upper()

        cred_color = "green" if credible else "red"
        write_fn(
            Text.assemble(
                bold_colored("│ ", "cyan"),
                bold("URL Credible: "),
                colored(str(credible), cred_color),
            )
        )

        score_color = (
            "green"
            if isinstance(score, int) and score >= 71
            else ("yellow" if isinstance(score, int) and score >= 51 else "red")
        )
        write_fn(
            Text.assemble(
                bold_colored("│ ", "cyan"),
                bold("Trust Score: "),
                colored(f"{score}/100", score_color),
            )
        )
        write_fn(
            Text.assemble(
                bold_colored("│ ", "cyan"),
                bold("Category: "),
                colored(category, "yellow"),
            )
        )
        write_fn(bold_colored("│", "cyan"))

        if claims := data.get("claims"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"), bold_colored("📌 CLAIMS", "yellow")
                )
            )
            for c in claims:
                write_fn(Text.assemble(bold_colored("│  ", "cyan"), Text(f"• {c}")))
            write_fn(bold_colored("│", "cyan"))

        if entities := data.get("entities"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"), bold_colored("🏷  ENTITIES", "yellow")
                )
            )
            write_fn(
                Text.assemble(bold_colored("│  ", "cyan"), Text(", ".join(entities)))
            )
            write_fn(bold_colored("│", "cyan"))

        if tags := data.get("context_tags"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"),
                    bold_colored("🗂  CONTEXT TAGS", "yellow"),
                )
            )
            write_fn(Text.assemble(bold_colored("│  ", "cyan"), Text(", ".join(tags))))
            write_fn(bold_colored("│", "cyan"))

        if queries := data.get("search_queries"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"),
                    bold_colored("🔍 SEARCH QUERIES", "yellow"),
                )
            )
            for q in queries:
                write_fn(Text.assemble(bold_colored("│  ", "cyan"), dim(f"» {q}")))
            write_fn(bold_colored("│", "cyan"))

        if meta := data.get("metadata"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"), bold_colored("ℹ  METADATA", "yellow")
                )
            )
            for k, v in meta.items():
                write_fn(Text.assemble(bold_colored("│  ", "cyan"), dim(f"{k}: {v}")))

        write_fn(
            bold_colored(
                "└──────────────────────────────────────────────────────────┘", "cyan"
            )
        )

    def render_gpt_response(self, raw: str, write_fn):
        try:
            data = json.loads(self.extract_json(raw))
        except json.JSONDecodeError:
            write_fn(
                colored(
                    "⚠ Could not parse GPT-OSS response as JSON — showing raw output:",
                    "yellow",
                )
            )
            for line in raw.splitlines():
                if line.strip():
                    write_fn(line)
            return raw

        write_fn(
            bold_colored(
                "┌─ GPT-OSS FACT-CHECK ─────────────────────────────────────┐", "cyan"
            )
        )

        if vp := data.get("verdict_preview"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"),
                    bold_colored("🔍 INITIAL VERDICT PREVIEW", "yellow"),
                )
            )
            write_fn(Text.assemble(bold_colored("│  ", "cyan"), Text(vp)))
            write_fn(bold_colored("│", "cyan"))

        if ss := data.get("search_summary"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"),
                    bold_colored("🌐 SEARCH SUMMARY", "yellow"),
                )
            )
            for line in ss.splitlines():
                write_fn(Text.assemble(bold_colored("│  ", "cyan"), Text(line)))
            write_fn(bold_colored("│", "cyan"))

        if findings := data.get("key_findings"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"),
                    bold_colored("📋 KEY FINDINGS", "yellow"),
                )
            )
            for f_ in findings:
                icon = (
                    colored("✔", "green")
                    if f_.get("supports_article")
                    else colored("✘", "red")
                )
                write_fn(
                    Text.assemble(
                        bold_colored("│  ", "cyan"),
                        icon,
                        Text(f" {f_.get('finding', '')}"),
                    )
                )
            write_fn(bold_colored("│", "cyan"))

        if contradictions := data.get("contradictions"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"), bold_colored("⚠ CONTRADICTIONS", "red")
                )
            )
            for c in contradictions:
                write_fn(
                    Text.assemble(bold_colored("│  ", "cyan"), colored(f"• {c}", "red"))
                )
            write_fn(bold_colored("│", "cyan"))

        if reasoning := data.get("reasoning"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"), bold_colored("🧠 REASONING", "yellow")
                )
            )
            for line in reasoning.splitlines():
                write_fn(Text.assemble(bold_colored("│  ", "cyan"), Text(line)))
            write_fn(bold_colored("│", "cyan"))

        if sources := data.get("sources"):
            write_fn(
                Text.assemble(
                    bold_colored("│ ", "cyan"), bold_colored("🔗 SOURCES", "yellow")
                )
            )
            for s in sources:
                write_fn(
                    Text.assemble(
                        bold_colored("│  ", "cyan"), bold(s.get("title", "Unknown"))
                    )
                )
                write_fn(
                    Text.assemble(bold_colored("│    ", "cyan"), dim(s.get("url", "")))
                )
                write_fn(
                    Text.assemble(
                        bold_colored("│    ", "cyan"), Text(s.get("relevance", ""))
                    )
                )
            write_fn(bold_colored("│", "cyan"))

        write_fn(
            bold_colored(
                "└──────────────────────────────────────────────────────────┘", "cyan"
            )
        )
        return raw

    def render_qwen_response(self, raw: str, write_fn):
        try:
            data = json.loads(self.extract_json(raw))
        except json.JSONDecodeError:
            write_fn(
                colored(
                    "⚠ Could not parse Qwen response as JSON — showing raw output:",
                    "yellow",
                )
            )
            for line in raw.splitlines():
                if line.strip():
                    write_fn(line)
            return

        VERDICT_COLORS = {
            "real news": "green",
            "misinformation": "red",
            "disinformation": "bright_red",
            "malinformation": "dark_orange",
        }
        verdict: str = data.get("verdict", "Unknown")
        confidence = data.get("confidence", "?")
        color = VERDICT_COLORS.get(verdict.lower(), "yellow")

        # Verdict banner
        bar = "═" * 59
        write_fn(bold_colored(f"╔{bar}╗", color))
        write_fn(bold_colored(f"║  {verdict.upper():<57}║", color))
        write_fn(
            bold_colored(
                f"║  Confidence: {confidence}/100{' ' * (44 - len(str(confidence)))}║",
                color,
            )
        )
        write_fn(bold_colored(f"╚{bar}╝", color))
        write_fn(Text(""))

        # Reasoning steps
        if steps := data.get("reasoning_steps"):
            write_fn(bold_colored("🧠 REASONING STEPS", "yellow"))
            for step in steps:
                write_fn(Text.assemble(Text("  "), Text(step)))
            write_fn(Text(""))

        # Summary
        if summary := data.get("summary"):
            write_fn(bold_colored("📝 SUMMARY", "yellow"))
            for line in summary.splitlines():
                write_fn(Text.assemble(Text("  "), Text(line)))
            write_fn(Text(""))

        # Key sources
        if sources := data.get("key_sources"):
            write_fn(bold_colored("🔗 KEY SOURCES", "yellow"))
            for s in sources:
                write_fn(
                    Text.assemble(Text("  "), bold(f"• {s.get('title', 'Unknown')}"))
                )
                write_fn(Text.assemble(Text("    "), dim(s.get("url", ""))))
                write_fn(Text.assemble(Text("    "), Text(s.get("note", ""))))
            write_fn(Text(""))

        # Caveats
        if caveats := data.get("caveats"):
            write_fn(bold_colored("⚠ CAVEATS", "yellow"))
            for c in caveats:
                write_fn(Text.assemble(Text("  "), dim(f"• {c}")))
            write_fn(Text(""))

    # ------------------------------------------------------------------
    # Main pipeline worker
    # ------------------------------------------------------------------

    @work(thread=True)
    def process_article(self, url: str, headline: str, paragraph: str):
        llama_log = self.query_one("#llama-log", RichLog)
        gpt_log = self.query_one("#gpt-log", RichLog)
        qwen_log = self.query_one("#qwen-log", RichLog)

        def write_llama(msg):
            self.call_from_thread(llama_log.write, msg)

        def write_gpt(msg):
            self.call_from_thread(gpt_log.write, msg)

        def write_qwen(msg):
            self.call_from_thread(qwen_log.write, msg)

        # Clear all panels
        self.call_from_thread(llama_log.clear)
        self.call_from_thread(gpt_log.clear)
        self.call_from_thread(qwen_log.clear)

        user_content = f"URL: {url}\nHeadline: {headline}\nParagraph: {paragraph}"

        try:
            if not GROQ_API_KEY:
                write_llama(colored("[Error] GROQ_API_KEY not found.", "red"))
                return

            groq_client = Groq(api_key=GROQ_API_KEY)

            # ==========================================
            # STEP 1a: Llama — extract structured schema
            # ==========================================
            write_llama(dim("⏳ Extracting schema..."))
            llama_completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            raw_llama = llama_completion.choices[0].message.content
            parsed_schema = json.loads(self.extract_json(raw_llama))

            self.call_from_thread(llama_log.clear)
            write_llama(Text(""))
            self.render_llama_schema(parsed_schema, write_llama)

            # ==========================================
            # STEP 1b: Llama writes human-like prompt
            # ==========================================
            prompt_gen = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": LLAMA_GPT_PROMPT_SYSTEM},
                    {"role": "user", "content": json.dumps(parsed_schema)},
                ],
                temperature=0.7,
            )
            gpt_user_prompt = prompt_gen.choices[0].message.content.strip()
            write_llama(Text(""))
            write_llama(bold_colored("📤 Prompt sent to GPT-OSS:", "cyan"))
            write_llama(dim(gpt_user_prompt))

            # ==========================================
            # STEP 2: GPT-OSS-120b — live search
            # ==========================================
            write_gpt(dim("⏳ Searching live web... (may take a moment)"))

            gpt_stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": GPT_OSS_SYSTEM_PROMPT},
                    {"role": "user", "content": gpt_user_prompt},
                ],
                temperature=1,
                max_completion_tokens=5000,
                top_p=1,
                reasoning_effort="medium",
                stream=True,
                stop=None,
                tools=[{"type": "browser_search"}],
            )

            gpt_response = self.stream_collect(gpt_stream)
            self.call_from_thread(gpt_log.clear)
            write_gpt(Text(""))
            self.render_gpt_response(gpt_response, write_gpt)

            # ==========================================
            # STEP 3: Qwen-QwQ — final verdict
            # ==========================================
            write_qwen(dim("⏳ Synthesizing final verdict... (may take a moment)"))

            qwen_user_prompt = (
                f"ORIGINAL ARTICLE:\n"
                f"URL: {url}\n"
                f"Headline: {headline}\n"
                f"Paragraph: {paragraph}\n\n"
                f"LLAMA INITIAL ANALYSIS:\n"
                f"Trust Score: {parsed_schema.get('arbitrary_trust_score')}\n"
                f"Category: {parsed_schema.get('information_category')}\n"
                f"Claims: {'; '.join(parsed_schema.get('claims', []))}\n"
                f"Entities: {'; '.join(parsed_schema.get('entities', []))}\n\n"
                f"GPT-OSS-120b FACT-CHECK RESULTS (JSON):\n{gpt_response}\n\n"
                f"Synthesize everything into a final verdict."
            )

            qwen_stream = groq_client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": QWEN_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": qwen_user_prompt},
                ],
                temperature=0.6,
                max_completion_tokens=2048,
                stream=True,
            )

            qwen_response = self.stream_collect(qwen_stream)
            self.call_from_thread(qwen_log.clear)
            write_qwen(Text(""))
            self.render_qwen_response(qwen_response, write_qwen)

            # Done notice in all panels
            done = dim("✓ Pipeline complete")
            write_llama(Text(""))
            write_llama(done)
            write_gpt(Text(""))
            write_gpt(done)
            write_qwen(Text(""))
            write_qwen(done)

        except json.JSONDecodeError as e:
            write_llama(colored(f"[Error] JSON parse failed: {e}", "red"))
        except Exception as e:
            err_lines = traceback.format_exc().splitlines()
            msg = colored(f"[Exception] {type(e).__name__}: {e}", "red")
            for w in (write_llama, write_gpt, write_qwen):
                w(msg)
            for line in err_lines[-6:]:
                write_llama(Text(line))


if __name__ == "__main__":
    app = PreprocessorApp()
    app.run()
