import os
import json
import traceback
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# API Clients
from groq import Groq

# UI and Utils
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Input, TextArea, Button, RichLog, Select
from textual import work
from rich.json import JSON
import requests

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

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

# Llama writes a natural journalist-style prompt for GPT-OSS to search against
LLAMA_GPT_PROMPT_SYSTEM = """You are helping prepare a search request for a fact-checking AI that has access to live web search.

You will receive a structured JSON schema extracted from a news article. Your job is to write a single, natural, conversational prompt — as if a curious and skeptical journalist typed it — that asks the AI to search for and verify the key claims.

Rules:
- Write in first person, as a human asking a question
- Keep it concise (3-5 sentences max)
- Naturally weave in the most important entities, claims, and search angles
- Do NOT mention JSON, schemas, or structured data
- Do NOT ask for a structured output format
- Output only the prompt text, nothing else"""

# GPT-OSS is instructed to return structured JSON for clean TUI rendering
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

# Qwen-QwQ final synthesis prompt — structured JSON for clean TUI rendering
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
# NewsAPI Client (used by Path 1 only)
# ---------------------------------------------------------
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"
MAX_TOTAL_ARTICLES = 5
REQUEST_TIMEOUT_SECONDS = 10


@dataclass
class NewsArticle:
    title: str
    source: str
    url: str
    description: str = ""
    published_at: str = ""

    @classmethod
    def from_api_response(cls, raw: dict) -> "NewsArticle":
        return cls(
            title=raw.get("title") or "Untitled",
            source=(raw.get("source") or {}).get("name") or "Unknown",
            url=raw.get("url") or "",
            description=raw.get("description") or "",
            published_at=raw.get("publishedAt") or "",
        )

    def format_for_log(self) -> str:
        lines = [f"  * {self.title}", f"    Source: {self.source}"]
        if self.published_at:
            lines.append(f"    Published: {self.published_at[:10]}")
        if self.description:
            lines.append(f"    {self.description[:120]}...")
        if self.url:
            lines.append(f"    URL: {self.url}")
        return "\n".join(lines)


@dataclass
class NewsSearchResult:
    articles: list[NewsArticle] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    queries_run: int = 0

    @property
    def has_articles(self) -> bool:
        return bool(self.articles)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


class NewsAPIClient:
    class NewsAPIError(Exception):
        pass

    def __init__(
        self, api_key: Optional[str] = None, max_articles: int = MAX_TOTAL_ARTICLES
    ):
        self.api_key = api_key or NEWS_API_KEY
        self.max_articles = max_articles
        if not self.api_key:
            raise NewsAPIClient.NewsAPIError("NEWS_API_KEY is missing in .env")

    def search_from_schema(self, schema: dict) -> NewsSearchResult:
        queries: list[str] = schema.get("search_queries", [])
        if not queries:
            return NewsSearchResult(errors=["Schema contained no search_queries."])

        articles_per_query = max(1, self.max_articles // len(queries))
        result = NewsSearchResult(queries_run=len(queries))

        for query in queries:
            try:
                params = {
                    "q": query,
                    "pageSize": articles_per_query,
                    "apiKey": self.api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                }
                response = requests.get(
                    NEWS_API_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
                )
                response.raise_for_status()
                fetched = [
                    NewsArticle.from_api_response(a)
                    for a in response.json().get("articles", [])
                ]
                result.articles.extend(fetched)
            except Exception as e:
                result.errors.append(f"Error for '{query}': {e}")

        seen, unique = set(), []
        for article in result.articles:
            if article.url not in seen:
                seen.add(article.url)
                unique.append(article)
        result.articles = unique[: self.max_articles]
        return result


# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------
class PreprocessorApp(App):
    CSS = """
    Screen { layout: vertical; background: #000000; color: #f0f0f0; }
    #app-grid { height: 1fr; padding: 0 1; layout: vertical; }
    Header, Footer { background: transparent; color: #61afef; text-style: bold; }
    #conversation-log { height: 1fr; border: solid #444444; margin: 1 0; padding: 0 1; }
    #input-area { height: auto; padding: 1 0; border-top: solid #444444; }
    Input, TextArea, Select { background: transparent; color: #f0f0f0; border: round #61afef; padding: 0 1; margin-bottom: 1; }
    Input:focus, TextArea:focus, Select:focus { border: round #98c379; }
    Button { background: transparent; color: #ffffff; border: round #61afef; padding: 0 1; width: 100%; }
    #url, #headline { height: 3; }
    #paragraph { height: 8; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="app-grid"):
            yield RichLog(id="conversation-log", wrap=True, auto_scroll=True)
            with Vertical(id="input-area"):
                yield Select(
                    options=[
                        ("Groq + News API", "groq_news"),
                        ("Llama -> GPT-OSS-120b -> Qwen", "groq_gpt"),
                    ],
                    value="groq_news",
                    id="api-selector",
                )
                yield Input(placeholder="Article URL (Optional)", id="url")
                yield Input(placeholder="Article Headline", id="headline")
                yield TextArea(
                    placeholder="Paste Article Paragraph Here", id="paragraph"
                )
                yield Button("Process Extraction", id="submit", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            api_choice = self.query_one("#api-selector", Select).value
            url = self.query_one("#url", Input).value
            headline = self.query_one("#headline", Input).value
            paragraph = self.query_one("#paragraph", TextArea).text
            self.process_article(api_choice, url, headline, paragraph)

    def extract_json(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence) :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def stream_collect(self, stream) -> str:
        """Consume a streaming Groq response and return the full concatenated text."""
        return "".join(chunk.choices[0].delta.content or "" for chunk in stream).strip()

    def render_gpt_response(self, raw: str, write_fn) -> str:
        """Parse GPT-OSS JSON response and render it richly in the TUI."""
        try:
            data = json.loads(self.extract_json(raw))
        except json.JSONDecodeError:
            # Fallback: just stream the raw text line by line
            write_fn(
                "[yellow]⚠ Could not parse GPT-OSS response as JSON — showing raw output:[/yellow]"
            )
            for line in raw.splitlines():
                if line.strip():
                    write_fn(line)
            return raw

        write_fn(
            "[bold cyan]┌─ GPT-OSS FACT-CHECK ─────────────────────────────────────┐[/bold cyan]"
        )

        if vp := data.get("verdict_preview"):
            write_fn(f"[bold yellow]│ 🔍 INITIAL VERDICT PREVIEW[/bold yellow]")
            write_fn(f"│  {vp}")
            write_fn("│")

        if ss := data.get("search_summary"):
            write_fn(f"[bold yellow]│ 🌐 SEARCH SUMMARY[/bold yellow]")
            for line in ss.splitlines():
                write_fn(f"│  {line}")
            write_fn("│")

        if findings := data.get("key_findings"):
            write_fn(f"[bold yellow]│ 📋 KEY FINDINGS[/bold yellow]")
            for f_ in findings:
                icon = (
                    "[green]✔[/green]" if f_.get("supports_article") else "[red]✘[/red]"
                )
                write_fn(f"│  {icon} {f_.get('finding', '')}")
            write_fn("│")

        if contradictions := data.get("contradictions"):
            write_fn(f"[bold red]│ ⚠ CONTRADICTIONS[/bold red]")
            for c in contradictions:
                write_fn(f"│  [red]• {c}[/red]")
            write_fn("│")

        if reasoning := data.get("reasoning"):
            write_fn(f"[bold yellow]│ 🧠 REASONING[/bold yellow]")
            for line in reasoning.splitlines():
                write_fn(f"│  {line}")
            write_fn("│")

        if sources := data.get("sources"):
            write_fn(f"[bold yellow]│ 🔗 SOURCES[/bold yellow]")
            for s in sources:
                write_fn(f"│  [bold]{s.get('title', 'Unknown')}[/bold]")
                write_fn(f"│    [dim]{s.get('url', '')}[/dim]")
                write_fn(f"│    {s.get('relevance', '')}")
            write_fn("│")

        write_fn(
            "[bold cyan]└──────────────────────────────────────────────────────────┘[/bold cyan]"
        )
        return raw

    def render_qwen_response(self, raw: str, write_fn):
        """Parse Qwen JSON response and render it richly in the TUI."""
        try:
            data = json.loads(self.extract_json(raw))
        except json.JSONDecodeError:
            write_fn(
                "[yellow]⚠ Could not parse Qwen response as JSON — showing raw output:[/yellow]"
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
        verdict = data.get("verdict", "Unknown")
        confidence = data.get("confidence", "?")
        color = VERDICT_COLORS.get(verdict.lower(), "yellow")

        write_fn(
            f"[bold {color}]╔═ QWEN FINAL VERDICT ══════════════════════════════════════╗[/bold {color}]"
        )
        write_fn(f"[bold {color}]║  {verdict.upper():<55}║[/bold {color}]")
        write_fn(
            f"[bold {color}]║  Confidence: {confidence}/100{' ' * (44 - len(str(confidence)))}║[/bold {color}]"
        )
        write_fn(
            f"[bold {color}]╚═══════════════════════════════════════════════════════════╝[/bold {color}]"
        )
        write_fn("")

        if steps := data.get("reasoning_steps"):
            write_fn("[bold yellow]🧠 REASONING STEPS[/bold yellow]")
            for step in steps:
                write_fn(f"  {step}")
            write_fn("")

        if summary := data.get("summary"):
            write_fn("[bold yellow]📝 SUMMARY[/bold yellow]")
            for line in summary.splitlines():
                write_fn(f"  {line}")
            write_fn("")

        if sources := data.get("key_sources"):
            write_fn("[bold yellow]🔗 KEY SOURCES[/bold yellow]")
            for s in sources:
                write_fn(f"  [bold]• {s.get('title', 'Unknown')}[/bold]")
                write_fn(f"    [dim]{s.get('url', '')}[/dim]")
                write_fn(f"    {s.get('note', '')}")
            write_fn("")

        if caveats := data.get("caveats"):
            write_fn("[bold yellow]⚠ CAVEATS[/bold yellow]")
            for c in caveats:
                write_fn(f"  [dim]• {c}[/dim]")
            write_fn("")

    @work(thread=True)
    def process_article(self, api_choice: str, url: str, headline: str, paragraph: str):
        log = self.query_one("#conversation-log", RichLog)

        def write(msg):
            self.call_from_thread(log.write, msg)

        self.call_from_thread(log.clear)
        write(f"[System] Selected Provider: {api_choice.upper()}")
        write(f"[User Input] URL: {url}")
        write(f"[User Input] Headline: {headline}")
        write(f"[User Input] Paragraph: {paragraph}\n")

        user_content = f"URL: {url}\nHeadline: {headline}\nParagraph: {paragraph}"

        try:
            if not GROQ_API_KEY:
                write("[Error] GROQ_API_KEY not found.")
                return

            groq_client = Groq(api_key=GROQ_API_KEY)

            # ==========================================
            # PATH 1: GROQ LLAMA + NEWSAPI
            # ==========================================
            if api_choice == "groq_news":
                write("[System] Calling Groq / Llama...")
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                )
                raw_resp = completion.choices[0].message.content
                parsed_json = json.loads(self.extract_json(raw_resp))
                write("[JSON Output]\n")
                write(JSON.from_data(parsed_json))

                is_credible = parsed_json.get("is_url_credible", False)
                if is_credible:
                    write("\n[Assessment] URL highly credible. Halting.")
                    return

                trust_score = parsed_json.get("arbitrary_trust_score", 50)
                info_category = parsed_json.get("information_category", "unknown")
                write(
                    f"\n[Secondary Evaluation] Score: {trust_score}/100 | Category: {info_category.upper()}"
                )

                if trust_score <= 50:
                    write(
                        f"[Warning] High likelihood of {info_category}. Aborting search."
                    )
                    return
                elif trust_score >= 71:
                    write("[Assessment] High likelihood of real news. Halting.")
                    return
                else:
                    write(
                        "[NewsAPI] Score is ambiguous (51-70). Searching for articles...\n"
                    )
                    news_client = NewsAPIClient()
                    result = news_client.search_from_schema(parsed_json)
                    if result.has_articles:
                        for article in result.articles:
                            write(article.format_for_log())
                            write("")
                    else:
                        write("[NewsAPI] No articles found.")

            # ==========================================
            # PATH 2: LLAMA -> GPT-OSS-120b -> QWEN-QWQ
            # ==========================================
            elif api_choice == "groq_gpt":
                # --- STEP 1a: Llama extracts structured schema ---
                write("[System] Step 1a: Llama extracting and structuring claims...")
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
                write("[Llama Extracted Schema]\n")
                write(JSON.from_data(parsed_schema))

                # --- STEP 1b: Llama writes a human-like search prompt for GPT-OSS ---
                write(
                    "\n[System] Step 1b: Llama writing search prompt for GPT-OSS-120b..."
                )
                prompt_gen = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": LLAMA_GPT_PROMPT_SYSTEM},
                        {"role": "user", "content": json.dumps(parsed_schema)},
                    ],
                    temperature=0.7,
                )
                gpt_user_prompt = prompt_gen.choices[0].message.content.strip()
                write(f"[Llama -> GPT-OSS Prompt]\n{gpt_user_prompt}")

                # --- STEP 2: GPT-OSS-120b live browser search + structured JSON ---
                write("\n[System] Step 2: GPT-OSS-120b searching and reasoning live...")
                write("[dim]⏳ Collecting response (this may take a moment)...[/dim]")

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
                self.render_gpt_response(gpt_response, write)

                # --- STEP 3: Qwen-QwQ final verdict, structured JSON ---
                write("\n[System] Step 3: Qwen-QwQ-32b synthesizing final verdict...")
                write("[dim]⏳ Collecting response (this may take a moment)...[/dim]")

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
                self.render_qwen_response(qwen_response, write)
                write(
                    "\n[dim][System] Pipeline complete: "
                    "Llama → schema | GPT-OSS-120b → live search | Qwen-QwQ → final verdict.[/dim]"
                )

        except json.JSONDecodeError as e:
            write(f"[Error] JSON parse failed: {e}\n")
        except Exception as e:
            err_lines = traceback.format_exc().splitlines()
            write(f"[Exception] {type(e).__name__}: {e}\n")
            for line in err_lines[-6:]:
                write(line)


if __name__ == "__main__":
    app = PreprocessorApp()
    app.run()
