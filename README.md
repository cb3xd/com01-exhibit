# TruthGuard AI

TruthGuard AI is a terminal-based tool (TUI) designed to analyze news articles, extract key claims and entities, and perform automated fact-checking using advanced LLMs and live news data.

## Features

- **Terminal User Interface (TUI):** Interactive and responsive terminal UI built with [Textual](https://textual.textualize.io/).
- **Dual Pipeline Analysis:**
  - **Path 1 (Groq + News API):** Uses Llama 3 to extract structured data and search for supporting/contradicting articles via [NewsAPI](https://newsapi.org/).
  - **Path 2 (Multi-LLM Synthesis):** Employs a sophisticated chain:
    - **Llama 3:** For initial claim extraction and search prompt generation.
    - **GPT-OSS-120b:** For live web search and deep reasoning.
    - **Qwen-QwQ:** For final authoritative verdict synthesis.
- **Credibility Assessment:** Automatically assigns trust scores and categorizes information (Real News, Misinformation, Disinformation, Malinformation).
- **Rich Logging:** Real-time, formatted output of the analysis process.

## Prerequisites

- Python 3.11 or higher
- [Groq API Key](https://console.groq.com/)
- [NewsAPI Key](https://newsapi.org/register) (Required for Path 1)

## Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd predictive-media-validator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
NEWS_API_KEY=your_news_api_key_here
```

## Usage

Run the application using Python:

```bash
python app.py
```

### Navigating the TUI
- **API Selector:** Choose between the "Groq + News API" or "Llama -> GPT-OSS -> Qwen" pipelines.
- **URL/Headline/Paragraph:** Enter the details of the article you want to validate.
- **Submit:** Click the "Process Extraction" button to start the analysis.
- **Logs:** View the step-by-step progress and final results in the log area.

## Running with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t predictive-media-validator .
   ```

2. **Run the container:**
   ```bash
   docker run -it --env-file .env predictive-media-validator
   ```
   *Note: The `-it` flag is necessary for the TUI to function correctly.*

## Project Structure

- `app.py`: Main application logic and TUI definition.
- `requirements.txt`: Python dependencies.
- `.env`: Environment variables (not tracked by git).
- `.dockerignore`: Files excluded from Docker builds.
- `Dockerfile`: Containerization configuration.
