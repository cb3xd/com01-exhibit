# TruthGuard AI

TruthGuard AI is a terminal-based tool (TUI) designed to analyze news articles, extract key claims and entities, and perform automated fact-checking using advanced LLMs and live news data.

## Features

- **Terminal User Interface (TUI):** Interactive and responsive terminal UI built with [Textual](https://textual.textualize.io/).
- **Multi-LLM Synthesis Pipeline:** Employs a sophisticated chain for deep analysis:
    - **Llama 3:** For initial claim extraction and search prompt generation.
    - **GPT-OSS-120b:** For live web search and deep reasoning.
    - **Qwen-QwQ:** For final authoritative verdict synthesis.
- **Credibility Assessment:** Automatically assigns trust scores and categorizes information.
- **Rich Logging:** Real-time, formatted output of the analysis process with high-fidelity terminal coloring.

## Prerequisites

- Python 3.11 or higher
- [Groq API Key](https://console.groq.com/)

## Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cb3xd/com01-exhibit
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

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

Run the application using Python:

```bash
python app.py
```

### Navigating the TUI
- **URL/Headline/Paragraph:** Enter the details of the article you want to validate.
- **Submit:** Click the "Run Pipeline" button to start the analysis.
- **Logs:** View the step-by-step progress and final results in the tri-panel log area.

## Running with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t truthguard-ai .
   ```

2. **Run the container:**
   ```bash
   docker run -it --env-file .env truthguard-ai
   ```
   *Note: The `-it` flag is mandatory for the TUI to render correctly and handle keyboard input.*

## Project Structure

- `app.py`: Main application logic and TUI definition.
- `requirements.txt`: Python dependencies.
- `.env`: Environment variables (not tracked by git).
- `.dockerignore`: Files excluded from Docker builds.
- `Dockerfile`: Containerization configuration.
