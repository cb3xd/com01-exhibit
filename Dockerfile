FROM python:3.11-slim

# Ensure terminal colors and Unicode are supported for the TUI
ENV TERM=xterm-256color
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]