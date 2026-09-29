# 🎬 AI Video Assistant

Turn a YouTube link or a video file into a transcript, a summary, a list of action items and decisions, and a chat you can ask questions.

Paste a URL, click **Analyse video**, and get:

- **Transcript** of the spoken audio (English with Whisper, Hinglish with Sarvam AI)
- **Title and summary**, written even for long videos by summarising in chunks
- **Action items**, **key decisions** and **open questions**
- **Chat with the video**: ask questions and get answers based only on what was said (RAG)
- **Exports**: download the report as Markdown and the transcript as text

## Screenshot

<!-- Add a screenshot: put it in docs/screenshot.png and uncomment the line below -->
<!-- ![App screenshot](docs/screenshot.png) -->

## How it works

```text
YouTube URL / uploaded file
        │
        ▼
Audio extraction and chunking      (yt-dlp, FFmpeg, pydub)
        │
        ▼
Transcription                      (Whisper locally / Sarvam AI for Hinglish)
        │
        ├──► Title, summary, action items, decisions, questions
        │        (LangChain + Groq LLM, chunked and rate-limited)
        │
        └──► Vector index          (HuggingFace embeddings + ChromaDB)
                 │
                 ▼
             Chat: retrieve relevant transcript parts, then answer with the LLM
```

## Tech stack

| Area | Tools |
|---|---|
| UI | Streamlit |
| Speech to text | OpenAI Whisper (local), Sarvam AI |
| LLM | Groq via LangChain |
| Embeddings and search | sentence-transformers, ChromaDB |
| Audio | yt-dlp, FFmpeg, pydub |

## Getting started

### Prerequisites

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
- A free [Groq API key](https://console.groq.com/keys)

### Install

```bash
git clone https://github.com/Shivam23EC188/AI-meeting-Assistant.git
cd AI-meeting-Assistant

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
# Optional: force a specific Groq model. If unset, the app picks one your key can use.
# GROQ_MODEL=llama-3.3-70b-versatile
# Only needed for Hinglish transcription:
# SARVAM_API_KEY=your_sarvam_key_here
```

Never commit `.env`. It is already listed in `.gitignore`.

### Run the web app

```bash
streamlit run app.py
```

Open http://localhost:8501, paste a YouTube URL (or upload a file), choose the language, and click **Analyse video**.

### Run from the command line

```bash
python main.py
```

You'll be asked for a URL or file path, and can chat with the video in the terminal afterwards.

## Project structure

```text
├── app.py                  # Streamlit web app
├── main.py                 # Command-line version of the pipeline
├── test.py                 # Quick end-to-end test script
├── requirements.txt
├── core/
│   ├── llm.py              # Shared, rate-limited Groq client
│   ├── transcriber.py      # Whisper / Sarvam transcription
│   ├── summarizer.py       # Chunked summary and title
│   ├── extractor.py        # Action items, decisions, open questions
│   ├── vector_store.py     # ChromaDB + embeddings
│   └── rag_engine.py       # Retrieval-augmented chat
└── utils/
    └── audio_processor.py  # Download, extract and chunk audio
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `401 Invalid API Key` | Check `GROQ_API_KEY` in `.env` (no quotes or spaces around `=`), then fully restart Streamlit. |
| `429 Rate limit exceeded` | Free-tier limits are tight. Lower `requests_per_second` in `core/llm.py`, or try a shorter video. |
| `model_not_found` | Groq changes its model list often. Remove `GROQ_MODEL` from `.env` to let the app auto-pick, or set one your key can use. |
| Transcription seems stuck | Whisper on CPU is slow and prints nothing while working. Use a smaller model (`base` or `tiny`) or a shorter clip. |
| `FP16 is not supported on CPU` | Harmless warning. It appears when no GPU is available. |
| Changes not taking effect | Fully stop and restart Streamlit so modules are reloaded. |

## Limitations

- Long videos take a while on CPU and may be throttled on free API tiers.
- Answers in chat are limited to what's in the transcript, so transcription errors carry through.
- Speaker names are not identified.

## Contributing

Issues and pull requests are welcome. Please don't include API keys, downloaded media or vector databases in commits.