# 🎥 AI Meeting Assistant

An AI-powered meeting and video analysis assistant that transforms long videos or meetings into structured, searchable, and interactive knowledge.

The application can transcribe videos, generate summaries, extract important meeting information, and allow users to ask questions about the video using Retrieval-Augmented Generation (RAG).

---

## ✨ Features

- 🎙️ **Automatic Transcription**
  - Transcribe English videos using Whisper.
  - Supports Hinglish processing using Sarvam AI.

- 📝 **AI-Powered Summarization**
  - Generate concise summaries of long meetings and videos.
  - Automatically generate a meaningful title.

- ✅ **Action Item Extraction**
  - Identify tasks and action items discussed during the meeting.

- 📌 **Decision Extraction**
  - Extract important decisions made during the meeting.

- ❓ **Question Extraction**
  - Identify questions raised during the meeting.

- 🔎 **Semantic Search**
  - Search through the transcript using vector embeddings.

- 💬 **Chat With Your Meeting**
  - Ask natural-language questions about the uploaded video.
  - Uses Retrieval-Augmented Generation (RAG) to retrieve relevant transcript sections before generating an answer.

- 🎬 **YouTube & Local Video Support**
  - Analyze videos from YouTube URLs.
  - Process local video files.

- 🖥️ **Streamlit Interface**
  - Simple and interactive web interface.

---

## 🧠 How It Works

The application follows an AI-powered processing pipeline:

```text
                    ┌─────────────────┐
                    │  YouTube /      │
                    │  Local Video    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Audio Extraction│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Transcription  │
                    │ Whisper /       │
                    │ Sarvam AI       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Transcript      │
                    │ Processing      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌───────────┐   ┌───────────┐
        │ Summary  │   │  Actions  │   │ Decisions │
        └──────────┘   └───────────┘   └───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Vector Database │
                    │    ChromaDB     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      RAG        │
                    │ Retrieval + LLM │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Interactive     │
                    │ Q&A / Chat      │
                    └─────────────────┘
