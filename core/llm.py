import os
import requests
from dotenv import load_dotenv, find_dotenv

# Load .env here so this file never depends on import order.
load_dotenv(find_dotenv(usecwd=True), override=True)

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

_api_key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Put it in a .env file in the project root "
        "as: GROQ_API_KEY=your_key_here"
    )

print(f"[llm] Groq key loaded: {_api_key[:4]}...{_api_key[-4:]} (len {len(_api_key)})")

# Tried in this order; the first one your key can access is used.
_PREFERRED = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
# Models that are not general chat models
_SKIP = ("whisper", "guard", "tts", "orpheus", "compound", "safeguard", "distil")


def _pick_model() -> str:
    forced = (os.getenv("GROQ_MODEL") or "").strip()
    if forced:
        return forced

    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {_api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
    except Exception as e:
        print(f"[llm] Could not list Groq models ({e}); defaulting to {_PREFERRED[0]}")
        return _PREFERRED[0]

    for name in _PREFERRED:
        if name in ids:
            return name

    for name in ids:
        if not any(s in name for s in _SKIP):
            return name

    raise RuntimeError(f"No usable chat model found for this key. Models seen: {ids}")


MODEL = _pick_model()
print(f"[llm] Using Groq model: {MODEL}")

_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,   # about 1 call every 4s
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

llm = ChatGroq(
    model=MODEL,
    api_key=_api_key,
    temperature=0.2,
    rate_limiter=_limiter,
    max_retries=6,
)