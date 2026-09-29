# Action items, decisions, open questions

from functools import lru_cache

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import llm  # shared, rate-limited client

_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=300)   # was 12000

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert meeting analyst. From the meeting transcript, extract:\n"
        "1. \"action_items\": tasks someone must do. Each item is an object with keys "
        "\"task\", \"owner\" (who is responsible, or \"Not specified\") and "
        "\"deadline\" (or \"Not specified\").\n"
        "2. \"key_decisions\": decisions that were made. Each item is a short string.\n"
        "3. \"open_questions\": unresolved questions or topics needing follow-up. "
        "Each item is a short string.\n\n"
        "Return ONLY valid JSON with exactly these three keys. "
        "Use an empty list for anything not found. Do not add commentary.",
    ),
    ("human", "{text}"),
])

_chain = _prompt | llm | JsonOutputParser()

_KEYS = ("action_items", "key_decisions", "open_questions")


def _norm(x) -> str:
    return " ".join(str(x).lower().split())


def _item_key(item) -> str:
    if isinstance(item, dict):
        return _norm(item.get("task", ""))
    return _norm(item)


def _extract_chunk(text: str) -> dict:
    try:
        data = _chain.invoke({"text": text})
    except OutputParserException:
        print("Warning: could not parse extractor output for one chunk; skipping it.")
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=4)
def _extract_all(transcript: str) -> dict:
    result = {k: [] for k in _KEYS}

    for chunk in _splitter.split_text(transcript):
        data = _extract_chunk(chunk)
        for k in _KEYS:
            items = data.get(k) or []
            if isinstance(items, list):
                result[k].extend(items)

    # Remove duplicates that appear in overlapping chunks
    for k in _KEYS:
        seen, unique = set(), []
        for item in result[k]:
            key = _item_key(item)
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        result[k] = unique

    return result


def _format_action(i: int, item) -> str:
    if isinstance(item, dict):
        task = str(item.get("task", "")).strip()
        owner = item.get("owner") or "Not specified"
        deadline = item.get("deadline") or "Not specified"
        return f"{i}. {task} (Owner: {owner}; Deadline: {deadline})"
    return f"{i}. {item}"


def _numbered(items: list, empty_msg: str, fmt=None) -> str:
    if not items:
        return empty_msg
    fmt = fmt or (lambda i, item: f"{i}. {item}")
    return "\n".join(fmt(i, item) for i, item in enumerate(items, 1))


# ── Public API (same names as before) ────────────────────────────────────────

def extract_all(transcript: str) -> dict:
    return _extract_all(transcript)


def extract_action_items(transcript: str) -> str:
    items = _extract_all(transcript)["action_items"]
    return _numbered(items, "No action items found.", _format_action)


def extract_key_decisions(transcript: str) -> str:
    items = _extract_all(transcript)["key_decisions"]
    return _numbered(items, "No key decisions found.")


def extract_questions(transcript: str) -> str:
    items = _extract_all(transcript)["open_questions"]
    return _numbered(items, "No open questions found.")