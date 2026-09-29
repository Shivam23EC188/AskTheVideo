from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm import llm   # shared, rate-limited client


_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=300)   # was 8000

_map_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

_combine_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert meeting summarizer. Combine these partial summaries "
         "into one final professional meeting summary in bullet points."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

_title_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Based on the meeting transcript, generate a short professional meeting title "
         "(max 8 words). Only return the title, nothing else."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)


def split_transcript(transcript: str) -> list:
    return _splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    chunks = split_transcript(transcript)

    # Short transcript: one call is enough
    if len(chunks) == 1:
        return _combine_chain.invoke({"text": chunks[0]})

    summaries = [_map_chain.invoke({"text": c}) for c in chunks]

    # If the partial summaries are still too long, reduce them again
    combined = "\n\n".join(summaries)
    while len(combined) > 6000:                                                   # was 8000
        parts = split_transcript(combined)
        combined = "\n\n".join(_map_chain.invoke({"text": p}) for p in parts)

    return _combine_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    return _title_chain.invoke({"text": transcript[:2000]})