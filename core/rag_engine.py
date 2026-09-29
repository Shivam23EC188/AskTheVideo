from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm import llm  # shared, rate-limited client
from core.vector_store import build_vector_store, load_vector_store, get_retriever


_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _make_chain(retriever):
    # Full LCEL RAG pipeline
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | _prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    return _make_chain(get_retriever(vector_store, k=4))


def load_rag_chain():
    vector_store = load_vector_store()
    return _make_chain(get_retriever(vector_store, k=4))  # was missing its arguments


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer