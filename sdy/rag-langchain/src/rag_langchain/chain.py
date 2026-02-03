"""LCEL RAG Chain — retriever | prompt | llm | parser (LangChain)."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_langchain.llm import ClaudeCodeLLM

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question based ONLY on the provided context. "
    "If the context doesn't contain relevant information, say so."
)

_RAG_TEMPLATE = """{system_prompt}

Context:
---
{context}
---

Question: {question}"""


@dataclass
class SourceInfo:
    """Simplified source information from a retrieval result."""

    text: str
    metadata: dict[str, str | int]


@dataclass
class QueryResult:
    """Result of a query operation."""

    answer: str
    sources: list[SourceInfo] = field(default_factory=list)


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a numbered context string."""
    if not docs:
        return "(No relevant documents found.)"
    lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] {doc.page_content}")
        lines.append(f"(source: {source})")
        lines.append("")
    return "\n".join(lines).strip()


def build_rag_chain(
    vector_store: Chroma,
    top_k: int = 5,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
):
    """Build an LCEL RAG chain: retriever | prompt | llm | parser.

    Args:
        vector_store: Chroma vector store to retrieve from.
        top_k: Maximum number of retrieved documents.
        system_prompt: System prompt for the LLM.

    Returns:
        A Runnable chain that takes {"question": str} and returns str.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    prompt = ChatPromptTemplate.from_template(_RAG_TEMPLATE)
    llm = ClaudeCodeLLM()
    parser = StrOutputParser()

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
            "system_prompt": lambda _: system_prompt,
        }
        | prompt
        | llm
        | parser
    )
    return chain


def query_with_chain(
    vector_store: Chroma,
    query: str,
    top_k: int = 5,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> QueryResult:
    """Query using the LCEL chain and return formatted results.

    Args:
        vector_store: Chroma vector store.
        query: Natural-language question.
        top_k: Maximum number of source documents.
        system_prompt: System prompt for the LLM.

    Returns:
        QueryResult with answer and source information.

    Raises:
        ValueError: If query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Retrieve sources separately for metadata
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    source_docs = retriever.invoke(query)

    chain = build_rag_chain(vector_store, top_k=top_k, system_prompt=system_prompt)
    answer = chain.invoke(query)

    sources = [
        SourceInfo(
            text=doc.page_content,
            metadata=dict(doc.metadata),
        )
        for doc in source_docs
    ]

    return QueryResult(answer=answer.strip(), sources=sources)
