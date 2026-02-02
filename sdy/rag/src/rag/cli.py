"""CLI entrypoint for RAG pipeline."""

import typer

from rag.pipeline import AskResult, IndexResult, ask_question, index_documents

app = typer.Typer(help="RAG Pipeline CLI")


@app.command()
def hello() -> None:
    """Smoke test — 환경 설정이 정상인지 확인."""
    typer.echo("RAG pipeline is ready.")


@app.command()
def index(
    data_dir: str = typer.Option("./data", help="Directory containing .txt files"),
    db_path: str = typer.Option("./chroma_db", help="ChromaDB persist directory"),
    collection_name: str = typer.Option("rag", help="Collection name"),
    chunk_size: int = typer.Option(500, help="Characters per chunk"),
    chunk_overlap: int = typer.Option(50, help="Overlap between chunks"),
) -> None:
    """Index documents: load -> split -> embed -> store."""
    typer.echo(f"Indexing documents from {data_dir} ...")
    try:
        result: IndexResult = index_documents(
            data_dir=data_dir,
            db_path=db_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"  Loaded {result.total_documents} documents")
    typer.echo(
        f"  Split into {result.total_chunks} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    typer.echo(f"  Stored in {result.db_path} (collection: {result.collection_name})")
    typer.echo(f"Done! {result.total_chunks} chunks indexed.")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask"),
    db_path: str = typer.Option("./chroma_db", help="ChromaDB persist directory"),
    collection_name: str = typer.Option("rag", help="Collection name"),
    top_k: int = typer.Option(5, help="Maximum number of results"),
    score_threshold: float = typer.Option(0.0, help="Minimum similarity score"),
) -> None:
    """Ask a single question against indexed documents."""
    typer.echo("Searching for relevant documents...")
    try:
        result: AskResult = ask_question(
            query=query,
            db_path=db_path,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"  Found {len(result.sources)} relevant chunks")
    typer.echo("")
    typer.echo("Answer:")
    typer.echo(f"  {result.answer}")
    typer.echo("")

    if result.sources:
        typer.echo("Sources:")
        for i, src in enumerate(result.sources, start=1):
            meta = src.document.metadata
            filename = meta.get("filename", "unknown")
            chunk_idx = meta.get("chunk_index", "?")
            score = f"{src.score:.2f}"
            typer.echo(f"  [{i}] {filename} (chunk {chunk_idx}, score: {score})")


@app.command()
def chat(
    db_path: str = typer.Option("./chroma_db", help="ChromaDB persist directory"),
    collection_name: str = typer.Option("rag", help="Collection name"),
    top_k: int = typer.Option(5, help="Maximum number of results"),
    score_threshold: float = typer.Option(0.0, help="Minimum similarity score"),
) -> None:
    """Interactive REPL — ask multiple questions."""
    typer.echo("RAG Chat (type 'quit' or 'exit' to stop)")
    typer.echo(f"DB: {db_path}")
    typer.echo("")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            typer.echo("Goodbye!")
            break

        if not query:
            continue

        typer.echo("Searching...")
        try:
            result: AskResult = ask_question(
                query=query,
                db_path=db_path,
                collection_name=collection_name,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except (ValueError, RuntimeError) as e:
            typer.echo(f"Error: {e}")
            continue

        typer.echo(f"  Found {len(result.sources)} relevant chunks")
        typer.echo("")
        typer.echo("Answer:")
        typer.echo(f"  {result.answer}")
        typer.echo("")

        if result.sources:
            typer.echo("Sources:")
            for i, src in enumerate(result.sources, start=1):
                meta = src.document.metadata
                filename = meta.get("filename", "unknown")
                chunk_idx = meta.get("chunk_index", "?")
                typer.echo(
                    f"  [{i}] {filename} (chunk {chunk_idx}, score: {src.score:.2f})"
                )
        typer.echo("")


if __name__ == "__main__":
    app()
