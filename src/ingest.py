# src/ingest.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddings import SBERTEmbeddings

PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"
INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index" / "faiss"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("📥 Carregando PDFs...")
    docs = PyPDFDirectoryLoader(str(PDF_DIR)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    print("🧮 Gerando embeddings...")
    embeddings = SBERTEmbeddings(name=EMB_MODEL_NAME)

    print("🏗️  Construindo índice FAISS...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(INDEX_DIR))

    print(f"✅ Índice salvo em {INDEX_DIR}")


if __name__ == "__main__":
    main()

