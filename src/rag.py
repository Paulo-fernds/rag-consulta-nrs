# src/rag.py
import os
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from embeddings import SBERTEmbeddings

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index" / "faiss"
# Prefer IPv4 loopback to avoid potential IPv6 localhost issues on Windows
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

SYSTEM_PROMPT = """Você é um assistente administrativo técnico. 
Responda de forma objetiva, direta e formal.
Responda apenas com base nos trechos fornecidos. 
Não use expressões como "parece", "pode estar relacionado" ou "indica".
Se faltar informação, diga: "Não há informação suficiente nos documentos fornecidos."
Inclua um parágrafo único, seguido de lista de fontes.
"""

USER_PROMPT = """Pergunta: {question}

Contexto:
{context}

Responda objetivamente e cite as fontes.
"""

def format_context(docs: List[Document], snippet_chars: int = 600):
    lines, used = [], []
    for d in docs:
        src = Path(d.metadata.get("source", "documento")).name
        raw_page = d.metadata.get("page")
        page_disp = "?"
        if isinstance(raw_page, int):
            page_disp = raw_page + 1
        elif isinstance(raw_page, str) and raw_page.isdigit():
            page_disp = int(raw_page) + 1

        snippet = d.page_content.replace("\n", " ")
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + "…"

        lines.append(f"- ({src}, p.{page_disp}) {snippet}")
        used.append(f"{src} (p.{page_disp})")
    return "\n".join(lines), list(dict.fromkeys(used))

class LocalRAG:
    def __init__(
        self,
        k: int = 3,
        model: str = "phi3:mini",
        num_ctx: int = 2048,
        num_predict: int = 256,
        use_mmr: bool = True,
        fetch_k: int | None = None,
        snippet_chars: int = 400,
    ):
        self.emb = SBERTEmbeddings()
        self.db = FAISS.load_local(str(INDEX_DIR), self.emb, allow_dangerous_deserialization=True)
        self.k = k
        self.use_mmr = use_mmr
        self.fetch_k = fetch_k
        self.snippet_chars = snippet_chars
        self.llm = Ollama(
            model=model,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )

    def _retrieve(self, question: str):
        if self.use_mmr:
            fetch_k = self.fetch_k or max(self.k * 4, 20)
            docs = self.db.max_marginal_relevance_search(question, k=self.k, fetch_k=fetch_k)
        else:
            docs = self.db.similarity_search(question, k=self.k)
        return docs

    def answer(self, question: str):
        docs = self._retrieve(question)
        context, sources = format_context(docs, snippet_chars=self.snippet_chars)
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(question=question, context=context)}"
        try:
            resp = self.llm.invoke(prompt)
        except Exception as e:
            raise RuntimeError(
                "Falha ao conectar no Ollama. Verifique se o serviço está ativo em "
                f"{OLLAMA_BASE_URL} e se o modelo está disponível. "
                "Dicas: execute 'ollama serve' e 'ollama pull qwen2.5:7b-instruct'.\n" 
                f"Detalhes: {e}"
            ) from e
        return {"answer": resp, "sources": sources}

    def stream_answer(self, question: str):
        docs = self._retrieve(question)
        context, sources = format_context(docs, snippet_chars=self.snippet_chars)
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(question=question, context=context)}"
        try:
            gen = self.llm.stream(prompt)
        except Exception as e:
            raise RuntimeError(
                "Falha ao conectar no Ollama. Verifique se o serviço está ativo em "
                f"{OLLAMA_BASE_URL} e se o modelo está disponível. "
                "Dicas: execute 'ollama serve' e 'ollama pull <modelo>'.\n"
                f"Detalhes: {e}"
            ) from e
        return gen, sources
