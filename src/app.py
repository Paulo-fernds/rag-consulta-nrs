import streamlit as st
from rag import LocalRAG

st.set_page_config(page_title="RAG Local de PDFs", layout="centered")
st.title("🧠 RAG Local de PDFs (FAISS + SBERT + Ollama)")

model = st.selectbox(
    "Modelo do Ollama",
    [
        "phi3:mini",
        "llama3.2:3b-instruct",
        "gemma2:2b-instruct",
        "qwen2.5:7b-instruct",
        "mistral:7b",
    ],
    index=0,
)

# Controles de desempenho
st.sidebar.header("Ajustes de Desempenho")
k = st.sidebar.slider("k (trechos recuperados)", 1, 8, 3)
use_mmr = st.sidebar.checkbox("Usar MMR (diversidade)", True)
snippet_chars = st.sidebar.slider("Tamanho do trecho no prompt", 200, 1000, 400, 50)
num_predict = st.sidebar.slider("Tokens máximos (num_predict)", 64, 1024, 256, 64)
num_ctx = st.sidebar.slider("Janela de contexto (num_ctx)", 1024, 4096, 2048, 256)
stream = st.sidebar.checkbox("Stream da resposta", True)

needs_new = (
    "rag" not in st.session_state
    or st.session_state.get("model") != model
    or st.session_state.get("k") != k
    or st.session_state.get("use_mmr") != use_mmr
    or st.session_state.get("snippet_chars") != snippet_chars
    or st.session_state.get("num_predict") != num_predict
    or st.session_state.get("num_ctx") != num_ctx
)
if needs_new:
    st.session_state["rag"] = LocalRAG(
        k=k,
        model=model,
        num_ctx=num_ctx,
        num_predict=num_predict,
        use_mmr=use_mmr,
        snippet_chars=snippet_chars,
    )
    st.session_state["model"] = model
    st.session_state["k"] = k
    st.session_state["use_mmr"] = use_mmr
    st.session_state["snippet_chars"] = snippet_chars
    st.session_state["num_predict"] = num_predict
    st.session_state["num_ctx"] = num_ctx

st.caption(f"🧭 Modelo ativo: {st.session_state['model']}")

q = st.text_input("Pergunte algo sobre seus PDFs:", placeholder="Ex.: Qual é a política de reembolso?")
if st.button("Perguntar") and q.strip():
    try:
        st.markdown("### Resposta")
        if stream:
            with st.spinner("Gerando (stream)..."):
                gen, sources = st.session_state["rag"].stream_answer(q.strip())
                st.write_stream(gen)
        else:
            with st.spinner("Consultando..."):
                out = st.session_state["rag"].answer(q.strip())
            st.write(out["answer"])
            sources = out["sources"]

        st.markdown("### Fontes")
        for s in sources:
            st.write(f"- {s}")
    except RuntimeError as e:
        st.error(str(e))
        st.info(
            "Verifique se o Ollama está rodando (ex.: 'ollama serve') e se o modelo "
            "selecionado foi baixado (ex.: 'ollama pull <modelo>')."
        )
