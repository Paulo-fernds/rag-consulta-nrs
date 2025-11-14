# 📚 RAG Local para Consulta de Normas Regulamentadoras  
### (FAISS + SBERT + Ollama + Streamlit)

Este projeto implementa um sistema de **Recuperação Aumentada por Geração (RAG)** totalmente **local**, permitindo consultas inteligentes a Normas Regulamentadoras (NRs) a partir de arquivos PDF.  

A solução combina:

- **FAISS** — busca vetorial eficiente  
- **SBERT (Sentence-BERT)** — geração de embeddings semânticos  
- **Ollama** — execução local de modelos LLM (phi3:mini, Llama3, Mistral, etc.)  
- **Streamlit** — interface gráfica simples e interativa  

O sistema responde perguntas em linguagem natural utilizando exclusivamente os trechos recuperados dos PDFs.

---

## ✨ Visão Geral

- **Objetivo:** facilitar consultas técnicas às NRs sem depender da internet.  
- **Entrada:** arquivos PDF das NRs.  
- **Saída:** respostas fundamentadas, junto com as fontes (nome do PDF + página).  
- **Privacidade:** todo o processamento é local.  
- **Flexibilidade:** qualquer NR pode ser adicionada à base.

---

## 🖥️ Interface (Streamlit)

A interface inclui controles para ajustar desempenho:

- `k` (trechos recuperados)  
- Diversidade MMR  
- Tamanho do trecho no prompt  
- Tokens máximos gerados  
- Janela de contexto  
- Stream de resposta  

### 📸 Exemplos

![Screenshot 3](https://github.com/user-attachments/assets/3df55501-8fd5-4b8b-b953-0b447c7fbdcf)
![Screenshot 1](https://github.com/user-attachments/assets/fe86a1d0-2665-4f4e-bd82-3cbeec620e83)
![Screenshot 2](https://github.com/user-attachments/assets/c58461c2-30ae-49c3-9784-d3036a645f52)

---

## 🧠 Como Funciona

### 1. Ingestão dos PDFs
- Leitura e extração dos textos.  
- Divisão em *chunks* com metadados (página, arquivo).

### 2. Geração de Embeddings
- Cada trecho é convertido em vetor usando **SBERT**.

### 3. Indexação com FAISS
- Os vetores são armazenados em um índice FAISS para busca rápida.

### 4. Consulta (RAG)
- A pergunta → é vetorizarada  
- FAISS → retorna os trechos mais relevantes  
- Ollama → gera a resposta usando somente esses trechos  
- Streamlit → exibe resposta + fontes  

---

## 📊 Exemplos de Respostas

Pergunta: *“O que é a NR-35?”*  
Resposta gerada com referência: *(nr-35-atualizada-2025.pdf, p.18)*

---

## 🛠️ Tecnologias Utilizadas

- **Python**  
- **FAISS**  
- **Sentence-BERT**  
- **Ollama**  
- **Streamlit**  

---

