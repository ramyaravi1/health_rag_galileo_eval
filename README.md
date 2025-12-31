# Healthcare RAG with Galileo Evaluation (health_rag_galileo_eval)

A simple **Retrieval-Augmented Generation (RAG)** chatbot for healthcare,
powered by **FLAN-T5** and evaluated using **Galileo**.

This project demonstrates:
- RAG with FAISS + sentence-transformers
- Local open-source LLM inference (FLAN-T5)
- Galileo tracing & evaluation
- Built-in + custom LLM-as-judge metrics
- Batch evaluation over multiple test queries

## Architecture Overview
User Query -> Vector Search (FAISS) -> Context Retrieval -> FLAN-T5 Generation -> Galileo Tracing -> Automated Evaluation (Metrics)

## Setup Instructions
### 1. Clone the repo - 
```bash
git clone https://github.com/your-username/health_rag_galileo_eval.git
cd healthcare-rag-galileo
