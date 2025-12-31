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

### Project Structure
```bash
health_rag_galileo_eval/
│
├── app.py (CLI Chat)
├── rag_cli.py (Production-Safe + Galileo Instrumented)
├── requirements.txt
├── main.ipynb
│
├── data/
│   └── healthcare_docs.txt (Healthcare Documents)
│
├── evaluations/
│   ├── test_cases.py (10-Example Evaluation Dataset (Synthetic))
│   └── medical_groundedness_metric.py (Custom LLM-as-Judge Metric)
│
└── run_evaluations.py
```
## Setup Instructions (implemented the entire Healthcare RAG + FLAN-T5 + Galileo evaluation project on Google Colab)
### 1. Clone the repo - 
```bash
git clone https://github.com/your-username/health_rag_galileo_eval.git
cd health_rag_galileo_eval
```
### 2. Sign up on Galileo - https://app.galileo.ai/sign-in and create an API key (Settings → API Keys → Create Key)
### 3. Upload the folder on your Google drive and open it on Colab
### 3. Open the jupyter notebook file - main.ipynb
### 4. Mount your Google Drive to the Colab environment
```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/health_rag
```
### 5. Set environment variables
```python
import os
os.environ["GALILEO_API_KEY"] = "your_api_key_here"
os.environ["GALILEO_PROJECT"] = "health-rag"
```
### 6. Install dependencies
```python
!pip install -r requirements.txt
```
### 7. Run RAG application with Galileo setup and enable Galileo built-in metrics
```python
!python app.py
```
### 8. Run Evaluations (10 Test Cases)
```python
!python run_evaluations.py
```
### 9. View results in Galileo
➡️ Project: health-rag
➡️ Log Stream: FLAN-T5-Runs-new

Here you’ll see:
-> Traces
-> Built-in metrics verdicts
-> Custom judge verdicts


