from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document

from transformers import pipeline

from galileo import galileo_context
from galileo.config import GalileoPythonConfig

from galileo import GalileoScorers
from galileo.log_streams import enable_metrics
from evaluations.medical_groundedness_metric import medical_groundedness_metric

galileo_context.init(
    project="health_rag",
    log_stream="FLAN-T5-Runs-new"
)

enable_metrics(
    project_name="health_rag",
    log_stream_name="FLAN-T5-Runs-new",
    metrics=[GalileoScorers.context_adherence, 	GalileoScorers.completeness, GalileoScorers.prompt_injection, medical_groundedness_metric]
)

logger = galileo_context.get_logger_instance()
logger.start_session()


# -------------------------
# Input validation
# -------------------------
class QuestionInput(BaseModel):
    question: str = Field(..., min_length=5)


# -------------------------
# Load documents
# -------------------------
def load_documents() -> List[Document]:
    with open("data/healthcare_docs.txt", "r") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    return [Document(page_content=chunk) for chunk in chunks]


# -------------------------
# Vector Store
# -------------------------
def build_vector_store(docs: List[Document]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


# -------------------------
# Load FLAN-T5
# -------------------------
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=128
    )


# -------------------------
# RAG Pipeline (Galileo-instrumented)
# -------------------------
def run_rag(question: str):
    try:
        validated = QuestionInput(question=question)
    except ValidationError as e:
        logger.start_trace(
            name="validation_error",
            input=question
        )
        logger.conclude(output=str(e))
        logger.flush()
        return f"Input error: {e}"

    # Start trace
    logger.start_trace(
        name="healthcare_rag_run",
        input=validated.question
    )

    docs = load_documents()
    vector_db = build_vector_store(docs)

    retrieved_docs = vector_db.similarity_search(
        validated.question,
        k=2
    )

    context = "\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{validated.question}
"""

    llm = load_llm()

    start_time_ns = datetime.now().timestamp() * 1_000_000_000
    response = llm(prompt)
    duration_ns = (
        datetime.now().timestamp() * 1_000_000_000
        - start_time_ns
    )

    answer = response[0]["generated_text"]

    # -------------------------
    # Galileo LLM Span
    # -------------------------
    logger.add_llm_span(
        input=[
            {"role": "user", "content": validated.question},
            {"role": "system", "content": context}
        ],
        output=answer,
        model="google/flan-t5-small",
        num_input_tokens=None,   # HF pipeline doesn't expose tokens
        num_output_tokens=None,
        total_tokens=None,
        duration_ns=duration_ns
    )

    logger.conclude(output=answer)
    logger.flush()

    return answer
