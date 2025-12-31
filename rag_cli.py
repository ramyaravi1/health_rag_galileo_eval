from pydantic import BaseModel, Field, ValidationError
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document

from transformers import pipeline


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

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    return [Document(page_content=chunk) for chunk in chunks]


# -------------------------
# Build Vector Store
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
# RAG Function
# -------------------------
def run_rag(question: str):
    try:
        validated = QuestionInput(question=question)
    except ValidationError as e:
        return f"Input error: {e}"

    docs = load_documents()
    vector_db = build_vector_store(docs)

    retrieved_docs = vector_db.similarity_search(validated.question, k=2)
    context = "\n".join(doc.page_content for doc in retrieved_docs)

    llm = load_llm()

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{validated.question}
"""

    try:
        response = llm(prompt)
        return response[0]["generated_text"]
    except Exception as e:
        return f"Unexpected error: {e}"
