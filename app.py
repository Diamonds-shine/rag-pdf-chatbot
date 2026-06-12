from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
st.title("Placement Preparation Assistant")

pdf = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if pdf:

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

    chunks = splitter.split_text(text)

    st.success(f"PDF split into {len(chunks)} chunks")

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        model = SentenceTransformer('all-MiniLM-L6-v2')

        chunk_embeddings = model.encode(chunks)

        dimension = chunk_embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(np.array(chunk_embeddings))

        question_embedding = model.encode([question])

        distances, indices = index.search(
        np.array(question_embedding),
        k=3
    )

    best_chunk = chunks[indices[0][0]]

    st.subheader("Best Match")
    st.write(best_chunk)