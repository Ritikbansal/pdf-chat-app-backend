"""
FastAPI Backend for PDF Processing with AI21 Integration
Install: pip install fastapi uvicorn python-multipart pypdf2 pinecone-client ai21 python-dotenv
"""

from ai21.models.chat import ChatMessage
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import io
import os
import uuid
from datetime import datetime
import time
from dotenv import load_dotenv
from ai21 import AI21Client
import requests
from pinecone import Pinecone, ServerlessSpec

JINA_API_TOKEN = os.getenv("JINA_API_TOKEN")
JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
AI21_API_KEY = os.getenv("AI21_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ai21_client = AI21Client(api_key=AI21_API_KEY)

load_dotenv()

app = FastAPI(title="PDF RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-nextjs-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
INDEX_NAME = "pdf-chat-index"


pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(10)

index = pc.Index(INDEX_NAME)


class PDFUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    document_id: str
    conversation_history: Optional[List[Message]] = []


class Citation(BaseModel):
    page: int
    text: str


class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]


def extract_text_from_pdf(pdf_file: bytes) -> List[dict]:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
    pages_data = []

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            pages_data.append({"page": page_num, "text": text.strip()})

    return pages_data


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by words"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def get_jina_embeddings(texts: list, task: str):
    """
    Generate embeddings using Jina AI.
    Each text in 'texts' returns a vector embedding.
    """
    if not JINA_API_TOKEN:
        raise HTTPException(status_code=500, detail="Jina API token not set")

    payload = {"model": "jina-embeddings-v3", "task": task, "input": texts}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_TOKEN}",
    }

    response = requests.post(JINA_EMBEDDINGS_URL, json=payload, headers=headers)
    response_data = response.json()

    if "data" not in response_data:
        raise HTTPException(
            status_code=500, detail="Jina API returned invalid response"
        )

    embeddings = [item["embedding"] for item in response_data["data"]]
    return embeddings


@app.post("/api/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process PDF file
    1. Extract text from PDF
    2. Chunk text into smaller pieces
    3. Generate embeddings using AI21
    4. Store in Pinecone vector database
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Read PDF file
        pdf_content = await file.read()

        # Generate unique document ID
        document_id = str(uuid.uuid4())

        # Extract text from PDF
        pages_data = extract_text_from_pdf(pdf_content)

        if not pages_data:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # Process each page
        all_chunks = []
        metadata_list = []

        for page_data in pages_data:
            page_num = page_data["page"]
            page_text = page_data["text"]

            # Chunk the page text
            chunks = chunk_text(page_text, chunk_size=500, overlap=50)
            print("chunks")
            print(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata_list.append(
                    {
                        "document_id": document_id,
                        "filename": file.filename,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "text": chunk[:500],  # Store first 500 chars in metadata
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        print("all_chunks", all_chunks)
        print("metadata_list", metadata_list)
        # Generate embeddings using AI21
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        try:
            embeddings = get_jina_embeddings(all_chunks, "retrieval.query")
        except Exception as e:
            print("Error generating embeddings:", e)
            raise HTTPException(
                status_code=500, detail=f"Error generating embeddings: {e}"
            )

        # Prepare vectors for Pinecone
        vectors = []
        for idx, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            vector_id = f"{document_id}_chunk_{idx}"
            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})
        print("vectors", vectors)
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch)
            time.sleep(0.5)  # Rate limiting

        print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")

        return PDFUploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_pages=len(pages_data),
            total_chunks=len(all_chunks),
            message="PDF uploaded and processed successfully",
        )

    except Exception as e:
        print(e)
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Generate query embedding
        query_embedding = get_jina_embeddings([request.message], "retrieval.query")[0]
        print("query_embedding", query_embedding)
        # 2. Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            filter={"document_id": request.document_id},
            include_metadata=True,
        )
        print("search_results", search_results)
        matches = getattr(search_results, "matches", [])
        if not matches:
            return ChatResponse(
                response="I couldn't find relevant information in the PDF to answer your question.",
                citations=[],
            )
        print(matches)
        # Filter irrelevant results by score threshold
        RELEVANCE_THRESHOLD = 0.05
        filtered_matches = [m for m in matches if m.score >= RELEVANCE_THRESHOLD]

        if not filtered_matches:
            return ChatResponse(
                response="I couldn't find relevant information in the PDF to answer your question.",
                citations=[],
            )

        # 3. Build context
        context_parts, citations_data = [], []
        for match in filtered_matches:
            meta = match.metadata
            page = meta.get("page", 0)
            text = meta.get("text", "")
            context_parts.append(f"[Page {page}] {text}")
            citations_data.append({"page": page, "text": text[:200]})

        context = "\n\n".join(context_parts)
        print("context", context)

        # 4. Build conversation history
        history_messages = [
            Message(**msg) if isinstance(msg, dict) else msg
            for msg in request.conversation_history[-3:]
        ]

        history_text = ""
        for msg in history_messages:
            history_text += f"{msg.role.capitalize()}: {msg.content.strip()}\n"

        # 5. Build system prompt
        prompt = f"""You are a helpful assistant that answers questions about PDF documents.

Previous conversation:
{history_text}

Context from the PDF:
{context}

User question: {request.message}

Provide a clear, concise answer using only the given context. If the context doesn't contain the answer, say so explicitly."""
        # 6. Generate AI21 response
        response = ai21_client.chat.completions.create(
            model="jamba-large",
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a helpful PDF question-answering assistant.",
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0.7,
            max_output_tokens=500,
        )

        # Access the assistant's message correctly
        answer = response.choices[0].message.content

        print("answer", answer)
        # 7. Unique citations
        unique_pages = {}
        for c in citations_data:
            if c["page"] not in unique_pages:
                unique_pages[c["page"]] = c

        citations = [
            Citation(page=p, text=c["text"]) for p, c in sorted(unique_pages.items())
        ]
        print("citations", citations)
        print("Answer type:", type(answer))
        print("Citations:", citations)
        for c in citations:
            print("Citation type:", type(c))

        citations_cleaned = [
            c if isinstance(c, Citation) else Citation(**c) for c in citations
        ]
        return ChatResponse(response=answer, citations=citations_cleaned)

    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    print("application started")
    uvicorn.run(app, host="0.0.0.0", port=8000)
