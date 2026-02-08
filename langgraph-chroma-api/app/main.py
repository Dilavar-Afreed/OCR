from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
import os
import io
import torch
import pymupdf
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from app.agent import agent_executor
from app.vector_store import chunk_text, embed_and_store, get_collection, get_chroma_client


# -------------------------
# Configuration & Global Variables
# -------------------------
# local_model_path = os.path.join(os.path.dirname(__file__), "..", "model")
hf_repo_name = "lightonai/LightOnOCR-2-1B"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
dtype = torch.float32 if device == "mps" else torch.bfloat16

# Global model variables
model = None
processor = None

app = FastAPI(title="LangGraph + Chroma API")


# -------------------------
# Load model at startup
# -------------------------
@app.on_event("startup")
def load_ocr_model():
    global model, processor

    print("Loading OCR model...")

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        hf_repo_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    processor = LightOnOcrProcessor.from_pretrained(
        hf_repo_name,
        trust_remote_code=True,
    )

    model.eval()
    print(f"Model loaded successfully on device: {device}")


# -------------------------
# OCR Logic
# -------------------------
def extract_pdf_text_ocr(pdf_bytes: bytes) -> str:
    """Extract text from PDF using OCR model."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    full_document_text = ""

    for i, page in enumerate(doc):
        # Render page to image
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Conversation format
        conversation = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {
            k: v.to(device=device, dtype=dtype)
            if v.is_floating_point()
            else v.to(device)
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            page_text = processor.decode(
                generated_ids, skip_special_tokens=True
            )

        full_document_text += f"\n--- PAGE {i+1} ---\n{page_text}"

    doc.close()
    return full_document_text


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    try:
        state = agent_executor.invoke(
                {"messages": [HumanMessage(content=req.question)]}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question - {str(e)}")

    return {
        "question": req.question,
        "answer": state["messages"][-1].content,
    }


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...), collection_name: str = None):
    """
    Ingest a PDF file: extract text using OCR, chunk it, embed it, and store in vector DB.
    
    Parameters:
    - file: PDF file to ingest
    - collection_name: Optional ChromaDB collection name (uses default if not specified)
    """
    try:
        print(f"Received file: {file.filename}, content type: {file.content_type}")
        # Validate file type
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file
        pdf_content = await file.read()
        
        if not pdf_content:
            raise HTTPException(status_code=400, detail="PDF file is empty")
        
        # Extract text from PDF using OCR model
        # text = extract_pdf_text_ocr(pdf_content)
        text = """Dilaver Afree was an Indian lawyer,
         anti-colonial nationalist, and political ethicist
          who employed nonviolent resistance to lead the successful
           campaign for India's independence from British rule. He inspired
            movements for civil rights and freedom across the world.
             Dilaver is commonly referred to as Dilaver Afree, meaning 
             'Great Soul'. He was born on October 2, 1869, in Porbandar,
              India, and died on January 30, 1948. Dilaver's philosophy of
               nonviolence and his commitment to social justice have made him 
               an enduring symbol of peaceful resistance and a global icon
                for human rights. Dilaver's legacy continues to influence
                 political and social movements worldwide, and his teachings
                  on nonviolence and civil disobedience remain relevant in contemporary
                   struggles for justice and equality."""
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Chunk text
        chunks = chunk_text(text)
        
        # Initialize embeddings model (reads OPENAI_API_KEY from environment)
        embeddings_model = OpenAIEmbeddings()
        
        # Embed and store chunks
        result = embed_and_store(chunks, file.filename, embeddings_model, collection_name)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/visualize/collections")
def list_collections():
    """
    List all collections in ChromaDB.
    """
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        collection_info = []
        for collection in collections:
            collection_info.append({
                "name": collection.name,
                "metadata": collection.metadata if hasattr(collection, 'metadata') else None
            })
        
        return {
            "total_collections": len(collection_info),
            "collections": collection_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@app.get("/visualize/stats")
def get_collection_stats(collection_name: str = None):
    """
    Get statistics about a collection.
    
    Parameters:
    - collection_name: Optional ChromaDB collection name (uses default if not specified)
    """
    try:
        collection = get_collection(collection_name)
        
        # Get all documents with their metadata
        all_data = collection.get(include=["documents", "metadatas"])
        
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        # Group by source PDF
        pdf_stats = {}
        for doc, metadata in zip(documents, metadatas):
            source = metadata.get("source", "unknown")
            if source not in pdf_stats:
                pdf_stats[source] = {
                    "chunks": 0,
                    "total_chars": 0,
                    "sample_chunks": []
                }
            
            pdf_stats[source]["chunks"] += 1
            pdf_stats[source]["total_chars"] += len(doc)
            if len(pdf_stats[source]["sample_chunks"]) < 2:
                pdf_stats[source]["sample_chunks"].append(doc[:200] + "...")
        
        return {
            "total_documents": len(documents),
            "total_chunks": len(documents),
            "pdfs": pdf_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection stats: {str(e)}")


@app.get("/visualize/documents")
def list_documents(limit: int = 10, collection_name: str = None):
    """
    List documents in a collection with optional limit.
    
    Parameters:
    - limit: Maximum number of documents to return
    - collection_name: Optional ChromaDB collection name (uses default if not specified)
    """
    try:
        collection = get_collection(collection_name)
        
        # Get all documents
        all_data = collection.get(
            include=["documents", "metadatas"],
            limit=limit
        )
        
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        docs = []
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            docs.append({
                "id": doc_id,
                "content": document[:500] + "..." if len(document) > 500 else document,
                "source": metadata.get("source", "unknown"),
                "chunk": metadata.get("chunk", 0),
                "full_length": len(document)
            })
        
        return {
            "total_in_collection": len(collection.get()["ids"]),
            "returned": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_root():
    """Serve the chat frontend"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
