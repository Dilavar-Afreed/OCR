import io
import torch
import pymupdf  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

# -------------------------
# Configuration
# -------------------------
local_model_path = "./model"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
dtype = torch.float32 if device == "mps" else torch.bfloat16

app = FastAPI(title="PDF OCR API")

# Global model variables
model = None
processor = None


# -------------------------
# Load model at startup
# -------------------------
@app.on_event("startup")
def load_model():
    global model, processor

    print("Loading OCR model...")

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    processor = LightOnOcrProcessor.from_pretrained(
        local_model_path,
        trust_remote_code=True,
    )

    model.eval()
    print("Model loaded successfully.")


# -------------------------
# OCR Logic
# -------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
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
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            page_text = processor.decode(
                generated_ids, skip_special_tokens=True
            )

        full_document_text += f"\n--- PAGE {i+1} ---\n{page_text}"

    doc.close()
    return full_document_text


# -------------------------
# API Endpoint
# -------------------------
@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    pdf_bytes = await file.read()

    try:
        extracted_text = extract_pdf_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(
        content={
            "filename": file.filename,
            "extracted_text": extracted_text,
        }
    )
