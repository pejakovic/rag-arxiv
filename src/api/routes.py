from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from rag.multimodal_rag import MultimodalRAG

router = APIRouter()
rag_system = MultimodalRAG(
    model_path="path/to/model",
    index_path="path/to/index"
)

class MultimodalResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    multimodal_context: Dict[str, Any]

@router.post("/query", response_model=MultimodalResponse)
async def query_endpoint(query: str):
    """Process a query and return multimodal response"""
    try:
        response = rag_system.query(query)
        return MultimodalResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_document")
async def process_document(pdf_path: str):
    """Process and index a new document"""
    try:
        rag_system.process_document(pdf_path)
        return {"status": "success", "message": "Document processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 