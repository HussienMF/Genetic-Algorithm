
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
import shutil
from typing import List
from ..config import UPLOAD_DIR, logger

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file and get basic metadata."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    temp_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")

        with temp_path.open("wb") as f:
            f.write(content)

        # Validate CSV
        df = pd.read_csv(temp_path)
        if df.empty:
            raise ValueError("CSV file is empty")

        # Return metadata
        return JSONResponse({
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "sample_data": df.head().to_dict(orient='records')
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@router.get("/datasets")
async def list_datasets():
    """List all uploaded datasets."""
    datasets = []
    for file in UPLOAD_DIR.glob("*.csv"):
        df = pd.read_csv(file)
        datasets.append({
            "name": file.stem,
            "size": file.stat().st_size,
            "rows": len(df),
            "columns": len(df.columns)
        })
    return {"datasets": datasets}

@router.delete("/dataset/{filename}")
async def delete_dataset(filename: str):
    """Delete an uploaded dataset."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        file_path.unlink()
        return {"message": f"Dataset '{filename}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")