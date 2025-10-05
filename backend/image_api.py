import os
import tempfile
import shutil
import zipfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from dotenv import load_dotenv

from utils.image_utils import anonymize_image, generate_image_description, generate_key_findings

load_dotenv()


app = FastAPI(title="Optive Image Anonymizer", version="1.0.0")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/anonymize")
async def anonymize(
    image: UploadFile = File(...),
    instructions: Optional[str] = Form(None),
) -> FileResponse:
    """
    Anonymize an image and generate description and key findings.
    Returns a zip file containing:
    - Anonymized image
    - Image description text file
    - Key findings text file
    """
    if image.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")

    # Read instructions from file if not provided
    if not instructions:
        instructions_path = os.path.join(os.getcwd(), "instructions.txt")
        if os.path.exists(instructions_path):
            with open(instructions_path, "r") as f:
                instructions = f.read()
        else:
            instructions = ""

    # Get API key for Gemini
    api_key = os.getenv("GEMINI_API_KEY")

    # Create a persistent temp dir; clean it after response is sent
    tmpdir = tempfile.mkdtemp(prefix="optive_")
    input_suffix = ".png" if (image.filename or "").lower().endswith(".png") else ".jpg"
    input_path = os.path.join(tmpdir, f"input{input_suffix}")
    output_path = os.path.join(tmpdir, f"output{input_suffix}")
    description_path = os.path.join(tmpdir, "description.txt")
    findings_path = os.path.join(tmpdir, "key_findings.txt")
    zip_path = os.path.join(tmpdir, "results.zip")

    # Save upload to disk
    content = await image.read()
    with open(input_path, "wb") as f:
        f.write(content)

    try:
        # 1. Run anonymization
        anonymize_image(input_path, output_path, instructions or "")

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to generate anonymized image")

        # 2. Generate image description
        description = generate_image_description(input_path, api_key)
        with open(description_path, "w", encoding="utf-8") as f:
            f.write("IMAGE DESCRIPTION\n")
            f.write("=" * 50 + "\n\n")
            f.write(description)

        # 3. Generate key findings
        findings = generate_key_findings(input_path, api_key)
        with open(findings_path, "w", encoding="utf-8") as f:
            f.write("KEY FINDINGS\n")
            f.write("=" * 50 + "\n\n")
            f.write(findings)

        # 4. Create zip file with all outputs
        filename_base = os.path.splitext(os.path.basename(image.filename or "image"))[0]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_path, f"{filename_base}_anonymized{input_suffix}")
            zipf.write(description_path, f"{filename_base}_description.txt")
            zipf.write(findings_path, f"{filename_base}_key_findings.txt")

        # Return the zip file and delete temp dir after response is sent
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{filename_base}_results.zip",
            background=BackgroundTask(lambda: shutil.rmtree(tmpdir, ignore_errors=True)),
        )

    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# To run locally: uvicorn api:app --reload