import os
import zipfile
import tempfile
import shutil
import backend34 as b34
from flask import app, jsonify, request, send_file, Flask
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS
from docling.document_converter import DocumentConverter, PdfFormatOption, PowerpointFormatOption, ImageFormatOption
from streamlit import progress

# --- GraniteDocling Extractor ---

# --- Extraction Functions ---
def extract_pdf_granite(file_path, upload_id=None):
    pdf_text = b34.extract_pdf(file_path)
    output = []
    if pdf_text:
        output.append(pdf_text)
    if upload_id:
        progress[upload_id] = 100
    return '\n'.join(output)

def extract_pptx_granite(file_path, upload_id=None):
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(
            lang=["en"],
            force_full_page_ocr=True
        )
    )
    format_option = PowerpointFormatOption(pipeline_options=pipeline_options)
    input_format = InputFormat.PPTX
    converter = DocumentConverter(format_options={input_format: format_option})
    doc = converter.convert(file_path).document
    

    # ...existing PPTX extraction logic...
    # (copy the PPTX block from above here)
    # For brevity, you can move the PPTX logic here
    # Return output as string
    # ...existing code...
    # (for now, just return doc.text)
    output = []
    texts = getattr(doc, "texts", [])
    para_map = {}
    for text_item in texts:
        parent = getattr(text_item, "parent", None)
        para_id = id(parent) if parent else None
        value = getattr(text_item, "text", None) or getattr(text_item, "orig", None)
        if value:
            para_map.setdefault(para_id, []).append(value)
    for para in para_map.values():
        line = ' '.join(para)
        if line and line not in output:
            output.append(line)

    tables = getattr(doc, "tables", [])
    def _safe_cell_text(cell):
        if cell is None:
            return ""
        raw = str(getattr(cell, "text", "") or getattr(cell, "orig", "") or "").replace("\r", " ").replace("\n", " ")
        return raw

    def _format_table_as_text(rows, header_row_index=None):
        if not rows:
            return []
        col_count = max(len(r) for r in rows)
        widths = [0] * col_count
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        lines = []
        for idx, r in enumerate(rows):
            padded = []
            for i in range(col_count):
                cell = r[i] if i < len(r) else ""
                padded.append(cell.ljust(widths[i]))
            lines.append("  ".join(padded).rstrip())
            if header_row_index is not None and idx == header_row_index:
                sep = []
                for w in widths:
                    sep.append("-" * w)
                lines.append("  ".join(sep).rstrip())
        return lines

    for table in tables:
        table_data = getattr(table, "data", None)
        collected_rows = []
        header_index = None
        header = getattr(table_data, "header", None)
        if header and isinstance(header, (list, tuple)):
            header_texts = [_safe_cell_text(cell) for cell in header]
            collected_rows.append(header_texts)
            header_index = 0
        cols = None
        if table_data and hasattr(table_data, "rows"):
            for row in getattr(table_data, "rows", []):
                row_text = [_safe_cell_text(cell) for cell in row]
                collected_rows.append(row_text)
        elif table_data and hasattr(table_data, "table_cells"):
            flat = [_safe_cell_text(cell) for cell in getattr(table_data, "table_cells")]
            cols = getattr(table_data, "columns", None) or getattr(table_data, "ncols", None) or None
            if cols and isinstance(cols, int) and cols > 0:
                for i in range(0, len(flat), cols):
                    collected_rows.append(flat[i:i+cols])
            else:
                for cell in flat:
                    collected_rows.append([cell])
        if not collected_rows:
            if table_data:
                if hasattr(table_data, "cells"):
                    for row in getattr(table_data, "cells"):
                        collected_rows.append([_safe_cell_text(cell) for cell in row])
                elif hasattr(table_data, "rows_data"):
                    for row in getattr(table_data, "rows_data"):
                        collected_rows.append([_safe_cell_text(cell) for cell in row])
        if collected_rows:
            output.append("")
            output.extend(_format_table_as_text(collected_rows, header_row_index=header_index))
            output.append("")

    with tempfile.TemporaryDirectory() as tmpdir:
        pptx_zip = os.path.join(tmpdir, "pptx.zip")
        shutil.copyfile(file_path, pptx_zip)
        with zipfile.ZipFile(pptx_zip, 'r') as zip_ref:
            media_files = [f for f in zip_ref.namelist() if f.startswith("ppt/media/") and not f.endswith("/")]
            for idx, media_file in enumerate(media_files):
                ext_img = os.path.splitext(media_file)[-1].lower()
                if ext_img in [".jpg", ".jpeg", ".png"]:
                    img_path = os.path.join(tmpdir, f"img_{idx}{ext_img}")
                    with open(img_path, "wb") as img_out:
                        img_out.write(zip_ref.read(media_file))
                    img_text = extract_image_granite(img_path)
                    output.append(f"Image from ppt/media ({media_file}):\n{img_text}")

    pictures = getattr(doc, "pictures", [])
    for idx, pic in enumerate(pictures):
        pic_texts = []
        for key in ["text", "ocr_text", "orig"]:
            value = getattr(pic, key, None)
            if value:
                pic_texts.append(str(value))
        annotations = getattr(pic, "annotations", None)
        if annotations:
            for ann in annotations:
                ann_text = getattr(ann, "text", None)
                if ann_text:
                    pic_texts.append(str(ann_text))
        pic_ref = getattr(pic, "self_ref", None)
        for text_item in texts:
            parent = getattr(text_item, "parent", None)
            cref = getattr(parent, "cref", None) if parent else None
            if cref == pic_ref:
                value = getattr(text_item, "text", None) or getattr(text_item, "orig", None)
                if value:
                    pic_texts.append(value)
        if pic_texts:
            output.append(f"Image {idx+1} OCR Text:\n" + '\n'.join(pic_texts))
    normal_text = getattr(doc, "text", None)
    if normal_text and normal_text not in output:
        output.append(normal_text)
    if upload_id:
        progress[upload_id] = 100
    return '\n'.join(output)

def extract_image_granite(file_path, upload_id=None):
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    ext = file_path.lower().split('.')[-1]
    scale_factor = 3.0
    with Image.open(file_path) as img:
        img = img.convert("L")
        img_np = np.array(img)
        img_np = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        img_np = cv2.fastNlMeansDenoising(img_np, None, h=30, templateWindowSize=7, searchWindowSize=21)
        img = Image.fromarray(img_np)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img_resized = img.resize(new_size, Image.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            img_resized.save(tmp.name)
            processed_file_path = tmp.name
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(
            lang=["en"],
            force_full_page_ocr=True
        )
    )
    format_option = ImageFormatOption(pipeline_options=pipeline_options)
    input_format = InputFormat.IMAGE
    file_path = processed_file_path
    converter = DocumentConverter(format_options={input_format: format_option})
    doc = converter.convert(file_path).document
    output_lines = []
    def _get_bbox_coords(item):
        bbox = getattr(item, "bbox", None)
        if not bbox or not isinstance(bbox, (list, tuple)):
            x0 = getattr(item, "x0", None) or getattr(item, "left", None)
            y0 = getattr(item, "y0", None) or getattr(item, "top", None)
            return (x0, y0) if x0 is not None and y0 is not None else None
        if len(bbox) >= 4:
            x = bbox[0]
            y = bbox[1]
        elif len(bbox) >= 2:
            x = bbox[0]
            y = bbox[1]
        else:
            return None
        return (x, y)
    doc_texts = getattr(doc, "texts", None) or getattr(doc, "ocr_texts", None) or getattr(doc, "items", None)
    tokens = []
    if doc_texts:
        for t in doc_texts:
            text = getattr(t, "text", None) or getattr(t, "orig", None) or getattr(t, "value", None)
            if not text:
                continue
            coords = _get_bbox_coords(t)
            x, y = (coords[0], coords[1]) if coords else (0.0, 0.0)
            tokens.append({"text": str(text), "x": float(x), "y": float(y), "obj": t})
    if tokens:
        tokens.sort(key=lambda it: (round(it["y"], 1), it["x"]))
        ys = [t["y"] for t in tokens]
        if ys:
            diffs = [abs(ys[i+1] - ys[i]) for i in range(len(ys)-1)] or [2.0]
            median_diff = diffs[len(diffs)//2] if diffs else 2.0
            y_threshold = max(1.0, median_diff * 0.6)
        else:
            y_threshold = 2.0
        lines_tokens = []
        current_line = [tokens[0]]
        current_y = tokens[0]["y"]
        for tk in tokens[1:]:
            if abs(tk["y"] - current_y) <= y_threshold:
                current_line.append(tk)
                current_y = (current_y + tk["y"]) / 2.0
            else:
                current_line.sort(key=lambda it: it["x"])
                lines_tokens.append(current_line)
                current_line = [tk]
                current_y = tk["y"]
        if current_line:
            current_line.sort(key=lambda it: it["x"])
            lines_tokens.append(current_line)
        for line in lines_tokens:
            row_text = " ".join([tk["text"] for tk in line]).strip()
            output_lines.append(row_text)
        if upload_id:
            progress[upload_id] = 100
        return "\n".join(output_lines)
    texts = []
    for element, _ in getattr(doc, "iterate_items", lambda: [])():
        value = getattr(element, "text", None) or getattr(element, "orig", None)
        if value:
            texts.append(str(value))
    if texts:
        return "\n".join(texts)
    if upload_id:
        progress[upload_id] = 100
    return "No text extracted from image."

# --- Dispatcher ---
def extract_with_granite_docling(file_path, upload_id=None):
    ext = file_path.lower().split('.')[-1]
    if ext == "pdf":
        return extract_pdf_granite(file_path, upload_id)
    elif ext == "pptx":
        return extract_pptx_granite(file_path, upload_id)
    elif ext in ["jpg", "jpeg", "png"]:
        return extract_image_granite(file_path, upload_id)
    else:
        raise ValueError("Unsupported file format for GraniteDocling.")
