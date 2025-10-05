import os
import re
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import easyocr

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    import spacy
    _HAS_PRESIDIO = True
except Exception:
    _HAS_PRESIDIO = False

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False


def _blur_region(image: np.ndarray, box: Tuple[int, int, int, int]) -> None:
   
    x, y, w, h = box
    x_end = min(x + w, image.shape[1])
    y_end = min(y + h, image.shape[0])
    x = max(x, 0)
    y = max(y, 0)
    roi = image[y:y_end, x:x_end]
    if roi.size == 0:
        return
    kx = max(11, (w // 5) * 2 + 1)
    ky = max(11, (h // 5) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
    image[y:y_end, x:x_end] = blurred


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
HANDLE_RE = re.compile(r"(^|\s)@[A-Za-z0-9_\.]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")
ID_LIKE_RE = re.compile(r"(?=.*\d)[A-Za-z0-9\-_/]{5,}")
ADDRESS_RE = re.compile(r"\b(\d{1,6}\s+)?([A-Za-z]+\s+){0,6}(street|st\.?|lane|ln\.?|road|rd\.?|avenue|ave\.?|blvd\.?|drive|dr\.?|way|court|ct\.?|circle|cir\.?|parkway|pkwy\.?|place|pl\.?|terrace|ter\.?|highway|hwy\.?|square|sq\.?)\b", re.I)
LONG_NUMBER_RE = re.compile(r"(?<!\d)(?:\d[ \-()]{0,2}){10,}(?!\d)")
BANK_KEYWORDS = ["bank", "iban", "ifsc", "swift", "routing", "account", "acct", "micr"]


LABEL_KEYWORDS = [
    "id", "employee id", "emp id", "issue date", "issued", "expiry",
    "expiration", "license", "licence", "dob", "date of birth", "phone",
    "mobile", "contact", "email", "account", "number", "no.", "code"
]


def _contains_sensitive_text(text: str, instructions: str) -> bool:
   
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if EMAIL_RE.search(t):
        return True
    if HANDLE_RE.search(t):
        return True
    if PHONE_RE.search(t):
        return True
    if ID_LIKE_RE.search(t):
        return True
    lowered = (instructions or "").lower()
    if any(k in lowered for k in ["email", "phone", "id", "date", "account", "code", "handle"]):
        if re.fullmatch(r"[A-Za-z]{2,}", t):
            return False
    return False


def _group_by_lines(results: List[Any]) -> List[List[Tuple[List[Tuple[int, int]], str, float]]]:
    
    items: List[Tuple[float, Any]] = []
    for item in results:
        bbox, text, conf = item
        ys = [p[1] for p in bbox]
        y_center = float(sum(ys) / 4.0)
        items.append((y_center, item))

    items.sort(key=lambda x: x[0])
    lines: List[List[Any]] = []
    threshold = 14.0 
    for y_center, item in items:
        if not lines:
            lines.append([item])
            continue
        last_line = lines[-1]
        last_y = sum(p[1] for p in last_line[0][0]) / 4.0
        if abs(y_center - last_y) <= threshold:
            last_line.append(item)
        else:
            lines.append([item])

    sorted_lines: List[List[Any]] = []
    for line in lines:
        line.sort(key=lambda it: min(p[0] for p in it[0]))
        sorted_lines.append(line)
    return sorted_lines


def _is_label_word(word: str) -> bool:
    w = word.strip().lower().rstrip(":")
    return any(w == k or w in k.split() for k in LABEL_KEYWORDS)


def _build_presidio_analyzer() -> AnalyzerEngine:
    if not _HAS_PRESIDIO:
        return None
    try:
        _ = spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine)


def _pii_entities_from_lines(lines: List[List[Any]]) -> List[Dict[str, Any]]:
   
    if not _HAS_PRESIDIO:
        return []

    analyzer = _build_presidio_analyzer()
    if analyzer is None:
        return []
    detections: List[Dict[str, Any]] = []
    for line_idx, line in enumerate(lines):
        # Build line string and char index map per token
        pieces: List[str] = []
        spans: List[Tuple[int, int, int]] = []  # (start, end, token_idx)
        cursor = 0
        for token_idx, (bbox, text, conf) in enumerate(line):
            t = (text or "").strip()
            if token_idx > 0:
                pieces.append(" ")
                cursor += 1
            pieces.append(t)
            spans.append((cursor, cursor + len(t), token_idx))
            cursor += len(t)

        line_text = "".join(pieces)
        if not line_text:
            continue
        results = analyzer.analyze(
            text=line_text,
            language="en",
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "LOCATION",
                "NRP",
                "IBAN_CODE",
                "CREDIT_CARD",
                "US_SSN",
                "DATE_TIME",
            ],
        )
        for r in results:
            # Map char span to token indexes on this line
            for start, end, token_idx in spans:
                if not (r.end <= start or r.start >= end):
                    detections.append({
                        "line_idx": line_idx,
                        "token_idx": token_idx,
                        "entity_type": r.entity_type,
                        "score": r.score,
                    })
    return detections


def _match_cross_token_patterns(lines: List[List[Any]]) -> List[Dict[str, int]]:
    """
    Detect emails/phones/long numbers/addresses that are split across tokens.
    Returns list of { line_idx, token_idx } to blur.
    """
    hits: List[Dict[str, int]] = []
    for li, line in enumerate(lines):
        # Build text and token boundaries
        tokens = [(bbox, (text or "").strip()) for bbox, text, _ in line]
        if not tokens:
            continue
        pieces: List[str] = []
        spans: List[Tuple[int, int, int]] = []
        cur = 0
        for idx, (_, t) in enumerate(tokens):
            if idx > 0:
                pieces.append(" ")
                cur += 1
            pieces.append(t)
            spans.append((cur, cur + len(t), idx))
            cur += len(t)
        txt = "".join(pieces)
        # Search patterns on the joined text
        patterns = [EMAIL_RE, PHONE_RE, LONG_NUMBER_RE, ADDRESS_RE]
        for pat in patterns:
            for m in pat.finditer(txt):
                ms, me = m.start(), m.end()
                for s, e, ti in spans:
                    if not (me <= s or ms >= e):
                        hits.append({"line_idx": li, "token_idx": ti})
    return hits


def _analyze_document_structure(text: str) -> str:
    """
    Analyze the document type and structure based on extracted text.
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        String describing the document type and structure
    """
    text_lower = text.lower()
    
    # Detect document type
    doc_type = "Unknown Document"
    if any(word in text_lower for word in ["dear", "sincerely", "regards", "letter"]):
        doc_type = "Formal Letter"
    elif any(word in text_lower for word in ["invoice", "bill", "amount due", "total:"]):
        doc_type = "Invoice/Bill"
    elif any(word in text_lower for word in ["resume", "curriculum vitae", "experience", "education", "skills"]):
        doc_type = "Resume/CV"
    elif any(word in text_lower for word in ["id", "identification", "license", "passport"]):
        doc_type = "Identification Document"
    elif any(word in text_lower for word in ["contract", "agreement", "terms", "conditions"]):
        doc_type = "Contract/Agreement"
    elif any(word in text_lower for word in ["report", "analysis", "findings", "conclusion"]):
        doc_type = "Report"
    elif any(word in text_lower for word in ["form", "application", "questionnaire"]):
        doc_type = "Form/Application"
    
    # Detect key information categories
    categories = []
    if EMAIL_RE.search(text):
        categories.append("Email addresses")
    if PHONE_RE.search(text):
        categories.append("Phone numbers")
    if any(word in text_lower for word in ["address", "street", "city", "state"]):
        categories.append("Physical addresses")
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
        categories.append("Dates")
    if re.search(r'\$\d+|\d+\.\d{2}', text):
        categories.append("Financial amounts")
    if any(word in text_lower for word in ["name", "mr.", "mrs.", "ms."]):
        categories.append("Personal names")
    
    # Build description
    description = f"Document Type: {doc_type}\n\n"
    
    if categories:
        description += "Information Categories Detected:\n"
        for cat in categories:
            description += f"  • {cat}\n"
    
    # Estimate document length
    word_count = len(text.split())
    description += f"\nDocument Length: Approximately {word_count} words"
    
    return description


def _extract_key_findings(text: str) -> str:
    """
    Extract key findings from the document text.
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        String with formatted key findings
    """
    findings = []
    
    # Extract emails
    emails = EMAIL_RE.findall(text)
    if emails:
        findings.append(f"• Email addresses detected: {len(emails)} found")
    
    # Extract phone numbers
    phones = PHONE_RE.findall(text)
    if phones:
        findings.append(f"• Phone numbers detected: {len(phones)} found")
    
    # Extract dates
    dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4}', text)
    if dates:
        findings.append(f"• Dates mentioned: {', '.join(dates[:3])}" + (" and more" if len(dates) > 3 else ""))
    
    # Extract monetary amounts
    amounts = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
    if amounts:
        findings.append(f"• Financial amounts: {', '.join(amounts[:3])}" + (" and more" if len(amounts) > 3 else ""))
    
    # Extract potential names (capitalized words)
    potential_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
    if potential_names:
        unique_names = list(set(potential_names))[:3]
        findings.append(f"• Names mentioned: {', '.join(unique_names)}" + (" and others" if len(potential_names) > 3 else ""))
    
    # Extract addresses
    addresses = ADDRESS_RE.findall(text)
    if addresses:
        findings.append(f"• Physical addresses detected: {len(addresses)} found")
    
    # Extract key sentences (first few sentences)
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if sentences:
        findings.append(f"\n• Main content summary:")
        for i, sentence in enumerate(sentences[:3], 1):
            # Truncate long sentences
            truncated = sentence[:150] + "..." if len(sentence) > 150 else sentence
            findings.append(f"  {i}. {truncated}")
    
    # Extract keywords (most common meaningful words)
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    word_freq = {}
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'there', 'would', 'could', 'should'}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    if word_freq:
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        findings.append(f"\n• Frequently mentioned terms: {', '.join([k for k, v in top_keywords])}")
    
    if not findings:
        findings.append("• No specific patterns detected in the document")
    
    return "\n".join(findings)


def _gemini_select_sensitive_tokens(lines: List[List[Any]], instructions: str, api_key: str) -> List[Dict[str, int]]:
    """
    Send OCR tokens as JSON to Gemini to select sensitive tokens. Returns list of
    { line_idx, token_idx } dicts to blur.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        import json
        payload_lines = []
        for li, line in enumerate(lines):
            payload_tokens = []
            for ti, (bbox, text, conf) in enumerate(line):
                payload_tokens.append({"idx": int(ti), "text": (text or "").strip()})
            payload_lines.append({"idx": int(li), "tokens": payload_tokens})
        payload = {"lines": payload_lines}
        system_prompt = (
            "Decide which OCR tokens contain PII based on the user instructions. "
            "Return ONLY JSON with an array of objects: {\"line_idx\": int, \"token_idx\": int}."
        )
        user_prompt = (
            f"Instructions:\n{instructions}\n\n"
            f"OCR tokens (as JSON):\n{json.dumps(payload)}\n\n"
            "Output JSON array only, no extra text."
        )
        resp = model.generate_content(system_prompt + "\n\n" + user_prompt)
        text = getattr(resp, "text", None)
        if not text:
            return []
        # Extract JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        data = json.loads(text[start:end+1])
        results: List[Dict[str, int]] = []
        for item in data:
            try:
                li = int(item.get("line_idx", -1))
                ti = int(item.get("token_idx", -1))
                if li >= 0 and ti >= 0:
                    results.append({"line_idx": li, "token_idx": ti})
            except Exception:
                continue
        return results
    except Exception:
        return []


def generate_image_description(input_path: str, api_key: str) -> str:
    """
    Generate a detailed description of the image using OCR + Gemini API.
    
    Args:
        input_path: Path to input image file.
        api_key: Gemini API key.
        
    Returns:
        String containing the image description.
    """
    try:
        # First, extract text from image using OCR
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            return "Error: Unable to read image file"
        
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_bgr)
        
        # Extract all text from OCR results
        extracted_text = []
        for bbox, text, conf in results:
            if isinstance(text, str) and text.strip():
                extracted_text.append(text.strip())
        
        ocr_text = " ".join(extracted_text)
        
        if not ocr_text:
            return "No text detected in the image. The image may be blank or contain only graphics."
        
        # Analyze the document type and structure
        doc_analysis = _analyze_document_structure(ocr_text)
        
        return f"""{doc_analysis}

Extracted Content:
{ocr_text}"""
        
    except Exception as e:
        return f"Error generating description: {str(e)}"


def generate_key_findings(input_path: str, api_key: str) -> str:
    """
    Extract key findings from the image using OCR + Gemini API.
    
    Args:
        input_path: Path to input image file.
        api_key: Gemini API key.
        
    Returns:
        String containing key findings.
    """
    try:
        # First, extract text from image using OCR
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            return "Error: Unable to read image file"
        
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_bgr)
        
        # Extract all text from OCR results
        extracted_text = []
        for bbox, text, conf in results:
            if isinstance(text, str) and text.strip():
                extracted_text.append(text.strip())
        
        ocr_text = " ".join(extracted_text)
        
        if not ocr_text:
            return "No text detected in the image. Unable to extract key findings."
        
        # Extract key findings from the text
        findings = _extract_key_findings(ocr_text)
        return findings
        
    except Exception as e:
        return f"Error generating key findings: {str(e)}"


def anonymize_image(input_path: str, output_path: str, instructions: str) -> None:
    """
    Anonymize an image locally by OCR-detecting sensitive text and blurring it.

    Args:
        input_path: Path to input image file.
        output_path: Path to write redacted image file.
        instructions: Natural-language guidance (currently heuristic-based).
    """
    try:
        # Read image with OpenCV (BGR)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"Failed to read image: {input_path}")
            return

        # Run EasyOCR (English by default; add more langs if needed)
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_bgr)

        # 1) Direct pattern-based redaction
        for bbox, text, conf in results:
            if not isinstance(text, str):
                continue
            if _contains_sensitive_text(text, instructions):
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x = max(min(xs), 0)
                y = max(min(ys), 0)
                w = max(max(xs) - x, 1)
                h = max(max(ys) - y, 1)
                _blur_region(image_bgr, (x, y, w, h))

        # 2) Label -> value redaction (erase value tokens to the right of label tokens)
        lines = _group_by_lines(results)
        for line in lines:
            # Build token list with boxes and words
            tokens: List[Dict[str, Any]] = []
            for bbox, text, conf in line:
                if not isinstance(text, str):
                    continue
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                token = {
                    "bbox": bbox,
                    "x": min(xs),
                    "y": min(ys),
                    "w": max(xs) - min(xs),
                    "h": max(ys) - min(ys),
                    "text": text.strip(),
                }
                tokens.append(token)

            # Scan for label tokens; redact tokens to their right on same line
            for idx, tok in enumerate(tokens):
                if _is_label_word(tok["text"]):
                    label_right_edge = tok["x"] + tok["w"] + 6  # small padding
                    for j in range(idx + 1, len(tokens)):
                        right_tok = tokens[j]
                        if right_tok["x"] >= label_right_edge:
                            _blur_region(image_bgr, (
                                right_tok["x"], right_tok["y"], right_tok["w"], right_tok["h"]
                            ))

        # 2b) Gemini-assisted token selection (OPTIONAL if key present)
        api_key = os.getenv("GEMINI_API_KEY")
        if _HAS_GEMINI and api_key:
            print("Using Gemini-assisted token selection")
            selections = _gemini_select_sensitive_tokens(lines, instructions, api_key)
            for sel in selections:
                li = sel.get("line_idx")
                ti = sel.get("token_idx")
                if li is None or ti is None:
                    continue
                if li < 0 or li >= len(lines):
                    continue
                line = lines[li]
                if ti < 0 or ti >= len(line):
                    continue
                bbox, text, conf = line[ti]
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x = max(min(xs), 0)
                y = max(min(ys), 0)
                w = max(max(xs) - x, 1)
                h = max(max(ys) - y, 1)
                _blur_region(image_bgr, (x, y, w, h))
        else:
            print("Gemini API not available - using heuristic-based detection only")

        # 3) NER-driven PII redaction using Presidio (names, addresses, etc.)
        if _HAS_PRESIDIO:
            ner_hits = _pii_entities_from_lines(lines)
            for hit in ner_hits:
                line = lines[hit["line_idx"]]
                bbox, text, conf = line[hit["token_idx"]]
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x = max(min(xs), 0)
                y = max(min(ys), 0)
                w = max(max(xs) - x, 1)
                h = max(max(ys) - y, 1)
                _blur_region(image_bgr, (x, y, w, h))

        # 3b) Cross-token pattern detection (emails/phones/etc. split over tokens)
        cross_hits = _match_cross_token_patterns(lines)
        for hit in cross_hits:
            line = lines[hit["line_idx"]]
            bbox, text, conf = line[hit["token_idx"]]
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x = max(min(xs), 0)
            y = max(min(ys), 0)
            w = max(max(xs) - x, 1)
            h = max(max(ys) - y, 1)
            _blur_region(image_bgr, (x, y, w, h))

        # 3c) Token-level redaction for email/phone/long numbers/addresses/bank keywords
        for line in lines:
            for bbox, text, conf in line:
                t = (text or "").strip()
                if not t:
                    continue
                is_sensitive = (
                    EMAIL_RE.search(t)
                    or PHONE_RE.search(t)
                    or LONG_NUMBER_RE.search(t)
                    or ADDRESS_RE.search(t)
                    or any(k in t.lower() for k in BANK_KEYWORDS)
                )
                if is_sensitive:
                    xs = [int(p[0]) for p in bbox]
                    ys = [int(p[1]) for p in bbox]
                    x = max(min(xs), 0)
                    y = max(min(ys), 0)
                    w = max(max(xs) - x, 1)
                    h = max(max(ys) - y, 1)
                    _blur_region(image_bgr, (x, y, w, h))

        # 3d) Name heuristic: blur only the name tokens, not whole line
        for line in lines:
            tokens = [(bbox, (text or "").strip()) for bbox, text, _ in line]
            for idx in range(len(tokens) - 1):
                w1 = tokens[idx][1]
                w2 = tokens[idx + 1][1]
                if (w1[:1].isupper() and w1[1:].islower()) and (w2[:1].isupper() and w2[1:].islower()):
                    for w in [tokens[idx], tokens[idx + 1]]:
                        bbox = w[0]
                        xs = [int(p[0]) for p in bbox]
                        ys = [int(p[1]) for p in bbox]
                        x = max(min(xs), 0)
                        y = max(min(ys), 0)
                        ww = max(max(xs) - x, 1)
                        hh = max(max(ys) - y, 1)
                        _blur_region(image_bgr, (x, y, ww, hh))

        # 4) Strong regex-based safety net remains (emails/phones/IDs/etc.)

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, image_bgr)
        if success:
            print(f"Anonymized image saved to {output_path}")
        else:
            print(f"Failed to save output image to {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")
