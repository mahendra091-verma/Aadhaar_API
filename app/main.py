from functools import lru_cache
from io import BytesIO
import json
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from Aadhaar_API.app.aadhar_extractor import AadhaarExtractor, build_extractor_from_env

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(25 * 1024 * 1024)))
MAX_DOWNLOAD_BYTES = max(MAX_IMAGE_BYTES, MAX_PDF_BYTES)
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))

ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", "artifacts"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AadhaarCompareResponse(BaseModel):
    beneficiary_id: str
    application_id: str
    beneficiary_aadhaar_number: str
    beneficiary_aadhaar_name: str
    front_aadhaar_name: str | None = None
    front_aadhaar_dob: str | None = None
    front_aadhaar_no: str | None = None
    front_aadhaar_gender: str | None = None
    front_aadhaar_photo: bool = False
    # back_aadhaar_name: str | None = None
    # back_aadhaar_dob: str | None = None
    # back_aadhaar_no: str | None = None
    # back_aadhaar_gender: str | None = None
    # back_aadhaar_photo: bool = False
    beneficiary_aadhaar_number_match: str
    beneficiary_aadhaar_name_match: str


def normalize_extract_result(result: dict | None) -> dict[str, Any]:
    if not result or result.get("valid_aadhaar") != "Yes":
        return {"name": None, "dob": None, "aadhaar_no": None, "gender": None, "photo": False}
    return {
        "name": result.get("name"),
        "dob": result.get("dob"),
        "aadhaar_no": result.get("aadhaar_number"),
        "gender": result.get("gender"),
        "photo": bool(result.get("photo")),
    }


def build_compare_response(
        beneficiary_id: str,
        application_id: str,
        beneficiary_aadhaar_number: str,
        beneficiary_aadhaar_name: str,
        front_aadhaar_result: dict | None,
        back_aadhaar_result: dict | None,
) -> AadhaarCompareResponse:
    front_aadhaar_data = normalize_extract_result(front_aadhaar_result)
    back_aadhaar_data = normalize_extract_result(back_aadhaar_result)

    # Normalize user inputs
    input_name_clean = str(beneficiary_aadhaar_name).lower().replace(" ", "")
    input_num_clean = re.sub(r"\D", "", str(beneficiary_aadhaar_number))

    # Normalize front card extraction values (Covers both Front File and Front URL)
    extracted_front_name = str(front_aadhaar_data.get("name") or "").lower().replace(" ", "")
    extracted_front_num = re.sub(r"\D", "", str(front_aadhaar_data.get("aadhaar_no") or ""))

    # Evaluate matches ONLY against front data (File or URL)
    name_match_status = "Yes" if (input_name_clean and extracted_front_name and input_name_clean == extracted_front_name) else "No"
    num_match_status = "Yes" if (input_num_clean and extracted_front_num and input_num_clean == extracted_front_num) else "No"

    return AadhaarCompareResponse(
        beneficiary_id=beneficiary_id,
        application_id=application_id,
        beneficiary_aadhaar_number=beneficiary_aadhaar_number,
        beneficiary_aadhaar_name=beneficiary_aadhaar_name,
        front_aadhaar_name=front_aadhaar_data["name"],
        front_aadhaar_dob=front_aadhaar_data["dob"],
        front_aadhaar_no=front_aadhaar_data["aadhaar_no"],
        front_aadhaar_gender=front_aadhaar_data["gender"],
        front_aadhaar_photo=bool(front_aadhaar_data["photo"]),
        # back_aadhaar_name=back_aadhaar_data["name"],
        # back_aadhaar_dob=back_aadhaar_data["dob"],
        # back_aadhaar_no=back_aadhaar_data["aadhaar_no"],
        # back_aadhaar_gender=back_aadhaar_data["gender"],
        # back_aadhaar_photo=bool(back_aadhaar_data["photo"]),
        beneficiary_aadhaar_number_match=num_match_status,
        beneficiary_aadhaar_name_match=name_match_status,
    )


def sanitize_name(value: str | int) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", str(value).strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "unknown"

def image_extension(image_bytes: bytes) -> str:
    with Image.open(BytesIO(image_bytes)) as image:
        image_format = (image.format or "jpg").lower()
    if image_format == "jpeg":
        return "jpg"
    return image_format

def is_pdf_payload(payload: bytes, content_type: str, url: str) -> bool:
    parsed_path = urlparse(url).path.lower()
    return (
        "application/pdf" in content_type.lower()
        or parsed_path.endswith(".pdf")
        or payload.startswith(b"%PDF")
    )

def pdf_to_first_page_image(pdf_bytes: bytes, field_name: str) -> bytes:
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail=f"{field_name} PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="PDF URL support requires PyMuPDF. Install it with: pip install PyMuPDF",
        ) from exc

    document = None
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if document.page_count < 1:
            raise HTTPException(status_code=400, detail=f"{field_name} PDF does not contain any pages.")

        page = document.load_page(0)
        render_dpis = [PDF_RENDER_DPI, 150, 100]
        seen_dpis: set[int] = set()

        for dpi in render_dpis:
            if dpi in seen_dpis:
                continue
            seen_dpis.add(dpi)
            zoom = dpi / 72
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image_bytes = pixmap.tobytes("png")
            if len(image_bytes) <= MAX_IMAGE_BYTES:
                ensure_image_bytes(image_bytes, field_name)
                return image_bytes

        raise HTTPException(
            status_code=400,
            detail=f"{field_name} PDF page rendered larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} PDF could not be converted to an image: {exc}") from exc
    finally:
        if document is not None:
            document.close()

def artifact_source_prefix(source_name: str) -> str:
    source_map = {
        "file1": "front_aadhaar_file",
        "file2": "back_aadhaar_file",
        "url1": "front_aadhaar_url",
        "url2": "back_aadhaar_url",
    }
    return source_map.get(source_name, source_name)

def json_response_prefix(source_name: str) -> str:
    if source_name in {"file1", "file2", "front_aadhaar_file", "back_aadhaar_file"}:
        return "json_aadhaar_file"
    if source_name in {"url1", "url2", "front_aadhaar_url", "back_aadhaar_url"}:
        return "json_aadhaar_url"
    return f"json_{source_name}"

def save_artifacts(
        beneficiary_id: str,
        application_id: str,
        beneficiary_aadhaar_number: str,
        beneficiary_aadhaar_name: str,
        images: list[tuple[str, bytes]],
        response_payload: BaseModel,
        response_source_name: str,
) -> Path:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    beneficiary_token = sanitize_name(beneficiary_id).lower()

    for source_name, image_bytes in images:
        artifact_source = artifact_source_prefix(source_name)
        ext = image_extension(image_bytes)
        image_name = f"{artifact_source}_{timestamp}_{beneficiary_token}.{ext}"
        (ARTIFACTS_ROOT / image_name).write_bytes(image_bytes)

    response_source = json_response_prefix(response_source_name)
    response_name = f"{response_source}_{timestamp}_{beneficiary_token}.json"
    response_path = ARTIFACTS_ROOT / response_name
    response_path.write_text(
        json.dumps(response_payload.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ARTIFACTS_ROOT


async def read_uploaded_image(file: UploadFile, field_name: str) -> tuple[str, bytes]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{field_name} is not an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"{field_name} is larger than 10 MB.")

    return field_name, image_bytes


def has_uploaded_file(file: UploadFile | None) -> bool:
    return file is not None and bool(file.filename)


def ensure_image_bytes(image_bytes: bytes, field_name: str) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} did not return a valid image.") from exc


def extract_image_url_from_html(html_text: str, page_url: str) -> str | None:
    meta_patterns = [
        r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image|twitter:image:src)["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:og:image|twitter:image|twitter:image:src)["\']',
        r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
    ]
    for pattern in meta_patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = urljoin(page_url, match.group(1).strip())
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            return candidate

    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_text, flags=re.IGNORECASE)
    if match:
        candidate = urljoin(page_url, match.group(1).strip())
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            return candidate

    match = re.search(r'<a[^>]+href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', html_text, flags=re.IGNORECASE)
    if match:
        candidate = urljoin(page_url, match.group(1).strip())
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            return candidate

    normalized_html = html_text.replace("\\/", "/")
    direct_url_patterns = [
        r"https://i\.pinimg\.com/[^\s\"'<>]+",
        r"https?://[^\s\"'<>]+\.(?:jpg|jpeg|png|webp|pdf)(?:\?[^\s\"'<>]*)?",
    ]
    for pattern in direct_url_patterns:
        match = re.search(pattern, normalized_html, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(0).strip()
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            return candidate

    return None


def fetch_url_bytes(url: str, field_name: str) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        },
    )
    try:
        with urlopen(request, timeout=8) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field_name} download is larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB.",
                )

            payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    except HTTPException:
        raise
    except URLError as exc:
        raise HTTPException(status_code=400, detail=f"Could not download {field_name}: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=400, detail=f"Timed out downloading {field_name}.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Network connection error fetching {field_name}: {exc}") from exc

    if not payload:
        raise HTTPException(status_code=400, detail=f"{field_name} URL returned an empty response.")
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} download is larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return payload, content_type


def read_url_image(url: str, field_name: str) -> tuple[str, bytes]:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an http or https URL.")
    is_pinterest = "pinterest." in parsed.netloc.lower()

    try:
        first_payload, first_content_type = fetch_url_bytes(url, field_name)
    except HTTPException as exc:
        if is_pinterest:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{field_name}: Pinterest page URL blocked. "
                    "Use direct image URL (i.pinimg.com/...) or upload the image as Front_Aadhaar_file/Back_Aadhaar_file."
                ),
            ) from exc
        raise
    try:
        ensure_image_bytes(first_payload, field_name)
        if len(first_payload) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"{field_name} image is larger than 10 MB.")
        return field_name, first_payload
    except HTTPException:
        if is_pdf_payload(first_payload, first_content_type, url):
            return field_name, pdf_to_first_page_image(first_payload, field_name)
        if "text/html" not in first_content_type:
            raise

    page_html = first_payload.decode("utf-8", errors="ignore")
    image_url = extract_image_url_from_html(page_html, url)
    if not image_url:
        if is_pinterest:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{field_name}: Pinterest page URL did not expose a downloadable image. "
                    "Use direct image URL (i.pinimg.com/...) or upload the image as Front_Aadhaar_file/Back_Aadhaar_file."
                ),
            )
        raise HTTPException(status_code=400, detail=f"{field_name} did not return a valid image.")

    image_payload, image_content_type = fetch_url_bytes(image_url, field_name)
    if is_pdf_payload(image_payload, image_content_type, image_url):
        return field_name, pdf_to_first_page_image(image_payload, field_name)
    ensure_image_bytes(image_payload, field_name)
    if len(image_payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"{field_name} image is larger than 10 MB.")
    return field_name, image_payload


def normalize_url_value(url: str) -> str:
    value = url.strip()
    if value.lower() in {"string", "url1", "url2", "Front_aadhaar_url", "Back_aadhaar_url"}:
        return ""
    return value

def resolve_side_source(
        file_value: UploadFile | None,
        url_value: str,
        file_source_name: str,
        url_source_name: str,
        side_label: str,
        is_required: bool = True,
) -> tuple[str | None, UploadFile | str | None]:
    has_file = has_uploaded_file(file_value)
    normalized_url = normalize_url_value(url_value)

    if has_file and normalized_url:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provide only one source for {side_label}: either {file_source_name} or {url_source_name}."
            ),
        )
    if has_file:
        if file_value is None:
            raise HTTPException(status_code=400, detail=f"{file_source_name} is invalid.")
        return file_source_name, file_value
    if normalized_url:
        return url_source_name, normalized_url

    if is_required:
        raise HTTPException(
            status_code=400,
            detail=f"{side_label} source is required. Provide {file_source_name} or {url_source_name}.",
        )
    return None, None

async def read_image_source(source_name: str, source: UploadFile | str) -> tuple[str, bytes]:
    if hasattr(source, "read"):
        return await read_uploaded_image(source, source_name)
    return read_url_image(source, source_name)


app = FastAPI(
    title="Aadhaar OCR API",
    version="1.0.0",
    description="Detect Aadhaar fields with YOLO and extract structured data with OCR.",
)


@lru_cache(maxsize=1)
def get_extractor() -> AadhaarExtractor:
    return build_extractor_from_env()


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_extractor()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok"}


@app.post("/identifyAadh6/")
async def extract_aadhaar(
        beneficiary_id: str = Form(...),
        application_id: str = Form(...),
        beneficiary_aadhaar_number: str = Form(...),
        beneficiary_aadhaar_name: str = Form(...),
        file1: UploadFile | None = File(default=None, alias="Front_Aadhaar_file"),
        file2: UploadFile | None = File(default=None, alias="Back_Aadhaar_file"),
        url1: str = Form(default="", alias="Front_Aadhaar_url"),
        url2: str = Form(default="", alias="Back_Aadhaar_url"),
) -> AadhaarCompareResponse:
    try:
        extractor = get_extractor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc

    processed_images: list[tuple[str, bytes]] = []

    # Front Aadhaar remains mandatory
    front_source_name, front_source = resolve_side_source(
        file1,
        url1,
        "file1",
        "url1",
        "Front Aadhaar",
        is_required=True,
    )

    # Back Aadhaar is now completely optional
    back_source_name, back_source = resolve_side_source(
        file2,
        url2,
        "file2",
        "url2",
        "Back Aadhaar",
        is_required=False,
    )

    # Process Front Aadhaar
    try:
        # front_source_name will never be None since is_required=True
        _, front_image_bytes = await read_image_source(front_source_name, front_source)  # type: ignore
        processed_images.append((front_source_name, front_image_bytes))  # type: ignore
        front_aadhaar_result = extractor.extract(front_image_bytes, include_photo=False)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{front_source_name}: Extraction failed: {exc}") from exc

    # Process Back Aadhaar conditionally (Only runs if a file or URL was provided)
    back_aadhaar_result = None
    if back_source_name and back_source:
        try:
            _, back_image_bytes = await read_image_source(back_source_name, back_source)
            processed_images.append((back_source_name, back_image_bytes))
            back_aadhaar_result = extractor.extract(back_image_bytes, include_photo=False)
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{back_source_name}: Extraction failed: {exc}") from exc

    final_response = build_compare_response(
        beneficiary_id=beneficiary_id,
        application_id=application_id,
        beneficiary_aadhaar_number=beneficiary_aadhaar_number,
        beneficiary_aadhaar_name=beneficiary_aadhaar_name,
        front_aadhaar_result=front_aadhaar_result,
        back_aadhaar_result=back_aadhaar_result,
    )

    try:
        save_artifacts(
            beneficiary_id,
            application_id,
            beneficiary_aadhaar_number,
            beneficiary_aadhaar_name,
            processed_images,
            final_response,
            front_source_name,  # type: ignore
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save artifacts: {exc}") from exc

    return final_response
