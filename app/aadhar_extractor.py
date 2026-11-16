import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO

FIELD_ALIASES = {
    "aadhaar": {"aadhaar", "aadhar", "aadhaar_card", "aadhar_card", "card"},
    "aadhaar_no": {
        "aadhaar_no",
        "aadhar_no",
        "aadhaar_number",
        "aadhar_number",
        "uid",
        "uidai_no",
        "number",
    },
    "dob": {"dob", "date_of_birth", "birth", "yob"},
    "gender": {"gender", "sex"},
    "name": {"name", "person_name", "fullname", "full_name"},
    "photo": {"photo", "face", "image"},
}


@dataclass(frozen=True)
class Detection:
    label: str
    field: str | None
    confidence: float
    box: tuple[int, int, int, int]


class AadhaarExtractor:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        ocr_languages: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = YOLO(str(path))
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.ocr_reader = easyocr.Reader(ocr_languages or ["en"], gpu=device != "cpu")

    def extract(self, image_bytes: bytes, include_photo: bool = False) -> dict[str, Any]:
        image = self._read_image(image_bytes)
        detections = self._detect(image)
        detections_by_field = self._best_detection_by_field(detections)

        raw_text_by_field: dict[str, str] = {}
        for field in ("name", "dob", "aadhaar_no", "gender"):
            detection = detections_by_field.get(field)
            if detection is None:
                continue
            crop = self._crop(image, detection.box, pad=8)
            raw_text_by_field[field] = self._ocr(crop)

        name = self._clean_name(raw_text_by_field.get("name", ""))
        dob = self._clean_dob(raw_text_by_field.get("dob", ""))
        aadhaar_no = self._clean_aadhaar_no(raw_text_by_field.get("aadhaar_no", ""))
        gender = self._clean_gender(raw_text_by_field.get("gender", ""))

        photo_found = "photo" in detections_by_field
        is_aadhaar_card = self._is_aadhaar_card(detections_by_field, aadhaar_no)
        valid_aadhaar = "Yes" if is_aadhaar_card and aadhaar_no else "No"
        # print("====",self.model.names)
        return {
            "valid_aadhaar": valid_aadhaar,
            "name": name,
            "dob": dob,
            "aadhaar_number": aadhaar_no,
            "gender": gender,
            "photo": photo_found,
        }

    def _detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        names = results[0].names
        detections: list[Detection] = []

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            label = str(names.get(class_id, class_id)).lower().strip()
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=label,
                    field=self._canonical_field(label),
                    confidence=confidence,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections

    def _best_detection_by_field(self, detections: list[Detection]) -> dict[str, Detection]:
        best: dict[str, Detection] = {}
        for detection in detections:
            if detection.field is None:
                continue
            current = best.get(detection.field)
            if current is None or detection.confidence > current.confidence:
                best[detection.field] = detection
        return best

    def _canonical_field(self, label: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        for field, aliases in FIELD_ALIASES.items():
            if normalized in aliases:
                return field
        return None

    def _is_aadhaar_card(self, detections_by_field: dict[str, Detection], aadhaar_no: str) -> bool:
        if "aadhaar" in detections_by_field:
            return True
        if aadhaar_no:
            return True
        required_field_hits = {"name", "dob", "gender", "aadhaar_no"} & detections_by_field.keys()
        return len(required_field_hits) >= 2

    def _read_image(self, image_bytes: bytes) -> np.ndarray:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _crop(self, image: np.ndarray, box: tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        return image[y1:y2, x1:x2]

    def _ocr(self, crop: np.ndarray) -> str:
        processed = self._preprocess_for_ocr(crop)
        results = self.ocr_reader.readtext(processed, detail=0, paragraph=False)
        return " ".join(text.strip() for text in results if text.strip())

    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = 2 if max(gray.shape[:2]) < 700 else 1
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _clean_aadhaar_no(self, text: str) -> str | None:
        digits = re.sub(r"\D", "", text)
        match = re.search(r"\d{12}", digits)
        if not match:
            return None
        value = match.group(0)
        return f"{value[0:4]} {value[4:8]} {value[8:12]}"

    def _clean_dob(self, text: str) -> str | None:
        cleaned = text.replace("O", "0").replace("o", "0").replace("l", "1")
        match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", cleaned)
        if match:
            day, month, year = match.groups()
            return f"{int(day):02d}/{int(month):02d}/{year}"
        year_match = re.search(r"\b(19|20)\d{2}\b", cleaned)
        return year_match.group(0) if year_match else None

    def _clean_gender(self, text: str) -> str | None:
        normalized = text.upper()
        if "FEMALE" in normalized or re.search(r"\bF\b", normalized):
            return "Female"
        if "MALE" in normalized or re.search(r"\bM\b", normalized):
            return "Male"
        if "TRANS" in normalized:
            return "Transgender"
        return None

    def _clean_name(self, text: str) -> str | None:
        text = re.sub(r"(?i)\b(name|dob|date of birth|male|female|government|india|aadhaar)\b", " ", text)
        text = re.sub(r"[^A-Za-z .]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 2:
            return None
        return text.title()


def build_extractor_from_env() -> AadhaarExtractor:
    load_dotenv()
    model_path = os.getenv("MODEL_PATH", "D:\\Aadhar_API\\Model\\best.pt")
    confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    languages = [
        item.strip()
        for item in os.getenv("OCR_LANGUAGES", "en").split(",")
        if item.strip()
    ]
    device = os.getenv("DEVICE", "cpu")
    return AadhaarExtractor(
        model_path=model_path,
        confidence_threshold=confidence,
        ocr_languages=languages,
        device=device,
    )