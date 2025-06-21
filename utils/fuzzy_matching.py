from rapidfuzz import process, fuzz
import easyocr
import numpy as np

reader = easyocr.Reader(['en'])  # language list


def combine_easyocr_text_ordered(results, sep=" "):
    # Sort by top-left corner (y, then x)
    sorted_results = sorted(results, key=lambda x: (x[0][1][1], x[0][0][0]))
    return sep.join([text for (_, text, _) in sorted_results])

def get_ocr_image(image):
    ocr_result = reader.readtext(np.array(image))
    ocr_result = combine_easyocr_text_ordered(ocr_result).strip().lower()
    return ocr_result


def smart_fuzzy_brand_match(ocr_text, brand_list, base_threshold=70):
    clean_ocr = ocr_text.strip()
    is_short = len(clean_ocr) < 4

    threshold = base_threshold + 15 if is_short else base_threshold

    match, score, _ = process.extractOne(clean_ocr, brand_list, scorer=fuzz.WRatio)
    
    # if is_short and clean_ocr.lower() not in [b.lower() for b in brand_list]:
    #     return "UNKNOWN", 0

    return (match, score) if score >= threshold else ("UNKNOWN", score)

def robust_fuzzy_match(ocr_text: str, db_text: str, base_threshold=75, min_text_len=3) -> str:
    ocr_clean = ocr_text.strip()
    if len(ocr_clean) < min_text_len:
        return "UNKNOWN"

    score = fuzz.WRatio(ocr_clean, db_text)
    partial_score = fuzz.partial_ratio(ocr_clean, db_text)

    if len(ocr_clean) <= 3 and score < base_threshold + 10:
        return "UNKNOWN"

    if score >= base_threshold and partial_score >= 80:
        return db_text

    return "UNKNOWN"