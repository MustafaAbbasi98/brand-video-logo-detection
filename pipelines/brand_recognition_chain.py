from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image
import uuid
from filtering.clip_filtering import CLIPModel, positive_prompts, negative_prompts, detect_brands_clip
import pandas as pd
import re
from utils.fuzzy_matching import combine_easyocr_text_ordered, get_ocr_image, robust_fuzzy_match
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
# --- 1. LogoImage Data Model ---

@dataclass
class LogoImage:
    id: str
    image: Image.Image
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(image: Image.Image, metadata: Dict[str, Any] = None) -> 'LogoImage':
        return LogoImage(
            id=str(uuid.uuid4()),
            image=image,
            metadata=metadata or {}
        )

# --- 2. Abstract Brand Recognition Technique ---

class BrandRecognitionTechnique(ABC):
    @abstractmethod
    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        """
        Given a list of LogoImage, returns a dict mapping image IDs to brand names (or 'UNKNOWN').
        """
        pass

# --- 3. Brand Recognizer Chain Engine ---

# class BrandRecognizer:
#     def __init__(self, techniques: List[BrandRecognitionTechnique]):
#         self.techniques = techniques

#     def recognize(self, logo_images: List[LogoImage]) -> Dict[str, Dict]:
#         remaining_images = {logo.id: logo for logo in logo_images}
#         final_results = {}

#         for technique in self.techniques:
#             if not remaining_images:
#                 break

#             predictions = technique.predict(list(remaining_images.values()))

#             for image_id, brand in predictions.items():
#                 if brand != 'UNKNOWN':
#                     logo = remaining_images[image_id]
#                     final_results[image_id] = {
#                         'brand': brand,
#                         'image': logo.image,
#                         'metadata': logo.metadata,
#                     }

#             # Only keep images still UNKNOWN
#             remaining_images = {
#                 img_id: logo for img_id, logo in remaining_images.items()
#                 if img_id not in final_results
#             }

#         # Mark any leftovers explicitly as UNKNOWN
#         for image_id, logo in remaining_images.items():
#             final_results[image_id] = {
#                 'brand': 'UNKNOWN',
#                 'image': logo.image,
#                 'metadata': logo.metadata,
#             }

#         return final_results


class BrandRecognizer:
    def __init__(self, techniques: List[BrandRecognitionTechnique]):
        self.techniques = techniques

    def recognize(self, logo_images: List[LogoImage]) -> Dict[str, Dict[str, Any]]:
        remaining_images = {logo.id: logo for logo in logo_images}
        final_results = {}

        for technique in self.techniques:
            if not remaining_images:
                break

            predictions = technique.predict(list(remaining_images.values()))

            processed = 0
            for image_id, prediction in predictions.items():
                if isinstance(prediction, str):
                    # Simple string prediction (just brand name)
                    brand = prediction
                    extra_info = {}
                elif isinstance(prediction, dict):
                    # Rich prediction with additional info
                    brand = prediction.get('brand', 'UNKNOWN')
                    extra_info = {k: v for k, v in prediction.items() if k != 'brand'}
                else:
                    raise ValueError("Prediction must be either str or dict.")

                if brand != 'UNKNOWN':
                    logo = remaining_images[image_id]
                    final_results[image_id] = {
                        'brand': brand,
                        'technique': technique.__class__.__name__,
                        'image': logo.image,
                        'metadata': logo.metadata,
                        **extra_info  # Add any extra fields like 'score', 'actual_text'
                    }
                    processed+= 1
            
            total_images = len(remaining_images)
            failed = total_images - processed

            print(f"[{technique.__class__.__name__}] Processed {processed} / {total_images} images successfully. {failed} remain unrecognized.")

                
            # Only keep logos that are still UNKNOWN
            remaining_images = {
                img_id: logo for img_id, logo in remaining_images.items()
                if img_id not in final_results
            }

        # Mark leftover images explicitly as UNKNOWN
        for image_id, logo in remaining_images.items():
            final_results[image_id] = {
                'brand': 'UNKNOWN',
                'technique': None,
                'image': logo.image,
                'metadata': logo.metadata
            }

        return final_results

# --- 4. CLIP-Based Brand Recognition Technique ---


class ClipTechnique(BrandRecognitionTechnique):
    def __init__(self, clip_model, clip_threshold: float = 0.8):
        """
        clip_model: CLIP model instance with run_examples method
        clip_threshold: Probability threshold for accepting predictions
        """
        self.clip_model = clip_model
        self.clip_threshold = clip_threshold
        self.brand_names = self._load_brand_prompts()

    def _load_brand_prompts(self) -> List[str]:
        # Load brands once during init
        top_brands_1000 = pd.read_csv("../Data/fortune1000_2024.csv")
        top_brands_2000 = pd.read_csv("../Data/Top2000CompaniesGlobally.csv")
        top_brands = pd.concat([top_brands_1000[['Company']], top_brands_2000['Company']])
        top_brands.drop_duplicates('Company', inplace=True)

        brand_names = [name + " logo" for name in top_brands["Company"].to_list()]
        brand_names += ["Other", "Not a logo"]
        return brand_names

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        results = {}

        if not logo_images:
            return results
        
        images = [li.image for li in logo_images]
        ids = [li.id for li in logo_images]

        # Run CLIP model
        clip_output = self.clip_model.run_examples(images, self.brand_names)
        logits = clip_output.logits_per_image.softmax(dim=-1)
        max_probs, max_indices = logits.max(dim=-1)

        for logo_id, prob, idx in zip(ids, max_probs, max_indices):
            if prob.item() >= self.clip_threshold:
                results[logo_id] = self.brand_names[idx]
            else:
                results[logo_id] = 'UNKNOWN'

        return results


# --- 5. OCR + Fuzzy Matching Brand Recognition Technique ---

class OcrFuzzyTechnique(BrandRecognitionTechnique):
    def __init__(self, ocr_func, fuzzy_matcher, known_brands: List[str], fuzzy_threshold: float = 0.8):
        self.ocr_func = ocr_func
        self.fuzzy_matcher = fuzzy_matcher
        self.known_brands = [b.lower().strip() for b in known_brands]
        self.fuzzy_threshold = fuzzy_threshold

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, str]:
        results = {}

        for logo in logo_images:
            ocr_text = self.ocr_func(logo.image)
            if not ocr_text or len(ocr_text) < 3:
                results[logo.id] = 'UNKNOWN'
                continue

            matched_brand, score = self.fuzzy_matcher(ocr_text, self.known_brands)
            # if score >= self.fuzzy_threshold:
            #     results[logo.id] = matched_brand
            # else:
            results[logo.id] = {
                'brand': matched_brand,
                'fuzzy_score': score,
                'ocr_text': ocr_text
            }

        return results


# --- 5. FAISS + OCR + Fuzzy Matching Brand Recognition Technique ---


class DatabaseFaissTechnique(BrandRecognitionTechnique):
    def __init__(self, db, ocr_reader, base_threshold: int = 75, faiss_threshold: float = 0.85, min_text_len: int = 3):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len

    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results = {}

        for logo in logo_images:
            search_results = self.db.search_logo(logo.image, k=5, threshold=self.faiss_threshold)
            ocr_text = self._extract_clean_text(logo.image)

            if not search_results:
                results[logo.id] = {
                    'brand': 'UNKNOWN',
                    'ocr_text': ocr_text,
                    'matched_text': None,
                    'score': 0.0
                }
                continue

            top_result = search_results[0]
            db_text = top_result['brand_name'].lower().strip()
            db_score = top_result['similarity']

            if not ocr_text and not db_text and db_score >= 0.93:
                results[logo.id] = {
                    'brand': db_text,
                    'ocr_text': '',
                    'matched_text': db_text,
                    'score': db_score
                }
            else:
                matched_text = robust_fuzzy_match(ocr_text, db_text, self.base_threshold, self.min_text_len)
                brand = matched_text if matched_text != "UNKNOWN" else "UNKNOWN"
                results[logo.id] = {
                    'brand': brand,
                    'ocr_text': ocr_text,
                    'matched_text': db_text,
                    'score': db_score
                }

        return results


class DatabaseFaissParallelTechnique(BrandRecognitionTechnique):
    def __init__(self, db, ocr_reader, base_threshold: int = 75, faiss_threshold: float = 0.85, min_text_len: int = 3, num_workers: int = None):
        self.db = db
        self.ocr_reader = ocr_reader
        self.base_threshold = base_threshold
        self.faiss_threshold = faiss_threshold
        self.min_text_len = min_text_len
        self.num_workers = num_workers if num_workers else max(1, mp.cpu_count() - 2)

    def _extract_clean_text(self, image: Image.Image) -> str:
        text = get_ocr_image(image).strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z\\s]", "", text)
        return text

    def _process_logo(self, logo: LogoImage):
        """Process a single LogoImage into a (logo_id, result_dict)"""
        search_results = self.db.search_logo(logo.image, k=5, threshold=self.faiss_threshold)
        ocr_text = self._extract_clean_text(logo.image)

        if not search_results:
            return logo.id, {
                'brand': 'UNKNOWN',
                'ocr_text': ocr_text,
                'matched_text': None,
                'score': 0.0
            }

        top_result = search_results[0]
        db_text = top_result['brand_name'].lower().strip()
        db_score = top_result['similarity']

        if not ocr_text and not db_text and db_score >= 0.93:
            return logo.id, {
                'brand': db_text,
                'ocr_text': '',
                'matched_text': db_text,
                'score': db_score
            }
        else:
            matched_text = robust_fuzzy_match(ocr_text, db_text, self.base_threshold, self.min_text_len)
            brand = matched_text if matched_text != "UNKNOWN" else "UNKNOWN"
            return logo.id, {
                'brand': brand,
                'ocr_text': ocr_text,
                'matched_text': db_text,
                'score': db_score
            }

    def predict(self, logo_images: List[LogoImage]) -> Dict[str, Any]:
        results = {}

        if not logo_images:
            return results

        with mp.Pool(self.num_workers) as pool:
            # Partial is not needed here, function has no extra args
            logo_result_pairs = pool.map(self._process_logo, logo_images)

        # Collect results
        for logo_id, result in logo_result_pairs:
            results[logo_id] = result

        return results
