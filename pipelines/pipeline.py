from utils.shot_detection import ShotDetector
from models.detector_model import LogoDetector
from models.description_model import Qwen2_5VLModel, Qwen2_5TextModel
from filtering.llm_filtering import process_image, LLMResult
from filtering.clip_filtering import CLIPModel, positive_prompts, negative_prompts, cluster_hard_cases, detect_brands_clip
from tqdm import tqdm
import numpy as np
import pandas as pd
import getpass
import math
from typing import List
from PIL import Image
import os
from utils.prompts import QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT, QWEN_TRANSCRIPT_LOGO_PROMPT
from models.audio_model import WhisperModel, convert_video_to_audio
from utils.fuzzy_matching import get_ocr_image, smart_fuzzy_brand_match
from brand_recognition_chain import BrandRecognizer, LogoImage, OcrFuzzyTechnique, ClipTechnique, DatabaseFaissTechnique, DatabaseFaissParallelTechnique
from models.faiss_db import LogoDatabase

def resize_images(images):
    """
    Resize each PIL image in the list if its width or height is less than 28 pixels.
    The image is scaled up uniformly so that both dimensions are at least 28 pixels,
    preserving the original aspect ratio.

    Parameters:
        images (list): A list of PIL.Image objects.
    
    Returns:
        list: A new list of PIL.Image objects with images resized if necessary.
    """
    resized = []
    for img in images:
        width, height = img.size
        if width < 28 or height < 28:
            # Calculate the scale factor to ensure both dimensions are at least 28.
            scale = max(28 / width, 28 / height)
            new_size = (math.ceil(width * scale), math.ceil(height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        resized.append(img)
    return resized

def extract_audio_brands(video_path):
    qwen_text_model = Qwen2_5TextModel()
    whisper = WhisperModel()
    #convert video to audio file
    convert_video_to_audio(video_path)

    #transcript extraction
    transcript, _ = whisper.run_example('logo_audio.wav')
    brand_string = qwen_text_model.run_example(
        QWEN_TRANSCRIPT_LOGO_PROMPT(transcript),
        QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT
    )
    audio_brands = brand_string.split('\n')
    audio_brands = [name.lower().strip() for name in audio_brands]
    return audio_brands

def filter_logos_with_clip(logo_images: List[LogoImage], clip_model, positive_prompts: List[str], negative_prompts: List[str]) -> List[LogoImage]:
    """
    Filters out noisy logos using CLIP model and clustering.
    
    Parameters:
        logo_images: List of LogoImage objects.
        clip_model: CLIP model instance to run.
        positive_prompts: List of positive prompts for CLIP filtering.
        negative_prompts: List of negative prompts for CLIP filtering.
    
    Returns:
        List of LogoImage objects after filtering.
    """
    if len(logo_images) == 0:
        return []
    
    raw_images = [li.image for li in logo_images]
    raw_scores = [li.metadata.get('score', 1.0) for li in logo_images]

    clip_filter_results_df = clip_model.process_images(
        raw_images,
        scores=raw_scores,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts
    )
    clip_filter_results_df, sorted_clusters = cluster_hard_cases(clip_filter_results_df)
    
    # if not sorted_clusters:
    #     # No clusters found, return all logos
    #     return logo_images

    exclude_cluster = sorted_clusters[0]

    # Identify logos to keep (not in the excluded cluster)
    include_logo_indices = clip_filter_results_df.loc[
        clip_filter_results_df['cluster'] != exclude_cluster, 'image_path'
    ].values
    include_logo_indices = [int(idx) for idx in include_logo_indices]

    filtered_logo_images = [logo_images[i] for i in include_logo_indices]

    return filtered_logo_images


def process_video(video_path, threshold, use_db=False):
    print('use_db', use_db)
    shot_detector = ShotDetector(video_path)
    detector  = LogoDetector()
    clip_model = CLIPModel()
    qwen_model = Qwen2_5VLModel()

    db_index_path =  "D:\\milestone 2\\faiss_db\\logo_index.faiss"
    metadata_path = "D:\\milestone 2\\faiss_db\\metadata.json"
    db = LogoDatabase(db_index_path, metadata_path, device='cpu', batch_size=32)

    audio_brands = extract_audio_brands(video_path)
    print(audio_brands)
    
    #Video Shot Detection
    frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
    print('Total Frames:', len(frames))

    #Object detection on all frames
    detector_results = detector.process_images(frames, threshold=0.02, nms_threshold=0.2, batch_size=16)
    # print(len(detector_results))
    # for result in detector_results:
    #     print(len(result['logos']))
    
    frame_results = []
    for result, timestamp, frame_no in tqdm(zip(detector_results, timestamps, frame_numbers)):
        
        logos, scores = result['logos'], result['scores']
        print("Total logos in frame: ", len(logos))
        if len(logos) <= 0:
            continue

        #Create LogoImage objects
        logo_images = []
        for idx, (logo, score) in enumerate(zip(logos, scores)):
            metadata = {
                'frame_number': frame_no,
                'timestamp': timestamp,
                'logo_index': idx,
                'score': score,
            }
            logo_image = LogoImage.create(logo, metadata)
            logo_images.append(logo_image)
        
        #Filter out easy examples using CLIP
        if len(logo_images) >= 10:
            logo_images = filter_logos_with_clip(logo_images, clip_model, positive_prompts, negative_prompts)
            print(f"Filtered logos kept for frame {frame_no}: {len(logo_images)}")
            
        if len(logo_images) == 0:
            continue
    
        # Then run recognizer (same as earlier)
        techniques = [
            ClipTechnique(clip_model, clip_threshold=0.8),
            OcrFuzzyTechnique(get_ocr_image, smart_fuzzy_brand_match, audio_brands, fuzzy_threshold=0.8),
            # DatabaseFaissTechnique(db, get_ocr_image),
        ]

        if use_db:
            techniques.append(DatabaseFaissTechnique(db, get_ocr_image))

        recognizer = BrandRecognizer(techniques)
        frame_recognition_results = recognizer.recognize(logo_images)

        frame_results.append({
            'frame_number': frame_no,
            'timestamp': timestamp,
            'results': frame_recognition_results
        })

    print(frame_results)
    return frame_results



def process_frame_results_merged(frame_results, default_end_offset: float = 2.0) -> pd.DataFrame:
    """
    Processes frame_results by merging multiple detections of the same brand across frames.

    Parameters:
        frame_results: List of frame-wise recognition results.
        default_end_offset: Default duration if no end detected.

    Returns:
        A pandas DataFrame with columns: ['logo', 'brand_name', 'start_timestamp', 'end_timestamp']
    """
    brand_records = {}

    for frame_result in frame_results:
        frame_timestamp = frame_result['timestamp']
        frame_items = frame_result['results']

        for logo_id, data in frame_items.items():
            brand = data['brand']
            if brand == 'UNKNOWN':
                continue  # Skip unknowns

            if brand not in brand_records:
                # First time seeing this brand
                brand_records[brand] = {
                    'logo': data['image'],
                    'start_timestamp': frame_timestamp,
                    'end_timestamp': frame_timestamp
                }
            else:
                # Update end timestamp if brand seen again
                brand_records[brand]['end_timestamp'] = frame_timestamp

    # Convert to DataFrame
    records = []
    for brand, info in brand_records.items():
        records.append({
            'logo': info['logo'],
            'brand_name': brand,
            'start_timestamp': info['start_timestamp'],
            'end_timestamp': info['end_timestamp']  # extend a little
        })

    df = pd.DataFrame(records)

    if not df.empty:
        df['brand_name'] = df['brand_name'].str.replace('\n', ' or ')
        df['info'] = "Brand Name(s): " + df['brand_name'] + ", Start Timestamp: " + df['start_timestamp'].astype(str) + ", End Timestamp: " + df['end_timestamp'].astype(str)

    return df


def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    video_path = "../videos/1.mp4"
    # stats_path= "C:/Users/Admin/Desktop/milestone 2/videos/video_2_stats.csv"
    process_video(video_path, 0.98)
    
if __name__ == "__main__":
    main()
        