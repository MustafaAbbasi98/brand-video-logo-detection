from utils.shot_detection import ShotDetector
from models.detector_model import LogoDetector
from models.description_model import Qwen2_5VLModel, Qwen2_5TextModel
from filtering.llm_filtering import process_image, LLMResult
from filtering.clip_filtering import CLIPModel, positive_prompts, negative_prompts, cluster_hard_cases, detect_brands_clip
from filtering.heuristic_filtering import FilterPipeline, area_aspect_filter, texture_filter, edge_density_filter, color_variance_filter
from tqdm import tqdm
import numpy as np
import pandas as pd
import getpass
import math
from typing import List
from PIL import Image
import os
import re
from utils.prompts import QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT, QWEN_TRANSCRIPT_LOGO_PROMPT
from models.audio_model import WhisperModel, convert_video_to_audio
from utils.fuzzy_matching import get_ocr_image, smart_fuzzy_brand_match, smart_fuzzy_brand_match_batch
from pipelines.brand_recognition_chain import BrandRecognizer, LogoImage, OcrFuzzyTechnique, OcrFuzzyBatchTechnique, ClipTechnique, DatabaseFaissTechnique, DatabaseFaissTechniqueBatch
from models.faiss_db import LogoDatabase, LogoDatabaseNew
# import albumentations as A





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
    # print(transcript)
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


# def process_video(video_path, threshold, use_db=False):
#     print('use_db', use_db)
#     shot_detector = ShotDetector(video_path)
#     detector  = LogoDetector()
#     clip_model = CLIPModel()
#     filter_pipeline = FilterPipeline([
#         area_aspect_filter,
#         texture_filter,
#         edge_density_filter,
#         color_variance_filter,
#     ])

#     # qwen_model = Qwen2_5VLModel()

#     db_index_path =  "D:\\milestone 2\\faiss_db\\logo_index.faiss"
#     metadata_path = "D:\\milestone 2\\faiss_db\\metadata.json"
#     db = LogoDatabaseNew(db_index_path, metadata_path, batch_size=64)

#     # audio_brands = extract_audio_brands(video_path)
#     audio_brands = ['sky', 'neoborocillina', 'storeroom', 'merluzzo', 'nurofen', 'monifarma', 'moneypharm', 'verishure', 'verisure', 'barbie', 'chanted evening', 'orogel', 'rai sport', 'rai 2', 'guinness world records']
#     # print(audio_brands)
    
#     #Video Shot Detection
#     frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
#     print('Total Frames:', len(frames))

#     #Object detection on all frames
#     detector_results = detector.process_images(frames, threshold=0.03, nms_threshold=0.3, batch_size=16)
#     # print(len(detector_results))
#     # for result in detector_results:
#     #     print(len(result['logos']))
    
#     frame_results = []
#     for result, timestamp, frame_no in tqdm(zip(detector_results, timestamps, frame_numbers)):
        
#         logos, scores = result['logos'], result['scores']
#         print("Total logos in frame: ", len(logos))
#         if len(logos) <= 0:
#             continue

#         #Create LogoImage objects
#         logo_images = []
#         for idx, (logo, score) in enumerate(zip(logos, scores)):
#             metadata = {
#                 'frame_number': frame_no,
#                 'timestamp': timestamp,
#                 'logo_index': idx,
#                 'score': score,
#             }
#             logo_image = LogoImage.create(logo, metadata)
#             logo_images.append(logo_image)
        
#         #Filter out easy examples using CLIP
#         if len(logo_images) >= 10:
#             # logo_images = filter_logos_with_clip(logo_images, clip_model, positive_prompts, negative_prompts)
#             logo_images = filter_pipeline.apply(logo_images)
#             print(f"Filtered logos kept for frame {frame_no}: {len(logo_images)}")
            
#         if len(logo_images) == 0:
#             continue
    
#         # Then run recognizer (same as earlier)
#         techniques = [
#             ClipTechnique(clip_model, clip_threshold=0.8),
#             OcrFuzzyBatchTechnique(get_ocr_image, smart_fuzzy_brand_match_batch, audio_brands, fuzzy_threshold=0.8),
#             # OcrFuzzyTechnique(get_ocr_image, smart_fuzzy_brand_match, audio_brands, fuzzy_threshold=0.8),
#             # DatabaseFaissTechnique(db, get_ocr_image),
#         ]

#         if use_db:
#             techniques.append(DatabaseFaissTechniqueBatch(db, get_ocr_image))

#         recognizer = BrandRecognizer(techniques)
#         frame_recognition_results = recognizer.recognize(logo_images)

#         frame_results.append({
#             'frame_number': frame_no,
#             'timestamp': timestamp,
#             'results': frame_recognition_results
#         })

#     print(frame_results)
#     return frame_results

def process_video(video_path, threshold, use_db=False):
    print('use_db', use_db)
    # --- Prepare output dirs & CSV accumulator ---
    base = os.path.splitext(os.path.basename(video_path))[0]
    sanitized = re.sub(r'[^A-Za-z0-9_\-]', '_', base)
    unknown_dir = os.path.join(sanitized, "unknowns")
    known_dir   = os.path.join(sanitized, "knowns")       
    os.makedirs(unknown_dir, exist_ok=True)
    os.makedirs(known_dir, exist_ok=True)         

    detected_rows = []  # will collect rows for CSV
    frame_results = []

    # --- Initialize your components ---
    shot_detector   = ShotDetector(video_path)
    detector        = LogoDetector()
    clip_model      = CLIPModel()

    #Heuristic-based filtering of false positives
    filter_pipeline = FilterPipeline([
        area_aspect_filter,
        texture_filter,
        edge_density_filter,
        color_variance_filter,
    ])

    db = LogoDatabaseNew(
        index_path="D:\\milestone 2\\faiss_database_with_italian_logos\\logo_index.faiss",
        metadata_path="D:\\milestone 2\\faiss_database_with_italian_logos\\metadata.json",
        batch_size=64,
        device='cpu', #Remove this if you want to use GPU
    )

    audio_brands = extract_audio_brands(video_path)
    # audio_brands = [
    #     'sky','neoborocillina','storeroom','merluzzo','nurofen',
    #     'monifarma','moneypharm','verishure','verisure','barbie',
    #     'chanted evening','orogel','rai sport','rai 2','guinness world records'
    # ]
    print(audio_brands)

    # 1) Shot detection → get frames, numbers, timestamps
    frames, frame_numbers, timestamps = shot_detector.process_video(quantile=threshold)
    print('Total Frames:', len(frames))

    # 2) Logo detection on all frames (batched)
    detector_results = detector.process_images(
        frames,
        threshold=0.03,
        nms_threshold=0.3,
        batch_size=16
    )

    # 3) Per-frame loop
    for frame, result, timestamp, frame_no in tqdm(
        zip(frames, detector_results, timestamps, frame_numbers),
        total=len(detector_results),
        desc="Processing frames"
    ):
        logos, scores = result['logos'], result['scores']
        if not logos:
            continue

        # Compute frame area once
        if isinstance(frame, Image.Image):
            fw, fh = frame.size
        else:  # assume numpy array HxWxC
            fh, fw = frame.shape[:2]
        frame_area = fw * fh

        # 3a) Build LogoImage list *with* percent_area in metadata
        logo_images = []
        for idx, (crop_img, det_score) in enumerate(zip(logos, scores)):
            w, h = (crop_img.size if isinstance(crop_img, Image.Image)
                    else (crop_img.shape[1], crop_img.shape[0]))
            percent_area = (w * h) / frame_area * 100.0

            metadata = {
                'frame_number':  frame_no,
                'timestamp':     timestamp,
                'logo_index':    idx,
                'detection_score': det_score,
                'percent_area':  percent_area,
            }
            logo_images.append(LogoImage.create(crop_img, metadata))

        # 3b) Optional filtering
        if len(logo_images) >= 10:
            logo_images = filter_pipeline.apply(logo_images)

        if not logo_images or len(logo_images)==0:
            continue

        # 3c) Recognition via your stacked techniques
        techniques = [
            ClipTechnique(clip_model, clip_threshold=0.8),
            OcrFuzzyBatchTechnique(get_ocr_image, smart_fuzzy_brand_match_batch, audio_brands, fuzzy_threshold=0.8),
        ]
        if use_db:
            techniques.append(DatabaseFaissTechniqueBatch(db, get_ocr_image))

        recognizer = BrandRecognizer(techniques)
        recs = recognizer.recognize(logo_images)

        # 3d) Enrich recs with percent_area, save unknowns & collect CSV rows
        for logo in logo_images:
            info = recs[logo.id]
            # add percent_area to the result
            info['percent_area'] = logo.metadata['percent_area']

            if info['brand'] == 'UNKNOWN':
                # save unknown crop
                fname = f"frame{frame_no}_logo{logo.metadata['logo_index']}.png"
                logo.image.save(os.path.join(unknown_dir, fname))
            else:
                # save known crop
                fname = f"{logo.id}.png"                             # ← new
                logo.image.save(os.path.join(known_dir, fname))      # ← new
                # queue for CSV
                detected_rows.append({
                    'frame_number':  logo.metadata['frame_number'],
                    'timestamp':     logo.metadata['timestamp'],
                    'brand':         info['brand'],
                    'percent_area':  logo.metadata['percent_area'],
                    'filename':     fname,
                })

        # 3e) Append to overall frame_results
        frame_results.append({
            'frame_number': frame_no,
            'timestamp':    timestamp,
            'results':      recs
        })

    # 4) Export known detections to CSV
    if detected_rows:
        df = pd.DataFrame(detected_rows)
        # csv_path = os.path.join(sanitized, f"{sanitized}_detected_logos.csv")
        csv_path = os.path.join(known_dir, f"{sanitized}_detected_logos.csv")
        df.to_csv(csv_path, index=False)
        print(f"▶ Exported {len(df)} detections to {csv_path}")
    
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
        