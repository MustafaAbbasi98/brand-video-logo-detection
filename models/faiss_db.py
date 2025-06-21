import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import faiss
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import pandas as pd
import torch
# from rapidfuzz import process, fuzz
# import easyocr
import re

def apply_augmentation(image: Image.Image, augmenter) -> Image.Image:
    """
    Apply an Albumentations augmenter to a PIL image.

    Args:
        image: PIL Image
        augmenter: Albumentations Compose or similar callable

    Returns:
        Augmented PIL Image
    """
    img_np = np.array(image)
    augmented = augmenter(image=img_np)['image']
    return Image.fromarray(augmented)



class LogoDatabase:
    """
    A production-ready logo similarity search system with batch processing.
    
    Features:
    - Batch embedding generation with configurable size
    - Direct PIL image input/output
    - Optimized metadata storage
    - Efficient duplicate checking
    - GPU acceleration support
    """
    
    def __init__(self, 
                 index_path: str = "logo_index.faiss",
                 metadata_path: str = "metadata.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        """
        Initialize the logo database.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load brand metadata
            device: Compute device for embedding model
            batch_size: Number of images to process simultaneously
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size
        
        # Initialize DINOv2 model
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large', use_fast=True)
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

        # Initialize CLIP model and processor (using the large variant)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self._init_index()

    def _init_index(self):
        """Initialize or load existing FAISS index"""

        # combined_dim = 1024 + 768
        combined_dim=768
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(combined_dim)  # DINOv2-large dimension
            self.metadata = []
        
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate DINOv2 embeddings for a batch of images.
        
        Args:
            images: List of PIL Image objects (RGB format)
            
        Returns:
            Normalized embedding vectors (batch_size x 1024)
        """
        try:
            # Process images through DINOv2
            # dino_inputs = self.dino_processor(images=images, return_tensors="pt").to(self.device)
            # with torch.no_grad():
            #     dino_outputs = self.dino_model(**dino_inputs)
            # # Obtain DINOv2 embeddings by averaging over tokens
            # dino_embeddings = dino_outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, 1024)
            
            # Process images through CLIP
            clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                clip_embeddings = self.clip_model.get_image_features(**clip_inputs)  # Shape: (batch_size, 768)
            
            # Combine the embeddings via concatenation
            # combined_embeddings = torch.cat([dino_embeddings, clip_embeddings], dim=1)  # Shape: (batch_size, 1792)
            combined_embeddings = clip_embeddings
            
            # Normalize the combined embeddings
            combined_embeddings = combined_embeddings.cpu().numpy()
            norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
            combined_embeddings_normalized = combined_embeddings / norms
            return combined_embeddings_normalized
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return np.array([])
        

    def add_logos(self, 
                 images: List[Image.Image], 
                 brand_names: List[str],
                 duplicate_threshold: float = 0.95):
        """
        Add new logos to the database with batch processing.
        
        Args:
            images: Iterable of PIL Image objects (RGB format)
            brand_names: Corresponding brand names
            duplicate_threshold: Similarity threshold for duplicate detection
        """
        images = list(images)
        if len(images) != len(brand_names):
            raise ValueError("Number of images and brand names must match")
            
        total_added = 0
        for i in tqdm(range(0, len(images), self.batch_size)):
            batch_images = images[i:i+self.batch_size]
            batch_brands = brand_names[i:i+self.batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self._embed_batch(batch_images)
            if batch_embeddings.size == 0:
                continue
                
            # Check for duplicates in existing database
            # if self.index.ntotal > 0:
            #     # Find nearest existing neighbors
            #     distances, _ = self.index.search(batch_embeddings, 1)
            #     similarities = 1 - distances.flatten() / 4
                
            #     # Mask for new entries
            #     mask = similarities < duplicate_threshold
            #     batch_embeddings = batch_embeddings[mask]
            #     batch_brands = [b for b, m in zip(batch_brands, mask) if m]
                
            # Add valid entries to index and metadata
            if batch_embeddings.size > 0:
                self.index.add(batch_embeddings.astype('float32'))
                self.metadata.extend(batch_brands)
                total_added += len(batch_brands)
                
        print(f"Added {total_added} new logos (skipped {len(images)-total_added} duplicates)")

    def search_logo(self, 
                   query_image: Image.Image, 
                   threshold: float = 0.85,
                   k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar logos in the database.
        
        Args:
            query_image: PIL Image object (RGB format)
            threshold: Minimum similarity score (0-1)
            k: Number of neighbors to retrieve
            
        Returns:
            List of (brand_name, similarity_score) tuples
        """
        # Process single image as batch of 1
        batch_embed = self._embed_batch([query_image])
        if batch_embed.size == 0:
            return []
            
        query_embed = batch_embed[0].astype('float32')
        distances, indices = self.index.search(np.expand_dims(query_embed, 0), k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist / 4  # Convert L2 to cosine-like similarity
            if similarity >= threshold and i < len(self.metadata):
                results.append({'index': i, 
                                'brand_name': self.metadata[i], 
                                'similarity': similarity
                               })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    def save(self):
        """Persist the index and metadata to disk"""
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def __len__(self):
        return len(self.metadata)

    @property
    def total_logos(self):
        return len(self.metadata)
    
    
class LogoDatabaseNew:
    """
    A production-ready logo similarity search system with batch processing.
    
    Features:
    - Batch embedding generation with configurable size
    - Direct PIL image input/output
    - Optimized metadata storage
    - Efficient duplicate checking
    - GPU acceleration support
    """
    
    def __init__(self, 
                 index_path: str = "logo_index.faiss",
                 metadata_path: str = "metadata.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        """
        Initialize the logo database.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load brand metadata
            device: Compute device for embedding model
            batch_size: Number of images to process simultaneously
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.device = device
        self.batch_size = batch_size
        
        # Initialize DINOv2 model
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

        # Initialize CLIP model and processor (using the large variant)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self._init_index()

    def _init_index(self):
        """Initialize or load existing FAISS index"""

        # combined_dim = 1024 + 768
        combined_dim=768
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(combined_dim)  # DINOv2-large dimension
            self.metadata = []
        
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def _embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate DINOv2 embeddings for a batch of images.
        
        Args:
            images: List of PIL Image objects (RGB format)
            
        Returns:
            Normalized embedding vectors (batch_size x 1024)
        """
        try:
            # Process images through DINOv2
            # dino_inputs = self.dino_processor(images=images, return_tensors="pt").to(self.device)
            # with torch.no_grad():
            #     dino_outputs = self.dino_model(**dino_inputs)
            # # Obtain DINOv2 embeddings by averaging over tokens
            # dino_embeddings = dino_outputs.last_hidden_state.mean(dim=1)  # Shape: (batch_size, 1024)
            
            # Process images through CLIP
            clip_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                clip_embeddings = self.clip_model.get_image_features(**clip_inputs)  # Shape: (batch_size, 768)
            
            # Combine the embeddings via concatenation
            # combined_embeddings = torch.cat([dino_embeddings, clip_embeddings], dim=1)  # Shape: (batch_size, 1792)
            combined_embeddings = clip_embeddings
            
            # Normalize the combined embeddings
            combined_embeddings = combined_embeddings.cpu().numpy()
            norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
            combined_embeddings_normalized = combined_embeddings / norms
            return combined_embeddings_normalized
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return np.array([])

    def add_logos(self, 
                  images: List[Image.Image], 
                  brand_names: List[str],
                  duplicate_threshold: float = 0.95,
                  augmentations = None,
                  num_augments: int = 10):
        """
        Add new logos to the database with optional augmentations.
    
        Args:
            images: List of PIL Image objects (RGB)
            brand_names: Corresponding brand names
            duplicate_threshold: Optional threshold for deduplication (unused currently)
            augmentations: Albumentations Compose object or similar callable
            num_augments: Number of augmentations to generate per image (if augmentations are provided)
        """
        if len(images) != len(brand_names):
            raise ValueError("Number of images and brand names must match")
    
        all_images = []
        all_brands = []
    
        for img, brand in zip(images, brand_names):
            all_images.append(img)
            all_brands.append(brand)
    
            # Apply augmentations if requested to store multiple variants along with original image
            if augmentations:
                for _ in range(num_augments):
                    aug_img = apply_augmentation(img, augmentations)
                    all_images.append(aug_img)
                    all_brands.append(brand)
    
        total_added = 0
        for i in tqdm(range(0, len(all_images), self.batch_size)):
            batch_images = all_images[i:i + self.batch_size]
            batch_brands = all_brands[i:i + self.batch_size]
    
            # Generate embeddings
            batch_embeddings = self._embed_batch(batch_images)
            if batch_embeddings.size == 0:
                continue
    
            # Add to index + metadata
            self.index.add(batch_embeddings.astype('float32'))
            self.metadata.extend(batch_brands)
            total_added += len(batch_brands)
    
        print(f"Added {total_added} new logos including augmentations.")



    def search_logo(self, 
                    query_image: Image.Image, 
                    threshold: float = 0.85,
                    k: int = 5,
                    augmentations = None,
                    num_augments: int = 5):
        """
        Search for similar logos in the database using optional test-time augmentation.

        Args:
            query_image: PIL Image (RGB)
            threshold: Minimum similarity score (0–1)
            k: Number of neighbors to retrieve
            augmentations: Albumentations Compose object (optional)
            num_augments: Number of augmented versions to include (if augmentations are provided)
    
        Returns:
            List of dictionaries with keys: 'index', 'brand_name', 'similarity'
        """
        # Build list of query images (original + augmentations)
        query_images = [query_image]

        #Apply test-time augmentations
        if augmentations:
            img_np = np.array(query_image)
            for _ in range(num_augments):
                aug_np = augmentations(image=img_np)['image']
                query_images.append(Image.fromarray(aug_np))
    
        # Generate embeddings in batch
        embeddings = self._embed_batch(query_images)
        if embeddings.size == 0:
            return []
    
        # Normalize each embedding, then average them, then re-normalize the average embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        avg_embedding = np.mean(normalized, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)
    
        # FAISS search
        distances, indices = self.index.search(np.expand_dims(avg_embedding.astype('float32'), 0), k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist / 4
            if similarity >= threshold and i < len(self.metadata):
                results.append({
                    'index': i,
                    'brand_name': self.metadata[i],
                    'similarity': similarity
                })
    
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    def search_logos(self,
                     images: List[Image.Image],
                     threshold: float = 0.85,
                     k: int = 5,
                     batch_size: int = 64,
                     augmentations = None,
                     num_augments: int = 5):
        
        """
        Perform batched similarity search for a list of logos with optional test-time augmentation.
    
        Args:
            images: List of PIL Image objects (RGB)
            threshold: Minimum similarity score (0 to 1)
            k: Number of nearest neighbors to retrieve per image
            batch_size: Number of image variants to embed per batch
            augmentations: Albumentations Compose object (optional)
            num_augments: Number of augmentations per image if augmentations provided
    
        Returns:
            List of results (one per original image). Each result is a list of dictionaries.
        """
        
        all_results = []
    
        for img in images:
            # Test-time augmentation Prepare augmented variants for this particular image
            variants = [img]
            if augmentations:
                img_np = np.array(img)
                for _ in range(num_augments):
                    aug_np = augmentations(image=img_np)['image']
                    variants.append(Image.fromarray(aug_np))
    
            # Embed all variants in batch with original
            emb = self._embed_batch(variants)
            if emb.size == 0:
                all_results.append([])
                continue
    
            # Normalize and average, then re-normalize average embedding
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            avg_emb = np.mean(emb, axis=0)
            avg_emb /= np.linalg.norm(avg_emb)
    
            # Search FAISS
            distances, indices = self.index.search(np.expand_dims(avg_emb.astype('float32'), 0), k)
            
            image_results = []
            for i, dist in zip(indices[0], distances[0]):
                similarity = 1 - dist / 4
                if similarity >= threshold and i < len(self.metadata):
                    image_results.append({
                        'index': i,
                        'brand_name': self.metadata[i],
                        'similarity': similarity
                    })
            
            all_results.append(sorted(image_results, key=lambda x: x['similarity'], reverse=True))
    
        return all_results



    def save(self):
        """Persist the index and metadata to disk"""
        index = self.index
        if self.device=='cuda':
            index = faiss.index_gpu_to_cpu(index)
            
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def __len__(self):
        return len(self.metadata)

    @property
    def total_logos(self):
        return len(self.metadata)
    



if __name__ == '__main__':
    index_path = "D:\\milestone 2\\faiss_db\\logo_index.faiss"
    metadata_path = "D:\\milestone 2\\faiss_db\\metadata.json"

    db = LogoDatabase(index_path, metadata_path, device='cpu', batch_size=32)


    logos_path = Path("D:\\milestone 2\\faiss_db\\logo_for_database\\logo_for_database")

    logo = Image.open(logos_path/'frosta'/'1.png').convert('RGB')

    result = db.search_logo(logo,
                            threshold=0.85,
                            k=5)
    
    print(result[0])