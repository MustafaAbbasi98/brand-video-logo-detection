import os
import json
import argparse
from typing import List, Dict
import torch
from moviepy import AudioFileClip, VideoFileClip
#import moviepy.editor as mp
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
from pathlib import Path

# class WhisperModel:
#     def __init__(self, model_id='openai/whisper-large-v3'):
#         self.model_id = model_id
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#         self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#         )
#         self.model.to(self.device)

#         self.processor = AutoProcessor.from_pretrained(model_id)

#         self.pipe = pipeline(
#             "automatic-speech-recognition",
#             model=self.model,
#             tokenizer=self.processor.tokenizer,
#             feature_extractor=self.processor.feature_extractor,
#             torch_dtype=torch_dtype,
#             device=self.device,
#             # chunk_length_s=15,
#         )

#     def run_example(self, audio_path:str, task:str = 'translate', **kwargs):
#         result = self.pipe(
#             audio_path,
#             return_timestamps=True,
#             chunk_length_s=25,
#             stride_length_s=5,
#             batch_size=16,
#             generate_kwargs = {
#                 'language': 'english',
#                 'task': task,
#                 'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
#                 'compression_ratio_threshold': 1.35,
#                 'logprob_threshold': -1.0,
#                 # 'no_speech_threshold': 0.6,
#                 'condition_on_prev_tokens': False,
#             }
#         )
#         chunks_df = pd.DataFrame(result['chunks'])
#         transcript = result['text']
#         return transcript, chunks_df

class WhisperModel:
    """
    Wrapper around HuggingFace Whisper v3 large for long-form
    transcription and translation (Italian→English by default).
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        chunk_length_s: int = 30,
        stride_length_s: int = None,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        """
        Initialize the WhisperModel.

        Args:
            model_id (str): HuggingFace model identifier.
            chunk_length_s (int): Segment length (seconds) for chunked processing.
            stride_length_s (int, optional): Overlap (seconds) between chunks.
                If None, defaults to chunk_length_s // 6.
            device (str, optional): "cuda" or "cpu". If None, auto-detects.
            dtype (torch.dtype, optional): torch.float16 or torch.float32.
                If None, uses float16 on GPU or float32 on CPU.
        """
        # 1️⃣ Determine compute device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 2️⃣ Determine tensor dtype
        self.dtype = dtype or (torch.float16 if "cuda" in self.device else torch.float32)

        # 3️⃣ Chunking parameters
        self.chunk_length_s = chunk_length_s
        # default stride = chunk_length_s / 6 for smooth overlap
        self.stride_length_s = stride_length_s if stride_length_s is not None else chunk_length_s // 6

        # 4️⃣ Load the model with low-memory options
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)

        # 5️⃣ Load processor (feature extractor + tokenizer)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # 6️⃣ Build the pipeline: chunked translation into English
        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=(self.stride_length_s, self.stride_length_s),
            batch_size=4,
            device=self.device,
            torch_dtype=self.dtype,
            return_timestamps=True,
            generate_kwargs = {
                'language': 'english',
                'task': "translate",
                'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
                'compression_ratio_threshold': 1.35,
                'logprob_threshold': -1.0,
                # 'no_speech_threshold': 0.6,
                # 'condition_on_prev_tokens': False,
            }        
        )

    def run_example(self, audio_path: str) -> str:
        """
        Transcribe and translate a long-form audio file.

        Args:
            audio_path (str): Path to the audio file (e.g., .mp3, .wav).

        Returns:
            str: The full English transcript.
        """
        # Run the pipeline; it will chunk, translate, then stitch results
        result = self.asr_pipeline(audio_path)
        chunks_df = pd.DataFrame(result['chunks'])
        transcript = result['text']
        return transcript, chunks_df
    
def convert_video_to_audio(video_path:str):
    with VideoFileClip(video_path) as video_clip:
        audio_clip = video_clip.audio
        audio_clip.write_audiofile('logo_audio.wav')


if __name__ == "__main__":
    video_path = Path('../videos/1.mp4')
    print('here')
    #convert_video_to_audio(video_path)

    audio_path = 'logo_audio.wav'

    model = WhisperModel()

    transcript, chunk_df = model.run_example(audio_path)
    print(transcript)