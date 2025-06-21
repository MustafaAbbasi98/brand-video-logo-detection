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

class WhisperModel:
    def __init__(self, model_id='openai/whisper-large-v3'):
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
            # chunk_length_s=15,
        )

    def run_example(self, audio_path:str, task:str = 'translate', **kwargs):
        result = self.pipe(
            audio_path,
            return_timestamps=True,
            chunk_length_s=25,
            stride_length_s=5,
            batch_size=16,
            generate_kwargs = {
                'language': 'english',
                'task': task,
                'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
                'compression_ratio_threshold': 1.35,
                'logprob_threshold': -1.0,
                # 'no_speech_threshold': 0.6,
                'condition_on_prev_tokens': False,
            }
        )
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