import json
import numpy as np
import pandas as pd
import random
import torch

from dataclasses import dataclass, field
from datasets import load_dataset, Audio, load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from typing import Any, Dict, List, Optional, Union

class CommonVoice():
    def __init__(self, data_files, processor):
        self.dataset = load_dataset("csv", data_files=data_files)["train"]
        self.processor = processor
        self.make_dataset()

    def remove_columns(self):
        self.dataset = self.dataset.remove_columns(["client_id", "up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"])
    
    def expand_filename(self):
        def add_filename_prefix(batch):
            batch["path"] = "cv-corpus-10.0-2022-07-04/nan-tw/clips/" + batch["path"]
            return batch
        self.dataset = self.dataset.map(add_filename_prefix)
    
    def make_audio(self):
        self.dataset = self.dataset.cast_column("path", Audio(sampling_rate=16000))
    
    def make_model_input(self):
        def tokenize_audio_and_label(batch):
            audio = batch["path"]

            # batched output is "un-batched"
            batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            batch["input_length"] = len(batch["input_values"])
    
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["tailo"]).input_ids
            return batch
        self.dataset = self.dataset.map(tokenize_audio_and_label, remove_columns=self.dataset.column_names)
    
    def make_dataset(self):
        self.remove_columns()
        self.expand_filename()
        self.make_audio()
        self.make_model_input()