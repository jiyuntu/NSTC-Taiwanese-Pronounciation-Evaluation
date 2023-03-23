import json
import numpy as np
import pandas as pd
import random
import torch

from dataclasses import dataclass, field
from datasets import load_dataset, Audio, load_metric, concatenate_datasets, disable_caching
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from typing import Any, Dict, List, Optional, Union

from build_vocab import build_vocab
from corpus import CommonVoice, Suisiann, TAT
from data_collator import DataCollatorCTCWithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

build_vocab()

# disable_caching()
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

common_voice_train = CommonVoice(["cv-corpus-10.0-2022-07-04/nan-tw/train-processed-2.csv"], processor).dataset
common_voice_test = CommonVoice(["cv-corpus-10.0-2022-07-04/nan-tw/test-processed-2.csv"], processor).dataset
# To avoid "out of memory" error, comment out these two lines
#max_input_length_in_sec = 5.0
#common_voice_train = common_voice_train.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

suisiann = Suisiann(["Suisiann/SuiSiann-processed.csv"], processor).dataset

tat_vol1_train = TAT(["TAT/TAT-Vol1-train/lavalier-processed.csv"], processor).dataset
tat_vol1_test = TAT(["TAT/TAT-Vol1-test/lavalier-processed.csv"], processor).dataset

tat_vol2_train = TAT(["TAT/TAT-Vol2-train/lavalier-processed.csv"], processor).dataset
tat_vol2_test = TAT(["TAT/TAT-Vol2-test/lavalier-processed.csv"], processor).dataset

assert common_voice_train.features.type == suisiann.features.type == tat_vol1_train.features.type == tat_vol2_train.features.type
assert common_voice_test.features.type == tat_vol1_test.features.type == tat_vol2_test.features.type
train_dataset = concatenate_datasets([common_voice_train, suisiann, tat_vol1_train, tat_vol2_train])
test_dataset = concatenate_datasets([common_voice_test, tat_vol1_test, tat_vol2_test])

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# TODO: adjust dropout, SpecAugment's masking dropout rate, layer dropout, 
# and the learning rate here.
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
model.freeze_feature_extractor()

output_dir = "wav2vec2-large-xls-r-300m-cv-suisiann-tat-vol12-lavalier-3"
training_args = TrainingArguments(
    output_dir=output_dir,
    group_by_length=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    dataloader_drop_last=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_wer",
    greater_is_better=False,
    num_train_epochs=100,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=500,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    report_to="wandb",
)
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)
trainer.train()