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

from corpus import CommonVoice, Recordings
from data_collator import DataCollatorCTCWithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wer(train_dataset, test_dataset, model, processor, data_collator):
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

    output_dir = "wav2vec2-large-xls-r-300m-mn-colab-eval"
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        gradient_checkpointing=True,
        fp16=True,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
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
    trainer.evaluate()

def inference(test_dataset, model, processor):
    input_dict = processor(test_dataset[0]["input_values"], return_tensors="pt", padding=True)
    logits = model(input_dict.input_values.to("cuda")).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]

    print("Prediction:")
    print(processor.decode(pred_ids))

def main():
    processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-xls-r-300m-mn-colab/checkpoint-6100")
    model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xls-r-300m-mn-colab/checkpoint-6100").to("cuda")
    # common_voice_train = CommonVoice(["cv-corpus-10.0-2022-07-04/nan-tw/train-processed.csv"], processor).dataset
    # common_voice_test = CommonVoice(["cv-corpus-10.0-2022-07-04/nan-tw/test-processed.csv"], processor).dataset
    # data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    # wer(common_voice_train, common_voice_test, model, processor, data_collator)
    jiyun_recordings = Recordings(["jiyun-corpus/test-processed.csv"], processor, "jiyun").dataset
    inference(jiyun_recordings, model, processor)

if __name__ == "__main__":
    main()