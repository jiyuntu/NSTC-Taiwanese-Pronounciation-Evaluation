import re

from datasets import load_dataset, Audio

class Corpus():
    def __init__(self, data_files, processor):
        self.dataset = load_dataset("csv", data_files=data_files)["train"]
        self.processor = processor
        self.make_dataset()

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
        self.make_audio()
        self.make_model_input()

class CommonVoice(Corpus):
    pass

class Recordings(Corpus):
    pass

class Suisiann(Corpus):
    pass

class TAT(Corpus):
    pass