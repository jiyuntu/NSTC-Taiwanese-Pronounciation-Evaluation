import re

from datasets import load_dataset, Audio

class Corpus():
    def __init__(self, data_files, processor):
        self.dataset = load_dataset("csv", data_files=data_files)["train"]
        self.processor = processor
        self.make_dataset()

    def is_valid_tailo(self, tailo):
        return all(re.match(r"^[a-zA-Z]+[0-9]$", w) for w in tailo.split(" "))

    def remove_tone_instances(self):
        self.dataset = self.dataset.filter(lambda example: self.is_valid_tailo(example["tailo"]))

    def remove_columns(self):
        self.dataset = self.dataset.remove_columns([c for c in self.dataset.column_names if c not in ["path", "tailo"]])

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
        self.dataset = self.dataset.map(tokenize_audio_and_label, remove_columns=self.dataset.column_names,
                                        keep_in_memory=True)

    def clean_up_cache_files(self):
        self.dataset.cleanup_cache_files()

    def make_dataset(self):
        self.remove_columns()
        self.remove_tone_instances()
        self.make_audio()
        self.make_model_input()
        self.clean_up_cache_files()

class CommonVoice(Corpus):
    pass

class Recordings(Corpus):
    def __init__(self, data_files, processor, name):
        self.name = name
        super().__init__(data_files, processor)

    def expand_filename(self):
        def add_filename_prefix(batch):
            batch["path"] = f"{self.name}-corpus/clips/" + batch["path"]
            return batch
        self.dataset = self.dataset.map(add_filename_prefix)

class Suisiann(Corpus):
    def remove_long_audios(self):
        # avoid OUT OF CUDA error
        self.dataset = self.dataset.filter(lambda example: float(example["length"]) <= 20.)

    def make_dataset(self):
        self.remove_long_audios()
        super().make_dataset()

class TAT(Corpus):
    pass