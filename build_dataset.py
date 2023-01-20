"""
Build the dataset outside the training script because we need to
manually transform the phonetic alphabet. Seems like 臺灣言語工具
skips some of the words, and I'm not sure why.
"""

import re

from datasets import load_dataset

from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音

class BuildCsv():
    def __init__(self, train_data_files, test_data_files, train_output_file, test_output_file):
        self.train_dataset = load_dataset("csv", data_files=train_data_files)["train"]
        self.test_dataset = load_dataset("csv", data_files=test_data_files)["train"]
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file

    def remove_punctuation_marks(self):
        def replace_with_space(batch):
            chars_to_remove_regex = "[\,\?\.\!\;\:\"\“\%\‘\”\'\-]"
            batch["tailo"] = re.sub(chars_to_remove_regex, " ", batch["tailo"]).lower()
            # multiple spaces to one
            tailo_list = batch["tailo"].split(" ")
            batch["tailo"] = " ".join([w for w in tailo_list if w]).strip()
            return batch
        self.train_dataset = self.train_dataset.map(replace_with_space)
        self.test_dataset = self.test_dataset.map(replace_with_space)

    def to_csv(self):
        self.train_dataset.to_csv(self.train_output_file, index=False)
        self.test_dataset.to_csv(self.test_output_file, index=False)

    def tailo_transform(self, 台羅):
        台羅物件 = 拆文分析器.建立句物件(台羅)
        return 台羅物件.轉音(臺灣閩南語羅馬字拼音).看語句()

    def is_valid_tailo(self, tailo):
        # str isalnum is different from Alphabetic property
        # https://docs.python.org/3/library/stdtypes.html#str.isalpha
        return tailo.replace(" ", "").encode("utf-8").isalnum()

    def remove_tone_instances(self):
        self.train_dataset = self.train_dataset.filter(lambda example: self.is_valid_tailo(example["tailo"]))
        self.test_dataset = self.test_dataset.filter(lambda example: self.is_valid_tailo(example["tailo"]))

class BuildCsvFromCommonVoice(BuildCsv):
    def __init__(self):
        super().__init__(["cv-corpus-10.0-2022-07-04/nan-tw/train.csv", "cv-corpus-10.0-2022-07-04/nan-tw/dev.csv"],
                         ["cv-corpus-10.0-2022-07-04/nan-tw/test.csv"],
                         "cv-corpus-10.0-2022-07-04/nan-tw/train-processed-2.csv",
                         "cv-corpus-10.0-2022-07-04/nan-tw/test-processed-2.csv")

    def tailo(self):
        # 台羅 to 台羅數字調
        # Some tone could not be transderred
        def transform(batch):
            begin = batch["sentence"].find("（")
            # Try to find ｜ but often fail.
            end = batch["sentence"].find("）")
            batch["tailo"] = self.tailo_transform(batch["sentence"][begin + 1:end].strip())
            if "|" in batch["tailo"]:
                batch["tailo"] = batch["tailo"].split("|")[0]
            return batch
        self.train_dataset = self.train_dataset.filter(lambda example: example["sentence"].find("（") >= 0)
        self.test_dataset = self.test_dataset.filter(lambda example: example["sentence"].find("（") >= 0)

        self.train_dataset = self.train_dataset.map(transform)
        self.test_dataset = self.test_dataset.map(transform)

    def build(self):
        self.tailo()
        self.remove_punctuation_marks()
        self.remove_tone_instances()
        self.to_csv()

class BuildCsvFromSuisiann(BuildCsv):
    def __init__(self):
        super().__init__(["Suisiann/SuiSiann.csv"], ["Suisiann/SuiSiann.csv"],
                          "Suisiann/SuiSiann-processed.csv", "Suisiann/SuiSiann-processed-2.csv")

    def expand_filename(self):
        def add_filename_prefix(batch):
            batch["path"] = f"Suisiann/" + batch["path"]
            return batch
        self.train_dataset = self.train_dataset.map(add_filename_prefix)
        self.test_dataset = self.test_dataset.map(add_filename_prefix)

    def tailo(self):
        def transform(batch):
            batch["tailo"] = self.tailo_transform(batch["Rome"])
            return batch
        self.train_dataset = self.train_dataset.map(transform)
        self.test_dataset = self.test_dataset.map(transform)

    def build(self):
        self.tailo()
        self.remove_punctuation_marks()
        self.remove_tone_instances()
        self.expand_filename()
        self.to_csv()

if __name__ == "__main__":
    # build_common_voice = BuildCsvFromCommonVoice()
    # build_common_voice.build()

    build_suisiann = BuildCsvFromSuisiann()
    build_suisiann.build()
