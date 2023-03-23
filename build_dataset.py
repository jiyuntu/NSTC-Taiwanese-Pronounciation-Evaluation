"""
Build the dataset outside the training script because we need to
manually transform the phonetic alphabet. Seems like 臺灣言語工具
skips some of the words, and I'm not sure why.
"""
import json
import os
import pandas as pd
import re

from datasets import load_dataset

from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音

class BuildCsv():
    def __init__(self, data_files, output_file):
        self.dataset = load_dataset("csv", data_files=data_files)["train"]
        self.output_file = output_file

    def remove_columns(self):
        self.dataset = self.dataset.remove_columns([c for c in self.dataset.column_names if c not in ["path", "tailo"]])

    def remove_punctuation_marks(self):
        def replace_with_space(batch):
            chars_to_remove_regex = "[-\",.!~()*;\':?！|、，。？‘－“”﹖（）「」｢｣：；ㄅㄆㄇ⋯──《》＊—／…+/=―’\%]"
            batch["tailo"] = re.sub(chars_to_remove_regex, " ", batch["tailo"]).lower()
            # multiple spaces to one
            tailo_list = batch["tailo"].split(" ")
            batch["tailo"] = " ".join([w for w in tailo_list if w]).strip()
            return batch
        self.dataset = self.dataset.map(replace_with_space)

    def is_valid_tailo(self, tailo):
        return all(re.match(r"^[a-zA-Z]+[0-9]$", w) for w in tailo.split(" "))

    def remove_tone_instances(self):
        self.dataset = self.dataset.filter(lambda example: self.is_valid_tailo(example["tailo"]))

    def to_csv(self):
        self.dataset.to_csv(self.output_file, index=False)

    def tailo_transform(self, 台羅):
        台羅物件 = 拆文分析器.建立句物件(台羅)
        return 台羅物件.轉音(臺灣閩南語羅馬字拼音).看語句()

class BuildCsvFromCommonVoice(BuildCsv):
    def expand_filename(self):
        def add_filename_prefix(batch):
            batch["path"] = "cv-corpus-10.0-2022-07-04/nan-tw/clips/" + batch["path"]
            return batch
        self.dataset = self.dataset.map(add_filename_prefix)

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
        self.dataset = self.dataset.filter(lambda example: example["sentence"].find("（") >= 0)

        self.dataset = self.dataset.map(transform)

    def build(self):
        self.tailo()
        self.remove_punctuation_marks()
        self.remove_tone_instances()
        self.expand_filename()
        self.remove_columns()
        self.to_csv()

class BuildCsvFromCommonVoiceTrain(BuildCsvFromCommonVoice):
    def __init__(self):
        super().__init__(["cv-corpus-10.0-2022-07-04/nan-tw/train.csv", "cv-corpus-10.0-2022-07-04/nan-tw/dev.csv"], "cv-corpus-10.0-2022-07-04/nan-tw/train-processed-2.csv")

class BuildCsvFromCommonVoiceTest(BuildCsvFromCommonVoice):
    def __init__(self):
        super().__init__(["cv-corpus-10.0-2022-07-04/nan-tw/test.csv"], "cv-corpus-10.0-2022-07-04/nan-tw/test-processed-2.csv")

class BuildCsvFromSuisiann(BuildCsv):
    def __init__(self):
        super().__init__(["Suisiann/SuiSiann.csv"], "Suisiann/SuiSiann-processed.csv")

    def remove_long_audios(self):
        # avoid OUT OF CUDA error
        self.dataset = self.dataset.filter(lambda example: float(example["length"]) <= 20.)

    def expand_filename(self):
        def add_filename_prefix(batch):
            batch["path"] = f"Suisiann/" + batch["path"]
            return batch
        self.dataset = self.dataset.map(add_filename_prefix)

    def tailo(self):
        def transform(batch):
            batch["tailo"] = self.tailo_transform(batch["Rome"])
            return batch
        self.dataset = self.dataset.map(transform)

    def build(self):
        self.tailo()
        self.remove_punctuation_marks()
        self.remove_tone_instances()
        self.expand_filename()
        self.remove_long_audios()
        self.remove_columns()
        self.to_csv()

class BuildCsvFromTAT(BuildCsv):
    def __init__(self, data_dir, key_dir, output_dir):
        self.data_dir = data_dir
        self.key_dir = key_dir
        self.output_dir = output_dir

    def pre_build(self):
        for environment in os.listdir(self.data_dir):
            environment_dir = os.path.join(self.data_dir, environment)
            if os.path.isdir(environment_dir) and environment != "json":
                environment_dir = os.path.join(environment_dir, "wav")

                path = []
                tailo = []
                for speaker in os.listdir(environment_dir):
                    data_dir = os.path.join(environment_dir, speaker)
                    key_dir = os.path.join(self.key_dir, speaker)
                    for wav in os.listdir(data_dir):
                        path.append(os.path.join(data_dir, wav))
                        key_file = wav[:wav.rfind("-")] + ".json"
                        key_file = os.path.join(key_dir, key_file)
                        with open(key_file, "r") as f:
                            tailo.append(json.load(f)['台羅'])
                df = pd.DataFrame({"path": path, "tailo": tailo})
                df.to_csv(os.path.join(self.output_dir, f"{environment}.csv"), index=False)

    def tailo(self):
        def transform(batch):
            batch["tailo"] = self.tailo_transform(batch["tailo"])
            return batch
        self.dataset = self.dataset.map(transform)

    def build(self):
        # self.pre_build()
        # super().__init__([os.path.join(self.output_dir, d) for d in ["XYH-6-X.csv", "XYH-6-Y.csv", "android.csv"]],
        #                  os.path.join(self.output_dir, "train-1.csv"))
        # self.tailo()
        # self.remove_punctuation_marks()
        # self.to_csv()
        # super().__init__([os.path.join(self.output_dir, d) for d in ["condenser.csv", "ios.csv", "lavalier.csv"]],
        #                  os.path.join(self.output_dir, "train-2.csv"))
        # self.tailo()
        # self.remove_punctuation_marks()
        # self.to_csv()
        super().__init__([os.path.join(self.output_dir, "lavalier.csv")],
                         os.path.join(self.output_dir, "lavalier-processed.csv"))
        self.tailo()
        self.remove_punctuation_marks()
        self.remove_tone_instances()
        self.remove_columns()
        self.to_csv()

if __name__ == "__main__":
    build_common_voice_train = BuildCsvFromCommonVoiceTrain()
    build_common_voice_train.build()
    build_common_voice_test = BuildCsvFromCommonVoiceTest()
    build_common_voice_test.build()

    build_suisiann = BuildCsvFromSuisiann()
    build_suisiann.build()

    build_TAT_Vol1_train = BuildCsvFromTAT("TAT/TAT-Vol1-train/d26eae87829adde551bf4b852f9da6b8c3c2db9b65b8b68870632a2db5f53e00-master/",
                                           "TAT/TAT-Vol1-train/d26eae87829adde551bf4b852f9da6b8c3c2db9b65b8b68870632a2db5f53e00-master/json",
                                           "TAT/TAT-Vol1-train")
    build_TAT_Vol1_train.build()

    build_TAT_Vol2_train = BuildCsvFromTAT("TAT/TAT-Vol2-train/f64f410744d9470ffe2d6b9ee6f042cdffcc42a745d2568146e8782ea828ff48-master",
                                           "TAT/TAT-Vol2-train/f64f410744d9470ffe2d6b9ee6f042cdffcc42a745d2568146e8782ea828ff48-master/json",
                                           "TAT/TAT-Vol2-train")
    build_TAT_Vol2_train.build()

    build_TAT_Vol1_test = BuildCsvFromTAT("TAT/TAT-Vol1-test/b7c7470e59e2a2df1bfd0a4705488ee6fe0c5c125de15cccdfab0e00d6c03dc0-master/",
                                          "TAT/TAT-Vol1-test-key/e6f47e008cc58b38596e6fdf2f50a0fea93fd10543e652522aeab3aa71355719-master/json/",
                                          "TAT/TAT-Vol1-test")
    build_TAT_Vol1_test.build()

    build_TAT_Vol2_test = BuildCsvFromTAT("TAT/TAT-Vol2-test/a73b320dc0d3a57c03f897eb28ca91e623c5ee635db59476ba3178c90b94019f-master/",
                                          "TAT/TAT-Vol2-test-key/480f5a496560ae4228bb7977ecf29b2c589d7a7aa6b609534566af8cbc229a9e-master/json/",
                                          "TAT/TAT-Vol2-test")
    build_TAT_Vol2_test.build()
