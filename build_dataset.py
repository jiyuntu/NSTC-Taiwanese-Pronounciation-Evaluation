"""
Build the dataset outside the training script because we need to
manually transform the phonetic alphabet. Seems like 臺灣言語工具
skips some of the words, and I'm not sure why.
"""

import re

from datasets import load_dataset

from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音

def tailo_transform(batch):
    begin = batch["sentence"].find("（")
    # Try to find ｜ but often fail.
    end = batch["sentence"].find("）")
    台羅 = batch["sentence"][begin + 1:end].strip()
    台羅物件 = 拆文分析器.建立句物件(台羅)
    batch["tailo"] = 台羅物件.轉音(臺灣閩南語羅馬字拼音).看語句()
    if "|" in batch["tailo"]:
        batch["tailo"] = batch["tailo"].split("|")[0]
    return batch

# replace - with space
def replace_dash_with_space(batch):
    chars_to_remove_regex = "[\,\?\.\!\;\:\"\“\%\‘\”\'\-]"
    batch["tailo"] = re.sub(chars_to_remove_regex, " ", batch["tailo"]).lower()
    # multiple spaces to one
    batch["tailo"] = " ".join(batch["tailo"].split(" ")).strip()
    return batch

common_voice_train = load_dataset("csv", data_files=["cv-corpus-10.0-2022-07-04/nan-tw/train.csv",
                                                     "cv-corpus-10.0-2022-07-04/nan-tw/dev.csv"])["train"]
common_voice_test = load_dataset("csv", data_files=["cv-corpus-10.0-2022-07-04/nan-tw/test.csv"])["train"]

# 台羅 to 台羅數字調
common_voice_train = common_voice_train.filter(lambda example: example["sentence"].find("（") >= 0)
common_voice_test = common_voice_test.filter(lambda example: example["sentence"].find("（") >= 0)

common_voice_train = common_voice_train.map(tailo_transform).map(replace_dash_with_space)
common_voice_test = common_voice_test.map(tailo_transform).map(replace_dash_with_space)

common_voice_train.to_csv("cv-corpus-10.0-2022-07-04/nan-tw/train-processed.csv", index=False)
common_voice_test.to_csv("cv-corpus-10.0-2022-07-04/nan-tw/test-processed.csv", index=False)