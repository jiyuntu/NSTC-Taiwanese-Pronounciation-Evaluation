import json

from datasets import load_dataset

def extract_all_chars(batch):
    all_text = " ".join(batch["tailo"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def build_vocab():
    train_dataset = load_dataset("csv", data_files=["cv-corpus-10.0-2022-07-04/nan-tw/train-processed-2.csv",
                                                    "Suisiann/SuiSiann-processed.csv",
                                                    "TAT/TAT-Vol1-train/lavalier-processed.csv",
                                                    "TAT/TAT-Vol2-train/lavalier-processed.csv"])["train"]
    test_dataset = load_dataset("csv", data_files=["cv-corpus-10.0-2022-07-04/nan-tw/test-processed-2.csv",
                                                   "TAT/TAT-Vol1-test/lavalier-processed.csv",
                                                   "TAT/TAT-Vol2-test/lavalier-processed.csv"])["train"]
    vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_dataset.column_names)
    vocab_test = test_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_dataset.column_names)
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

if __name__ == "__main__":
    build_vocab()


