import torch

from datasets import load_metric
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from corpus import CommonVoice, Recordings, Suisiann, TAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(test_dataset, model, processor):
    for data in test_dataset:
        input_dict = processor(data["input_values"], return_tensors="pt", padding=True)
        logits = model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]

        print("Prediction:")
        print(processor.decode(pred_ids))

def wer(test_dataset, model, processor):
    wer_metric = load_metric("wer")
    def map_to_result(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        try:
            batch["pred_str"] = processor.batch_decode(pred_ids)[0].replace("[PAD]", "")
            batch["text"] = processor.decode(batch["labels"], group_tokens=False)
        except:
            batch["pred_str"] = ""
            batch["text"] = ""

        return batch

    results = test_dataset.map(map_to_result, remove_columns=test_dataset.column_names)
    results = results.filter(lambda example: len(example["pred_str"]) > 0 and len(example["text"]) > 0)
    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

def main():
    processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-xls-r-300m-mn-colab/checkpoint-6100")
    model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xls-r-300m-mn-colab/checkpoint-6100").to("cuda")
    #data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    #common_voice_train = CommonVoice(["cv-corpus-10.0-2022-07-04/nan-tw/train-processed.csv"], processor).dataset
    common_voice_test = CommonVoice(["cv-corpus-10.0-2022-07-04/nan-tw/test-processed-2.csv"], processor).dataset
    #suisiann_test = Suisiann(["Suisiann/SuiSiann-processed.csv"], processor).dataset
    #tat_vol1_eval = TAT(["TAT/TAT-Vol1-eval.csv"], processor).dataset
    #jiyun_recordings = Recordings(["jiyun-corpus/test-processed.csv"], processor, "jiyun").dataset
    #inference(suisiann_test, model, processor)
    wer(common_voice_test, model, processor)

if __name__ == "__main__":
    main()