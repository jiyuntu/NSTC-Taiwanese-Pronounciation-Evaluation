# https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
import json
import torch
import torchaudio

from dataclasses import dataclass
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from corpus import CommonVoice, Recordings, Suisiann, TAT

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-xls-r-300m-cv-suisiann-tat-vol12-lavalier/checkpoint-1576")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xls-r-300m-cv-suisiann-tat-vol12-lavalier/checkpoint-1576").to(device)

jiyun_recordings = Recordings(["jiyun-corpus/test-processed.csv"], processor).dataset
input_dict = processor(jiyun_recordings[0]["input_values"], return_tensors="pt", padding=True)
logits = model(input_dict.input_values.to("cuda")).logits
emissions = torch.log_softmax(logits, dim=-1)

emission = emissions[0].to(device).detach()

transcript = "kong1|un2"
dictionary = processor.tokenizer.get_vocab()

tokens = [dictionary[c] for c in transcript]

def get_trellis(emission, tokens, blank_id=29):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(device)
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

trellis = get_trellis(emission, tokens)

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


path = backtrack(trellis, emission, tokens)
for p in path:
    print(p)