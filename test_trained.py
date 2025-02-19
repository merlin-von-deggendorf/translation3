
import torch
import random
import gzip

from aatranslator import Trainer

device = torch.device('cuda')
max_len = 650
trainer = Trainer(device,embed_size=16,hidden_size=1024,num_heads=16,max_len=max_len)
trainer.load_model("c:/data/training/links.model")
sequences:list[list[str]] = []
sequences.append([])
sequences.append([])
# load all sequences

with gzip.open('c:/data/training/links.txt.gz', "rt", encoding="utf-8-sig") as file:
    is_even = False
    line1:str
    line2:str
    for line in file:
        if is_even:
            line2 = line.strip()
            # make sure both lines are smaller than 500 characters
            if line1.__len__()<max_len and line2.__len__()<max_len:
                sequences[0].append(line1)
                sequences[1].append(line2)
            is_even = False
        else:
            line1 = line.strip()
            is_even = True
        # if end is even, we should raise an exception
    if is_even:
        raise Exception("Odd number of lines in file")

accumulated_translation_performance = 0
accumulated_original_performance = 0
sample_size = 500
for val in range(sample_size):
    # pick a random value from the training sample
    idx = random.randint(0, len(sequences[0])-1)
    translation_performance,original_performance=trainer.compare_identity(sequences[0][idx],sequences[1][idx])
    accumulated_translation_performance += translation_performance
    accumulated_original_performance += original_performance
    print(f" original: {original_performance} ||| translation: {translation_performance} ||| average original: {accumulated_original_performance/(val+1)} ||| average translation: {accumulated_translation_performance/(val+1)} sample number: {val+1} out of {sample_size}")

print(f"average original performance: {accumulated_original_performance/sample_size} average translation performance: {accumulated_translation_performance/sample_size}")
