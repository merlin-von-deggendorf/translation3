
import gzip
import torch
import random

from aatranslatorregular import Trainer
import os

trainingdata="c:/data/training/links.txt.gz"
evaldata="c:/data/training/links.eval.txt.gz"
model_path="c:/data/training/droput.model"
train_mode=False
max_len=650


device = torch.device('cuda')

trainer = Trainer(device,embed_size=16,hidden_size=1024,num_heads=16,max_len=max_len,dropout=0.3)
if os.path.exists(model_path):
    trainer.load_model(model_path)

if train_mode:
    trainer.load_dataset(trainingdata)
    trainer.train(num_epochs=1,batch_size=16)
    trainer.save_model(model_path)
else:
    sequences:list[list[str]] = []
    sequences.append([])
    sequences.append([])
    # load all sequences

    with gzip.open(evaldata, "rt", encoding="utf-8-sig") as file:
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
    len1 = len(sequences[0])
    for val in range(len1):
        # pick a random value from the training sample
        translation_performance,original_performance=trainer.compare_identity(sequences[0][val],sequences[1][val])
        accumulated_translation_performance += translation_performance
        accumulated_original_performance += original_performance
        print(f" original: {original_performance} ||| translation: {translation_performance} ||| average original: {accumulated_original_performance/(val+1)} ||| average translation: {accumulated_translation_performance/(val+1)} sample number: {val+1} out of {len1}")

    print(f"average original performance: {accumulated_original_performance/len1} average translation performance: {accumulated_translation_performance/len1}")



