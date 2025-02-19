
import torch
import random

from aatranslator import Trainer

device = torch.device('cuda')

trainer = Trainer(device,embed_size=16,hidden_size=512,num_heads=16,max_len=200)
trainer.load_dataset("c:/data/training/links.txt.gz")
trainer.train(num_epochs=1,batch_size=50,eval_count=100)
# pick random value from the training set
for val in range(5):
    # pick a random value from the training sample
    idx = random.randint(0, len(trainer.training_samples)-1)
    trainer.translate_evaluation(idx)


