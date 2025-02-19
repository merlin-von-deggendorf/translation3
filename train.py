
import torch
import random

from aatranslator import Trainer

device = torch.device('cuda')

trainer = Trainer(device,embed_size=16,hidden_size=1024,num_heads=16,max_len=650)
trainer.load_dataset("c:/data/training/links.txt.gz")
trainer.train(num_epochs=6,batch_size=30)
trainer.save_model("c:/data/training/links.model")


