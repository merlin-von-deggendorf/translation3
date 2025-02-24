from multilanguagereader import Persistence
from model import StringPairsFile
import torch
import random


device = torch.device('cuda')
pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict(200)
# print dict sizes
print(f"Language 1 dictionary size: {spf.languages[0].word_dict.__len__()}")
print(f"Language 2 dictionary size: {spf.languages[1].word_dict.__len__()}")

# import simplemodel
# simplemodel.train_model(spf,device)

# import attentionmodel
# attentionmodel.train(spf,device)
    
from rnnattention import Trainer

trainer = Trainer(spf,device,embed_size=512,hidden_size=1024,num_heads=64)
trainer.train(num_epochs=3,batch_size=8000)
for val in range(5):
    i=random.randint(0,spf.languages[0].sentences.__len__())
    print(spf.languages[0].sentences[i])
    print(spf.languages[1].sentences[i])
    print(f'predicted: {trainer.translate(spf.languages[0].sentences[i])}')

