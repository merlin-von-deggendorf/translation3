from multilanguagereader import Persistence
from model import StringPairsFile
import torch
import random


device = torch.device('cuda')
pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict(50)

# import simplemodel
# simplemodel.train_model(spf,device)

# import attentionmodel
# attentionmodel.train(spf,device)
    
from rnnattention import Trainer
trainer = Trainer(spf,device)
trainer.train(num_epochs=10,eval_count=100,train_eval_count=1)
for val in range(5):
    i=random.randint(0,spf.languages[0].sentences.__len__())
    print(spf.languages[0].sentences[i])
    print(spf.languages[1].sentences[i])
    print(f'predicted: {trainer.translate(spf.languages[0].sentences[i])}')

