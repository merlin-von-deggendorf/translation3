from multilanguagereader import Persistence
from model import StringPairsFile
import torch


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
trainer.train(num_epochs=1,eval_count=100,train_eval_count=1)