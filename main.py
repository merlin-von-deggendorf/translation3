from multilanguagereader import Persistence
from model import StringPairsFile
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import random
from simplemodel import Encoder,Decoder,Seq2Seq
import simplemodel


device = torch.device('cuda')
pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict(50)
simplemodel.train_model(spf,device)


    