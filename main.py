from multilanguagereader import Persistence
from model import StringPairsFile,TranslationDataset
from torch.utils.data import DataLoader
# stringpairs=StringPairs()
# persisitance.split('deu','eng',stringpairs)
# stringpairs.print_strings()


pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict()
dataset=TranslationDataset(spf)
dataloader = DataLoader(dataset, batch_size=10000, shuffle=True, collate_fn=dataset.collate_fn)

for src_batch, tgt_batch, src_lengths in dataloader:
    print(src_batch)
    print(tgt_batch)
    print(src_lengths)
    break