import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from model import StringPairsFile


class TranslationDataset(Dataset):
    def __init__(self,string_pairs_file:StringPairsFile):
        self.string_pairs_file=string_pairs_file

    def __len__(self):
        return self.string_pairs_file.languages[0].tokenized_sentences.__len__()

    def __getitem__(self, idx):
        return torch.tensor(self.string_pairs_file.languages[0].tokenized_sentences[idx],dtype=torch.long), torch.tensor(self.string_pairs_file.languages[1].tokenized_sentences[idx],dtype=torch.long)
    
    def collate_fn(self,batch):
        """
        Collate function to be used with DataLoader.
    
        Args:
            batch (list of tuples): Each tuple contains (src_tensor, tgt_tensor).
    
        Returns:
            src_padded: Padded source sequences tensor [batch_size, src_max_len].
            tgt_padded: Padded target sequences tensor [batch_size, tgt_max_len].
        """
        src, tgt = zip(*batch)
        src_padded = pad_sequence(src,batch_first=True, padding_value=self.string_pairs_file.languages[0].pad_tokken)
        tgt_padded = pad_sequence(tgt,batch_first=True, padding_value=self.string_pairs_file.languages[1].pad_tokken)
        return src_padded, tgt_padded