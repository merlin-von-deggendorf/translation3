import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from model import StringPairsFile
from torch.utils.data import DataLoader
import torch.optim as optim
import math



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
    
# Learned Positional Encoding
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, spf:StringPairsFile, d_model):
        super().__init__()
        self.embedding = nn.Embedding(spf.languages[0].max_length+2, d_model)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embed = self.embedding(positions)
        return x + pos_embed


# Batch-first Transformer Model with Learned Positional Encoding
class TransformerModelBatchFirst(nn.Module):
    def __init__(self, spf:StringPairsFile, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_embed = nn.Embedding(spf.languages[0].word_dict.__len__(), d_model)
        self.tgt_embed = nn.Embedding(spf.languages[1].word_dict.__len__(), d_model)
        
        self.pos_encoder = LearnedPositionalEncoding(spf, d_model)
        self.pos_decoder = LearnedPositionalEncoding(spf, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, spf.languages[1].word_dict.__len__())
        self.d_model = d_model

    def forward(self, src, tgt, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Embed and apply learned positional encoding
        src = self.src_embed(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)
        
        memory = self.transformer_encoder(src,
                                          mask=None,
                                          src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_decoder(tgt, memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=src_key_padding_mask)
        output = self.fc_out(output)
        return output

    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train(spf:StringPairsFile,device, 
            batch_size=6000,
            learning_rate=0.001,
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3, 
            dim_feedforward=512,
            dropout=0.1):
    
    dataset = TranslationDataset(spf,d_model=d_model,nhead=nhead,num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward,dropout=dropout)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model=TransformerModelBatchFirst(spf)
    model=model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=spf.languages[1].pad_tokken)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Training loop (simplified)
    model.train()
    for epoch in range(20):  # small number of epochs for demonstration
        batchnr=0
        batch_count=len(dataloader)
        for src_batch, tgt_batch in dataloader:
            src_batch=src_batch.to(device)
            tgt_batch=tgt_batch.to(device)

            # Prepare target input and output sequences
            tgt_input = tgt_batch[:, :-1]
            tgt_out = tgt_batch[:, 1:]

            tgt_seq_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

            optimizer.zero_grad()
            output = model(src_batch, tgt_input, tgt_mask=tgt_mask)
            # Reshape output and target for loss computation
            output = output.reshape(-1, spf.languages[1].word_dict.__len__())
            tgt_out = tgt_out.reshape(-1)
            loss = criterion(output, tgt_out)
            loss.backward()
            optimizer.step()
            print(f'batch {batchnr} of {batch_count} done loss: {loss.item():.4f}')
            batchnr+=1
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    
    
