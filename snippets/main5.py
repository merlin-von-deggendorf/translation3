import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from multilanguagereader import Persistence
from model import StringPairsFile
import torch

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
# ----- Data Preparation -----

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, spf:StringPairsFile, embed_size, hidden_size, num_heads):
        super(Seq2SeqAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder_embedding = nn.Embedding(spf.languages[0].word_dict.__len__(), embed_size, padding_idx=spf.languages[0].pad_tokken)
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(spf.languages[1].word_dict.__len__(), embed_size, padding_idx=spf.languages[1].pad_tokken)
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, spf.languages[1].word_dict.__len__())
        
    def forward(self, src, tgt):
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, hidden = self.encoder_rnn(embedded_src)
        
        embedded_tgt = self.decoder_embedding(tgt)
        decoder_outputs, _ = self.decoder_rnn(embedded_tgt, hidden)
        
        attn_output, attn_weights = self.attention(query=decoder_outputs,
                                                   key=encoder_outputs,
                                                   value=encoder_outputs)
        
        output = self.fc_out(attn_output)
        return output

device = torch.device('cuda')
pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict(50)
translation_dataset = TranslationDataset(spf)


batch_size = 6000
dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True, 
                        collate_fn=translation_dataset.collate_fn)

embed_size = 512
hidden_size = 1024
num_heads = 64
learning_rate = 0.001
num_epochs = 3

model = Seq2SeqAttentionModel(spf, embed_size, hidden_size, num_heads)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=spf.languages[1].pad_tokken)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    total_batches = len(dataloader)
    current_batch = 0
    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        tgt_input = tgt_batch[:, :-1]
        tgt_target = tgt_batch[:, 1:]
        
        optimizer.zero_grad()
        output = model(src_batch, tgt_input)
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_target = tgt_target.contiguous().view(-1)
        
        loss = criterion(output, tgt_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Batch {current_batch+1}/{total_batches}, Loss: {loss.item():.4f}")
        current_batch += 1
        
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# def translate_sentence(model, sentence, src_vocab, tgt_vocab, inv_tgt_vocab, max_len=10):
#     model.eval()
#     with torch.no_grad():
#         src_tokens = torch.tensor(preprocess_source(sentence, src_vocab), dtype=torch.long).unsqueeze(0)
#         tgt_input = torch.tensor([[tgt_vocab["<sos>"]]] , dtype=torch.long)
        
#         for _ in range(max_len):
#             output = model(src_tokens, tgt_input)
#             next_token_logits = output[0, -1, :]
#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
#             tgt_input = torch.cat((tgt_input, next_token), dim=1)
#             if next_token.item() == tgt_vocab["<eos>"]:
#                 break
        
#         translated_tokens = tgt_input.squeeze().tolist()
#         words = []
#         for token in translated_tokens:
#             if token == tgt_vocab["<sos>"] or token == tgt_vocab["<eos>"]:
#                 continue
#             words.append(inv_tgt_vocab.get(token, "<unk>"))
#         return ' '.join(words)

# # Test the updated model with <sos> and <eos> tokens on the source side
# test_sentence = "good morning"
# translation = translate_sentence(model, test_sentence, src_vocab, tgt_vocab, inv_tgt_vocab)
# print(f"Source: {test_sentence}")
# print(f"Translation: {translation}")
