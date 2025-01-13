import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample synthetic dataset of paired sentences
data_pairs = [
    ("hello", "bonjour"),
    ("world", "monde"),
    ("how are you", "comment ça va"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
    # add more pairs as needed
]

# Build a simple vocabulary from dataset
def build_vocab(pairs):
    vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
    idx = len(vocab)
    for src, tgt in pairs:
        for token in src.split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
        for token in tgt.split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

vocab = build_vocab(data_pairs)
vocab_size = len(vocab)

# Tokenization and conversion to indices
def tokenize(sentence, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in sentence.split()]

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_tokens = [self.vocab["<sos>"]] + tokenize(src, self.vocab) + [self.vocab["<eos>"]]
        tgt_tokens = [self.vocab["<sos>"]] + tokenize(tgt, self.vocab) + [self.vocab["<eos>"]]
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

# Create DataLoader
dataset = TranslationDataset(data_pairs, vocab)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=lambda batch: zip(*batch))

# Learned Positional Encoding
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embed = self.embedding(positions)
        return x + pos_embed

# Batch-first Transformer Model with Learned Positional Encoding
class TransformerModelBatchFirst(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, max_len=100):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        
        self.pos_encoder = LearnedPositionalEncoding(max_len, d_model)
        self.pos_decoder = LearnedPositionalEncoding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
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

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len = 50  # maximum sequence length expected
model = TransformerModelBatchFirst(vocab_size, max_len=max_seq_len).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Utility function to create target masks (for autoregressive decoding)
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Training loop (simplified)
model.train()
for epoch in range(20):  # small number of epochs for demonstration
    for src_batch, tgt_batch in dataloader:
        # Pad sequences in the batch to the same length
        src_batch = list(src_batch)
        tgt_batch = list(tgt_batch)
        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=vocab["<pad>"]).to(device)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=vocab["<pad>"]).to(device)

        # Prepare target input and output sequences
        tgt_input = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]

        tgt_seq_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

        optimizer.zero_grad()
        output = model(src_batch, tgt_input, tgt_mask=tgt_mask)
        # Reshape output and target for loss computation
        output = output.reshape(-1, vocab_size)
        tgt_out = tgt_out.reshape(-1)
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
