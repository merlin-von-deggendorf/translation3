import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ----- Data Preparation -----

# Sample dataset: English-French pairs
data = [
    ("hello", "hola"),
    ("how are you", "cómo estás"),
    ("i am fine", "estoy bien"),
    ("thank you", "gracias"),
    ("goodbye", "adiós"),
    ("see you later", "hasta luego"),
    ("what is your name", "cómo te llamas"),
    ("nice to meet you", "encantado de conocerte"),
    ("i love you", "te amo"),
    ("have a nice day", "que tengas un buen día")
]

def build_vocab(sentences):
    vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
    idx = 4
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

src_sentences = [pair[0] for pair in data]
tgt_sentences = [pair[1] for pair in data]

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

def tokenize(sentence, vocab):
    return [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]

def preprocess_source(sentence, vocab):
    return [vocab["<sos>"]] + tokenize(sentence, vocab) + [vocab["<eos>"]]

def preprocess_target(sentence, vocab):
    return [vocab["<sos>"]] + tokenize(sentence, vocab) + [vocab["<eos>"]]

dataset_pairs = []
for src, tgt in data:
    src_tokens = torch.tensor(preprocess_source(src, src_vocab), dtype=torch.long)
    tgt_tokens = torch.tensor(preprocess_target(tgt, tgt_vocab), dtype=torch.long)
    dataset_pairs.append((src_tokens, tgt_tokens))

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

translation_dataset = TranslationDataset(dataset_pairs)

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=src_vocab["<pad>"])
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_vocab["<pad>"])
    return src_padded, tgt_padded

batch_size = 2
dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True, 
                        collate_fn=collate_fn)

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, num_heads):
        super(Seq2SeqAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=src_vocab["<pad>"])
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=tgt_vocab["<pad>"])
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, tgt_vocab_size)
        
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

embed_size = 32
hidden_size = 64
num_heads = 4
learning_rate = 0.001
num_epochs = 300

model = Seq2SeqAttentionModel(src_vocab_size, tgt_vocab_size, embed_size, hidden_size, num_heads)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for src_batch, tgt_batch in dataloader:
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
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def translate_sentence(model, sentence, src_vocab, tgt_vocab, inv_tgt_vocab, max_len=10):
    model.eval()
    with torch.no_grad():
        src_tokens = torch.tensor(preprocess_source(sentence, src_vocab), dtype=torch.long).unsqueeze(0)
        tgt_input = torch.tensor([[tgt_vocab["<sos>"]]] , dtype=torch.long)
        
        for _ in range(max_len):
            output = model(src_tokens, tgt_input)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            tgt_input = torch.cat((tgt_input, next_token), dim=1)
            if next_token.item() == tgt_vocab["<eos>"]:
                break
        
        translated_tokens = tgt_input.squeeze().tolist()
        words = []
        for token in translated_tokens:
            if token == tgt_vocab["<sos>"] or token == tgt_vocab["<eos>"]:
                continue
            words.append(inv_tgt_vocab.get(token, "<unk>"))
        return ' '.join(words)

# Test the updated model with <sos> and <eos> tokens on the source side
test_sentence = "good morning"
translation = translate_sentence(model, test_sentence, src_vocab, tgt_vocab, inv_tgt_vocab)
print(f"Source: {test_sentence}")
print(f"Translation: {translation}")
