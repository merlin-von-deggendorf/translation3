import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

# Hyperparameters
INPUT_SIZE = 256   # Size of the source vocabulary
OUTPUT_SIZE = 256  # Size of the target vocabulary
EMBED_SIZE = 256   # Embedding size
HIDDEN_SIZE = 512  # Hidden size for RNNs
NUM_LAYERS = 1     # Number of layers in RNNs
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
TEACHER_FORCING_RATIO = 0.5

# Sample data: list of (source_sentence, target_sentence) tuples
data = [
    ("hello", "hola"),
    ("how are you", "cómo estás"),
    ("i am fine", "estoy bien"),
    ("thank you", "gracias"),
    ("goodbye", "adiós")
]

# Build vocabularies
def build_vocab(data):
    source_vocab = set()
    target_vocab = set()
    for src, tgt in data:
        source_vocab.update(src.lower().split())
        target_vocab.update(tgt.lower().split())
    source_vocab = {word: idx+2 for idx, word in enumerate(sorted(source_vocab))}
    target_vocab = {word: idx+2 for idx, word in enumerate(sorted(target_vocab))}
    # Add special tokens
    source_vocab["<pad>"] = 0
    source_vocab["<sos>"] = 1
    target_vocab["<pad>"] = 0
    target_vocab["<sos>"] = 1
    target_vocab["<eos>"] = 2
    return source_vocab, target_vocab

source_vocab, target_vocab = build_vocab(data)
inv_target_vocab = {v: k for k, v in target_vocab.items()}

# Function to convert sentences to tensor of word indices
def sentence_to_indices(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[word] for word in sentence.lower().split()] + [vocab.get("<eos>", len(vocab)+1)]

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x, lengths):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_size]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)  # outputs is PackedSequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden  # outputs: [batch_size, seq_len, hidden_size]

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        # x: [batch_size] (current input token)
        x = x.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(x)  # [batch_size, 1, embed_size]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hidden_size]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_size]
        return prediction, hidden

# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder outputs
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # First input to the decoder is the <sos> tokens
        input = trg[:,0]
        
        for t in range(1, trg_len):
            # Decode
            output, hidden = self.decoder(input, hidden)
            outputs[:,t] = output
            # Decide whether to do teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # [batch_size]
            input = trg[:,t] if teacher_force else top1
        
        return outputs

# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(INPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
decoder = Decoder(OUTPUT_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Prepare training data
def prepare_batch(data, source_vocab, target_vocab):
    src = []
    trg = []
    src_lengths = []
    for (s, t) in data:
        src_indices = sentence_to_indices(s, source_vocab)
        trg_indices = sentence_to_indices(t, target_vocab)
        src.append(src_indices)
        trg.append(trg_indices)
        src_lengths.append(len(src_indices))
    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in src], batch_first=True, padding_value=source_vocab["<pad>"])
    trg_padded = nn.utils.rnn.pad_sequence([torch.tensor(t) for t in trg], batch_first=True, padding_value=target_vocab["<pad>"])
    return src_padded.to(device), trg_padded.to(device), src_lengths

# Training
for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    src_batch, trg_batch, src_lengths = prepare_batch(data, source_vocab, target_vocab)
    outputs = model(src_batch, src_lengths, trg_batch, TEACHER_FORCING_RATIO)
    # outputs: [batch_size, trg_len, output_dim]
    # trg_batch: [batch_size, trg_len]
    
    # Reshape for loss computation
    outputs = outputs[:,1:].reshape(-1, OUTPUT_SIZE)
    trg = trg_batch[:,1:].reshape(-1)
    
    loss = criterion(outputs, trg)
    loss.backward()
    
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# Inference
def translate(sentence, model, source_vocab, target_vocab, inv_target_vocab, max_len=10):
    model.eval()
    with torch.no_grad():
        src_indices = sentence_to_indices(sentence, source_vocab)
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
        src_length = [len(src_indices)]
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)
        
        # Initialize decoder input with <sos>
        input = torch.tensor([target_vocab["<sos>"]]).to(device)
        translated = []
        
        for _ in range(max_len):
            output, hidden = model.decoder(input, hidden)
            top1 = output.argmax(1).item()
            if top1 == target_vocab.get("<eos>", -1):
                break
            translated.append(inv_target_vocab.get(top1, "<unk>"))
            input = torch.tensor([top1]).to(device)
        
        return ' '.join(translated)

# Example translation
test_sentence = "hello"
translation = translate(test_sentence, model, source_vocab, target_vocab, inv_target_vocab)
print(f"Translation of '{test_sentence}': {translation}")

test_sentence = "how are you"
translation = translate(test_sentence, model, source_vocab, target_vocab, inv_target_vocab)
print(f"Translation of '{test_sentence}': {translation}")

test_sentence = "thank you"
translation = translate(test_sentence, model, source_vocab, target_vocab, inv_target_vocab)
print(f"Translation of '{test_sentence}': {translation}")
