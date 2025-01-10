import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

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
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size,padding_value, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=padding_value)
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