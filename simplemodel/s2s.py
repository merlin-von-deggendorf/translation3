import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from model import StringPairsFile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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
        src_lengths = [len(s) for s in src]
        src_padded = pad_sequence(src,batch_first=True, padding_value=self.string_pairs_file.languages[0].pad_tokken)
        tgt_padded = pad_sequence(tgt,batch_first=True, padding_value=self.string_pairs_file.languages[1].pad_tokken)
        return src_padded, tgt_padded , src_lengths

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
    
def train_model(spf:StringPairsFile,device):
    EMBED_SIZE=256
    HIDDEN_SIZE=512    
    TEACHER_FORCE_RATIO=0.5
    LEARNING_RATE=0.001
    NUM_EPOCHS=3
    BATCH_SIZE=5000
    dataset=TranslationDataset(spf)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
    print(spf.languages[0].word_dict.__len__())

    OUTPUT_SIZE=spf.languages[1].word_dict.__len__()
    INPUT_SIZE=spf.languages[0].word_dict.__len__()

    encoder=Encoder(INPUT_SIZE,EMBED_SIZE,HIDDEN_SIZE,1).to(device)
    decoder=Decoder(OUTPUT_SIZE,EMBED_SIZE,HIDDEN_SIZE,spf.languages[1].pad_tokken,1).to(device)
    model=Seq2Seq(encoder,decoder,device).to(device)


    criterion = nn.CrossEntropyLoss(ignore_index=spf.languages[1].pad_tokken)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # calculate number of batches
    batch_count = len(dataloader)
    # Training
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        batch_nr=0
        for batch_idx, (src_batch, tgt_batch, src_lengths) in enumerate(dataloader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(src_batch, src_lengths, tgt_batch, TEACHER_FORCE_RATIO)
            # outputs: [batch_size, trg_len, output_dim]
            # tgt_batch: [batch_size, trg_len]

            # Reshape for loss computation
            outputs = outputs[:,1:].reshape(-1, OUTPUT_SIZE)
            trg = tgt_batch[:,1:].reshape(-1)

            loss = criterion(outputs, trg)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            batch_nr+=1
            print(f"Batch [{batch_nr}/{batch_count}], Loss: {loss.item():.4f} epoch {epoch+1}/{NUM_EPOCHS}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")