from multilanguagereader import Persistence
from model import StringPairsFile,TranslationDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import random
from simplemodel import Encoder,Decoder,Seq2Seq
# stringpairs=StringPairs()
# persisitance.split('deu','eng',stringpairs)
# stringpairs.print_strings()

    

EMBED_SIZE=256
HIDDEN_SIZE=512    
TEACHER_FORCE_RATIO=0.5
LEARNING_RATE=0.001
NUM_EPOCHS=3
BATCH_SIZE=1000

# Initialize model, loss, optimizer
device = torch.device('cuda')
pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict(3)
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
        print(f"Batch [{batch_nr}/{batch_count}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    