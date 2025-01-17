import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model import StringPairsFile
import torch
from torch.utils.data import random_split

class TranslationDataset(Dataset):
    def __init__(self,string_pairs_file:StringPairsFile):
        self.string_pairs_file=string_pairs_file

    def __len__(self):
        return self.string_pairs_file.languages[0].tokenized_sentences.__len__()

    def __getitem__(self, idx):
        return torch.tensor(self.string_pairs_file.languages[0].tokenized_sentences[idx],dtype=torch.long), torch.tensor(self.string_pairs_file.languages[1].tokenized_sentences[idx],dtype=torch.long)
    
    def collate_fn(self,batch):
        
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
    
class Trainer:

    def __init__(self,spf:StringPairsFile, device,
                embed_size = 512,
                hidden_size = 1024,
                num_heads = 64):
        self.spf = spf
        self.device = device
        model = Seq2SeqAttentionModel(spf, embed_size, hidden_size, num_heads)
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=spf.languages[1].pad_tokken)
        
    def train(self,
                batch_size = 6000,
                learning_rate = 0.001,
                num_epochs = 3 ,
                eval_count=0,train_eval_count=0 ):
        dataset = TranslationDataset(self.spf)
        self.collate_fn = dataset.collate_fn
        if eval_count>0:
            train_size = len(dataset) - eval_count
            train_dataset, test_dataset = random_split(dataset, [train_size, eval_count])
            self.train_dataset(train_dataset,batch_size,learning_rate,num_epochs)
            self.evaluate_dataset(test_dataset,batch_size)
            if train_eval_count>0:
                self.train_dataset(test_dataset,batch_size,learning_rate,train_eval_count)
                self.train_dataset(dataset,batch_size,learning_rate,1)
        else:
            dataset = TranslationDataset(self.spf)
            self.train_dataset(dataset,batch_size,learning_rate,num_epochs)

    def train_dataset(self,dataset,batch_size,learning_rate,num_epochs):
        self.model.train()
        device = self.device
        translation_dataset = dataset
        
        model = self.model
        criterion = self.criterion
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        dataloader = DataLoader(translation_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=self.collate_fn)


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
                print(f"Epoch {epoch+1} of {num_epochs} : Batch {current_batch+1}/{total_batches}, Loss: {loss.item():.4f}")
                current_batch += 1
                
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # src sentence needs to be normalized first
    def translate(self, src_sentence, max_len=50):
        model = self.model
        device = self.device
        spf = self.spf

        # Ensure the reverse dictionary for target language is generated
        spf.languages[1].generate_reverse_dict()
        reverse_dict = spf.languages[1].reverse_dict

        # Tokenize and convert the source sentence to tensor
        tokenized_src = spf.languages[0].tokenize_sentence(src_sentence)


        src_tensor = torch.tensor(tokenized_src, dtype=torch.long, device=device).unsqueeze(0)  # shape: [1, seq_len]

        # Encode the source sentence
        with torch.no_grad():
            model.eval()  # set model to evaluation mode
            embedded_src = model.encoder_embedding(src_tensor)
            encoder_outputs, hidden = model.encoder_rnn(embedded_src)

            # Prepare the initial decoder input with the BOS token
            bos_token = spf.languages[1].bos_token  # Ensure bos_token is defined
            eos_token = spf.languages[1].eos_token  # Ensure eos_token is defined
            decoder_input = torch.tensor([[bos_token]], dtype=torch.long, device=device)

            translated_tokens = []
            
            # Greedy decoding loop
            for _ in range(max_len):
                embedded_dec_in = model.decoder_embedding(decoder_input)
                decoder_outputs, hidden = model.decoder_rnn(embedded_dec_in, hidden)
                
                # Apply attention mechanism
                attn_output, _ = model.attention(query=decoder_outputs,
                                                 key=encoder_outputs,
                                                 value=encoder_outputs)
                output = model.fc_out(attn_output)  # shape: [1, 1, vocab_size]
                
                # Get the token with highest probability
                next_token = output.argmax(-1)  # shape: [1, 1]
                next_token_id = next_token.item()
                
                # Stop if EOS is predicted
                if next_token_id == eos_token:
                    break
                
                translated_tokens.append(next_token_id)
                
                # Prepare next input for decoder
                decoder_input = next_token

            # Convert token ids to words using reverse dictionary
            translated_sentence = [reverse_dict[token] for token in translated_tokens if token in reverse_dict]
            
            model.train()  # revert model back to training mode if needed
            
        return ' '.join(translated_sentence)

        
    def evaluate_dataset(self, dataset, batch_size):
        model = self.model
        criterion = self.criterion
        device = self.device

        # Set the model to evaluation mode
        model.eval()

        # Create a DataLoader for the evaluation dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=self.collate_fn)

        total_loss = 0.0
        total_batches = len(dataloader)

        with torch.no_grad():
            for src_batch, tgt_batch in dataloader:
                # Move batches to the specified device
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                # Prepare decoder input and target output
                tgt_input = tgt_batch[:, :-1]
                tgt_target = tgt_batch[:, 1:]

                # Forward pass through the model
                output = model(src_batch, tgt_input)

                # Reshape outputs and targets to compute loss
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_target = tgt_target.contiguous().view(-1)

                # Compute loss
                loss = criterion(output, tgt_target)
                total_loss += loss.item()

        # Compute and print average loss over all batches
        avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        print(f"Evaluation Loss: {avg_loss:.4f}")
        model.train()


