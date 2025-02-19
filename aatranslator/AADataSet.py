import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model import StringPairsFile
import torch
import gzip
from torch.utils.data import random_split
from aatranslator import Converter
from aatranslator import NeedlemanWunsch
from aatranslator import NeedlemanWunschRatio


class AADataSet(Dataset):
    def __init__(self,max_len=500):
        self.sequences:list[list[int]] = []
        self.sequences.append([])
        self.sequences.append([])
        self.converter:Converter=Converter()
        self.PAD_TOKEN = self.converter.PAD
        self.path:str = ''
        self.max_len = max_len
        
                     
    def load(self,path:str):
        self.path = path
        is_even = False
        line1:str
        line2:str
        with gzip.open(self.path, "rt", encoding="utf-8-sig") as file:
            for line in file:
                if is_even:
                    line2 = line.strip()
                    # make sure both lines are smaller than 500 characters
                    if line1.__len__()<self.max_len and line2.__len__()<self.max_len:
                        self.sequences[0].append(self.converter.convert_2_list(line1,True))
                        self.sequences[1].append(self.converter.convert_2_list(line2,True))
                    is_even = False
                else:
                    line1 = line.strip()
                    is_even = True
        # if end is even, we should raise an exception
        if is_even:
            raise Exception("Odd number of lines in file")
        
    def __len__(self):
        return self.sequences[0].__len__()

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[0][idx],dtype=torch.long), torch.tensor(self.sequences[1][idx],dtype=torch.long)
    
    def collate_fn(self,batch):
        
        src, tgt = zip(*batch)
        src_padded = pad_sequence(src,batch_first=True, padding_value=self.PAD_TOKEN)
        tgt_padded = pad_sequence(tgt,batch_first=True, padding_value=self.PAD_TOKEN)
        return src_padded, tgt_padded 
    

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads,converter:Converter):
        super(Seq2SeqAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        
     
        self.embedding = nn.Embedding(converter.length, embed_size, padding_idx=converter.PAD)
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, converter.length)
        
    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        encoder_outputs, hidden = self.encoder_rnn(embedded_src)
        
        embedded_tgt = self.embedding(tgt)
        decoder_outputs, _ = self.decoder_rnn(embedded_tgt, hidden)
        
        attn_output, attn_weights = self.attention(query=decoder_outputs,
                                                   key=encoder_outputs,
                                                   value=encoder_outputs)
        
        output = self.fc_out(attn_output)
        return output
    
class Trainer:

    def __init__(self, device,
                embed_size = 512,
                hidden_size = 1024,
                num_heads = 64,max_len=500):
        self.aa_dataset:AADataSet = AADataSet(max_len)
        self.converter = self.aa_dataset.converter
        self.device = device
        model = Seq2SeqAttentionModel( embed_size, hidden_size, num_heads,self.converter)
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.converter.PAD)
    def load_dataset(self,path:str):

        self.aa_dataset.load(path)

    def train(self,
                batch_size = 6000,
                learning_rate = 0.001,
                num_epochs = 3 ,
                eval_count=0,train_eval_count=0 ):
        self.collate_fn = self.aa_dataset.collate_fn
        if eval_count>0:
            train_size = len(self.aa_dataset) - eval_count
            train_dataset, test_dataset = random_split(self.aa_dataset, [train_size, eval_count])
            self.training_samples=train_dataset
            self.train_dataset(train_dataset,batch_size,learning_rate,num_epochs)
            self.evaluate_dataset(test_dataset,batch_size)
            if train_eval_count>0:
                self.train_dataset(test_dataset,batch_size,learning_rate,train_eval_count)
                self.train_dataset(self.aa_dataset,batch_size,learning_rate,1)
        else:
            self.train_dataset(self.aa_dataset,batch_size,learning_rate,num_epochs)

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
                if(current_batch%10==0):
                    print(f"Epoch {epoch+1} of {num_epochs} : Batch {current_batch+1}/{total_batches}, Loss: {loss.item():.4f}")
                current_batch += 1
                
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
            
          
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
        
    def translate(self, src_sentence):
        model = self.model
        device = self.device
        model.eval()

        # 1) Tokenize and convert the source sentence to a tensor of shape [1 x src_length]
        tokenized_src = self.converter.convert_2_list(src_sentence,False)
        src_tensor = torch.tensor(tokenized_src, dtype=torch.long).unsqueeze(0).to(device)

        # 2) Run the encoder to get encoder outputs and the final hidden state
        with torch.no_grad():
            embedded_src = model.embedding(src_tensor)
            encoder_outputs, hidden = model.encoder_rnn(embedded_src)

            # 3) Greedy decoding
            # Start the decoder input with the <sos> (start of sentence) token.
            # Make sure you replace `sos_tokken` and `eos_tokken` with whatever names you have used.
            next_token = self.converter.BOS
            decoded_tokens = []

            for _ in range(self.aa_dataset.max_len):
                # Prepare the decoder input of shape [1 x 1]
                tgt_input = torch.tensor([[next_token]], dtype=torch.long).to(device)

                # Pass through the decoder
                embedded_tgt = model.embedding(tgt_input)
                decoder_output, hidden = model.decoder_rnn(embedded_tgt, hidden)
                
                # Apply attention over the encoder outputs
                attn_output, attn_weights = model.attention(
                    query=decoder_output,
                    key=encoder_outputs,
                    value=encoder_outputs
                )
                
                # Get the distribution over the target vocabulary
                output = model.fc_out(attn_output)  # shape: [1, 1, vocab_size]
                
                # Greedy pick the token with the highest logit
                next_token = torch.argmax(output, dim=-1).item()

                # If we hit <eos>, stop decoding
                if next_token == self.converter.EOS:
                    break

                decoded_tokens.append(next_token)

            # 4) Convert token IDs back to words using `reverse_dict`
            

            return self.converter.convert_2_str(decoded_tokens)
        
    def translate_evaluation(self, idx:int):
        sample = self.training_samples[idx]
        # translate the sample
        # pick only the input part of the sample
        input = sample[0]
        # remove sos and eos tokens
        input = input[1:-1]
        # convert input tensor to python list
        input = input.tolist()
        # convert input to string
        input = self.converter.convert_2_str(input)
        print("Input: ",input)
        # translate the input
        output = self.translate(input)
        print("Output: ",output)
        # calculate the real output
        real_output = sample[1]
        # remove sos and eos tokens
        real_output = real_output[1:-1]
        # convert real output tensor to python list
        real_output = real_output.tolist()
        # convert real output to string
        real_output = self.converter.convert_2_str(real_output)
        print("Real Output: ",real_output)
        # calculate the distance between the real output and the output
        distance = NeedlemanWunsch(output,real_output)
        print(f'distance: {distance} sequence length: {len(real_output)} percentage: {(distance/len(real_output)):.2f}')
    
    def compare_identity(self,input:str,output:str):
        # translate the input
        translation = self.translate(input)
        # calculate the distance between the real output and the output
        distance2translation = NeedlemanWunschRatio(translation,output)
        distanceinput2output=NeedlemanWunschRatio(input,output)
        return distance2translation,distanceinput2output

    
    def save_model(self,path:str):
        torch.save(self.model.state_dict(), path)
    def load_model(self,path:str):
        self.model.load_state_dict(torch.load(path))
    
    
