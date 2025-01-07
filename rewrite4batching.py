import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Hyperparameters
INPUT_SIZE = 256   # Size of the source vocabulary
OUTPUT_SIZE = 256  # Size of the target vocabulary
EMBED_SIZE = 256   # Embedding size
HIDDEN_SIZE = 512  # Hidden size for RNNs
NUM_LAYERS = 1     # Number of layers in RNNs
LEARNING_RATE = 0.001
NUM_EPOCHS = 20    # Increased epochs for better learning
TEACHER_FORCING_RATIO = 0.5
BATCH_SIZE = 20     # Example batch size

# Sample data: list of (source_sentence, target_sentence) tuples
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

# Build vocabularies
def build_vocab(data):
    source_vocab = set()
    target_vocab = set()
    for src, tgt in data:
        source_vocab.update(src.lower().split())
        target_vocab.update(tgt.lower().split())
    source_vocab = {word: idx+4 for idx, word in enumerate(sorted(source_vocab))}
    target_vocab = {word: idx+4 for idx, word in enumerate(sorted(target_vocab))}
    # Add special tokens
    source_vocab["<pad>"] = 0
    source_vocab["<sos>"] = 1
    source_vocab["<eos>"] = 2
    source_vocab["<unk>"] = 3
    target_vocab["<pad>"] = 0
    target_vocab["<sos>"] = 1
    target_vocab["<eos>"] = 2
    target_vocab["<unk>"] = 3
    return source_vocab, target_vocab

source_vocab, target_vocab = build_vocab(data)
inv_target_vocab = {v: k for k, v in target_vocab.items()}

# Function to convert sentences to tensor of word indices
def sentence_to_indices(sentence, vocab):
    return [vocab.get("<sos>", 1)] + [vocab.get(word, vocab["<unk>"]) for word in sentence.lower().split()] + [vocab.get("<eos>", 2)]

class TranslationDataset(Dataset):
    def __init__(self, data, source_vocab, target_vocab):
        """
        Args:
            data (list of tuples): List containing (source_sentence, target_sentence).
            source_vocab (dict): Vocabulary for the source language.
            target_vocab (dict): Vocabulary for the target language.
        """
        self.data = data
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data[idx]
        src_indices = sentence_to_indices(src_sentence, self.source_vocab)
        tgt_indices = sentence_to_indices(tgt_sentence, self.target_vocab)
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    """
    Collate function to be used with DataLoader.

    Args:
        batch (list of tuples): Each tuple contains (src_tensor, tgt_tensor).

    Returns:
        src_padded: Padded source sequences tensor [batch_size, src_max_len].
        tgt_padded: Padded target sequences tensor [batch_size, tgt_max_len].
        src_lengths: List of actual lengths of source sequences.
    """
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [len(s) for s in src_batch]
    tgt_lengths = [len(t) for t in tgt_batch]

    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=source_vocab["<pad>"])
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=target_vocab["<pad>"])

    return src_padded, tgt_padded, src_lengths

# Initialize Dataset and DataLoader
dataset = TranslationDataset(data, source_vocab, target_vocab)
    
dataloader = DataLoader(dataset, batch_size=10000, shuffle=True, collate_fn=collate_fn)

for src_batch, tgt_batch, src_lengths in dataloader:
    print(src_batch)
    print(tgt_batch)
    print(src_lengths)
    break