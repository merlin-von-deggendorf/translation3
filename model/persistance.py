
import gzip
import regex
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class StringPairsFile:
    def __init__(self, id):
        self.id = id
        self.path = f"d:/translations/pairs/{self.id}.txt.gz"
        self.languages:list[Language] = [Language(0),Language(0)]
    def does_exist(self):
        return os.path.exists(self.path)

    def add_pair(self, s1, s2):
        # replace tabs and newlines with spaces
        s1 = self.normalize(s1)
        s2 = self.normalize(s2)
        self.languages[0].sentences.append(s1)
        self.languages[1].sentences.append(s2)
    def normalize(self, text):
        return regex.sub(r'[^\p{L}\p{N}]+', ' ', text).strip().lower()
    
    def save(self):
        # open compressed file
        with gzip.open(self.path, "wt",encoding='utf-8') as f:
            for i in range(len(self.languages[0].sentences)):
                s1 = self.languages[0].sentences[i]
                s2 = self.languages[1].sentences[i]
                f.write(f"{s1}\t{s2}\n")
    
    def load(self,max_len):
        self.languages:list[Language] = [Language(max_len),Language(max_len)]
        with gzip.open(self.path, "rt", encoding='utf-8') as f:
            for line in f:
                s1, s2 = line.strip().split("\t")
                # make sure the sentences are not too long

                b1=self.languages[0].set_current_words(s1)
                b2=self.languages[1].set_current_words(s2)
                if b1 and b2:
                    for lang in self.languages:
                        lang.append_current_words()
                else:
                    # reset words
                    for lang in self.languages:
                        lang.reset_current_words()
    def generate_dict(self,min_count=1):
        for lang in self.languages:
            lang.generate_count_dictionary()
        for i in range(self.languages[0].sentences.__len__()):
            
            if self.languages[0].check_word_count(i,min_count) and self.languages[1].check_word_count(i,min_count):
                for lang in self.languages:
                    lang.reduced_sentences.append(lang.sentences[i])
        for lang in self.languages:
            lang.tokenize()
    
                
            

class Language:
    def __init__(self,max_length):
        self.sentences = []
        self.reduced_sentences = []
        self.current_words:list[str]=None
        self.max_length=max_length
        self.tokenized_sentences=None
        self.reverse_dict=None
        self.word_dict=None
    def set_current_words(self,sentence) ->bool:
        self.current_words=sentence.split()
        return len(self.current_words)<=self.max_length
    def reset_current_words(self):
        self.current_words=None

    def append_current_words(self):
        self.sentences.append(self.current_words)

    def generate_count_dictionary(self):

        self.word_counts = {}
        for sentence in self.sentences:
            for word in sentence:
                value=self.word_counts.get(word,0)
                self.word_counts[word]=value+1
    def check_word_count(self,index:int,min_count:int):
        for word in self.sentences[index]:
            if self.word_counts[word]<min_count:
                return False
        return True     
    def tokenize(self):
        self.sentences=self.reduced_sentences
        self.reduced_sentences=None
        self.word_dict = {}
        self.pad_tokken=0
        self.word_dict["<PAD>"] = self.pad_tokken
        self.sos_token=1
        self.word_dict["<SOS>"] = self.sos_token
        self.eos_token=2
        self.word_dict["<EOS>"] = self.eos_token
        self.mask_token=3
        self.word_dict["<MASK>"] = self.mask_token
        self.generate_count_dictionary()
        for word, count in self.word_counts.items():
            self.word_dict[word] = len(self.word_dict)

        # tokenize sequences
        self.tokenized_sentences = []
        for sentence in self.sentences:
            sentence_tokens = []
            sentence_tokens.append(self.sos_token)
            for word in sentence:
                index=self.word_dict.get(word, None)
                if index is None:
                    # exit
                    print(f"Word {word} not found in dictionary")
                    exit()

                sentence_tokens.append(index)
            sentence_tokens.append(self.eos_token)
            self.tokenized_sentences.append(sentence_tokens)
    def tokenize_sentence(self,sentence):
        sentence_tokens = []
        sentence_tokens.append(self.sos_token)
        for word in sentence:
            index=self.word_dict.get(word, None)
            if index is None:
                # exit
                print(f"Word {word} not found in dictionary")
                exit()

            sentence_tokens.append(index)
        sentence_tokens.append(self.eos_token)
        return sentence_tokens
    def generate_reverse_dict(self):
        self.reverse_dict = {v: k for k, v in self.word_dict.items()}
        return self.reverse_dict
    


                  