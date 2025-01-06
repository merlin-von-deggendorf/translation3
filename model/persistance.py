
import gzip
import regex
import os

class StringPairsFile:
    def __init__(self, id):
        self.id = id
        self.path = f"d:/translations/pairs/{self.id}.txt.gz"
        self.languages:list[Language] = None
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
            lang.generate_dict(min_count)
            

class Language:
    def __init__(self,max_length):
        self.sentences = []
        self.current_words:list[str]=None
        self.max_length=max_length
    def set_current_words(self,sentence) ->bool:
        self.current_words=sentence.split()
        return len(self.current_words)<=self.max_length
    def reset_current_words(self):
        self.current_words=None

    def append_current_words(self):
        self.sentences.append(self.current_words)
    
    def generate_dict(self,min_count=1):
        word_counts = {}
        for sentence in self.sentences:
            for word in sentence:
                value=word_counts.get(word,0)
                word_counts[word]=value+1
        self.word_dict = {}
        self.word_dict["<PAD>"] = 0
        self.pad_tokken=0
        self.word_dict["<SOS>"] = 1
        self.sos_token=1
        self.word_dict["<EOS>"] = 2
        self.eos_token=2
        self.word_dict["<UNK>"] = 3
        self.unk_token=3
        self.word_dict["<MASK>"] = 4
        self.mask_token=4
        for word, count in word_counts.items():
            if count >= min_count:
                self.word_dict[word] = len(self.word_dict)

        # tokenize sequences
        self.tokenized_sentences = []
        for sentence in self.sentences:
            sentence_tokens = []
            for word in sentence:
                sentence_tokens.append(self.word_dict.get(word, self.unk_token))
            self.tokenized_sentences.append(sentence_tokens)
                  