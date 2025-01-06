import unicodedata
import re
import regex
import unicodedata

class StringPairs:

    def __init__(self):
        self.pairs = []
    def add_strings(self,s1,s2):
        s1=self.normalizeString(s1)
        s2=self.normalizeString(s2)
        self.pairs.append((s1,s2))
        
    def transform_strings(self):
        pass
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = regex.sub(r"([.!?])", r" \1", s)  # Add space before punctuation
        s = regex.sub(r"[^\p{L}]+", r" ", s)  # Keep all Unicode letters
        return s.strip()
    def print_strings(self):
        for s1,s2 in self.pairs:
            print(f'{s1}\t{s2}')
        pass
    def save(self,path):
        with open(path,'w') as f:
            for s1,s2 in self.pairs:
                f.write(f'{s1}\t{s2}\n')
        pass
    
    

