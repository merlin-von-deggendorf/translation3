from multilanguagereader.linkreader import Links,Link
from multilanguagereader.sentencereader import Sentences,Sentence
from model import StringPairsFile

import typing
if typing.TYPE_CHECKING:
    from model import StringPairs

  

class Persistence:
    def __init__(self,lang1:str,lang2:str):
        if lang1==lang2:
            raise ValueError("Languages must be different")
        if lang1<=lang2:
            self.lang1=lang1
            self.lang2=lang2
        else:
            self.lang1=lang2
            self.lang2=lang1
        self.spf=StringPairsFile(f"trans_{self.lang1}_{self.lang2}")
    def generate_pair(self):
        # generate 
        sentences=Sentences()
        links=Links()
        
        for link in links.links:
            s1=sentences.senteces.get(link.id1)
            s2=sentences.senteces.get(link.id2)
            if s1 and s2:
                if s1.lang == self.lang1 and s2.lang == self.lang2 or s1.lang == self.lang2 and s2.lang == self.lang1:
                    self.spf.add_pair(s1.text,s2.text)
        self.spf.save()
    def get_sentece_pairs(self):
        # if it doesn't exist, generate it
        if not self.spf.does_exist():
            self.generate_pair()
        
        return self.spf