from multilanguagereader import Persistence
from model import StringPairs
from model import StringPairsFile
from multilanguagereader import Sentences
# stringpairs=StringPairs()
# persisitance.split('deu','eng',stringpairs)
# stringpairs.print_strings()


pers=Persistence('deu','eng')	
spf:StringPairsFile=pers.get_sentece_pairs()
spf.load(10)
spf.generate_dict()
for language in spf.languages:
    # print first 10 tokenized sentences
    print(language.tokenized_sentences[:10])