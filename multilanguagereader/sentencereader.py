import gzip

 
class Sentence:
    def __init__(self, id, lang, text):
        self.id = id
        self.lang = lang
        self.text = text

class Sentences:
    def __init__(self):
        with gzip.open('d:/translations/sentences.csv.gz', 'rt',encoding='utf-8') as file:
            self.senteces = {}
            for line in file:
                line=line.strip()
                # split by tab
                parts=line.split('\t')
                s=Sentence(parts[0],parts[1],parts[2])
                self.senteces[s.id]=s
    def extract_languages(self):
        langs=set()
        for s in self.senteces.values():
            langs.add(s.lang)
        return langs
    

if __name__ == '__main__':
    sentece=Sentences()
    langs=sentece.extract_languages()
    for lang in langs:
        print(lang)
