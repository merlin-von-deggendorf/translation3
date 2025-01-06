import gzip

class Link:
    def __init__(self, id1, id2):
        self.id1 = id1
        self.id2 = id2

class Links:
    def __init__(self):
        self.links=[]
        with gzip.open('d:/translations/links.csv.gz', 'rt',encoding='utf-8') as file:
            for line in file:
                line=line.strip()
                splits=line.split('\t')
                l=Link(splits[0],splits[1])
                self.links.append(l)

if __name__ == '__main__':
    links=Links()
    
