import tkinter as tk
import gzip

metapath='c:/data/enzymes.txt.gz'
# create simple Frame

class Taxonomy:
    def __init__(self,taxid):
        self.taxid=taxid
        self.sequences=[]
        self.type=None #0=unknown, 1=Eukryota,2=prokaryota
    
class Sequence:
    def __init__(self,sequence,taxonomie):
        self.sequence=sequence
        self.reactions:list[Reaction]=[]
        self.taxonomie:Taxonomy=taxonomie
class Reaction:
    def __init__(self,reaction,taxonomy:Taxonomy,metabolite_dict:dict[str,set[Taxonomy]]):    
        self.reaction=reaction
        # split reaction by = and +
        substring=reaction.split(" = ")
        for substr in substring:
            substr=substr.strip()
            metaboites=substr.split(" + ")
            for metaboite in metaboites:
                metaboite=metaboite.strip()
                # check if metaboite is already in dictionary
                taxonomies=metabolite_dict.get(metaboite)
                if taxonomies==None:
                    taxonomies=set()
                    metabolite_dict[metaboite]=taxonomies
                taxonomies.add(taxonomy)
                
class MetaboliteDB:
    def __init__(self):
        # load the text of the metabolites
        # read line by line
        linecount = 0
        # print first 1000 lines
        self.taxonomies:dict[int,Taxonomy]={}
        self.metaboliteDict:dict[str,set[Taxonomy]]=dict()

        with gzip.open(metapath, "rt", encoding="utf-8-sig") as file:
            # read line by line
            while True:
                # sequence
                sequence = file.readline().strip()
                if not sequence:
                    break
                linecount += 1
                taxid = file.readline().strip()
                reaction_count = file.readline()
                reactions:list[Reaction] = []
                # check if taxid is already in dictionary
                taxon=self.taxonomies.get(int(taxid))
                if taxon==None:
                    taxon=Taxonomy(int(taxid))
                    self.taxonomies[int(taxid)]=taxon
                seq=Sequence(sequence,taxon)
                taxon.sequences.append(seq)
                for i in range(int(reaction_count)):
                    reaction_string = file.readline().strip()
                    reaction=Reaction(reaction_string,taxon,self.metaboliteDict)
                    reactions.append(reaction)
                seq.reactions=reactions
    
    def search_metabolite(self,metabolite):
        # find metabolite in dictionary
        taxonomies=self.metaboliteDict.get(metabolite)
        for taxonomy in taxonomies:
            print(taxonomy.taxid)
    def list_taxonomies(self):
        print(f'Number of taxonomies: {len(self.taxonomies)}')


mDB=MetaboliteDB()
mDB.list_taxonomies()
