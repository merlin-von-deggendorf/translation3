import tkinter as tk
import gzip
from typing import TYPE_CHECKING
from metasearch import TaxDB
# import TaxDB for recognition by editor

    
# create simple Frame

class Taxonomy:
    def __init__(self,taxid,type:int):
        self.taxid=taxid
        self.sequences:list[Sequence]=[]
        self.type:int=type #0=unknown, 1=Eukryota,2=prokaryota
    
class Sequence:
    def __init__(self,sequence,taxonomie):
        self.sequence=sequence
        self.taxonomie:Taxonomy=taxonomie
        self.reactions:list[Reaction]=[]
class Reaction:
    def __init__(self,reaction,taxonomy:Taxonomy,metabolite_dict:dict[str,set[Taxonomy]]):
        self.sides:list[Side]=[]
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
class Side:
    def __init__(self):
        pass
class Metabolite:
    def __init__(self):
        self.depth=-1
                
class MetaboliteDB:
    def __init__(self,metapath:str,taxDB:TaxDB):
        self.reactionDB=[]
        self.metaboliteDB:dict[str,Metabolite]=dict()
        self.taxonomies:dict[int,Taxonomy]=dict()
        with gzip.open(metapath, "rt", encoding="utf-8-sig") as file:
            # read line by line
            while True:
                # sequence
                sequence = file.readline().strip()
                if not sequence:
                    break
                linecount += 1
                taxid = file.readline().strip()
                tid=int(taxid)
                reaction_count = file.readline()
                reactions:list[Reaction] = []
                # check if taxid is already in dictionary
                taxon=self.taxonomies.get(tid)
                if taxon==None:
                    taxtype=taxDB.get_base_type(tid)
                    taxon=Taxonomy(tid,taxtype)
                    self.taxonomies[int(taxid)]=taxon
                seq=Sequence(sequence,taxon)
                taxon.sequences.append(seq)
                for i in range(int(reaction_count)):
                    reaction_string = file.readline().strip()
                    reaction=Reaction(reaction_string,taxon,self.metaboliteDict)
                    reactions.append(reaction)
                seq.reactions=reactions
        print(f'Prokaryota: {pro} Eukaryota: {eu} Unknown: {unknown}')
    


