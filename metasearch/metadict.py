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
    def __init__(self,reaction,metaboliteDB:dict[str,'Metabolite']):
        self.depth=-1
        self.sides:list[Side]=[]
        self.reaction=reaction
        self.sequences:list[Sequence]=[]
        self.tax_type:int=-1
        # split reaction by = and +
        substring=reaction.split(" = ")
        sidecount=len(substring)
        if sidecount<1 or sidecount>2:
            raise Exception(f"Invalid reaction {reaction}")
        for substr in substring:
            substr=substr.strip()
            side=Side()
            self.sides.append(side)
            metabolites=substr.split(" + ")
            for metabolite in metabolites:
                metabolite=metabolite.strip()
                met=metaboliteDB.get(metabolite)
                if met==None:
                    met=Metabolite()
                    metaboliteDB[metabolite]=met
                side.metabolites.append(met)

    def append_sequence(self,sequence:Sequence):
        self.sequences.append(sequence)
    def extract_tax_types(self):
        self.tax_type=-1
        for seq in self.sequences:
            if seq.taxonomie.type>self.tax_type:
                self.tax_type=seq.taxonomie.type
    def remove_from_set(self,met_set:set['Metabolite']):
        for side in self.sides:
            side.remove_from_set(met_set)
            

class Side:
    def __init__(self):
        self.metabolites:list[Metabolite]=[]
    def remove_from_set(self,met_set:set['Metabolite']):
        for met in self.metabolites:
            try:
                met_set.remove(met)
            except KeyError:
                pass
            
class Metabolite:
    def __init__(self):
        self.depth=-1
                
class MetaboliteDB:
    def __init__(self,metapath:str,taxDB:TaxDB):
        self.reactionDB:dict[str,Reaction]=dict()
        self.metaboliteDB:dict[str,Metabolite]=dict()
        self.taxonomieDB:dict[int,Taxonomy]=dict()
        self.unprocessedMetabolites:set[Metabolite]=set()
        self.unprocessedReactions:list[Reaction]=[]
        with gzip.open(metapath, "rt", encoding="utf-8-sig") as file:
            # read line by line
            while True:
                # sequence
                sequence = file.readline().strip()
                if not sequence:
                    break
                taxid = file.readline().strip()
                tid=int(taxid)
                reaction_count = file.readline()
                reactions:list[Reaction] = []
                # check if taxid is already in dictionary
                taxon=self.taxonomieDB.get(tid)
                if taxon==None:
                    taxtype=taxDB.get_base_type(tid)
                    taxon=Taxonomy(tid,taxtype)
                    self.taxonomieDB[int(taxid)]=taxon
                seq=Sequence(sequence,taxon)
                taxon.sequences.append(seq)
                for i in range(int(reaction_count)):
                    reaction_string = file.readline().strip()
                    reaction=self.reactionDB.get(reaction_string)
                    if reaction==None:
                        reaction=Reaction(reaction_string,self.metaboliteDB)
                        self.reactionDB[reaction_string]=reaction
                    reaction.append_sequence(seq)
                seq.reactions=reactions
        print(f'MetaboliteDB loaded with {len(self.reactionDB)} reactions and {len(self.metaboliteDB)} metabolites')
    def index(self):
        print("Indexing reactions")
        # add all reactions and metabolites to unprocessed lists
        for reaction in self.reactionDB.values():
            reaction.extract_tax_types()
            if reaction.tax_type==-1:
                print(f"Reaction {reaction.reaction} has no tax type")
            elif reaction.tax_type>0:
                self.unprocessedReactions.append(reaction)
        for metabolite in self.metaboliteDB.values():
            self.unprocessedMetabolites.add(metabolite)
        # traverse reactions in reverse order so we can remove them from the list
        precount=len(self.unprocessedMetabolites)
        for i in range(len(self.unprocessedReactions)-1,-1,-1):
            # if reaction is prokaryotic, we remove all metabolites from the list
            unprocessed=self.unprocessedReactions[i]
            if unprocessed.tax_type==2:
                unprocessed.remove_from_set(self.unprocessedMetabolites)
        postcount=len(self.unprocessedMetabolites)
        print(f"before: {precount} after: {postcount}")
        
    


