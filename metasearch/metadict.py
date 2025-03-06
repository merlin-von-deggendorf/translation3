import tkinter as tk
import gzip
from typing import TYPE_CHECKING
from metasearch import TaxDB,TaxType


class Taxonomy:
    def __init__(self,taxid,type:TaxType):
        self.taxid=taxid
        self.sequences:list[Sequence]=[]
        self.type:TaxType=type #0=unknown, 1=Eukryota,2=prokaryota
    
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
        self.tax_type:TaxType=TaxType.UNKNOWN
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
        self.tax_type=TaxType.UNKNOWN
        for seq in self.sequences:
            if seq.taxonomie.type>self.tax_type:
                self.tax_type=seq.taxonomie.type
    def add_2_set(self,met_set:set['Metabolite']):
        for side in self.sides:
            side.add_2_set(met_set)
    def check_metabolites(self,available:set['Metabolite']):
        for side in self.sides:
            for met in side.metabolites:
                if met not in available:
                    return False
        return True
    def check_xor_side(self,metset:set['Metabolite'])->bool:
        if len(self.sides)!=2:
            raise Exception("Invalid reaction")
        if self.sides[0].check_side(metset) and (not self.sides[1].check_side(metset)):
            return True
        if self.sides[1].check_side(metset) and (not self.sides[0].check_side(metset)):
            return True
        return False
        
    
 
    def check_side(self, available: set['Metabolite'], met_buffer: set['Metabolite']) -> bool:
        if len(self.sides) != 2:
            raise Exception("Invalid reaction")
        if self.sides[0].check_side(available) or self.sides[1].check_side(available):
            self.sides[0].add_2_set(met_buffer)
            self.sides[1].add_2_set(met_buffer)
            return True
        else:
            return False

        


class Side:
    def __init__(self):
        self.metabolites:list[Metabolite]=[]
    def add_2_set(self,met_set:set['Metabolite']):
        for met in self.metabolites:
            met_set.add(met)
    def check_side(self,available:set['Metabolite'])->bool:
        """
        Check if all metabolites of the side are available.
        Returns True if all metabolites are available, False otherwise.
        """
        for met in self.metabolites:
            if met not in available:
                return False
        return True
            
class Metabolite:
    def __init__(self):
        self.depth=-1



class MetaboliteDB:
    def __init__(self,metapath:str,taxDB:TaxDB):
        self.reactionDB:dict[str,Reaction]=dict()
        self.metaboliteDB:dict[str,Metabolite]=dict()
        self.taxonomieDB:dict[int,Taxonomy]=dict()
        self.unprocessedReactions:list[Reaction]=[]
        self.met_buffer:set[Metabolite]=set()
        self.available_metabolites:set[Metabolite]=set()
        self.reactionsBYdepth:list[list[Reaction]]=[]
        self.metabolitesBYdepth:list[set[Metabolite]]=[]
        self.unique_metabolites_by_depth:list[set[Metabolite]]=[]
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
                    taxtype:TaxType=taxDB.get_base_type(tid)
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
        unknown_count=0
        valid_count=0
        invalid_count=0
        for reaction in self.reactionDB.values():
            reaction.extract_tax_types()
            if reaction.tax_type==TaxType.UNKNOWN:
                unknown_count+=1
            elif reaction.tax_type==TaxType.EUKARYOTA and len(reaction.sides)<2:
                invalid_count+=1
            elif reaction.tax_type==TaxType.PROKARYOTA or reaction.tax_type==TaxType.EUKARYOTA:
                valid_count+=1
                self.unprocessedReactions.append(reaction)
            else:
                raise Exception("Invalid tax type")
                    
        print(f"Unknown reactions: {unknown_count} valid reactions: {valid_count} invalid reactions: {invalid_count} length of unprocessed reactions: {len(self.unprocessedReactions)}")
                    

        
        self.met_buffer.clear()
        # Iterate over indices in reverse order.
        for i in range(len(self.unprocessedReactions) - 1, -1, -1):
            reaction = self.unprocessedReactions[i]
            reaction.depth=-1
            if reaction.tax_type == TaxType.PROKARYOTA:
                # Remove metabolites related to the reaction
                reaction.add_2_set(self.met_buffer)
                # Remove the reaction from the list
                reaction.depth = 0
                self.unprocessedReactions.pop(i)
        # add buffer to available metabolites
        self.update_and_clear_buffer()
        depth=1
        while True:

            for i in range(len(self.unprocessedReactions) - 1, -1, -1):
                reaction = self.unprocessedReactions[i]
                if reaction.check_side(self.available_metabolites,self.met_buffer):
                    reaction.depth = depth
                    self.unprocessedReactions.pop(i)

                
            if len(self.met_buffer)==0:
                break
            else:
                self.update_and_clear_buffer()
                depth+=1
        print(f"Indexing done maximum depth: {depth}")
        for i in range(depth):
            self.reactionsBYdepth.append([])
            self.metabolitesBYdepth.append(set())
        print(f"Indexing reactions by depth")
        

        for reaction in self.reactionDB.values():
            self.reactionsBYdepth[reaction.depth].append(reaction)
            # add metabolites of reaction to metabolites by depth
            for side in reaction.sides:
                for met in side.metabolites:
                    self.metabolitesBYdepth[reaction.depth].add(met)
        # add each previous depth to the current depth
        for i in range(1,depth):
            self.metabolitesBYdepth[i].update(self.metabolitesBYdepth[i-1])
        # get unique metabolites for each depth
        self.unique_metabolites_by_depth.append(self.metabolitesBYdepth[0])
        for i in range(1,depth):
            self.unique_metabolites_by_depth.append(self.metabolitesBYdepth[i].difference(self.metabolitesBYdepth[i-1]))
        print(f"Indexing done")
        
        
        for i in range(depth):
            print(f"Depth {i} has {len(self.reactionsBYdepth[i])} reactions")
            print(f"Depth {i} has {len(self.metabolitesBYdepth[i])} metabolites")
            print(f"Depth {i} has {len(self.unique_metabolites_by_depth[i])} unique metabolites")
    def extract_relevant(self):
        # get all unique metabolites of depth 1
        relevant_metabolites=self.unique_metabolites_by_depth[1]
        reaction_candidates=self.reactionsBYdepth[1]
        # find all reactions that contain one of the relevant metabolites
        relevant=set()
        for met in relevant_metabolites:
            for reaction in reaction_candidates:
                if reaction.check_xor_side(self.unique_metabolites_by_depth[0]):
                    relevant.add(reaction)
        print(f"Found {len(relevant)} relevant reactions")
        

    def update_and_clear_buffer(self):
        self.available_metabolites.update(self.met_buffer)
        self.met_buffer.clear()





