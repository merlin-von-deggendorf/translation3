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
        # print all metabolites
        print('data loaded')
        self.window = tk.Tk()
        self.window.title("Metabolite Translator")
        self.window.geometry("800x600")
        # write linecount to the window
        self.label = tk.Label(self.window, text=f"Number of metabolites: {linecount}")
        self.label.grid(row=0, column=0)
        # create input field
        self.entry = tk.Entry(self.window)
        self.entry.grid(row=1, column=0)
        # create button
        self.button = tk.Button(self.window, text="Search Metabolite", command=self.search_metabolite)
        self.button.grid(row=1, column=1)
        # create textarea which can be scrolled
        self.textarea = tk.Text(self.window)
        # disable word wrap
        self.textarea.config(wrap="none")
        self.textarea.grid(row=2, column=0, columnspan=2)
        # create scrollbar
        self.scrollbar = tk.Scrollbar(self.window, orient="vertical", command=self.textarea.yview)
        self.scrollbar.grid(row=2, column=2, sticky="ns")
        self.textarea.config(yscrollcommand=self.scrollbar.set)
        # also add scrollbar for x-axis
        self.scrollbarx = tk.Scrollbar(self.window, orient="horizontal", command=self.textarea.xview)
        self.scrollbarx.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.textarea.config(xscrollcommand=self.scrollbarx.set)

        
        # print all metabolites to the textarea
        for metabolite in self.metaboliteDict.keys():
            self.textarea.insert(tk.END, metabolite + "\n")
        self.window.mainloop()
    
    def search_metabolite(self):
        # get the text from the input field
        metabolite = self.entry.get()
        # find metabolite in dictionary
        taxonomies=self.metaboliteDict.get(metabolite)
        for taxonomy in taxonomies:
            print(taxonomy.taxid)

class MetaboliteGui:
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
        # print all metabolites
        print('data loaded')
        self.window = tk.Tk()
        self.window.title("Metabolite Translator")
        self.window.geometry("800x600")
        # write linecount to the window
        self.label = tk.Label(self.window, text=f"Number of metabolites: {linecount}")
        self.label.grid(row=0, column=0)
        # create input field
        self.entry = tk.Entry(self.window)
        self.entry.grid(row=1, column=0)
        # create button
        self.button = tk.Button(self.window, text="Search Metabolite", command=self.search_metabolite)
        self.button.grid(row=1, column=1)
        # create textarea which can be scrolled
        self.textarea = tk.Text(self.window)
        # disable word wrap
        self.textarea.config(wrap="none")
        self.textarea.grid(row=2, column=0, columnspan=2)
        # create scrollbar
        self.scrollbar = tk.Scrollbar(self.window, orient="vertical", command=self.textarea.yview)
        self.scrollbar.grid(row=2, column=2, sticky="ns")
        self.textarea.config(yscrollcommand=self.scrollbar.set)
        # also add scrollbar for x-axis
        self.scrollbarx = tk.Scrollbar(self.window, orient="horizontal", command=self.textarea.xview)
        self.scrollbarx.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.textarea.config(xscrollcommand=self.scrollbarx.set)

        
        # print all metabolites to the textarea
        for metabolite in self.metaboliteDict.keys():
            self.textarea.insert(tk.END, metabolite + "\n")
        self.window.mainloop()
    
    def search_metabolite(self):
        # get the text from the input field
        metabolite = self.entry.get()
        # find metabolite in dictionary
        taxonomies=self.metaboliteDict.get(metabolite)
        for taxonomy in taxonomies:
            print(taxonomy.taxid)
        

if __name__ == "__main__":
    gui = MetaboliteGui()