class TaxDB:
    def __init__(self,path:str):
        self.nodes:dict[int,'Node']={}
        # get nodes.dmp
        nodespath=path+"nodes.dmp"
        mergedpath=path+"merged.dmp"
        namespath=path+"names.dmp"
        self.read_nodes(nodespath)
        self.read_merged(mergedpath)
        # self.read_names(namespath)

    def read_nodes(self,nodepath:str):
        # read all nodes line by line
        with open(nodepath) as file:
            while True:
                line=file.readline()
                if not line:
                    break
                # split line by |
                line=line.split("|")
                # get taxid
                taxid=int(line[0].strip())
                # get parent taxid
                parent=int(line[1].strip())
                # create node
                node:Node=self.nodes.get(taxid)
                parent_node:Node=self.nodes.get(parent)

                if node is None:
                    node=Node(taxid)
                    self.nodes[taxid]=node
                if parent_node is None:
                    parent_node=Node(parent)
                    self.nodes[parent]=parent_node
                if node.parent is None:
                    node.parent=parent_node
                else:
                    raise Exception("Node already has a parent")

            
    def read_merged(self,mergedpath:str):
        # read all merged line by line
        with open(mergedpath) as file:
            while True:
                line=file.readline()
                if not line:
                    break
                # split line by |
                line=line.split("|")
                # get taxid
                old=int(line[0].strip())
                # get parent taxid
                new=int(line[1].strip())
                node=self.nodes.get(new)
                if node is None:
                    print("Node not found")
                self.nodes[old]=node
    def read_names(self,namespath:str):
        # read all names line by line
        with open(namespath) as file:
            while True:
                line=file.readline()
                if not line:
                    break
                # split line by |
                line=line.split("|")
                # get taxid
                taxid=int(line[0].strip())
                # get name
                unique_name=line[1].strip()
                # check if unique name is available
                unique=line[2].strip()
                # check if unique contains anything but whitespace
                
                if unique!="":
                    # get name
                    unique_name=unique
    def get_base_type(self,taxid:int)->int: #0=unknown, 1=Eukryota,2=prokaryota
    #      public enum RootType
    # {
    #     Prokaryote = 2759, Eukaryote = 2, unknown = -1
    # }
        node=self.nodes.get(taxid)
        while node is not None:
            if node.taxid==2759:
                return 2
            if node.taxid==2:
                return 1
            node=node.parent
        return 0

class Node:
    def __init__(self,taxid:int):
        self.taxid=taxid
        self.parent:Node=None