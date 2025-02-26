from metasearch import TaxDB
from metasearch import MetaboliteDB

tax=TaxDB("d:/uniprot/")
metDB=MetaboliteDB('c:/data/enzymes.txt.gz',tax)
print("TaxDB and MetaboliteDB loaded")
