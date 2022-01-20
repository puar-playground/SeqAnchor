from Bio import Entrez, GenBank
import urllib3

Entrez.email = "jchen378@buffalo.edu"
info = Entrez.esearch(db="nucleotide", term="SARS-CoV-2[ORGN]complete+genome", rettype="gb", retmax=1000, idtype="acc")
print('search done')
# record = Entrez.read(info)
# count = record['Count']
# info = Entrez.esearch(db="nucleotide", term="SARS-CoV-2[ORGN]complete+genome", rettype="gb", retmax=count, idtype="acc")
record = Entrez.read(info)
print(len(record['IdList']))
print(record['IdList'])

for id in record['IdList']:

    handle = Entrez.efetch(db="nucleotide", id=id, rettype="gb", retmode="text")
    data = handle.read()
    handle.close()

    out_handle = open('./Covid-19/' + id + '.db', 'w')
    out_handle.write(data)
    out_handle.close()

    # break
