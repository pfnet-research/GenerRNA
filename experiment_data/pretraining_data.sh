wget -O ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz

gunzip rnacentral_active.fasta.gz

# deduplication (mmseqs2 required)
mmseqs createdb rnacentral_active.fasta rnacentral_db
mmseqs cluster rnacentral_db rnacentral_cluster tmp --min-seq-id 0.8 -c 0.8
mmseqs createsubdb rnacentral_cluster rnacentral_db rnacentral_rep
mmseqs convert2fasta rnacentral_rep rnacentral_clustered.fasta
rm -rf tmp
