g++ sort.cpp -o sort
./fromDirectToUndirect P1a.tsv P1a.tsv
./sort P1a.tsv
./fromDirectToUndirect U1a.tsv U1a.tsv
./sort U1a.tsv
./fromDirectToUndirect V2a.tsv V2a.tsv
./sort V2a.tsv
./fromDirectToUndirect V1r.tsv V1r.tsv
./sort V1r.tsv
./fromDirectToUndirect graph500-scale25-ef16_adj.mmio graph500-scale25-ef16_adj.mmio
./sort graph500-scale25-ef16_adj.mmio