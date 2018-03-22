# DNA nucleotide sequence feature exploration

## Background
In this work, we focused on exploring different representation approaches
for nucleotide sequences of a piece of DNA sequence.
We first sample sequences from exosome coding region and other non-coding
regions. 

### [Assumption]: 
The effectiveness of nucleotide sequence representation is evaluated
with the classification performance of exosome coding region and random non-coding region.

### [Representation]:
* One-hot encoding
* k-mer embedding

1. One-hot encoding experiments

| bin size |  sample num |  CNN  |  MLP |
|----------|-------------|-------|------|
| 100      | 372027 * 2  | 82.2% | 78.8%|
| 1000     |   3983 * 2  | 87%   | 74.8% | 


2. k-mer embedding experiments



