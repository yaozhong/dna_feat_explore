## DNA nucleotide sequence feature exploration

### Background
In this work, we focused on exploring different represenation approaches
for nucleotide sequences of a piece of DNA sequence.
We first sample sequences from exosome coding region and other non-coding
regions. 

#### [Assumption]: 
The effectiveness of sequence representation is measured with the classificaton performance of
exsome coding and non-coding sequences. 

#### [Represeantions]:
* One-hot coding
* k-mer embedding

##### One-hot coding:
| bin Size  |  Data Size | CNN | MLP |
| ---- |  ---: | :--:  | :---  |
|1000 |  3984*2  |  88%      |  77.8%  |
|100 |  372027 * 2 |  82.2%  | 78.4%   |
