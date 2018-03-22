# Basic data utility functions to process genomic data

from __future__ import division
import pyfaidx
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt

def cal_gc(seq, nlen):
	seq = seq.upper()
	gc = seq.count("G") + seq.count("C")
	return gc/nlen

def load_gtf(bedFile):
	bDic = {}
	total_region_cov = 0
	rgLens = []
	with open(bedFile) as f:
		for line in f:
			L = line.strip().split()

			if(len(L) != 3):
				print "[Warning]: Please check the format of gtf!! 3 columns please"
				exit(-1)

			if(bDic.has_key(L[0]) == False):
				bDic[L[0]] = []

			bDic[L[0]].append((int(L[1]), int(L[2])))
			rgLen = int(L[2]) - int(L[1])
			total_region_cov += rgLen
			rgLens.append(rgLen)

	print("** GTF file cover the region [{} nt]".format(total_region_cov))
	print("** minimum seq len:[{}], maximum seq len:[{}]".format(min(rgLens), max(rgLens)))

	return bDic

# assume the index is already sorted
def get_nc_region(chr_range_dic, exom_dic):

	nc_dic = {}

	for chr in chr_range_dic.keys():
		rgs = []
		region = [chr, chr_range_dic[chr][0][0], chr_range_dic[chr][0][1]]
		s_idx = chr_range_dic[chr][0][0]

		for rg in exom_dic[chr]:
			c_idx = rg[0]
			if(c_idx > s_idx):
				rgs.append((s_idx,c_idx))
				s_idx = rg[1]+1

		# add last elements into the lists
		rgs.append((s_idx, chr_range_dic[chr][0][1]))
		nc_dic[chr] = rgs

	return nc_dic
		
def get_chr_portion_noXY(chr_range_dic):

	lens = {}
	total_nt = 0
	for chr in chr_range_dic.keys():
		if chr not in ["X", "Y"]:
			l =  chr_range_dic[chr][0][1] + 1
			lens[chr] = l
			total_nt += l

	for chr in lens.keys():
		lens[chr] = lens[chr]/total_nt

	return lens

#############################################################
# function definition
#############################################################
class GENOMICSEQ:
	def __init__(self, ref_fasta, exome_gtf, chr_range_gtf):

		print("@ Loading reference genomic data ... ")
		self.gtf_dic = load_gtf(exome_gtf)
		self.fa_file = ref_fasta
		self.chr_range = load_gtf(chr_range_gtf)		
		# generate non_coding region lists
		self.nc_rg_dic = get_nc_region(self.chr_range, self.gtf_dic)

	# loading sequences form given regions
	def sample_ex_seqs(self, nlen=1000):
		seqs = []
		with pyfaidx.Fasta(self.fa_file, as_raw=True) as fa_file:

			for chr in self.gtf_dic.keys():
				# filter out X, Y chromesome
				if chr not in ['X', 'Y']: 
					for rg in self.gtf_dic[chr]:
						clen = rg[1] - rg[0] 
						if(clen >= nlen):
							seq = fa_file[chr][rg[0]:(rg[1]+1)]
							if seq.count("N") > 0:
								print("[WARNING!]: Chr{},[{},{}] has contains N...".format(chr,rg[0],rg[1]))
							# split long sequences into small sequences.
							for i in range(clen//nlen):
								i = i * nlen
								seqs.append(seq[i:i+nlen])
			print("** Total qualified {}nt seq is : {}".format(nlen,len(seqs)))
		return(seqs)

    # sample genomic sequences in non coding regions. 
	def sample_non_ex_seqs(self, nTotal=100000, nlen=1000):

		random.seed(100)
		seqs = []
		chr_portion = get_chr_portion_noXY(self.chr_range)

		with pyfaidx.Fasta(self.fa_file, as_raw=True) as fa_file:
			for chr in chr_portion.keys():
				target_n = round(chr_portion[chr]*nTotal)
				n = 0
				nc_rg = self.nc_rg_dic[chr]
				random.shuffle(nc_rg)
				# processing nc_regions
				for rg in nc_rg:

					if(n >= target_n):
						break

					clen = rg[1] - rg[0] 
					if(clen >= nlen):
						seq = fa_file[chr][rg[0]:(rg[1]+1)]

						if seq.count("N") == 0:
							for i in range(clen//nlen):
								i = i * nlen
								seqs.append(seq[i:i+nlen])
								n += 1
								if (n >= target_n):
									break

		return(seqs)

# sequence to one-hot encoding
def seq2onehot(seq):

	# note in the fasta file M, R is also contained
	CHARS = 'ACGT'
	dic = {'A':0, 'a':0, 'C':1, 'c':1, 'G':2, 'g':2, 'T':3, 't':3}
	CHARS_COUNT = len(CHARS)
	seqLen = len(seq)

	res = np.zeros((CHARS_COUNT, seqLen), dtype=np.float32) # original is uint8

	for i in xrange(seqLen):
		if seq[i] in dic:
			res[dic[seq[i]], i] = 1

	return res


# preparting the data for the deep learning training
def getData(nlen=100):
	
	ref_fasta="/data/Dataset/1000GP/reference/hs37d5.fa"
	exome_gtf="/data/Dataset/1000GP/phase3/p3_ref/exome_pull_down_targets/20130108.exome.targets.bed"
	chr_range_gtf="../ncEmbed_util/chr_range_h37.gtf"

	g = GENOMICSEQ(ref_fasta, exome_gtf, chr_range_gtf)
	ex_seqs = g.sample_ex_seqs(nlen)
	nc_seqs = g.sample_non_ex_seqs(len(ex_seqs),nlen)
	print("Final generated nc seq is [{}]".format(len(nc_seqs)))

	#ex_gcs = [cal_gc(x, nlen) for x in ex_seqs]
	#nc_gcs = [cal_gc(x, nlen) for x in nc_seqs]
	#plt.hist(ex_gcs, 200, facecolor='blue')
	#plt.hist(nc_gcs, 200, facecolor='red')
	#plt.savefig("../gc_hist.png")

	return({"ex":ex_seqs, "nc":nc_seqs})


#if __name__ == "__main__":
#	getData()



	



