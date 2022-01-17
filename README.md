# :anchor: SeqAnchor :anchor:

## Goal
We will use the ASM network structure to learn a set of patterns that captures biological sequence homology and use the model to do multiple sequence alignment. The model will be trained in an unsupervised way by manipulating the 3rd and 4th standardized moments of the channel activation distribution. Patterns are forced to find their best matches on different inputs in the same order without overlapping.

![Test Image 1](gray_spike.png)
![Test Image 1](gray_anchor.png)

## Data
Greengenes, 16S rRNA dataset, length ~1400, diverse\n
Zymo, 16S rRNA dataset, length ~1400, less diverse
Silva 23S rRNA dataset, length ~3000

Bralibase, A dataset for MSA benchmark, we will explore it.
ITS dataset, I have played with it before, more work need to be done. They share a clear pattern on bigger picture, but might be too diverse locally.


## Benchmark
We plan to use the Average-of-Pairs (AP) score, the Mean-Square-Error (MSE) of pairwise alignment distance as the accuracy measures. We will compare our model with Muscle, MAFFT, CLUSTAL Omega, Kalign, T-Coffee on 4 or 5 datasets and report their accuracy and efficiency.
