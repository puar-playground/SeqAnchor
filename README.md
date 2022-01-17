# :anchor: SeqAnchor :anchor:

## Summary
We will use the ASM network structure to learn a set of patterns that captures biological sequence homology and use the model to do Multiple Sequence Alignment (MSA). The model will be trained in an unsupervised way by manipulating the 3rd and 4th standardized moments of the channel activation distribution. Patterns are forced to find their best matches on different inputs in the same order without overlapping.

![Test Image 1](gray_spike.png)
![Test Image 1](gray_anchor.png)

## Data
Greengenes, 16S rRNA dataset, length ~1400, diverse<br />
Zymo, 16S rRNA dataset, length ~1400, less diverse<br />
Silva 23S rRNA dataset, length ~3000<br />

Bralibase, A dataset for MSA benchmark, we will explore it.<br />
ITS dataset, I have played with it before, more work need to be done. They share a clear pattern on bigger picture, but might be too diverse locally.<br />

## Benchmark
We plan to use the Average-of-Pairs (AP) score, the Mean-Square-Error (MSE) of pairwise alignment distance as the accuracy measures. We will compare our model with Muscle, MAFFT, CLUSTAL Omega, Kalign, T-Coffee on 4 or 5 datasets and report their accuracy and efficiency.

## Other Experiment
Maybe read more papers, I have uploaded 2. I am thinking to do some more CS-style experiment and publish it as a CS paper. <br />
1. We could compare the ASM structure with CNN structure. Since CNN is not robust to insertion/deletions, it will find less homology block (less anchors).<br />
2. We could do parameter sensitivity experiment to see how the pattern length affect performance.
3. We could use the anchor position imformation to design embeddings and train classifier. (this is actually useless) But we could only use the interval lengths as input to train a classifier. If it is capable of discriminating sequences on a reasonable taxonomy level, it could be something to present in the paper.
4. A more challenging task could be using our method to do MSA on Virus genomes. we might be able to download genome sequences of SARS-COV-2 (Covid 19). They are ~30000 bps long. So we need to modify our model and code for efficiency. (This is an eye-catching keyword)
