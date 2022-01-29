# :anchor: SeqAnchor :anchor:

## Summary
We will use the ASM network structure to learn a set of patterns that captures biological sequence homology and use the model to do Multiple Sequence Alignment (MSA). The model will be trained in an unsupervised way by manipulating the 3rd and 4th standardized moments of the channel activation distribution. Patterns are forced to find their best matches on different inputs in the same order without overlapping.

![Test Image 1](gray_spike.png)
![Test Image 1](gray_anchor.png)

## Datasets
1. Greengenes, 16S rRNA dataset, length ~1400, diverse<br />
2. Zymo, 16S rRNA dataset, length ~1400, less diverse<br />
3. Silva 23S rRNA dataset, length ~3000 :x: <br />
4. Bralibase, A dataset for MSA benchmark, we will explore it.<br />
5. ITS dataset, I have played with it before, more work need to be done. They share a clear pattern on bigger picture, but might be too diverse locally.<br />
6. SARS-CoV-2 complete genome sequences from NCBI

## Benchmark
We plan to use the Average-of-Pairs (AP) score, the Mean-Square-Error (MSE) of pairwise alignment distance as the accuracy measures. We will compare our model with Muscle, MAFFT, CLUSTAL Omega, Kalign, T-Coffee on 4 or 5 datasets and report their accuracy and efficiency.

## Other Experiment
Maybe read more papers, I have uploaded 2. I am thinking to do some more CS-style experiment and publish it as a CS paper. <br />
1. We could compare the ASM structure with CNN structure. Since CNN is not robust to insertion/deletions, it will find less homology block (less anchors).<br />
2. We could do parameter sensitivity experiment to see how the pattern length affect performance.
3. We could use the anchor position imformation to design embeddings and train classifier. (this is actually useless) But we could only use the interval lengths as input to train a classifier. If it is capable of discriminating sequences on a reasonable taxonomy level, it could be something to present in the paper.
4. A more challenging task could be using our method to do MSA on Virus genomes. we might be able to download genome sequences of SARS-COV-2 (Covid 19). They are ~30000 bps long. So we need to modify our model and code for efficiency. (This is an eye-catching keyword)


## Python Virtualenv Preparation
Install virtualenv for python3
```
sudo pip3 install virtualenv
```
Create a virtual environment named venv3 or prefered directory
```
virtualenv -p python3 ~/venv3
```
Activate the python virtual environment and install packages.
```
source ./venv3/bin/activate
pip install -r environment.txt
```
Use the 'setup.py' script build the Cython program 'ASM.so'
```
cd src
python setup.py build_ext --inplace
```

## Benchmark work
Please download and install:<br />
:mag: Kalign:<br />
https://github.com/timolassmann/kalign <br />
:mag: T-Coffee:<br />
https://tcoffee.readthedocs.io/en/latest/tcoffee_installation.html<br />
:ballot_box_with_check: Muscle:<br />
https://www.drive5.com/muscle/<br />
:ballot_box_with_check: MAFFT:<br />
https://mafft.cbrc.jp/alignment/software/<br />
:ballot_box_with_check: Clustal Omega:<br />
http://www.clustal.org/omega/<br />


Then please use the default parameter settings to generate result on the 2 datasets:<br />
For datasets:<br />
./Dataset/Greengenes/train/gg_{0...9}.fa <br />
./Dataset/zymo/zymo_{0...9}.fa <br />
