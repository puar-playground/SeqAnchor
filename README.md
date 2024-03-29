# :anchor: SeqAnchor :anchor:

This is a toy demo using the ASM network structure to learn a set of patterns that captures biological sequence homology and use the model to do Multiple Sequence Alignment (MSA). The model will be trained in an unsupervised way by manipulating the 3rd and 4th standardized moments of the channel activation distribution. Patterns are forced to find their best matches on different inputs in the same order without overlapping.

![Test Image 1](gray_spike.png)
![Test Image 1](gray_anchor.png)


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


|                                     Datasets                                     | Classification   Tasks in MoleculeNet |      |         |       |             |       |      |      |      |      |
|:--------------------------------------------------------------------------------:|:-------------------------------------:|:----:|:-------:|:-----:|:-----------:|:-----:|:----:|:----:|:----:|:----:|
|                                                                                  |                  BBBP                 | BACE | ClinTox | Tox21 | ToxC   last | SIDER |  HIV | PCBA |  MUV | MEAN |
|                                Standard CL Eq. (1)                               |                  69.3                 | 81.5 |   84.1  |  75.5 |     63.4    |  58.9 | 78.3 | 84.1 | 72.5 | 75.2 |
|                               Standard CL+ 3D loss                               |                  75.1                 | 86.8 |   87.9  |  78.9 |     68.5    |  62.8 | 81.8 | 88.0 | 77.1 | 78.1 |
|                       Standard CL + Probabilistic Framework                      |                  74.1                 | 86.3 |   88.2  |  79.5 |     68.2    |  63.1 | 82.5 | 88.4 | 77.1 | 78.6 |
|                                Standard CL + Both                                |                  76.7                 | 88,2 |   89.4  |  80.1 |     69.9    |  63.6 | 83.0 | 89.6 | 79.0 | 80,1 |
|                       Naive CL using Info NCE loss function                      |                   　                  |      |         |       |             |       |      |      |      |      |
| Table 1: MolecularNet classification performance with five   different baselines |                                       |      |         |       |             |       |      |      |      |      |
