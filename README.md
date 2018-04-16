# ML PROJECT

This is the code implemented for the paper "Security of Latent Dirichlet Distribution" by Shike Mei and Xiaojin Zhu.
The paper can be found here: http://pages.cs.wisc.edu/~jerryzhu/machineteaching/pub/aistatsAttackLDA.pdf

The main code is in the ldaAttack_main.py. To run the code, simply run python ldaAttack_main.py.
The code only works on MAC/Ubuntu systems. The project has been completed using Blei's C code for LDA which can be found here:
https://github.com/blei-lab/lda-c/blob/master/readme.txt

The folder lda-c contains Blei's code. Minimal changes have been performed to the code to facilitate float corpus and so on. The attached corpus is the Congress Corpus and the code works for that corpus only. One can change the corpus by placing a corpus of his choice in the /convote_v1.1 in the correct folder and mention the folder name in the python file ldaAttack_main.py.
