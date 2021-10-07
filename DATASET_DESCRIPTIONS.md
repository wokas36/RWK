# README for Dataset Descriptions
=============================================================================================================
We use tweleve benchmark datasets grouped in two catagories: 
(1) Graphs with discrete attributes
(2) Graphs with continuous attributes

=============================================================================================================
## (1) Graphs with discrete attributes
=============================================================================================================

==================================== MUTAG ==================================================================

The MUTAG dataset consists of 188 chemical compounds divided into two 
classes according to their mutagenic effect on a bacterium. 

The chemical data was obtained form http://cdb.ics.uci.edu and converted 
to graphs, where vertices represent atoms and edges represent chemical 
bonds. Explicit hydrogen atoms have been removed and vertices are labeled
by atom type and edges by bond type (Single, Double, Triple or Aromatic).

Node labels:

  0  C
  1  N
  2  O
  3  F
  4  I
  5  Cl
  6  Br

Edge labels:

  0  Aromatic
  1  Single
  2  Double
  3  Triple

================================= PTC =======================================================================

The PTC dataset contains compounds labeled according to carcinogenicity 
on rodents divided into male mice (MM), male rats (MR), female mice (FM)
and female rats (FR). The PTC-MR dataset consists of 344 compounds.

The chemical data was obtained form http://www.predictive-toxicology.org/ptc/
and converted to graphs, where vertices represent atoms and edges 
represent chemical bonds. Explicit hydrogen atoms have been removed and
vertices are labeled by atom type and edges by bond type (Single, Double,
Triple or Aromatic).

Node labels:

  0  In
  1  P
  2  O
  3  N
  4  Na
  5  C
  6  Cl
  7  S
  8  Br
  9  F
  10  K
  11  Cu
  12  Zn
  13  I
  14  Ba
  15  Sn
  16  Pb
  17  Ca

Edge labels:

  0  Triple
  1  Double
  2  Single
  3  Aromatic

======================================= D&D ==================================================================

D&D is a dataset of 1178 protein structures (Dobson and Doig, 2003). Each protein is 
represented by a graph, in which the nodes are amino acids and two nodes are connected 
by an edge if they are less than 6 Angstroms apart. The prediction task is to classify 
the protein structures into enzymes and non-enzymes.


======================================= NCI1 and NCI109 ======================================================

NCI1 and NCI109 represent two balanced subsets of datasets of chemical compounds screened 
for activity against non-small cell lung cancer and ovarian cancer cell lines respectively
(Wale and Karypis (2006) and http://pubchem.ncbi.nlm.nih.gov). 
The NCI1 and NCI109 datasets consist of 4110 and 4127 chemical compounds, respectively.

======================================= COLLAB ===============================================================

The COLLAB dataset is a social network dataset and it represents the scientific collaborations. 
it is derived from three public collaboration datasets: 

(1) Astro Physics
(2) Condensed Matter Physics 
(3) High Energy Physics

Each graph of this dataset represents the ego-network for different researches on each field. 
Usually, social network datasets has not any node attributes, thus, we construct the discrete labels 
for nodes by one-hot encoding of the node degrees.
The COLLAB dataset consists of 5000 ego-networks.


===============================================================================================================
## (2) Graphs with continuous attributes
===============================================================================================================

======================================= COX2 and BZR===========================================================

BZR and COX2 are derived from small molecules, where each class label represent a certain biological 
properties such as activity and toxicity against cancer cells. The edges and vertices of a graph associates 
with the chemical bond an atoms, respectively. 
The COX2 and BZR datasets consist of 467 and 405 chemical compounds, respectively.

======================================= COX2-MD =============================================================== 

Dataset derived from the chemical compound dataset COX2 which comes with 
3D coordinates. COX2-MD has continuous edge weights and node attributes.
We generated complete graphs from the compounds, where  edges are attributed 
with distances and labeled with the chemical bond type (single, double, triple or aromatic).
The COX2-MD dataset consists of 303 chemical compounds.

======================================= BZR-MD ================================================================ 

Dataset derived from the chemical compound dataset BZR which comes with 
3D coordinates. BZR-MD has continuous edge weights and node attributes.
We generated complete graphs from the compounds, where edges are attributed 
with distances and labeled with the chemical bond type (single, double, triple or aromatic).
The BZR-MD dataset consists of 306 chemical compounds.

======================================= PROTEINS ==============================================================

PROTEINS are derived from protein tertiary structures, where nodes are secondary structure elements, 
which belongs to the amino acid sequences. It consists of three class labels representation: 
(1) Turn
(2) Sheet 
(3) Helix

The PROTEINS dataset consists of 1113 protein structures.

======================================= ENZYMES =============================================================== 

ENZYMES is a dataset of protein tertiary structures obtained from (Borgwardt et al., 2005) 
consisting of 600 enzymes from the BRENDA enzyme database (Schomburg et al., 2004). 
In this case the task is to correctly assign each enzyme to one of the 6 EC top-level 
classes.

========================================== References =========================================================

[1] Debnath, A.K., Lopez de Compadre, R.L., Debnath, G., Shusterman, A.J., and Hansch, C.
Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds.
Correlation with molecular orbital energies and hydrophobicity. J. Med. Chem. 34(2):786-797 (1991).

[2] Kriege, N., Mutzel, P.: Subgraph matching kernels for attributed graphs. In: Proceedings
of the 29th International Conference on Machine Learning (ICML-2012) (2012).

[3] P. D. Dobson and A. J. Doig. Distinguishing enzyme structures from non-enzymes without 
alignments. J. Mol. Biol., 330(4):771–783, Jul 2003.

[4] Sutherland, J. J.; O'Brien, L. A. & Weaver, D. F. Spline-fitting with a
genetic algorithm: a method for developing classification structure-activity
relationships. J. Chem. Inf. Comput. Sci., 2003, 43, 1906-1915

[5] K. M. Borgwardt, C. S. Ong, S. Schoenauer, S. V. N. Vishwanathan, A. J. Smola, and H. P. 
Kriegel. Protein function prediction via graph kernels. Bioinformatics, 21(Suppl 1):i47–i56, 
Jun 2005.

[6] I. Schomburg, A. Chang, C. Ebeling, M. Gremse, C. Heldt, G. Huhn, and D. Schomburg. Brenda, 
the enzyme database: updates and major new developments. Nucleic Acids Research, 32D:431–433, 2004.
