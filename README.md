# coloredDAGs
A python library for colored Gaussian DAG models.

Current methods support learning BPEC-DAGs (blocked, properly edge-colored DAGs).  Edge colors are used to represent that two edges in the DAG carry the same causal effect. In a BPEC-DAG two arrows have the same color only if they have the same target (i.e., point into the same node). The learned colors therefore cluster the direct causes of each node into communities of variables that have similar effects on their common target node. 

To generate a random BPEC-DAG you may use the function e_coloredDAGmodel() in coloredDAGgenerator.py. 

To learn a BPEC-DAG from data, you may use the GECS (Greedy Edge-Colored Search) algorithm gecs() in ecDAGlearn.py.  Data should be in the form of an numpy array. The output of gecs() is a list in which the first element is the transpose of the adjacency matrix of the learned DAG.  The second element is the learned edge-coloring stored as a dictionary of color classes. Each edge in the dictionary is stored as a list where the first element of the list is the head node of the edge.
