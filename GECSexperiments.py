# Importing Packages
import numpy as np
import random
from numpy.random import default_rng

import ges
from ecDAGlearn import gecs
from coloredDAGgenerator import e_coloredDAGmodel
from ecmoves import (edgeList, nonEdgeList)


# compute structural hamming distance
def getSHD(A, B):
    shd = np.sum(np.logical_xor(A.astype(bool), B.astype(bool)))
    return shd

#compute edge true positive rate:
def edgeTPR(A, B):
    #B is taken as the ground truth
    B = B.astype(bool).astype(int)
    Bedges = edgeList(B)
    Aedges = edgeList(A)
    truepositives = []
    for e in Aedges:
        if Bedges.count(e) != 0:
            truepositives += [e]
    numTP = len(truepositives)
    numTrueEdges = len(Bedges)
    if numTrueEdges != 0:
        TPR = numTP / numTrueEdges
    else:
        TPR = -1

    return TPR

#computed edge true negative rate:
def edgeTNR(A, B):
    # B is taken as the ground truth
    Bnonedges = nonEdgeList(B)
    Anonedges = nonEdgeList(A)
    truenegatives = []
    for e in Anonedges:
        if Bnonedges.count(e) != 0:
            truenegatives += [e]
    numTN = len(truenegatives)
    numTrueNonEdges = len(Bnonedges)
    if numTrueNonEdges !=0:
        TNR = numTN / numTrueNonEdges
    else:
        TNR = -1

    return TNR

# colored edge true positives
def ecTPR(coloringA, coloringB):
    #coloringB is treated as the true coloring.
    Akeys = list(coloringA.keys())
    Bkeys = list(coloringB.keys())
    TPcount = 0
    Pcount = 0
    for c in Akeys:
        colorclass = coloringA[c]
        classsize = len(colorclass)
        for i in range(classsize):
            for j in range(i):
                for d in Bkeys:
                    if coloringB[d].count(colorclass[i]) != 0:
                        if coloringB[d].count(colorclass[j]) != 0:
                            TPcount += 1
    for d in Bkeys:
        bclasssize = len(coloringB[d])
        Pcount += bclasssize * (bclasssize - 1) / 2

    if Pcount != 0:
        colorTPR = TPcount / Pcount
    else:
        colorTPR = -1

    return colorTPR


# colored edge accuracy
def ecACC(A, coloringA, coloringB):
    numnodes = A.shape[0]
    TP = ecTP(coloringA, coloringB)
    TN = ecTN(coloringA, coloringB)
    total = numnodes * (numnodes - 1) / 2
    return (TP + TN) / total


# Simulations:

# Set number of desired experiments, nodes for tree and samples to be drawn.
num_exp = 25
num_nodes = 10
num_samples = 1000
# choose the minimum number of colors you would like the graph to have
# in each family.  When the randomly generated graph has sufficiently
# many edges in the family to use the specified number of colors it will.
# otherwise, it corrects the number of colors according to the number of edges
# in the family so that each color class can have at least two elements.
# Giving a list of different number of colors runs the above experimental settings for
# each coloring number in the list.
colorcounts = [2, 3, 4, 5, 6, 7, 8, 9] # for experiments on 10 nodes
# colorcounts = [2, 3, 4, 5] # for experiments on 6 nodes
num_colors = len(colorcounts)

# choose a list of probabilities of an edge between two nodes in the underlying
# (uncolored) DAGs to be generated.  Note that if minparents > 1 then edges may be
# added at random to the ER-DAG generated with the specified probability until each
# node with at least one parent in the initial ER-DAG has the specified number of parents.
probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_probs = len(probs)

# Seeds for data-generation (seed 57 used for exps on 10 nodes dagloop2)
seed = 57
random.seed(seed)
rng = default_rng(seed)
seeds = np.array(
    [[[np.random.randint(1000) for i in range(num_exp)] for j in range(num_probs)] for k in range(num_colors)]
)

# Set up storage for results
GECS_graphs = np.empty((num_colors, num_probs,  num_exp, num_nodes, num_nodes), int)
GECS_colorings = [[[0 for i in range(num_exp)] for j in range(num_probs)] for k in range(num_colors)]
GECS_shd = np.empty((num_colors, num_probs, num_exp), int)
GECS_TPR = np.empty((num_colors, num_probs, num_exp), float)
GECS_TNR = np.empty((num_colors, num_probs, num_exp), float)
GECS_cTPR = np.empty((num_colors, num_probs, num_exp), float)

GES_graphs = np.empty((num_colors, num_probs, num_exp, num_nodes, num_nodes), int)
GES_shd = np.empty((num_colors, num_probs, num_exp), int)
GES_TPR = np.empty((num_colors, num_probs, num_exp), float)
GES_TNR = np.empty((num_colors, num_probs, num_exp), float)


true_graphs = np.empty((num_colors, num_probs, num_exp, num_nodes, num_nodes), float)
true_colorings = [[[0 for i in range(num_exp)] for j in range(num_probs)] for k in range(num_colors)]

#Generate data, run simulations and store results
for c_idx in range(num_colors):
    for pr_idx in range(num_probs):
        for e_idx in range(num_exp):
            idx = int(e_idx)
            print('current color:', colorcounts[c_idx], 'current prob:', probs[pr_idx], 'experiment:', idx)

            # True model
            samples, graph, coloring, effects, errorvars = e_coloredDAGmodel(num_samples,
                                                                             num_nodes,
                                                                             prob=probs[pr_idx],
                                                                             minparents=2,
                                                                             numcolors=colorcounts[c_idx],
                                                                             seed=seeds[c_idx][pr_idx][idx])
            true_graphs[c_idx][pr_idx][idx] = graph - np.eye(num_nodes)
            true_colorings[c_idx][pr_idx][idx] = coloring

            # GECS
            GECSgraph, GECScoloring = gecs(samples)
            GECS_graphs[c_idx][pr_idx][idx] = GECSgraph
            GECS_colorings[c_idx][pr_idx][idx] = GECScoloring
            GECS_shd[c_idx][pr_idx][idx] = getSHD(GECSgraph, graph - np.eye(num_nodes))
            GECS_TPR[c_idx][pr_idx][idx] = edgeTPR(GECSgraph, graph - np.eye(num_nodes))
            GECS_TNR[c_idx][pr_idx][idx] = edgeTNR(GECSgraph, graph - np.eye(num_nodes))
            GECS_cTPR[c_idx][pr_idx][idx] = ecTPR(GECScoloring, coloring)

            # GES
            ges_graph, score = ges.fit_bic(samples)
            GES_graphs[c_idx][pr_idx][idx] = ges_graph
            GES_shd[c_idx][pr_idx][idx] = getSHD(ges_graph.T, graph - np.eye(num_nodes))
            GES_TPR[c_idx][pr_idx][idx] = edgeTPR(ges_graph.T, graph - np.eye(num_nodes))
            GES_TNR[c_idx][pr_idx][idx] = edgeTNR(ges_graph.T, graph - np.eye(num_nodes))

            np.savez(
                "gecs_exp_results_10_nodes_1000.npz",
                GECS_graphs=GECS_graphs,
                GECS_colorings=GECS_colorings,
                GECS_shd=GECS_shd,
                GECS_TPR=GECS_TPR,
                GECS_TNR=GECS_TNR,
                GECS_cTPR=GECS_cTPR,
                GES_graphs=GES_graphs,
                GES_shd=GES_shd,
                GES_TPR=GES_TPR,
                GES_TNR=GES_TNR,
                true_graphs=true_graphs,
                true_colorings=true_colorings
            )