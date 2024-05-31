import numpy as np
import copy
from edgeColorScore import score
from ecmoves import (
addColor,
addEdge,
moveEdge,
reverseEdge,
mergeColors,
removeEdge,
splitColor,
removeColor
)


# loop over phases:
def ecDAGloop(samples, A, coloring, BIC):

    same = 0
    while same == 0:
        proposalA, proposalc = addColor(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC

        proposalA, proposalc = splitColor(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC
        else:
            same = 1

    same = 0
    while same == 0:

        proposalA, proposalc = addEdge(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC

        proposalA, proposalc = moveEdge(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC

        proposalA, proposalc = reverseEdge(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC

        proposalA, proposalc = removeEdge(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC
        else:
            same = 1

    same = 0
    while same == 0:
        proposalA, proposalc = mergeColors(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC

        proposalA, proposalc = removeColor(A, coloring, samples)
        proposalBIC = score(samples, proposalA, proposalc)
        if proposalBIC > BIC:
            A = proposalA
            coloring = proposalc
            BIC = proposalBIC
        else:
            same = 1

    return A, coloring, BIC


# GECS algorithm:
def gecs(samples):
    numnodes = samples.shape[1]
    copyA = np.zeros([numnodes, numnodes])
    copycoloring = {}
    BIC = score(samples, copyA, copycoloring)

    same = 0
    while same == 0:
        newA, newcoloring, newBIC = ecDAGloop(samples, copyA, copycoloring, BIC)
        if newBIC == BIC:
            same = 1
        else:
            BIC = newBIC
            copycoloring = copy.deepcopy(newcoloring)
            copyA = copy.deepcopy(newA)

    return copyA, copycoloring
