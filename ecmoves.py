import numpy as np
import copy
from edgeColorScore import score

# check to make sure coloring edge list agrees with adjmat edgelist:
def coloringCheck(A, coloring):
    numnodes = A.shape[0]
    Abool = A.astype(bool)
    coloringkeys = list(coloring.keys())
    coloringedges = []
    for c in coloringkeys:
        coloringedges += coloring[c]
    C = np.zeros([numnodes, numnodes])
    for e in coloringedges:
        C[e[0], e[1]] = 1
    Cbool = C.astype(bool)
    shd = np.sum(np.logical_xor(Abool, Cbool))
    return shd

# check if a matrix is the adjacency matrix of a DAG:
def acyclic(A):
    decision = 0 # 0 represents acyclic
    numnodes = A.shape[0]
    P = A
    for i in range(numnodes-1):
        P = P @ A
    if np.sum(P) != 0:
        decision = 1
    return decision

def getFamily(i, A):
    numnodes = A.shape[0]
    fam = [j for j in range(numnodes) if A[i, j] == 1]
    return fam

# get color classes for family of j:
def getFamilyColors(j, coloring):
    coloringkeys = list(coloring.keys())
    familycolorkeys = []

    for c in coloringkeys:
        if coloring[c][0][0] == j:
            familycolorkeys += [c]

    return familycolorkeys

# get list of nonedges in adjacency matrix:
def nonEdgeList(A):
    numnodes = A.shape[0]
    nonedgelist = []

    for i in range(numnodes):
        for j in range(numnodes):
            if j != i:
                if A[i, j] == 0:
                    nonedgelist += [[i, j]]

    return nonedgelist

# get list of edges in adjacency matrix:
def edgeList(A):
    numnodes = A.shape[0]
    edgelist = []

    for i in range(numnodes):
        for j in range(numnodes):
            if j != i:
                if A[i, j] == 1:
                    edgelist += [[i, j]]

    return edgelist


# moves for increasing parameter count:

# add a color class:
def addColor(A, coloring, samples):
    numnodes = A.shape[0]
    B = copy.deepcopy(A)
    newcoloring = copy.deepcopy(coloring)

    additions = []
    for idx in range(numnodes):
        nonfamily = [j for j in range(numnodes) if B[idx, j] == 0]
        nonfamily.pop(nonfamily.index(idx))
        numnonfamily = len(nonfamily)
        if numnonfamily > 1:
            for j in range(numnonfamily):
                for k in range(j):
                    p1 = nonfamily[j]
                    p2 = nonfamily[k]
                    B[idx, p1] = 1
                    B[idx, p2] = 1
                    if acyclic(B) == 0:
                        curr_colors = list(newcoloring.keys())
                        new_key = len(curr_colors) + 1
                        while curr_colors.count(new_key) != 0:
                            new_key += 1
                        newcoloring.update({new_key: [[idx, p1], [idx, p2]]})
                        BIC = score(samples, B, newcoloring)
                        additions += [[idx, p1, p2, BIC]]
                        newcoloring.pop(new_key)
                        B[idx, p1] = 0
                        B[idx, p2] = 0
                    else:
                        B[idx, p1] = 0
                        B[idx, p2] = 0

    if len(additions) > 0:
        best = 0
        BIC = additions[0][3]
        for i in range(len(additions)):
            if additions[i][3] > BIC:
                BIC = additions[i][3]
                best = i
        bestnode = additions[best][0]
        bestp1 = additions[best][1]
        bestp2 = additions[best][2]
        B[bestnode, bestp1] = 1
        B[bestnode, bestp2] = 1
        bestcurr_colors = list(newcoloring.keys())
        bestnew_key = len(bestcurr_colors) + 1
        while bestcurr_colors.count(bestnew_key) != 0:
            bestnew_key += 1
        newcoloring.update({bestnew_key: [[bestnode, bestp1], [bestnode, bestp2]]})

    return B, newcoloring

# add an edge to an existing color class:
def addEdge(A, coloring, samples):
    numnodes = A.shape[0]
    newcoloring = copy.deepcopy(coloring)
    B = copy.deepcopy(A)

    nonedgelist = []
    for i in range(numnodes):
        for j in range(numnodes):
            if j != i:
                if B[i, j] == 0:
                    nonedgelist += [[i, j]]

    additions = []
    for edge in nonedgelist:
        pa = edge[1]
        ch = edge[0]
        B[ch, pa] = 1
        if acyclic(B) == 0:
            ch_familycolors = getFamilyColors(ch, newcoloring)
            numfamilycolors = len(ch_familycolors)
            if numfamilycolors > 0:
                for c in ch_familycolors:
                    newcoloring.update({c: newcoloring[c] + [edge]})
                    BIC = score(samples, B, newcoloring)
                    additions += [[edge, c, BIC]]
                    B[ch, pa] = 0
                    newcoloring.update({c: [x for x in newcoloring[c] if x[1] != pa]})
            else:
                B[ch, pa] = 0
        else:
            B[ch, pa] = 0

    if len(additions) > 0:
        best = 0
        BIC = additions[0][2]
        for i in range(len(additions)):
            if additions[i][2] > BIC:
                best = i
                BIC = additions[i][2]
        bestedge = additions[best][0]
        bestcolor = additions[best][1]
        bestpa = bestedge[1]
        bestch = bestedge[0]
        B[bestch, bestpa] = 1
        newcoloring.update({bestcolor: newcoloring[bestcolor] + [bestedge]})

    return B, newcoloring


# moves that keep parameter count the same:

# move an edge between color classes in the same family:
def moveEdge(A, coloring, samples):
    #moves an edge within the same family
    numnodes = A.shape[0]
    newcoloring = copy.deepcopy(coloring)
    B = np.zeros([numnodes, numnodes])
    for i in range(numnodes):
        for j in range(numnodes):
            if A[i, j] == 1:
                B[i, j] = 1

    moves = []
    for idx in range(numnodes):
        familycolors = getFamilyColors(idx, newcoloring)
        if len(familycolors) > 1:
            for c in familycolors:
                cedges = newcoloring[c]
                if len(cedges) > 2:
                    othercolors = [d for d in familycolors if d != c]
                    for edge in cedges:
                        for d in othercolors:
                            newcoloring.update({d: newcoloring[d] + [edge]})
                            newcoloring.update({c: [x for x in newcoloring[c] if x.count(edge[1]) == 0]})
                            newBIC = score(samples, B, newcoloring)
                            moves += [[edge, c, d, newBIC]]
                            newcoloring.update({c: newcoloring[c] + [edge]})
                            newcoloring.update({d: [x for x in newcoloring[d] if x.count(edge[1]) == 0]})

    if len(moves) != 0:
        best = 0
        BIC = moves[0][3]
        for i in range(len(moves)):
            if moves[i][3] > BIC:
                BIC = moves[i][3]
                best = i

        bestedge = moves[best][0]
        c = moves[best][1]
        d = moves[best][2]
        newcoloring.update({d: newcoloring[d] + [bestedge]})
        newcoloring.update({c: [x for x in newcoloring[c] if x.count(bestedge[1]) == 0]})

    return B, newcoloring

# reverse an edge:
def reverseEdge(A, coloring, samples):
    # removes an edge from one family and adds it to another
    numnodes = A.shape[0]
    newcoloring = copy.deepcopy(coloring)
    B = np.zeros([numnodes, numnodes])
    for i in range(numnodes):
        for j in range(numnodes):
            if A[i, j] == 1:
                B[i, j] = 1

    edges = edgeList(B)
    reversable = []
    for edge in edges:
        B[edge[0], edge[1]] = 0
        B[edge[1], edge[0]] = 1
        if acyclic(B) == 0:
            familycolors = getFamilyColors(edge[0], newcoloring)
            newfamilycolors = getFamilyColors(edge[1], newcoloring)
            for c in familycolors:
                if newcoloring[c].count(edge) != 0:
                    cidx = c
            if len(newcoloring[cidx]) > 2:
                if len(newfamilycolors) > 0:
                    for newc in newfamilycolors:
                        newcoloring.update({newc: newcoloring[newc] + [[edge[1], edge[0]]]})
                        newcoloring.update({cidx: [x for x in newcoloring[cidx] if x.count(edge[1]) == 0]})
                        BIC = score(samples, B, newcoloring)
                        reversable += [[edge, cidx, newc, BIC]]
                        newcoloring.update({cidx: newcoloring[cidx] + [edge]})
                        newcoloring.update({newc: [x for x in newcoloring[newc] if x.count(edge[0]) == 0]})
                        B[edge[0], edge[1]] = 1
                        B[edge[1], edge[0]] = 0
                else:
                    B[edge[0], edge[1]] = 1
                    B[edge[1], edge[0]] = 0
            else:
                B[edge[0], edge[1]] = 1
                B[edge[1], edge[0]] = 0
        else:
            B[edge[0], edge[1]] = 1
            B[edge[1], edge[0]] = 0

    if len(reversable) != 0:
        BIC = reversable[0][3]
        best = 0
        for i in range(len(reversable)):
            if reversable[i][3] > BIC:
                BIC = reversable[i][3]
                best = i

        bestedge = reversable[best][0]
        cidx = reversable[best][1]
        newc = reversable[best][2]
        newcoloring.update({newc: newcoloring[newc] + [[bestedge[1], bestedge[0]]]})
        newcoloring.update({cidx: [x for x in newcoloring[cidx] if x.count(bestedge[1]) == 0]})
        B[bestedge[0], bestedge[1]] = 0
        B[bestedge[1], bestedge[0]] = 1

    return B, newcoloring

# moves that decrease parameter count:

# merge two color classes:
def mergeColors(A, coloring, samples):
    numnodes = A.shape[0]
    B = copy.deepcopy(A)
    newcoloring = copy.deepcopy(coloring)

    merges = []
    for i in range(numnodes):
        familycolors = getFamilyColors(i, newcoloring)
        numfamcolors = len(familycolors)
        if numfamcolors > 1:
            for a in range(numfamcolors):
                for b in range(a):
                    colora = familycolors[a]
                    colorb = familycolors[b]
                    ca = [x for x in newcoloring[colora]]
                    cb = [x for x in newcoloring[colorb]]
                    cmerge = ca + cb
                    newcoloring.update({colora: cmerge})
                    newcoloring.pop(colorb)
                    BIC = score(samples, B, newcoloring)
                    merges += [[colora, colorb, BIC]]
                    newcoloring.update({colora: ca})
                    newcoloring.update({colorb: cb})

    if len(merges) > 0:
        best = 0
        BIC = merges[0][2]
        for i in range(len(merges)):
            if merges[i][2] > BIC:
                best = i
                BIC = merges[i][2]
        bestcolora = merges[best][0]
        bestcolorb = merges[best][1]
        bestca = [x for x in newcoloring[bestcolora]]
        bestcb = [x for x in newcoloring[bestcolorb]]
        bestcmerge = bestca + bestcb
        newcoloring.update({bestcolora: bestcmerge})
        newcoloring.pop(bestcolorb)

    return B, newcoloring

def removeEdge(A, coloring, samples):
    B = copy.deepcopy(A)
    newcoloring = copy.deepcopy(coloring)

    edges = edgeList(B)
    removals = []
    for edge in edges:
        #get color class of edge:
        ch = edge[0]
        pa = edge[1]
        ch_colorfamily = getFamilyColors(ch, newcoloring)
        for c in ch_colorfamily:
            if newcoloring[c].count(edge) != 0:
                cidx = c
        colorclass = newcoloring[cidx]
        edgeindex = colorclass.index(edge)
        classsize = len(colorclass)
        if classsize > 2:
            B[ch, pa] = 0
            colorclass.pop(edgeindex)
            newcoloring.update({cidx: colorclass})
            BIC = score(samples, B, newcoloring)
            removals += [[edge, cidx, edgeindex, BIC]]
            B[ch, pa] = 1
            newcoloring.update({cidx: newcoloring[cidx] + [edge]})

    if len(removals) > 0:
        best = 0
        BIC = removals[0][3]
        for i in range(len(removals)):
            if removals[i][3] > BIC:
                best = i
                BIC = removals[i][3]
        bestedge = removals[best][0]
        bestch = bestedge[0]
        bestpa = bestedge[1]
        best_ch_colorfamily = getFamilyColors(bestch, newcoloring)
        for c in best_ch_colorfamily:
            if newcoloring[c].count(bestedge) != 0:
                bestcidx = c
        bestcolorclass = newcoloring[bestcidx]
        bestedgeindex = bestcolorclass.index(bestedge)
        bestcolorclass.pop(bestedgeindex)
        B[bestch, bestpa] = 0
        newcoloring.update({bestcidx: bestcolorclass})

    return B, newcoloring


# split a color class:
def splitColor(A, coloring, samples):
    B = copy.deepcopy(A)
    newcoloring = copy.deepcopy(coloring)

    splits = []
    coloringkeys = list(newcoloring.keys())
    if len(coloringkeys) != 0:
        new_key = max(coloringkeys) + 1
        for c in coloringkeys:
            colorclass = newcoloring[c]
            classsize = len(colorclass)
            if classsize > 4:
                for e1_idx in range(classsize):
                    for e2_idx in range(e1_idx):
                        aug_colorclass = [e for e in colorclass if e != colorclass[e1_idx]]
                        aug_colorclass = [e for e in aug_colorclass if e != colorclass[e2_idx]]
                        new_colorclass = [colorclass[e1_idx], colorclass[e2_idx]]
                        newcoloring.update({c: aug_colorclass})
                        newcoloring.update({new_key: new_colorclass})
                        BIC = score(samples, B, newcoloring)
                        splits += [[c, aug_colorclass, new_colorclass, BIC]]
                        newcoloring.pop(new_key)
                        newcoloring.update({c: colorclass})

        if len(splits) > 0:
            best = 0
            BIC = splits[0][3]
            for i in range(len(splits)):
                if splits[i][3] > BIC:
                    best = i
                    BIC = splits[i][3]
            bestc = splits[best][0]
            bestaug_colorclass = splits[best][1]
            bestnew_colorclass = splits[best][2]
            newcoloring.update({bestc: bestaug_colorclass})
            newcoloring.update({new_key: bestnew_colorclass})

    return B, newcoloring
