import numpy as np
from numpy.random import default_rng
from scipy import stats


def getParents(i, graph):
    parents = []
    for j in range(i):
        if graph[i,j] != 0:
            parents += [j]
    return parents

def edgeGenerator(numvars, prob, minparents=2, seed=54):
    np.random.seed(seed)

    L = np.zeros([numvars, numvars])

    for i in range(numvars):
        L[i, i] = 1
        if i > 1:
            for j in range(i):
                L[i, j] = -stats.bernoulli.rvs(size=1, p=prob)

    parents = [getParents(i, L) for i in range(numvars)]
    nonparents = [[j for j in range(i) if parents[i].count(j) == 0] for i in range(numvars)]
    numParents = [len(pa) for pa in parents]

    for i in range(numvars):
        if numParents[i] != 0:
            while numParents[i] < minparents:
                pidx = np.random.randint(len(nonparents[i]))
                newparent = nonparents[i][pidx]
                L[i, newparent] = -1
                parents[i] += [newparent]
                nonparents[i].pop(pidx)
                numParents[i] = len(parents[i])

    return L

# builds a random DAG on numvars nodes with edge probability prob
# then assigns each edge to a color class
# The color classes always contain at least two elements
def e_colorGenerator(numvars, prob, minparents = 2, numcolors=1, seed=54):
    np.random.seed(seed)
    seeds = [np.random.randint(1000) for i in range(2)]
    numcolorsinit = numcolors
    G = edgeGenerator(numvars, prob, minparents, seed=seeds[1])
    parents = [getParents(i, G) for i in range(numvars)]
    numparents = [len(pa) for pa in parents]

    edgeList = []
    for i in range(numvars):
        for j in range(i):
            if G[i, j] == -1:
                edgeList += [[i, j]]

    coloringList = []
    coloringListByVertex = []

    np.random.seed(seeds[0])

    for i in range(numvars):
        if numparents[i] != 0:
            if numparents[i] / 2 < numcolors:
                print('Warning: half the number of edges in G is less than the number of colors. '
                      'Replacing numcolors with  numedges // 2')
                numcolors = numparents[i] // 2

            if numcolors > 1:
                localcoloringList = [[] for i in range(numcolors)]
                localcoloringListByVertex = [[] for i in range(numcolors)]

                for c in range(numcolors):
                    for k in range(2):
                        pidx = np.random.randint(numparents[i])
                        localcoloringList[c] += [[i, parents[i][pidx]]]
                        localcoloringListByVertex[c] += [i]
                        parents[i].pop(pidx)
                        numparents[i] = len(parents[i])

                while len(parents[i]) != 0:
                    for p in parents[i]:
                        cidx = np.random.randint(numcolors)
                        localcoloringList[cidx] += [[i, p]]
                        localcoloringListByVertex[cidx] += [i]
                        parents[i].pop(parents[i].index(p))

                coloringList += localcoloringList
                coloringListByVertex += localcoloringListByVertex
            else:
                coloringList += [[[i, p] for p in parents[i]]]
                coloringListByVertex += [[p for p in parents[i]]]

            numcolors = numcolorsinit

    coloring = {}
    coloringByVertex = {}
    for i in range(len(coloringList)):
        coloring.update({i: coloringList[i]})
        coloringByVertex.update({i: coloringListByVertex[i]})

    return G, coloring, coloringByVertex

def e_coloredDAGmodel(samplesize, numvars, prob, minparents = 2, numcolors=1, seed=54):
    np.random.seed(seed)
    seeds = [np.random.randint(1000) for i in range(2)]

    coloredG = e_colorGenerator(numvars, prob, minparents=minparents, numcolors=numcolors, seed=seeds[1])
    G = coloredG[0]
    coloring = coloredG[1]
    colorKeys = list(coloring.keys())

    effects = dict()
    for i in colorKeys:
        # assign a causal effect to the edges with color i in range [-1,0.25)\cup(0.25,1]
        effect = np.random.choice([np.random.uniform(-1, -0.25), np.random.uniform(0.25, 1)])
        effects.update({i: effect})

        iEdges = coloring[i]
        for e in iEdges:
            G[e[0], e[1]] = -effect

    # generate error variances for each node in the interval [0.25, 1.75]:
    np.random.seed(seeds[0])
    errorVariances = [np.random.uniform(0.25, 1.75) for i in range(numvars)]

    errorSamps = np.zeros([numvars, samplesize])

    for i in range(numvars):
        errorSamps[i] = stats.norm.rvs(size=samplesize, loc=0, scale=errorVariances[i])

    samps = np.matmul(np.linalg.inv(G), errorSamps)
    samps = np.transpose(samps)

    return samps, G, coloring, effects, errorVariances