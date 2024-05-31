import numpy as np

def getParents(i, graph):
    parents = []
    for j in range(i):
        if graph[i, j] != 0:
            parents += [j]
    return parents

def getSubmatrix(array, C):
    return np.array([[array[i][j] for j in C] for i in C])

def centerData(samples):
    return samples - np.mean(samples, axis=0)

def local_mle(samples, i, parents, coloring):
    numcolors = list(coloring.keys())
    centeredSamples = centerData(samples)
    Y = centeredSamples[:, i]
    numnodes = samples.shape[1]
    numsamples = samples.shape[0]
    b = np.zeros(numnodes)


    if len(parents) > 0:
        localcolors = {}
        localcolorcoeffs = {}
        # extract local colors for family of i:
        for c in numcolors:
            cvertexlist = []
            for e in coloring[c]:
                if e[0] == i:
                    cvertexlist += [e[1]]
            if len(cvertexlist) != 0:
                localcolors.update({c: cvertexlist})

        # sum columns in parent data matrix for each color:
        localcolorkeys = list(localcolors.keys())
        numlocalcolors = len(localcolorkeys)
        D = np.zeros([numlocalcolors, numsamples])
        for c in localcolorkeys:
            X = np.atleast_2d(centeredSamples[:, localcolors[c]])
            Z = np.array([np.sum(X, axis=1)])
            D[localcolorkeys.index(c)] = Z
        D = D.transpose()


        #Run regression:
        X = np.atleast_2d(centeredSamples[:, parents])
        coef = np.linalg.lstsq(D, Y, rcond=None)[0]

        for i in range(len(coef)):
            localcolorcoeffs.update({localcolorkeys[i]: coef[i]})
        for j in parents:
            for c in localcolorkeys:
                if localcolors[c].count(j) != 0:
                    b[j] = localcolorcoeffs[c]

        sigma = np.var(Y - X @ b.transpose()[parents])
    else:
        sigma = np.var(Y, ddof=0)

    if sigma <= 0:
        sigma = abs(np.finfo(float).eps)

    return b, sigma

def full_mle(samples, A, coloring):
    numnodes = samples.shape[1]
    B = np.zeros(A.shape)
    omegas = np.zeros(numnodes)
    for j in range(numnodes):
        parents = np.where(A[j, :] != 0)[0]
        B[j, :], omegas[j] = local_mle(samples, j, parents, coloring)
    return B, omegas

def score(samples, A, coloring):
    numnodes = samples.shape[1]
    numsamples = samples.shape[0]
    numcolors = len(list(coloring.keys()))
    centeredSamples = centerData(samples)
    # Compute MLE
    B, omegas = full_mle(samples, A, coloring)
    K = np.diag(1 / omegas)
    I_B = np.eye(numnodes) - B
    log_term = -numsamples * np.log(np.linalg.det(np.linalg.inv(I_B) @ np.diag(omegas) @ np.linalg.inv(I_B.T)))

    inv_cov = I_B.T @ K @ I_B
    cov_term = 0
    for i, x in enumerate(centeredSamples):
        cov_term += x @ inv_cov @ x
    likelihood = log_term + (-1/2) * cov_term

    l0_term = (1 / 2) * (numcolors + numcolors)*(np.log(numsamples))
    score = (likelihood - l0_term)

    return score