import numpy as np
from sklearn.preprocessing import minmax_scale
from VCA import vca


def normalize(inputdata, flag):
    X = np.zeros((inputdata.shape[0], inputdata.shape[1]))
    M = []
    m = []
    if flag == 1:
        for i in range(inputdata.shape[1]):
            M.append(np.max(inputdata[:, i]))
            m.append(np.min(inputdata[:, i]))
        M = np.array(M)
        m = np.array(m)
        index = (M == m) & (M != 0)  # Find out the columns with equal data values but not 0
        for i in range(inputdata.shape[1]):
            X[:, i] = minmax_scale(inputdata[:, i])
        X[:, index] = np.ones((inputdata.shape[0], inputdata.shape[1]))[:, index]

    if flag == 2:
        L = np.sqrt(np.sum(inputdata**2, axis=0))
        index = (L != 0)
        X[:, index] = inputdata[:, index]/L[index]

    return X


def FUMI_unmix(inputdata, E, flag, Eps):
    M = E.shape[1]
    N = inputdata.shape[1]

    if M > 1:
        DP = Eps*np.eye(M)  # M * M
        U = np.linalg.pinv(np.dot(E.T, E)+DP)  # M * M
        V = np.dot(E.T, inputdata)  # M * N
        if flag == 0:
            P = np.dot(U, V)  # M * N
        elif flag == 1:
            temp1 = 1 - np.ones((1, M)).dot(U).dot(V)  # 1 * N
            temp2 = np.ones((1, M)).dot(U).dot(np.ones((M, 1)))  # 1 * 1
            temp3 = temp1/temp2  # 1 * N
            temp4 = V + np.ones((M, 1)).dot(temp3)  # M * N
            P = U.dot(temp4)  # M * N
        else:
            P = np.zeros((M, N))

        z = (P < 0)  # M * N
        while np.sum(z) > 0:
            zz = np.unique(z, axis=1)  # M * ?
            for i in range(zz.shape[1]):
                if np.sum(zz[:, i]) > 0:
                    eLocs = np.argwhere(1-zz[:, i]).reshape((1, -1))[0]
                    rZZi = np.tile(zz[:, i].reshape((-1, 1)), (1, N))
                    inds = np.all(z == rZZi, axis=0)
                    P_temp = FUMI_unmix(inputdata[:, inds], E[:, eLocs], flag, Eps)
                    P_temp2 = np.zeros((zz.shape[0], np.sum(inds)))
                    P_temp2[eLocs, :] = P_temp
                    P[:, inds] = P_temp2
            z = (P < 0)

    else:
        P = np.ones((M, N))

    return P


def VCA_initialize(x, labels, args):

    # extrtact data according labels
    index_plus = np.reshape(np.argwhere(labels == 1), (1, -1))[0]  # index of elements with labels == 1
    index_minus = np.reshape(np.argwhere(labels == 0), (1, -1))[0]  # index of elements with labels == 0
    x_plus = x[:, index_plus]  # pixels with labels == 1
    x_minus = x[:, index_minus]  # pixels with labels == 0
    N_plus = x_plus.shape[1]
    N_minus = x_minus.shape[1]

    # pre process data
    if args.normflag == 1:
        data_pre = normalize(x, 1)
    elif args.normflag == 2:
        data_pre = normalize(x, 2)
    else:
        data_pre = x

    # E initialization
    E_minus = vca(x_minus, args.M)  # VCA to extract background endmembers
    P_plus_unmix = FUMI_unmix(x_plus, E_minus, 1, args.Eps)
    syn_x_plus = E_minus.dot(P_plus_unmix)
    unmix_diff = np.sqrt(np.sum((x_plus - syn_x_plus)**2, axis=0))
    idx_et = np.argmax(unmix_diff)
    e_t = x_plus[:, idx_et].reshape((-1, 1))

    E = np.concatenate((e_t, E_minus), axis=1)

    if args.flagE == 2:
        E = normalize(E, 2)  # if normalization constraint on endmember is added, normalize E as well

    # P initialization
    # use mean value to initialize proportion velues according labels
    P = np.zeros((args.M+1, len(labels)))
    P_plus = np.ones((args.M+1, N_plus))*(1/(args.M+1))
    P_minus = np.ones((args.M, N_minus))*(1/args.M)
    P[:, index_plus] = P_plus
    P[1:, index_minus] = P_minus

    return data_pre, E, P


if __name__ == '__main__':
    exp = np.array([[12,3,6,14,100,5,0],[67,35,6,9,54,13,0],[31,2,6,23,33,90,0]])
    print(exp)
    ans=normalize(exp,2)
    ans=np.round(ans,2)
    print(ans)