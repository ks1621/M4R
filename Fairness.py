import numpy as np

def chosen_arms(self):
    # Compute the arm frequency from the selection counts
    ca = self.run_logs[self.run_count-1]["Arms selected"]
    s = []
    freq = []
    for c in range(self.C):
        freq.append(np.zeros(self.K))
        for k in range(self.K):
            if k in list(ca[c][0]):
                freq[c][k] = ca[c][1][np.where(ca[c][0] == k)[0][0]]
        s.append(sum(freq[c]))
        freq[c] = freq[c] / sum(freq[c])
       
    return freq, s


def Gini_fairness(self):
    # Calculate the Gini index
    df = self.df.copy()
    assert df.shape[1] == self.K
    ret = []
    for m in range(df.shape[0]):
        den = 2 * self.K
        num = 0
        for i in range(self.K):
            for j in range(self.K):
                num += np.abs(df[m, i] - df[m, j])
        ret.append(num/den)

    retn = []
    for n in range(int(df.shape[0]/self.C)):
        retnn = 0
        for nn in range(self.C):
            retnn += ret[self.C * n + nn]
        retn.append(retnn)

    return np.array(retn) / self.C


def pair_consistency(estimates, prob_array):
    # Calculate consistency
    n = len(prob_array)
    num = 0
    for i in range(n):
        for j in range(i):
            if estimates[i] >= estimates[j]:
                if prob_array[i] >= prob_array[j]:
                    num += 1
            else:
                if prob_array[i] <= prob_array[j]:
                    num += 1

    return num / (n*(n-1)/2)


def mix(self):
    # Add Gini and Harmonic mean
    df = self.df.copy()
    assert df.shape[1] == self.K
    ret = []
    for m in range(df.shape[0]):
        den = 0
        for i in range(self.K):
            den += 1 / max(1e-6, df[m, i])
        ret.append(den)

    ret = [1 - self.K ** 2 / x for x in ret]
    retn = []
    for n in range(int(df.shape[0]/self.C)):
        retnn = 0
        for nn in range(self.C):
            retnn += ret[self.C * n + nn]
        retn.append(retnn)

    return np.array(retn) / self.C / self.K