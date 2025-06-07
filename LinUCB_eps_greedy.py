import numpy as np
from Fairness import pair_consistency

def LinUCB_eps_greedy(self, la=0.5, delta=0.05/2, noise=0.5, T=1000, steps=100, eps=0.25, fixed=[False, -1]):
    # Performs linear UCB algorithm
    # Initialize
    peps = np.log(5)/np.log(100*self.K)
    A = [np.identity(self.d) * la for k in range(self.K)]
    b = [np.zeros(self.d) for k in range(self.K)]
    real_theta = self.arms
    estimate_theta = [np.zeros(self.d) for k in range(self.K)]

    X = [int(np.floor(i * T / steps)) for i in range(1, steps+1)]
    Y = []
    assert len(np.unique(X)) == steps

    reward = 0
    max_reward = 0

    # Setting up a dict to record results
    arm_record = {}
    for c in range(self.C):
        arm_record[c] = []
    
    # Consistency
    cons = []
    cons2 = []

    # Loop for T 
    for t in range(T):
        p_ = []
        c_ = []

        # Randomly picks a context 
        if fixed[0]:
            context_t = self.fixed_contexts[fixed[1]][t]
            c_index = self.fixed_context_index[fixed[1]][t]
        else:
            if len(self.context) > 0:
                c_index = np.random.choice(np.arange(0, self.C))
                context_t = self.context[c_index]
            else:
                c_index = -1
                context_t = np.random.random(self.d)

        # Apply context for each arm
        for k in range(self.K):
            A_inv = np.linalg.inv(A[k])
            estimate_theta[k] = A_inv @ b[k]

            # Calculate the learning rate
            B = self.p
            B_ = np.linalg.norm(context_t)
            alpha = (la * B)**0.5 + noise * np.sqrt(2*np.log(1/delta) + self.d*np.log(1 + T*B_**2/self.d/la))

            # Calculate the upper confidence bound
            p = np.dot(estimate_theta[k], context_t) + alpha * np.sqrt(np.dot(context_t, A_inv @ context_t))
            p_.append(p)
            c_.append(np.dot(estimate_theta[k], context_t) )

        # Choose action with highest bound
        max_p = max(p_)
        max_inds = [j for j, p__ in enumerate(p_) if p__ == max_p]

        # In case of ties, we choose randomly from them
        if len(max_inds) > 1:
            max_ind = np.random.choice(max_inds)
        else:
            max_ind = max_inds[0]

        # Eps-Greedy, we have probability eps of choosing the arms randomly
        eps_p = np.random.uniform(0, 1)
        if eps_p < eps  / (t+1) ** peps:
            max_ind = np.random.choice(np.arange(self.K))

        # Consistency calculations
        prob = np.array([1 if i in max_inds else 0 for i in range(self.K)]) 
        prob = prob/sum(prob) *  (1- eps  / (t+1) ** peps)
        prob += np.ones(self.K) * eps  / (t+1) ** peps

        cons.append(pair_consistency(c_, prob/sum(prob)))

        arm_record[max(0, c_index)].append(max_ind)

        # Observe reward
        max_e = real_theta[max_ind]
        r = np.random.normal(np.dot(max_e, context_t), noise)
        reward += np.dot(max_e, context_t)

        # Append the best outcome 
        if c_index == -1:
            rewards_t = [float(np.dot(aa, context_t)) for aa in self.arms]
            max_reward_t = max(rewards_t)

            max_reward += max_reward_t

        else:
            max_reward += self.max_rewards[c_index]

        # Update
        A[max_ind] = A[max_ind] + np.outer(context_t, context_t)
        b[max_ind] = b[max_ind] + r * context_t
        if t+1 in X:
            Y.append(float(max_reward - reward))

            # Consistency 
            ma = -1 * min(10, len(cons))
            cons2.append(np.mean(np.array(cons[ma:])))


    # Best arm for each context
    best_estimated_arms = []
    best_arms = []
    if c_index != -1:
        for ccc in range(self.C):
            estimates = [np.dot(self.context[ccc], estimate_theta[aaa]) for aaa in range(self.K)]
            best = np.argmax(estimates)
            best_estimated_arms.append((ccc, int(best), float(estimates[best])))
            best_arms.append((ccc, int(self.max_arms[ccc]), float(self.max_rewards[ccc])))

    # Record
    self.run_logs[self.run_count] = {"Regret": float((max_reward - reward)/max_reward),
                                     "Arms selected": [np.unique(arm_record[cccc], return_counts=True) 
                                                       for cccc in range(self.C)],
                                     "Best estimated arms": best_estimated_arms,
                                     "Best arms": best_arms,
                                     "Consistency": np.array(cons2)}
    self.run_count += 1

    return estimate_theta, real_theta, float(max_reward - reward), X, list(np.array(Y)/max_reward)
