import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale


def p_before_maxregret(self, alg, noise, max_T=500, la=1, delta=0.05/2, repeats=10):
    # Calculate the highest parameter needed to achieve certain regret. It is used so the plot looks nicer
    # (The lines begin and end near the same regret)
    mean_regret = (np.mean(np.max(self.rewards, axis=1)) - np.mean(self.rewards)) / np.mean(np.max(self.rewards, axis=1))
    if mean_regret == 0:
        mean_regret = 0.2 * (0.1 * self.d)**(-1/3)
    max_regret =  mean_regret / 2
    print(max_regret)
    count = 0
    
    if alg == 'eps-greedy':
        try:
            eps = self.greedy_eps
        except:
            # Attempt to find a close estimate of the parameter that achives the mean_regret.
            # Similar to gradient descent, but without calculating the gradient.
            # The parameter will increase until it exceeds the mean_regret threshold, and it will decrease in a smaller increment
            # in an opposite direction when it crosses the threshold everytime.
            # Automatically ends after 100 iterations.
            eps = 0.1
            direction = 1
            while True:
                count += 1
                eps = max(0, eps * (1 + direction * 0.1))
                regret = 0
                freq = np.zeros((self.C, self.K))
                for r in range(repeats):
                    self.LinUCB_eps_greedy(la, delta, noise, T=max_T, steps=0, eps=eps, fixed=[True, r])
                    regret += self.run_logs[self.run_count-1]["Regret"]
                    ca = self.chosen_arms()
                    
                    for c in range(self.C):
                        freq[c] += ca[0][c]

                self.df = freq/repeats

                E = self.mix() + self.Gini_fairness()
                regret = regret / repeats

                if np.abs(direction) < 0.1 :
                    break

                if count > 100:
                    break
            
                if regret < max_regret:
                    direction = np.abs(direction)
                else:
                    direction = np.abs(direction) * -0.95
            
            self.greedy_eps = eps
        print(f"{alg} complete")
        return eps

    elif alg == 'cost':
        try:
            beta = self.cost_beta
        except:
            # Attempt to find a close estimate of the parameter that achives the mean_regret.
            # Similar to gradient descent, but without calculating the gradient.
            # The parameter will increase until it exceeds the mean_regret threshold, and it will decrease in a smaller increment
            # in an opposite direction when it crosses the threshold everytime.
            # Automatically ends after 100 iterations.
            beta = 0.1
            direction = 1
            while True:
                count += 1
                beta = max(0, beta * (1 + direction * 0.1))
                regret = 0
                freq = np.zeros((self.C, self.K))

                for r in range(repeats):
                    self.LinUCB_cost(la, delta, noise, T=max_T, steps=0, beta=beta, fixed=[True, r])
                    regret += self.run_logs[self.run_count-1]["Regret"]
                    ca = self.chosen_arms()
                    
                    for c in range(self.C):
                        freq[c] += ca[0][c]

            
                self.df = freq/repeats

                E = self.mix() + self.Gini_fairness()
                regret = regret / repeats

                if np.abs(direction) < 0.1 :
                    break
            
                if count > 100:
                    break

                if regret < max_regret:
                    direction = np.abs(direction)
                else:
                    direction = np.abs(direction) * -0.95
        
            self.cost_beta = beta
        print(f"{alg} complete")
        return beta

    elif alg == 'queue':
        try:
            q_p = self.queue_q_p
        except:
            # Attempt to find a close estimate of the parameter that achives the mean_regret.
            # Similar to gradient descent, but without calculating the gradient.
            # The parameter will increase until it exceeds the mean_regret threshold, and it will decrease in a smaller increment
            # in an opposite direction when it crosses the threshold everytime.
            # Automatically ends after 100 iterations.
            q_w = 1
            q_p = 0.1
            direction = 1
            while True:
                count += 1
                q_p = max(0, q_p * (1 + direction * 0.1))
                regret = 0
                freq = np.zeros((self.C, self.K))

                for r in range(repeats):
                    self.LinUCB_queue(la, delta, noise, T=max_T, steps=0, q_p=q_p, q_w=q_w, fixed=[True, r])
                    regret += self.run_logs[self.run_count-1]["Regret"]
                    ca = self.chosen_arms()
                    
                    for c in range(self.C):
                        freq[c] += ca[0][c]

                self.df = freq/repeats

                E = self.mix() + self.Gini_fairness()
                regret = regret / repeats

                if np.abs(direction) < 0.1 :
                    break

                if count > 100:
                    break
            
                if regret < max_regret:
                    direction = np.abs(direction)
                else:
                    direction = np.abs(direction) * -0.95
            
            self.queue_q_p = q_p
        print(f"{alg} complete")
        return q_p

    elif alg == 'bounds':
        try:
            wid = self.bounds_wid
        except:
            # Attempt to find a close estimate of the parameter that achives the mean_regret.
            # Similar to gradient descent, but without calculating the gradient.
            # The parameter will increase until it exceeds the mean_regret threshold, and it will decrease in a smaller increment
            # in an opposite direction when it crosses the threshold everytime.
            # Automatically ends after 100 iterations.
            wid = 0.1
            direction = 1
            while True:
                count += 1
                wid = max(0, wid * (1 + direction * 0.1))
                regret = 0
                freq = np.zeros((self.C, self.K))

                for r in range(repeats):
                    self.LinUCB_fairbounds(la, delta, noise, T=max_T, steps=0, wid=wid, fixed=[True, r])
                    regret += self.run_logs[self.run_count-1]["Regret"]
                    ca = self.chosen_arms()
                    
                    for c in range(self.C):
                        freq[c] += ca[0][c]

                self.df = freq/repeats

                E = self.mix() + self.Gini_fairness()
                regret = regret / repeats

                if np.abs(direction) < 0.1 :
                    break

                if count > 100:
                    break
            
                if regret < max_regret:
                    direction = np.abs(direction)
                else:
                    direction = np.abs(direction) * -0.95
        
            self.bounds_wid = wid
        print(f"{alg} complete")
        return wid

    elif alg == 'm':
        try:
            m_p = self.m_p
        except:
            # Attempt to find a close estimate of the parameter that achives the mean_regret.
            # Similar to gradient descent, but without calculating the gradient.
            # The parameter will increase until it exceeds the mean_regret threshold, and it will decrease in a smaller increment
            # in an opposite direction when it crosses the threshold everytime.
            # Automatically ends after 100 iterations.
            m_p = 0.1
            direction = 1
            while True:
                count += 1
                m_p = max(0, m_p * (1 + direction * 0.1))
                regret = 0
                freq = np.zeros((self.C, self.K))

                for r in range(repeats):
                    self.LinUCB_m(la, delta, noise, T=max_T, steps=0, p=m_p, fixed=[True, r])
                    regret += self.run_logs[self.run_count-1]["Regret"]
                    ca = self.chosen_arms()
                    
                    for c in range(self.C):
                        freq[c] += ca[0][c]

                self.df = freq/repeats

                E = self.mix() + self.Gini_fairness()
                regret = regret / repeats

                if np.abs(direction) < 0.1 :
                    break

                if count > 100:
                    break
            
                if regret < max_regret:
                    direction = np.abs(direction)
                else:
                    direction = np.abs(direction) * -0.95
        
            self.m_p = m_p
        print(f"{alg} complete")
        return m_p
    
    elif alg == 'm2':
        try:
            m_p2 = self.m_p2
        except:
            # Attempt to find a close estimate of the parameter that achives the mean_regret.
            # Similar to gradient descent, but without calculating the gradient.
            # The parameter will increase until it exceeds the mean_regret threshold, and it will decrease in a smaller increment
            # in an opposite direction when it crosses the threshold everytime.
            # Automatically ends after 100 iterations.
            m_p2 = 0.1
            direction = 1
            while True:
                count += 1
                m_p2 = max(0, m_p2 * (1 + direction * 0.1))
                regret = 0
                freq = np.zeros((self.C, self.K))

                for r in range(repeats):
                    self.LinUCB_m2(la, delta, noise, T=max_T, steps=0, p=m_p2, fixed=[True, r])
                    regret += self.run_logs[self.run_count-1]["Regret"]
                    ca = self.chosen_arms()
                    
                    for c in range(self.C):
                        freq[c] += ca[0][c]

                self.df = freq/repeats

                E = self.mix() + self.Gini_fairness()
                regret = regret / repeats

                if np.abs(direction) < 0.1 :
                    break

                if count > 100:
                    break
            
                if regret < max_regret:
                    direction = np.abs(direction)
                else:
                    direction = np.abs(direction) * -0.95
        
            self.m_p2 = m_p2
        print(f"{alg} complete")
        return m_p2   



def rf_curve(self, alg=[], max_T=500, la=1, delta=0.05/2, noisef=0.1, repeats=10, return_results=False):
    # Plot the fairness metric against the regret.
    # If return_results=True, it will return the dependent and independent variables in a vector
    noise = np.mean(self.rewards) * noisef
    self.noise = noise
    ret = []
    errret = []
    label_lists = []
    

    if len(alg) == 0:
        print("please select an algorithm")
        return NotImplementedError
    
    if 'eps-greedy' in alg:
        # Will use these 20 points to form the plot
        eps_list = np.linspace(0, self.p_before_maxregret('eps-greedy', noise, max_T=max_T, la=la, delta=delta, repeats=repeats), 20)
        regret_list = []
        regret_err_list = []
        fairness_list = []
        fairness_err_list = []

        for eps_i in eps_list:
            # Run the algorithm using the parameters from eps_list and record the results
            regret = []
            E = []
            for r in range(repeats):
                freq = np.zeros((self.C, self.K))
                self.LinUCB_eps_greedy(la, delta, noise, T=max_T, steps=0, eps=eps_i, fixed=[True, r])
                regret.append(self.run_logs[self.run_count-1]["Regret"])
                ca = self.chosen_arms()
            
                for c in range(self.C):
                    freq[c] += ca[0][c]

                self.df = freq
                ee = self.mix() + self.Gini_fairness()
                E.append(ee[0])

            regret_err = np.std(regret, ddof=1) / np.sqrt(repeats)
            E_err = np.std(E, ddof=1) / np.sqrt(repeats)

            regret = np.mean(regret)
            E = np.mean(E)

            regret_list.append(regret)
            regret_err_list.append(regret_err)
            fairness_list.append(E)
            fairness_err_list.append(E_err)

        ret.append([fairness_list, regret_list])
        errret.append([fairness_err_list, regret_err_list])
        label_lists.append('ε-greedy')

    if 'cost' in alg:     
        # Will use these 20 points to form the plot 
        beta_list = np.linspace(0, self.p_before_maxregret('cost', noise, max_T=max_T, la=la, delta=delta, repeats=repeats), 20)
        regret_list = []
        regret_err_list = []
        fairness_list = []
        fairness_err_list = []

        for beta_i in beta_list:
            # Run the algorithm using the parameters from eps_list and record the results
            regret = []
            E = []
            for r in range(repeats):
                freq = np.zeros((self.C, self.K))
                self.LinUCB_cost(la, delta, noise, T=max_T, steps=0, beta=beta_i, fixed=[True, r])
                regret.append(self.run_logs[self.run_count-1]["Regret"])
                ca = self.chosen_arms()
            
                for c in range(self.C):
                    freq[c] += ca[0][c]

                self.df = freq
                ee = self.mix() + self.Gini_fairness()
                E.append(ee[0])

            regret_err = np.std(regret, ddof=1) / np.sqrt(repeats)
            E_err = np.std(E, ddof=1) / np.sqrt(repeats)

            regret = np.mean(regret)
            E = np.mean(E)

            regret_list.append(regret)
            regret_err_list.append(regret_err)
            fairness_list.append(E)
            fairness_err_list.append(E_err)

            if beta_i == beta_list[-1]:
                self.repeat0 = True

        ret.append([fairness_list, regret_list])
        errret.append([fairness_err_list, regret_err_list])
        label_lists.append('cost')

    if 'queue' in alg:
        # Fixed parameter
        q_w = 1
        # Will use these 20 points to form the plot 
        q_p_list = np.linspace(0, self.p_before_maxregret('queue', noise, max_T=max_T, la=la, delta=delta, repeats=repeats), 20)
        regret_list = []
        regret_err_list = []
        fairness_list = []
        fairness_err_list = []

        for q_p_i in q_p_list:
            # Run the algorithm using the parameters from eps_list and record the results
            regret = []
            E = []
            for r in range(repeats):
                freq = np.zeros((self.C, self.K))
                self.LinUCB_queue(la, delta, noise, T=max_T, steps=0, q_p=q_p_i, q_w=q_w, fixed=[True, r])
                regret.append(self.run_logs[self.run_count-1]["Regret"])
                ca = self.chosen_arms()
            
                for c in range(self.C):
                    freq[c] += ca[0][c]
          
                self.df = freq
                ee = self.mix() + self.Gini_fairness()
                E.append(ee[0])

            regret_err = np.std(regret, ddof=1) / np.sqrt(repeats)
            E_err = np.std(E, ddof=1) / np.sqrt(repeats)

            regret = np.mean(regret)
            E = np.mean(E)

            regret_list.append(regret)
            regret_err_list.append(regret_err)
            fairness_list.append(E)
            fairness_err_list.append(E_err)
        ret.append([fairness_list, regret_list])
        errret.append([fairness_err_list, regret_err_list])
        label_lists.append('queue')
        
    if 'bounds' in alg:
        # Will use these 20 points to form the plot 
        wid_list = np.linspace(0, self.p_before_maxregret('bounds', noise, max_T=max_T, la=la, delta=delta, repeats=repeats), 20)
        regret_list = []
        regret_err_list = []
        fairness_list = []
        fairness_err_list = []

        for wid_i in wid_list:
            # Run the algorithm using the parameters from eps_list and record the results
            regret = []
            E = []
            
            for r in range(repeats):
                freq = np.zeros((self.C, self.K))
                self.LinUCB_fairbounds(la, delta, noise, T=max_T, steps=0, wid=wid_i, fixed=[True, r])
                regret.append(self.run_logs[self.run_count-1]["Regret"])
                ca = self.chosen_arms()
            
                for c in range(self.C):
                    freq[c] += ca[0][c]

                self.df = freq
                ee = self.mix() + self.Gini_fairness()
                E.append(ee[0])

            regret_err = np.std(regret, ddof=1) / np.sqrt(repeats)
            E_err = np.std(E, ddof=1) / np.sqrt(repeats)

            regret = np.mean(regret)
            E = np.mean(E)

            regret_list.append(regret)
            regret_err_list.append(regret_err)
            fairness_list.append(E)
            fairness_err_list.append(E_err)
        ret.append([fairness_list, regret_list])
        errret.append([fairness_err_list, regret_err_list])
        label_lists.append('bounds')
    
    if "m" in alg:
        # Not used
        # Will use these 20 points to form the plot 
        m_p_list = np.linspace(0, self.p_before_maxregret('m', noise, max_T=max_T, la=la, delta=delta, repeats=repeats), 20)
        regret_list = []
        regret_err_list = []
        fairness_list = []
        fairness_err_list = []

        for m_p in m_p_list:
            # Run the algorithm using the parameters from eps_list and record the results
            regret = []
            E = []
            
            for r in range(repeats):
                freq = np.zeros((self.C, self.K))
                self.LinUCB_m(la, delta, noise, T=max_T, steps=0, p=m_p, fixed=[True, r])
                regret.append(self.run_logs[self.run_count-1]["Regret"])
                ca = self.chosen_arms()
            
                for c in range(self.C):
                    freq[c] += ca[0][c]

                self.df = freq
                ee = self.mix() + self.Gini_fairness()
                E.append(ee[0])

            regret_err = np.std(regret, ddof=1) / np.sqrt(repeats)
            E_err = np.std(E, ddof=1) / np.sqrt(repeats)

            regret = np.mean(regret)
            E = np.mean(E)

            regret_list.append(regret)
            regret_err_list.append(regret_err)
            fairness_list.append(E)
            fairness_err_list.append(E_err)
        ret.append([fairness_list, regret_list])
        errret.append([fairness_err_list, regret_err_list])
        label_lists.append('explore')

    if "m2" in alg:
        # Will use these 20 points to form the plot 
        m_p2_list = np.linspace(0, self.p_before_maxregret('m2', noise, max_T=max_T, la=la, delta=delta, repeats=repeats), 20)
        regret_list = []
        regret_err_list = []
        fairness_list = []
        fairness_err_list = []

        for m_p2 in m_p2_list:
            # Run the algorithm using the parameters from eps_list and record the results
            regret = []
            E = []
            
            for r in range(repeats):
                freq = np.zeros((self.C, self.K))
                self.LinUCB_m2(la, delta, noise, T=max_T, steps=0, p=m_p2, fixed=[True, r])
                regret.append(self.run_logs[self.run_count-1]["Regret"])
                ca = self.chosen_arms()
            
                for c in range(self.C):
                    freq[c] += ca[0][c]

                self.df = freq
                ee = self.mix() + self.Gini_fairness()
                E.append(ee[0])

            regret_err = np.std(regret, ddof=1) / np.sqrt(repeats)
            E_err = np.std(E, ddof=1) / np.sqrt(repeats)

            regret = np.mean(regret)
            E = np.mean(E)

            regret_list.append(regret)
            regret_err_list.append(regret_err)
            fairness_list.append(E)
            fairness_err_list.append(E_err)
        ret.append([fairness_list, regret_list])
        errret.append([fairness_err_list, regret_err_list])
        label_lists.append('explore2')

    if return_results:
            return ret
            
    plt.figure()
    for z in range(len(label_lists)):
        plt.errorbar(np.array(ret[z][1]), np.array(ret[z][0]), xerr=np.array(errret[z][1]), yerr=np.array(errret[z][0]),
                    label=label_lists[z], linestyle='-', marker='o', markersize=2)


    # plt.gca().set_yscale(FuncScale(plt.gca(), (fyscale, byscale)))
    plt.grid()
    plt.legend()
    plt.ylabel("Fairness-Spread")
    plt.xlabel("Regret")
    plt.title("Relationship between fairness measure and regret")
    plt.show()

        




        

