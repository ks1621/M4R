import numpy as np
from LinUCB import LinUCB
from Thompson import Thompson
from RegretCurves import regret_curves
from Fairness import chosen_arms, Gini_fairness, mix

from LinUCB_eps_greedy import LinUCB_eps_greedy
from Thompson_eps_greedy import Thompson_eps_greedy

from LinUCB_cost import LinUCB_cost
from Thompson_cost import Thompson_cost

from LinUCB_fairbounds import LinUCB_fairbounds

from LinUCB_queue import LinUCB_queue

from ConsCurves import cons_curves

from GD import hp_tuning

from RFcurve import p_before_maxregret, rf_curve

from RegretCurvesQueue import regret_curves_queue
from RegretCurvesBounds import regret_curves_bounds

from LinUCB_m import LinUCB_m
from LinUCB_m2 import LinUCB_m2
from Context import generate_context

class Bandits:

    def __init__(self, K=0, d=2, C=1, arms=np.array([]), context=np.array([]), p=10, seed=0):
        np.random.seed(seed)
        self.p = p

        # Contexts
        if C == -1:
            # Infinite context
            self.C = 1
            self.context = []
            
        else:
        # Finite
            if len(context) == 0:
                # Dimension of contexts
                self.C = C

                # Generate contexts
                context = [np.random.random(d)for c in range(C)]
                self.context = context

            else:
                # Number of contexts
                self.C = context.shape[0]

                # Load contexts
                self.context = [context[c, :] for c in range(self.C)]


        # Dimension of the arms
        self.d = d

        # Number of arms
        if len(arms) == 0:
            if K == 0:
                raise ValueError("Invalid Number of Arms")
            else:
                self.K = K
                self.arms = [np.random.random(d) * self.p for k in range(K)] 

        else:
            self.arms = arms
            self.K = len(arms)

        if C != -1:
        # Calculating rewards
            self.rewards = []
            self.max_arms = []
            self.max_rewards = []
            for cc in range(self.C):
                context_c = self.context[cc]
                rewards_c = [float(np.dot(aa, context_c)) for aa in self.arms]
                self.rewards.append(rewards_c)
                max_arm_c = np.argmax(rewards_c)
                max_reward_c = rewards_c[max_arm_c]
                self.max_arms.append(max_arm_c)
                self.max_rewards.append(max_reward_c)
        else:
            self.rewards = [[self.p * self.d / 4]]

        # Run logs
        self.run_logs = {}
        self.run_count = 0

        # Fixed contexts for experiments
        self.fixed_contexts = []
        self.fixed_context_index = []

Bandits.LinUCB = LinUCB
Bandits.Thompson = Thompson
Bandits.regret_curves = regret_curves
Bandits.chosen_arms = chosen_arms

Bandits.LinUCB_eps_greedy = LinUCB_eps_greedy
Bandits.Thompson_eps_greedy = Thompson_eps_greedy

Bandits.LinUCB_cost = LinUCB_cost
Bandits.Thompson_cost = Thompson_cost

Bandits.Gini_fairness = Gini_fairness

Bandits.LinUCB_fairbounds = LinUCB_fairbounds

Bandits.LinUCB_queue = LinUCB_queue

Bandits.cons_curves = cons_curves

Bandits.mix = mix

Bandits.hp_tuning = hp_tuning

Bandits.rf_curve = rf_curve
Bandits.p_before_maxregret = p_before_maxregret

Bandits.regret_curves_queue = regret_curves_queue
Bandits.regret_curves_bounds = regret_curves_bounds

Bandits.LinUCB_m = LinUCB_m
Bandits.LinUCB_m2 = LinUCB_m2
Bandits.generate_context = generate_context