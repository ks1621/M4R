import numpy as np

def generate_context(self, max_T, repeats, seed=0):
    # Generate random context vectors 
    np.random.seed(seed)
    if len(self.context)== 0:
        for r in range(repeats):
            self.fixed_contexts.append([np.random.random(self.d) for t in range(max_T)])
            self.fixed_context_index.append([-1 for t in range(max_T)])
    else: 
        for r in range(repeats):
            c_r = []
            i_r = []
            for t in range(max_T):
                c_index = np.random.choice(np.arange(0, self.C))
                c_r.append(self.context[c_index])
                i_r.append(c_index)
            self.fixed_contexts.append(c_r)
            self.fixed_context_index.append(i_r)