import numpy as np
np.random.seed(17)

# for K = 2 , you should use the following parameters to initialize
# transition matrix
Initial_A_2 = np.array([
    [0.4,0.6],
    [0.6,0.4]
])

# emission matrix
Initial_phi_2 = np.array([
    [0.5, 0.1, 0.2, 0.2],
    [0.1, 0.5, 0.1, 0.3]
])

# Initial state
initial_K2 = [0.5,0.5]


# for K = 4 , you should use the following parameters to initialize
# transition matrix
Initial_A_4 = np.array([
    [0.3, 0.1, 0.2, 0.4],
    [0.1, 0.2, 0.4, 0.3],
    [0.2, 0.4, 0.3, 0.1],
    [0.4, 0.3, 0.1, 0.2]]
)

# emission matrix
Initial_phi_4 = np.array([
    [0.5, 0.1, 0.2, 0.2],
    [0.1, 0.5, 0.1, 0.3],
    [0.1, 0.2, 0.5, 0.2],
    [0.3, 0.1, 0.1, 0.5]
])

# Initial state
initial_K4 = [0.25,0.25,0.25,0.25]


## This part is used for computing the forward backward algorithm

# in our code we will have that A = 0, C = 1, G = 2, T = 3
# so we have the following initial lists of data

x1 = [1,1,3,0,1,0,1,2,1,0]
x2 = [1,3,0,1,2,1,0,0,3]

## Sequence 1 and 2 first step

# This is the forward part
def forward_part(transition, emission, observations,initial):
    no_obs = len(observations)
    states = emission.shape[0]
    #print(states)
    trial_alpha_seq = [{}]
    for y in range(states):
        trial_alpha_seq[0][y] = initial[y]*emission[y][observations[0]]
    #print(trial_alpha_seq)
    for t in range(1,no_obs):
        trial_alpha_seq.append({})
        for y in range(states):
            trial_alpha_seq[t][y] = sum((trial_alpha_seq[t-1][y0]*transition[y0][y]*emission[y][observations[t]]) for y0 in range(states))
    prob = sum((trial_alpha_seq[len(observations)-1][s]) for s in range(states)) 
    return prob,trial_alpha_seq

trial_initial = np.array([0.80,0.20])
transition = np.array([[0.80,0.20],[0.30,0.70]])
emission = np.array([[0.80,0.20],[0.10,0.90]])
observations = [0,1,0]
#print(forward_part(transition, emission, observations,trial_initial))
#print(forward_part(Initial_A_4, Initial_phi_4, x1,initial_K4))

# This is the backward part
def backward_part(transition, emission, observations,initial):
    trial_beta_seq = [{} for t in range(len(observations))]
    T = len(observations)
    states = emission.shape[0]
    for y in range(states):
        trial_beta_seq[T-1][y] = 1
    for t in reversed(range(T-1)):
        for y in range(states):
            trial_beta_seq[t][y] = sum((trial_beta_seq[t+1][y1]*transition[y][y1]*emission[y1][observations[t+1]]) for y1 in range(states))
    prob = sum((initial[y]*emission[y][observations[0]]*trial_beta_seq[0][y]) for y in range(states))
    return prob,trial_beta_seq

#print(backward_part(transition, emission,observations,trial_initial))

## This part is the viterbi algorithm
def viterbi(transition,emission, observations, initial):
    vit = [{}]
    path = {}
    states = emission.shape[0]
    ## Base cases
    for y in range(states):
        vit[0][y] = initial[y] * emission[y][observations[0]]
        path[y] = [y]
    ## Viterby part for t>0
    for t in range(1, len(observations)):
        vit.append({})
        newpath = {}
        for y in range(states):
            (prob,state) = max((vit[t-1][y0]*transition[y0][y]*emission[y][observations[t]],y0) for y0 in range(states))
            vit[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    #print(vit)
    n = 0
    if len(observations)!= 1:
        n = t
    (prob,state) = max((vit[n][y],y) for y in range(states))
    return (prob, path[state])

#print(viterbi(transition,emission,observations,trial_initial))

### E step  ####

trial_initial = np.array([0.80,0.20])
transition = np.array([[0.80,0.20],[0.30,0.70]])
emission = np.array([[0.80,0.20],[0.10,0.90]])
observations = {0:[0,1,0]}

#trial_initial = initial_K4
#transition = Initial_A_4
#emission = Initial_phi_4

#observations = {0: x1,1:x2}
no_obs_1 = len(x1)
no_obs_2 = len(x2)
states = [0,1,2,3,4]
#gammas = [{} for l in range(states)]  
#gammas = np.zeros(states)
#print(states)
#zis = [{} for l in range(no_obs_1+no_obs_2 - 1)]
#print(gammas)
gamma_1 = [{} for t in range(no_obs_1)]
zi_1 = [{} for t in range(no_obs_1 - 1)] 
gamma_2 = [{} for t in range(no_obs_2)]
zi_2 = [{} for t in range(no_obs_2 - 1)] 

#print(len(observations.keys()))

for j in range(len(observations.keys())):
    current_observation = observations[j]
    gammas = [{} for l in range(len(current_observation))]  
    #gammas = np.zeros(states)
    #print(states)
    zis = [{} for l in range(len(current_observation)-1)]

    #print(current_observation)
    prob_fwd,fwd = forward_part(transition,emission,current_observation,trial_initial)
    prob_bkwd,bkwd = backward_part(transition,emission,current_observation,trial_initial)
    #print(fwd)
    #print(bkwd)
    #print(len(fwd))
    product_fwd_bkwd = np.zeros(len(fwd))
    for l in range(len(fwd)):
        current_fwd = fwd[l]
        current_bkwd = bkwd[len(fwd)-1-l]
        #print(current_fwd)
        #print(current_bkwd)
        current_sum = 0
        for i in range(len(current_fwd.keys())):
            current_sum = current_sum + current_fwd[i]*current_bkwd[i]
        #print(current_sum)
        product_fwd_bkwd[l] = current_sum

    for t in range(len(current_observation)):
        for y in states:
            gammas[t][y] = (fwd[t][y] * [t][y]) / product_fwd_bkwd[y]
            if t == 0:
                trial_initial[y] = gammas[t][y]
            #compute zi values up to T - 1
            if t == len(current_observation) - 1:
                continue
            zis[t][y] = {}
            for y1 in states:
                zis[t][y][y1] = fwd[t][y] * transition[y][y1] * emission[y1][current_observation[t + 1]] * bkwd[t + 1][y1] / product_fwd_bkwd[y]


    for l in range(len(fwd)):
        current_fwd = fwd[l]
        current_bkwd = bkwd[len(fwd)-1-l]
        for k in range(len(current_fwd)):
            #print('This is current' + str(current_fwd[k]*current_bkwd[k]))
            #print('This is product ' + str(product_fwd_bkwd[k]))
            gammas[k][l] = current_fwd[k]*current_bkwd[k]/product_fwd_bkwd[k] 
            #print('This is gammas' +str(gammas))
    #print(product_fwd_bkwd)
        zis[l][k] = {}
        for y1 in range(len(current_fwd)):
            #print(fwd[l][k])
            #print(transition[k][y1])
            #print(emission[y1][fwd[l+1]])
            #print(current_bkwd[l+1][y1])
            zis[l][k][y1] =fwd[l][k]*transition[k][y1]*emission[y1][current_observation[l+1]] * bkwd[l+1][y1]/product_fwd_bkwd[k]
    print(gammas)
    
    psi_matrix = np.zeros((states,states))
    for l in range(no_obs_1 + no_obs_2):
        psi_matrix = np.zeros((states,states))
        for i in range(states):
            zis[l][i] = {}
            for s in range(states):
                print(current_bkwd[l])
                print(emission[s][current_observation[l+1]])
                print(current_fwd[l])
                print(transition[i][s])
                print(product_fwd_bkwd[l])
                print(zis)
                zis[l][i][s] = current_fwd[i]*transition[i][s]*emission[s][current_observation[l+1]]*current_bkwd[l + 1]/product_fwd_bkwd[l]
#                zi[t][y][y1] = fwd[t][y]*transition[y][y1]*emission[y1][observations[t+1]] * bkwd[t+1][y1]/sum_fwd_bkwd


#print(current_observation)

def forward_backward(transition,emission,observations,initial):
    no_obs = len(observations)
    states = emission.shape[0]   
    gamma = [{} for t in range(no_obs)]
    zi = [{} for t in range(no_obs - 1)] 
    prob_fwd,fwd = forward_part(transition,emission,observations,initial)
    prob_bkwd,bkwd = backward_part(transition,emission,observations,initial)
    sum_fwd_bkwd = sum(fwd[s][no_obs-1]*bkwd[s][no_obs-1] for s in range(states))

    ### E part

    for t in range(no_obs):
        for y in range(states):
            gamma[t][y] = (fwd[t][y] * bkwd[t][y])/sum_fwd_bkwd
            if t == 0:
                initial[y] = gamma[t][y]
            if t == no_obs - 1:
                continue
            zi[t][y] = {}
            for y1 in range(states):
                zi[t][y][y1] = fwd[t][y]*transition[y][y1]*emission[y1][observations[t+1]] * bkwd[t+1][y1]/sum_fwd_bkwd

    ## M part
    for y in range(states):
        for y1 in range(states):
            val = sum([zi[t][y][y1] for t in range(no_obs-1)])
            val /= sum([zi[t][y]] for t in range(no_obs-1))
            transition[y][y1] = val

    for y in range(states):
        for k in range(emission.shape[1]):
            val = 0.0
            for t in range(no_obs):
                if obs[t] == k :
                    val += gamma[t][y]                 
            val /= sum([gamma[t][y] for t in range(no_obs)])
            emission[y][k] = val
    return

#print(forward_backward(transition,emission,observations,trial_initial))