import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import time
# from numpy import random as rnd

N_cars = 20 # Max number of cars at each rental
m = 5 # Max movement of cars per night

rental = 10 # Reward for renting a car
move = 2 # Cost of moving a car

gamma = 0.9 # Discount factor

# Adaptations
free_shuttles = 1 # Nr of cars allowed to move for free from site 1 to site 2 every night
excess_park = 10 # Parking limit on each site, if exceeded, pay parking fee (independent on how many excess cars)
parking_fee = 4

### Distribution for supply and demand follow Poisson: lambda^n/n! * exp(-lambda)
lambda_demand_1 = 3 # Lambda for requests
lambda_demand_2 = 4 # --||--
lambda_supply_1 = 3 # Lambda for returns
lambda_supply_2 = 2 # --||--

states = np.arange(N_cars+1)
demand_1 = lambda_demand_1**states / factorial(states) * np.exp(-lambda_demand_1)
demand_2 = lambda_demand_2**states / factorial(states) * np.exp(-lambda_demand_2)
supply_1 = lambda_supply_1**states / factorial(states) * np.exp(-lambda_supply_1)
supply_2 = lambda_supply_2**states / factorial(states) * np.exp(-lambda_supply_2)
demand_1[-1] = 1-np.sum(demand_1[:-1]) # To account for if more than N_cars customer show up
demand_2[-1] = 1-np.sum(demand_2[:-1]) # --||--
supply_1[-1] = 1-np.sum(supply_1[:-1]) # To account for if more cars are returned than there is parking (returned to national branch)
supply_2[-1] = 1-np.sum(supply_2[:-1]) # --||--

script_start = time.time()

## Create transition matrices to cover all cases. Two methods, one which is memory intensive (not recommended),
## and anoter which is more CPU intensive
T1 = np.zeros((N_cars+1,N_cars+1))
T2 = np.zeros((N_cars+1,N_cars+1))

for n in range(N_cars+1):
    # Flip truncated demand (since probability of increase from renting out is zero) and ensure total mass sums to 1
    site1_after_loss = np.flip(demand_1[:n+1]).copy() # Excluding the trailing zeros up to element nr N_cars+1
    site1_after_loss[0] = 1 - np.sum(site1_after_loss[1:]) # If more than n customers show up, zero cars will remain

    # Compute convolution for fulfilled demand (capped by N_cars)
    site1_prime = np.convolve(site1_after_loss, supply_1, mode='full')[:N_cars+1]
    
    # Correct the final bin (when insurge of cars from returns exceeds N_cars)
    tail_weights = 1 - np.cumsum(supply_1[:-1]) # To make sure probabilities add up to one
    site1_prime[-1] = site1_after_loss @ np.pad(np.flip(tail_weights), (0, 1), constant_values=1)[:len(site1_after_loss)]
    
    # Do the same for the second site
    site2_after_loss = np.flip(demand_2[:n+1]).copy()
    site2_after_loss[0] = 1 - np.sum(site2_after_loss[1:])
    site2_prime = np.convolve(site2_after_loss, supply_2, mode='full')[:N_cars+1]
    tail_weights = 1 - np.cumsum(supply_2[:-1]) # To make sure probabilities add up to one
    site2_prime[-1] = site2_after_loss @ np.pad(np.flip(tail_weights), (0, 1), constant_values=1)[:len(site2_after_loss)]
    
    # state_matrix = site1_prime[:,None] @ site2_prime[None,:]

    T1[n,:] = site1_prime
    T2[n,:] = site2_prime

# Cache row vectors for T1 and T2 to optimise
cached_T1 = {}
cached_T2 = {}

def get_state_matrix(n1, n2):
    if n1 not in cached_T1:
        cached_T1[n1] = T1[n1, :][:, None]
    if n2 not in cached_T2:
        cached_T2[n2] = T2[n2, :][None, :]
    return cached_T1[n1] @ cached_T2[n2]

def get_state_value(V, n1, n2, action):
    n1_prime = int(n1 - action)
    n2_prime = int(n2 + action)

    # Adaptation 1: free_shuttles free moves from site 1 to site 2
    if action >= 0:
        cost = move*abs(max(action - free_shuttles, 0))
    else:
        cost = move*abs(action)
    # Adaptation 2: parking fees if you exceed excess_park on either site
    if n1_prime > excess_park:
        cost += parking_fee
    if n2_prime > excess_park:
        cost += parking_fee

    # Get a matrix for the probabilities of s_prime (the state the following night)
    state_matrix = get_state_matrix(n1_prime, n2_prime) # Shape (N_cars+1, N_cars+1)
    # state_matrix = T[n1_prime, n2_prime, :, :] # Shape (N_cars+1, N_cars+1)

    # We want to calculate the sum over s' and r for p(s',r)*(r+gamma*V(s')) = p(s')p(r)*(r+gamma*V(s'))
    # = sum_r p(r) * sum_s' p(s') (r+gamma*V(s'))
    # = sum_r p(r) * [sum_s' p(s') r + sum_s' p(s') gamma*V(s')]
    # = sum_r p(r) * [r sum_s' p(s') + sum_s' p(s') gamma*V(s')]
    # = sum_r p(r) * [r + sum_s' p(s') gamma*V(s')]
    # Start with the profit matrix (the reward)
    help_M = np.minimum(n1_prime,states)[:,None] + np.minimum(n2_prime,states)[None,:]
    R = rental*help_M - cost # Shape (N+1, N+1)

    # Then calculate the sum over s' for p(s')*gamma*V(s')
    inner_sum = np.sum(state_matrix * (gamma*V))

    # Now the final sum
    V_s = np.sum(demand * (R + inner_sum))

    return V_s


## Initialization
V = np.zeros((N_cars+1, N_cars+1)) # State = nr of cars at each location
policy = np.zeros((N_cars+1, N_cars+1)) # Could also be randomly initialized
demand = (demand_1[:,None]@demand_2[None,:]) # Probability matrix for nr of requests at each site

## Policy evaluation
def policy_eval(V, policy):
    tol = 1e-4
    delta = 100
    start = time.time()
    while delta>tol:
        delta = 0
        for n1 in range(N_cars+1):
            for n2 in range(N_cars+1):
                # The new state is determined by the action in policy and the distributions for returns and requests
                # The reward follows the minimum of the available cars and the request distribution
                V_old = V[n1,n2].copy()

                action = policy[n1,n2] # Positive int: move car from 1 to 2, negative: from 2 to 1

                V_new = get_state_value(V, n1, n2, action)
                
                V[n1,n2] = V_new

                delta = max(delta, abs(V[n1,n2]-V_old))

        # print(f'Current delta: {delta} (with target {tol})')

    end = time.time()
    # print(f'Elpased time for policy evaluation: {end-start} s')
    return V

## Policy improvement
policy_stable = False
policies = []
while not policy_stable:
    print('We continue...')
    V = policy_eval(V, policy)
    policy_stable = True
    policies.append(policy.copy())
    for n1 in range(N_cars+1):
        for n2 in range(N_cars+1):
            old_action = policy[n1,n2]
            V_list = []
            min_a = -min(m, n2, N_cars - n1)
            max_a = min(m, n1, N_cars - n2)
            for a in range(min_a, max_a + 1):
                V_a = get_state_value(V, n1, n2, a)

                V_list.append(V_a)

            a_best = np.argmax(V_list) + min_a
            policy[n1,n2] = a_best
            if a_best != old_action: # This could be a cause for a bug where two actions give the same state-value
                policy_stable = False

script_end = time.time()
print(f"Total time elapsed: {script_end-script_start}")

# for i, p in enumerate(policies):
#     plt.figure()
#     plt.imshow(p, origin = 'lower')
#     plt.colorbar()
#     plt.title(f'Policy nr {i}')

plt.figure()
plt.imshow(policy, origin = 'lower')
plt.colorbar()
plt.title('Final policy')
plt.show()
        