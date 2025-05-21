import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from helper_functions import gen_snake_grid, get_states_next_state_idxs, policy_iteration, grid_to_state_4_tuple, update_snake_grid

# ----------- Main Script ------------
np.random.seed(5)

# Specify number of non-terminal states and actions of the game
nbr_states = 4136
nbr_actions = 3

# Define size of the snake grid (N-by-N).
N = 7

# Define length of the snake (will be placed at the center, pointing in random direction).
snake_len = 3

# Define initial number of apples.
nbr_apples = 1

# ----- YOU MAY CHANGE SETTINGS BELOW UNLESS OTHERWISE NOTIFIED! ----------

# Updates per second (when watching the agent play).
updates_per_sec = 50

# Stuff related to the learning agent
rewards = {'default': 0, 'apple': 1, 'death': -1}
# Specify settings
# Tolerance in policy evaluation - ALLOWED TO BE CHANGED
# Experiment with different tolerances (try as small as 1e-4, up to as
# large as 1e4). Does the tolerance affect the final policy (SEE EXERCISE 5)?
pol_eval_tol = 1e-4

# Discount factor gamma - ALLOWED TO BE CHANGED
# Experiment with gamm; set it to 0, 1 and some values in (0,1). 
# What happens in the respective cases (SEE EXERCISE 5)?
gamm = 0.95

# ------- DO NOT CHANGE ANYTHING BELOW THIS LINE! -----------------------
# ------- BUT DON'T FORGET TO IMPLEMENT policy_iteration.m --------------

# Load or generate state representations
try:
    states = np.load('states.npy')
    next_state_idxs = np.load('next_state_idxs.npy')
    print('Successfully loaded states, next_state_idxs!')
except FileNotFoundError:
    print('Getting state and next state representations!')
    states, next_state_idxs = get_states_next_state_idxs(nbr_states, nbr_actions, N)
    np.save('states.npy', states)
    np.save('next_state_idxs.npy', next_state_idxs)
    print('Successfully saved states and next_state_idxs!')

# Run policy iteration
print('Running policy iteration!')
start_time = time.time()
values, policy, nbr_pol_iter, nbr_pol_eval = policy_iteration(pol_eval_tol, next_state_idxs, rewards, gamm)
end_time = time.time()
print(f'Policy iteration done! Number of policy iterations: {nbr_pol_iter}')
print(f'Number of policy evaluations: {nbr_pol_eval}, elapsed time: {end_time - start_time} seconds')

# Generate initial snake grid and show it
grid, head_loc = gen_snake_grid(N, snake_len, nbr_apples)
score = 0
grid_show = np.copy(grid)
grid_show[grid_show > 0] = 1
prev_grid_show = np.copy(grid_show)
print('Running the small Snake game!')
fig, ax = plt.subplots()

# Main game loop
while True:
    state = grid_to_state_4_tuple(grid)
    state_idx = np.where((states[:, 0] == state[0]) &
                         (states[:, 1] == state[1]) &
                         (states[:, 2] == state[2]) &
                         (states[:, 3] == state[3]))[0][0]
    action = policy[state_idx]

    # time.sleep(1 / updates_per_sec)

    grid, score, terminate = update_snake_grid(grid, snake_len, score, action)

    if terminate:
        print(f'GAME OVER! SCORE: {score}')
        break

    grid_show = np.copy(grid)
    grid_show[grid_show > 0] = 1
    ax.clear()
    ax.imshow(grid_show, animated=True)
    ax.set_title('Current score: ' + str(score))

    plt.pause(0.1)
