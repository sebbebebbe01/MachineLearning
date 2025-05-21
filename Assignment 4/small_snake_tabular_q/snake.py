import numpy as np
from matplotlib import pyplot as plt
import time

from helper_functions import get_states, gen_snake_grid, grid_to_state_4_tuple, update_snake_grid

# Main script
# --------- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED! -------

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
# Specify whether to test the agent or not (False --> train agent)
test_agent = True

# Updates per second (when watching the agent play).
updates_per_sec = 50

# Set visualization settings
show_fraction = 1 # 1: show everything, 0: show nothing, 0.1: show every tenth, and so on

# Stuff related to the learning agent
# Stuff related to learning agent (YOU SHOULD EXPERIMENT A LOT WITH THESE
# SETTINGS - SEE EXERCISE 6).
rewards = {'default': -1, 'apple': 2, 'death': -50}
gamm = 0.9 # Discount factor in q learning
alph = 0.5 # learning rate in Q learning
eps = 0.01 # Random action selection probability in epsilon-greedy Q-learning
alph_update_iter = 100 #0: Never update alpha, Positive integer k: Update alpha every kth episode
alph_update_factor = 0.95 # At alpha update: new alpha = old alpha * alph_update_factor
eps_update_iter = 0 # 0: Never update eps, Positive integer k: Update eps every kth episode
eps_update_factor = 0.5 # At eps update: new eps = old eps * eps_update_factor

# ------- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED --------
# ------- (FAR DOWN YOU WILL IMPLEMENT Q-UPDATES, SO DO THAT) -----------

if test_agent:
    nbr_ep = 1
    alph = 0
    eps = 0
    # Load agent Q-values for testing
    Q_vals = np.load('Q_vals.npy')
    print('Testing agent!')
    print('Successfully loaded Q-values!')
else:
    nbr_ep = 5000
    Q_vals = np.random.randn(nbr_states, nbr_actions)
    print('Training agent!')

Q_vals_init = Q_vals

# Display options
pause_time = 1 / updates_per_sec
show_every_kth = round(1 / show_fraction) if show_fraction else 0

# Set up state representations.
if test_agent:
    states = np.load('states.npy')
    print('Successfully loaded states!')
else:
    print('Getting state representation!')
    start_time = time.time()
    states = get_states(nbr_states, N)
    end_time = time.time()
    print(f'Done getting state representation! Elapsed time: {end_time - start_time} seconds')
    np.save('states.npy', states)
    print('Successfully saved states!')

# Keep track of high score, minimum score, and store all scores.
top_score = 0
min_score = 500000
all_scores = np.nan * np.ones(nbr_ep)
fig, ax = plt.subplots()

# Main loop
for i in range(1, nbr_ep + 1):
    # Display episode information
    if not test_agent:
        print(f'EPISODE: {i} / {nbr_ep}')

    # Check if learning rate and/or eps should decrease
    if alph_update_iter > 0 and i % alph_update_iter == 0:
        print('LOWERING ALPH!')
        alph *= alph_update_factor
        print(alph)
    if eps_update_iter > 0 and i % eps_update_iter == 0:
        print('LOWERING EPS!')
        eps *= eps_update_factor
        print(eps)

    # Generate initial snake grid and possibly show it
    grid, head_loc = gen_snake_grid(N, snake_len, nbr_apples)
    score = 0
    grid_show = grid.copy()
    grid_show[grid_show > 0] = 1

    if show_fraction and i % show_every_kth == 0:
        ax.imshow(grid_show, animated=True)

    # If test mode: Print score
    if test_agent:
        print(f'Current score: {score}')
        nbr_actions_since_last_apple = 0

    # Run an episode of the game
    while True:
        # Get state information
        state = grid_to_state_4_tuple(grid)
        state_idx = int(np.where(np.all(states[:, :4] == state, axis=1))[0])

        # epsilon-greedy action selection
        if np.random.rand() < eps:
            action = np.random.randint(3)
        else:
            action = np.argmax(Q_vals[state_idx, :])

        # Possibly pause for a while
        if show_fraction and i % show_every_kth == 0:
            time.sleep(pause_time)

        # Update state
        prev_score = score
        grid, score, reward, terminate = update_snake_grid(grid, snake_len, score, rewards, action)

        # If test mode: Print score if it increased
        if test_agent:
            if terminate:
                print('Agent died...')
                if score < 250:
                    print('... PLEASE TRY AGAIN (you should be able to get at least score 250 prior to dying)')
                else:
                    print('... SUCCESS! You got a score of at least 250 before dying (feel free to try increasing score further if you want)')
                print(f'Got test score (number of apples eaten): {score}')
            if score > prev_score:
                nbr_actions_since_last_apple = 0
                print(f'Current score: {score}')
            else:
                nbr_actions_since_last_apple += 1

                # Check if we seem to be stuck in a loop (at test time)
                if nbr_actions_since_last_apple > 250:
                    print('Agent seems stuck in an infinite loop...')
                    if score < 250:
                        print('... PLEASE TRY AGAIN (you should be able to get at least score 250 prior to getting stuck in a loop / dying)')
                    else:
                        print('... SUCCESS! You got a score of at least 250 before getting stuck in such loop (feel free to try increasing score further if you want)')
                    print(f'Got test score (number of apples eaten): {score}')
                    print('Press ctrl+c in the terminal to terminate!')
                    time.sleep(1000)

        # Check for termination
        if terminate:
            # Q-value update for terminal state
            # FILL IN THE BLANKS TO IMPLEMENT THE Q-UPDATE BELOW (SEE SLIDES)
            # Maybe useful: alph, reward, Q_vals(state_idx, action) [recall that
            # we set future Q-values at terminal states equal to zero].
            # Hint: Q(s,a) <-- (1 - alpha) * Q(s,a) + sample
            # can be rewritten as Q(s,a) <-- Q(s,a) + alpha * (sample - Q(s,a))

            sample = reward # next state is terminal, V(s') = 0
            pred = Q_vals[state_idx, action]
            td_err = sample - pred # don't change this.
            Q_vals[state_idx, action] += alph * td_err

            # -- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED ---
            # -- (IMPLEMENT NON-TERMINAL Q-UPDATE FURTHER DOWN) -----------

            # Insert score into container
            all_scores[i - 1] = score

            # Display information
            if not test_agent:
                print(f'GAME OVER! SCORE: {score}')
                print(f'AVERAGE SCORE SO FAR: {np.mean(all_scores[:i])}')
                if i >= 10:
                    print(f'AVERAGE SCORE LAST 10: {np.mean(all_scores[i - 10:i])}')
                if i >= 100:
                    print(f'AVERAGE SCORE LAST 100: {np.mean(all_scores[i - 100:i])}')
                if score > top_score:
                    print(f'NEW HIGH SCORE! {score}')
                    top_score = score
                if score < min_score:
                    print(f'NEW SMALLEST SCORE! {score}')
                    min_score = score

            # Terminate
            break

        # Update what to show on the screen
        grid_show = grid.copy()
        grid_show[grid_show > 0] = 1
        if show_fraction and i % show_every_kth == 0:
            ax.clear()
            ax.imshow(grid_show, animated=True)
            ax.set_title('Current score: ' + str(score))
            plt.pause(0.1)

        # Check the next state and associated next state index
        next_state = grid_to_state_4_tuple(grid)
        next_state_idx = np.where(np.all(states[:, :4] == next_state, axis=1))[0]

        # Q-value update for non-terminal state
        # FILL IN THE BLANKS TO IMPLEMENT THE Q-UPDATE BELOW (SEE SLIDES) 
        # Maybe useful: alph, max, reward, gamm, Q_vals(next_state_idx, :), 
        # Q_vals(state_idx, action)
        # Hint: Q(s,a) <-- (1 - alpha) * Q(s,a) + sample
        # can be rewritten as Q(s,a) <-- Q(s,a) + alpha * (sample - Q(s,a))
        sample = reward + gamm * np.max(Q_vals[next_state_idx,:])
        pred = Q_vals[state_idx, action]
        td_err = sample - pred # don't change this!
        Q_vals[state_idx, action] += alph * td_err

# Finally, save agent Q-values (if not in test mode already)
if not test_agent:
    np.save('Q_vals.npy', Q_vals)
    print('Successfully saved Q-values!')
    print('Done training agent!')
    print('You may try to set test_agent = True to test the agent if you want')
else:
    print('Done testing agent!')