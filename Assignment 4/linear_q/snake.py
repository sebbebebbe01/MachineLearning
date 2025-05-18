import time

import numpy as np
from matplotlib import pyplot as plt

from helper_functions import gen_snake_grid, extract_state_action_features, Q_fun, update_snake_grid

def main():
    np.random.seed(5)

    N = 30
    snake_len_init = 10
    nbr_apples = 1

    # ----- YOU MAY CHANGE SETTINGS BELOW UNLESS OTHERWISE NOTIFIED! ----------
    test_agent = False

    updates_per_sec = 20
    show_fraction = 0

    # Stuff related to learning agent (YOU SHOULD EXPERIMENT A LOT WITH THESE
    # SETTINGS - SEE EXERCISE 7)
    nbr_feats = 3
    rewards = {'default': 0, 'apple': 1, 'death': -1}
    gamm = 0.99
    alph = 0.5
    eps = 0.5

    alph_update_iter = 0
    alph_update_factor = 0.5
    eps_update_iter = 0
    eps_update_factor = 0.5

    # Initial weights. REMEMBER: weights should be set as 1's and -1's in a
    # BAD WAY with respect to your chosen features (see Exercise 8) .
    init_weights = np.random.randn(nbr_feats,1); # replace with your logic
    #np.array([[-1], [-1], [-1]])  

    # ------- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED ---------
    # ------- (FAR DOWN YOU WILL IMPLEMENT Q WEIGHTS UPDATES, SO DO THAT) ----

    if test_agent:
        nbr_ep = 100
        alph = 0
        eps = 0
        weights = np.load('weights.npy')
        print('Testing agent!')
        print('Successfully loaded Q-weights!')
    else:
        nbr_ep = 5000
        weights = init_weights
        print('Training agent!')

    pause_time = 1 / updates_per_sec
    show_every_kth = round(1 / show_fraction) if show_fraction else 3.1415

    top_score = 0
    min_score = 500
    all_scores = np.empty(nbr_ep)
    fig, ax = plt.subplots()

    for i in range(1, nbr_ep + 1):
        print('EPISODE:', i, '/', nbr_ep)
        if not test_agent:
            print('WEIGHTS: ')
            print(weights)
        print('------------------')

        if alph_update_iter > 0 and i % alph_update_iter == 0:
            print('LOWERING ALPH!')
            alph *= alph_update_factor
            print(alph)

        if eps_update_iter > 0 and i % eps_update_iter == 0:
            print('LOWERING EPS!')
            eps *= eps_update_factor
            print(eps)

        close_all = True
        snake_len = snake_len_init
        grid, head_loc = gen_snake_grid(N, snake_len, nbr_apples)
        score = 0
        prev_head_loc_m, prev_head_loc_n = np.where(grid == snake_len)
        prev_head_loc = np.array([prev_head_loc_m[0], prev_head_loc_n[0]])
        prev_head_loc_agent = prev_head_loc
        prev_grid_show = grid.copy()
        prev_grid_show[prev_grid_show > 0] = 1
        grid_show = prev_grid_show.copy()
        if i % show_every_kth == 0:
            ax.imshow(grid_show, animated=True)
            #plt.show()
        nbr_actions_since_last_apple = 0
        t = 0

        while True:
            if t == 0:
                state_action_feats, prev_grid_show, prev_head_loc_agent = extract_state_action_features(
                    prev_grid_show, grid_show, prev_head_loc_agent, nbr_feats)
            else:
                state_action_feats = state_action_feats_future
                prev_grid_show = prev_grid_show_future
                prev_head_loc_agent = prev_head_loc_agent_future

            if np.random.rand() < eps:
                action = np.random.randint(1, 4)
            else:
                Q_vals = Q_fun(weights, state_action_feats)
                action = np.argmax(Q_vals) + 1

            prev_score = score
            grid, head_loc, prev_head_loc, snake_len, score, reward, terminate = update_snake_grid(
                grid, head_loc, prev_head_loc, snake_len, score, rewards, action)

            if test_agent:
                if score == prev_score:
                    nbr_actions_since_last_apple += 1
                else:
                    nbr_actions_since_last_apple = 0

                if nbr_actions_since_last_apple > 10000:
                    print('Agent seems stuck in an infinite loop - PLEASE TRY AGAIN!')
                    print('Press ctrl+c in terminal to terminate!')
                    time.sleep(1000)
            if terminate:
                # FILL IN THE BLANKS TO IMPLEMENT THE Q WEIGHTS UPDATE BELOW (SEE SLIDES) 
                # Maybe useful: alph, reward, Q_fun(weights, state_action_feats, action),
                # state_action_feats[:, action-1] [recall that
                # we set future Q-values at terminal states equal to zero]
                # NOTE: since the features have a different dimension you have to
                # apply .reshape([nbr_feats, 1]) to it when adding to weights
                target = None
                pred = None
                td_err = target - pred
                weights = weights #+ Something
                # -- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED ---
                # -- (IMPLEMENT NON-TERMINAL Q WEIGHTS UPDATE FURTHER DOWN) ---

                all_scores[i - 1] = score

                print('GAME OVER! SCORE:', score)
                print('AVERAGE SCORE SO FAR:', np.mean(all_scores[:i]))
                if i >= 10:
                    print('AVERAGE SCORE LAST 10:', np.mean(all_scores[i - 10:i]))
                if i >= 100:
                    print('AVERAGE SCORE LAST 100:', np.mean(all_scores[i - 100:i]))
                if score > top_score:
                    print('NEW HIGH SCORE!', score)
                    top_score = score
                if score < min_score:
                    print('NEW SMALLEST SCORE!', score)
                    min_score = score
                break
            grid_show = grid.copy()
            grid_show[grid_show > 0] = 1
            if i % show_every_kth == 0:
                ax.clear()
                ax.imshow(grid_show, animated=True)
                ax.set_title('Current score: ' + str(score))
                plt.pause(0.1)

            state_action_feats_future, prev_grid_show_future, prev_head_loc_agent_future = extract_state_action_features(
                prev_grid_show, grid_show, prev_head_loc_agent, nbr_feats)
            
            # FILL IN THE BLANKS TO IMPLEMENT THE Q WEIGHTS UPDATE BELOW (SEE SLIDES) 
            #
            # Maybe useful: alph, max, reward, gamm, Q_fun(weights, state_action_feats_future),
            # Q_fun(weights, state_action_feats, action), state_action_feats[:, action-1]
            # NOTE: since the features have a different dimension you have to
            # apply .reshape([nbr_feats, 1]) to it when adding to weights
            target = None
            pred = None
            td_err = target - pred # Do not change
            weights = weights #+ Something

            # ------------ DO NOT CHANGE ANYTHING BELOW ----------------------

            t += 1

    if not test_agent:
        np.save('weights.npy', weights)
        print('Successfully saved Q-weights!')
        print('Done training agent!')
        print('You may try to set test_agent = True to test agent if you want')
    else:
        print('----------------------------------')
        print('Final weights:')
        print(weights)
        mean_score = np.mean(all_scores)
        print('Mean score after 100 episodes:', mean_score)
        if mean_score < 35:
            print('... PLEASE TRY AGAIN (you should be able to get at least average score 35)')
        else:
            print('... SUCCESS! You got average score at least 35 (feel free to try increasing this further if you want)')
        print('Done testing agent!')

if __name__ == "__main__":
    main()