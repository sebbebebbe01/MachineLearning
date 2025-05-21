import numpy as np

def gen_snake_grid(N=7, snake_len=3, nbr_apples=1):
    # Default settings
    if snake_len is None:
        snake_len = int(np.floor(N / 10))

    # Create empty grid
    grid = np.zeros((N, N))

    # Add walls in the borders of the grid
    grid[0:N, 0] = 1
    grid[0:N, N - 1] = 1
    grid[0, 0:N] = 1
    grid[N - 1, 0:N] = 1

    # Insert snake with its head in the center, facing a random direction (north/east/south/west)
    head_loc = [int(N / 2), int(N / 2)]
    snake_rot = np.random.randint(1, 5)  # 1: north, 2: east, 3: south, 4: west

    if snake_rot == 1:
        tail_loc = [head_loc[0] + snake_len - 1, head_loc[1]]
        grid[head_loc[0]:tail_loc[0] + 1, head_loc[1]] = np.arange(snake_len + 1, 1, -1)
    elif snake_rot == 2:
        tail_loc = [head_loc[0], head_loc[1] - snake_len + 1]
        grid[head_loc[0], tail_loc[1]:head_loc[1] + 1] = np.arange(2, snake_len + 2)
    elif snake_rot == 3:
        tail_loc = [head_loc[0] - snake_len + 1, head_loc[1]]
        grid[tail_loc[0]:head_loc[0] + 1, head_loc[1]] = np.arange(2, snake_len + 2)
    else:
        tail_loc = [head_loc[0], head_loc[1] + snake_len - 1]
        grid[head_loc[0], head_loc[1]:tail_loc[1] + 1] = np.arange(snake_len + 1, 1, -1)

    # Keep track of occupied locations (so that no conflicts happen in apple generation)
    occ_locs = np.argwhere(grid > 0)

    # Insert apples into snake_grid, at random locations
    for i in range(nbr_apples):
        # Sample random location to put apple makes sure that each apple
        # has a separate location and doesn't collide with the snake
        while True:
            apple_loc = np.random.randint(0, N - 1, size=(1, 2))  # N-1, since cannot be at walls
            if not np.any(np.all(occ_locs == apple_loc, axis=1)):
                break

        # Update occ_locs and insert apple into snake_grid
        occ_locs[i + snake_len, :] = apple_loc
        grid[apple_loc[0][0], apple_loc[0][1]] = -1

    return grid, head_loc

def get_next_head_loc(head_loc, prev_head_loc, action):
    # Extract stuff
    head_loc_m = head_loc[0]
    head_loc_n = head_loc[1]
    prev_head_loc_m = prev_head_loc[0]
    prev_head_loc_n = prev_head_loc[1]

    # Compute next head location based on movement direction and action
    if prev_head_loc_m > head_loc_m and prev_head_loc_n == head_loc_n:  # movement direction: NORTH
        if action == 1:  # LEFT
            next_head_loc = [head_loc_m, head_loc_n - 1]
        elif action == 2:  # FORWARD
            next_head_loc = [head_loc_m - 1, head_loc_n]
        else:  # RIGHT
            next_head_loc = [head_loc_m, head_loc_n + 1]
    elif prev_head_loc_m == head_loc_m and prev_head_loc_n < head_loc_n:  # movement direction: EAST
        if action == 1:  # LEFT
            next_head_loc = [head_loc_m - 1, head_loc_n]
        elif action == 2:  # FORWARD
            next_head_loc = [head_loc_m, head_loc_n + 1]
        else:  # RIGHT
            next_head_loc = [head_loc_m + 1, head_loc_n]
    elif prev_head_loc_m < head_loc_m and prev_head_loc_n == head_loc_n:  # movement direction: SOUTH
        if action == 1:  # LEFT
            next_head_loc = [head_loc_m, head_loc_n + 1]
        elif action == 2:  # FORWARD
            next_head_loc = [head_loc_m + 1, head_loc_n]
        else:  # RIGHT
            next_head_loc = [head_loc_m, head_loc_n - 1]
    else:  # movement direction: WEST
        if action == 1:  # LEFT
            next_head_loc = [head_loc_m + 1, head_loc_n]
        elif action == 2:  # FORWARD
            next_head_loc = [head_loc_m, head_loc_n - 1]
        else:  # RIGHT
            next_head_loc = [head_loc_m - 1, head_loc_n]

    return next_head_loc

def get_next_state(state_4_tuple, N, action=None):
    """
    Function to infer the next state based on the current state and action.

    Parameters:
    - state_4_tuple: Valid state-4-tuple of the form [apple_loc, head_loc, body1_loc, body2_loc]
    - N: The size of the game grid (an N-by-N matrix)
    - action: Integer in {1, 2, 3} representing the action taken by the agent (default: [1, 2, 3])

    Returns:
    - next_state_4_tuple: Either a 1-by-4 vector (if action provided) or a 3-by-4 matrix (if action not provided)
    - terminal_apple: Either a 1-by-1 logical or a 3-by-3 logical matrix
    - terminal_death: Either a 1-by-1 logical or a 3-by-3 logical matrix
    """

    # Default settings (compute for all actions by default)
    if action is None:
        action = [0, 1, 2]

    # Get the number of actions we want to compute next_state_4_tuple for
    nbr_actions = len(action)

    # Default next_state_4_tuple (will be irrelevant if termination occurs)
    next_state_4_tuple = np.full((nbr_actions, 4), np.nan)

    # Default terminal logicals (assume non-terminal by default)
    terminal_apple = np.zeros((nbr_actions,), dtype=bool)
    terminal_death = np.zeros((nbr_actions,), dtype=bool)

    # Extract relevant things (note: linear indexes in column-major order)
    apple_loc, head_loc, body1_loc, body2_loc = state_4_tuple

    # Transform linear indexing to matrix indexing
    # Note: N-2, since the states are considered only in the interior grid
    apple_loc_m, apple_loc_n = np.unravel_index(int(apple_loc), (N - 2, N - 2), order='F')
    head_loc_m, head_loc_n = np.unravel_index(int(head_loc), (N - 2, N - 2), order='F')
    body1_loc_m, body1_loc_n = np.unravel_index(int(body1_loc), (N - 2, N - 2), order='F')

    # Compute where the next head location ends up being (check collisions or apple eating)
    for i in range(len(action)):
        # Get the current action
        act = action[i]

        # Get the next head location based on the chosen action
        next_head_loc = get_next_head_loc((head_loc_m, head_loc_n), (body1_loc_m, body1_loc_n), act)
        next_head_loc_m, next_head_loc_n = next_head_loc

        # Check if the action yields an apple-terminal state
        if next_head_loc_m == apple_loc_m and next_head_loc_n == apple_loc_n:
            terminal_apple[i] = True
            continue

        # Check if the action yields a death-terminal state
        if next_head_loc_m == -1 or next_head_loc_m == N - 2 or \
                next_head_loc_n == -1 or next_head_loc_n == N - 2:
            terminal_death[i] = True
            continue

        # Now we know that the next state is not terminal - let's find out what the next state is!
        next_state_4_tuple[i, 0] = apple_loc
        next_state_4_tuple[i, 1] = np.ravel_multi_index((next_head_loc_m, next_head_loc_n), (N - 2, N - 2), order='F')
        next_state_4_tuple[i, 2] = head_loc
        next_state_4_tuple[i, 3] = body1_loc

    return next_state_4_tuple, terminal_apple, terminal_death

def grid_to_state_4_tuple(grid):
    """
    Based on what the grid looks like, decide which of the 4136 different
    (non-terminal) states it corresponds to in the small Snake game.

    Parameters:
    - grid: N-by-N matrix representing the small Snake game at a fixed time-step

    Returns:
    - state_4_tuple: Valid state-4-tuple corresponding to the grid of the form
                     [apple_loc, head_loc, body1_loc, body2_loc],
                     where the entries are linear indexes in column-major order
    """

    # Get the size of the grid
    N = grid.shape[0]

    # Get the inner grid (only interior states are considered; terminal states do
    # not need to be explicitly stored - their values are always zero)
    grid_inner = grid[1:N-1, 1:N-1]

    # Find the location of apple, head, body1, body2 (column-major order)
    apple_loc = np.where(grid_inner == -1)  # Inner representation of apple
    apple_loc = apple_loc[1]*(N-2)+apple_loc[0]
    head_loc = np.where(grid_inner == 4)  # Inner representation of head
    head_loc = head_loc[1]*(N-2)+head_loc[0]
    body1_loc = np.where(grid_inner == 3)  # Inner representation of body part 1
    body1_loc = body1_loc[1]*(N-2)+body1_loc[0]
    body2_loc = np.where(grid_inner == 2)  # Inner representation of body part 2
    body2_loc = body2_loc[1]*(N-2)+body2_loc[0]

    # Form the state-4-tuple
    state_4_tuple = [int(apple_loc), int(head_loc), int(body1_loc), int(body2_loc)]

    return state_4_tuple

def is_valid_state(state_4_tuple, N):
    """
    Function to check if a given state is a valid configuration in the small Snake game.

    Parameters:
    - state_4_tuple: Candidate state-4-tuple of the form
                     [apple_loc, head_loc, body1_loc, body2_loc],
                     where the entries are linear indexes in column-major order
    - N: The size of the game grid (an N-by-N matrix)

    Returns:
    - is_valid: True is returned if and only if state_4_tuple is a valid configuration/state
    """

    # Assume invalid by default
    is_valid = False

    # Extract relevant things (note: linear indexes in column major order)
    apple_loc, head_loc, body1_loc, body2_loc = state_4_tuple

    # Check that all four parts are in different positions
    # (if any two pieces are in the same position, we return False since it is an invalid state)
    if apple_loc == head_loc or apple_loc == body1_loc or apple_loc == body2_loc or \
            head_loc == body1_loc or head_loc == body2_loc or body1_loc == body2_loc:
        return is_valid

    # Transform linear indexing to matrix indexing
    # Note: N-2, since the states are considered only in the interior grid
    head_loc_m, head_loc_n = np.unravel_index(head_loc, (N - 2, N - 2), order='F')
    body1_loc_m, body1_loc_n = np.unravel_index(body1_loc, (N - 2, N - 2), order='F')
    body2_loc_m, body2_loc_n = np.unravel_index(body2_loc, (N - 2, N - 2), order='F')

    # Check that the snake is connected (if not connected, return False, since it is an invalid state)
    # 1) Check that body1 is adjacent to head
    if not ((body1_loc_m - 1 == head_loc_m and body1_loc_n == head_loc_n) or
            (body1_loc_m + 1 == head_loc_m and body1_loc_n == head_loc_n) or
            (body1_loc_m == head_loc_m and body1_loc_n + 1 == head_loc_n) or
            (body1_loc_m == head_loc_m and body1_loc_n - 1 == head_loc_n)):
        return is_valid

    # 2) Check that body2 is adjacent to body1
    if not ((body2_loc_m - 1 == body1_loc_m and body2_loc_n == body1_loc_n) or
            (body2_loc_m + 1 == body1_loc_m and body2_loc_n == body1_loc_n) or
            (body2_loc_m == body1_loc_m and body2_loc_n + 1 == body1_loc_n) or
            (body2_loc_m == body1_loc_m and body2_loc_n - 1 == body1_loc_n)):
        return is_valid

    # If we've come this far, then we have a valid state
    is_valid = True
    return is_valid

def update_snake_grid(grid, snake_len, score, action=2):
    """
    Function to update the state of the game and give appropriate reward
    to the learning agent.

    Parameters:
    - grid: The game screen (N-by-N matrix)
    - snake_len: The length of the snake (positive integer)
    - score: The current score of the game (nonnegative integer)
    - action: Integer in {0, 1, 2} representing the action taken by the
              agent, where 0: left, 1: forward, 2: right. Default: 1

    Returns:
    - grid: N-by-N matrix representing the game screen following the action
    - score: The score of the game after the action (remains the same as the input score,
             except when an apple was eaten)
    - reward: The reward received by the agent after taking the action.
              Will be precisely one of x, y, or z in rewards (see input rewards).
              In particular, x is received if the snake did not die nor eat an apple;
              y is received if the snake ate an apple; and z is received if the snake died.
    - terminate: True if the snake died by the action; otherwise False.
                 The game will end if the returned terminate is True; else it will continue.
    """

    # Default settings
    if action is None:
        action = 2  # Move forward by default

    # Will not terminate by default
    terminate = False

    # Get size of grid
    N = grid.shape[0]

    # Extract head and previous head locations
    head_loc = np.argwhere(grid == snake_len + 1)[0]
    prev_head_loc = np.argwhere(grid == snake_len)[0]

    # Compute where the next head location ends up being
    next_head_loc = get_next_head_loc(head_loc, prev_head_loc, action)
    next_head_loc_m, next_head_loc_n = next_head_loc

    # Update game state based on chosen action
    if grid[next_head_loc_m, next_head_loc_n] > 0:  # If snake collides, terminate the game
        terminate = True              # Will terminate
        #reward = rewards['death']     # Give death reward
    else:
        eats_apple = grid[next_head_loc_m, next_head_loc_n] == -1
        grid[next_head_loc_m, next_head_loc_n] = snake_len + 1  # Internal representation
        snake = np.argwhere(grid > 1)  # Find snake
        tail_idx = np.argmin(grid[snake[:,0], snake[:,1]])  # The tail is found due to internal representation
        grid[snake[:,0], snake[:,1]] -= 1  # Internal representation
        grid[snake[tail_idx, 0], snake[tail_idx, 1]] = 0  # The previous tail location becomes unoccupied
        grid[next_head_loc_m, next_head_loc_n] = snake_len + 1  # The new head location
        if eats_apple:
            occ_locs = np.argwhere(grid)  # Find occupied pixels, so that the new apple ends up in an unoccupied pixel
            while True:
                new_apple_loc = np.random.randint(N - 1, size=(1, 2))  # N-1, since cannot be at the wall
                if np.sum(np.all(occ_locs == new_apple_loc, axis=1)) == 0:  # Sampled unoccupied pixel location
                    break
            grid[new_apple_loc[0, 0], new_apple_loc[0, 1]] = -1  # Apple represented by -1
            score += 1  # Increase score by 1
            #reward = rewards['apple']  # Give apple reward

    return grid, score, terminate

def get_states_next_state_idxs(nbr_states, nbr_actions, N):
    states = np.full((nbr_states, 4), np.nan)  # Will be filled with meaningful stuff
    ctr = 0  # Keeps track of position in states

    for apple_loc in range((N - 2)**2):  # Iterate over apple locations
        for head_loc in range((N - 2)**2):  # Iterate over head locations
            for body1_loc in range((N - 2)**2):  # Iterate over body-part 1 locations
                for body2_loc in range((N - 2)**2):  # Iterate over body-part 2 locations
                    
                    # Current state-4-tuple candidate
                    state = np.array([apple_loc, head_loc, body1_loc, body2_loc])
                    
                    # If it is a valid state, insert into state representation 
                    if is_valid_state(state, N):
                        states[ctr, :] = state
                        ctr += 1

    # Check the three possible following states for all the states
    next_state_idxs = np.full((nbr_states, nbr_actions), np.nan)

    for i in range(nbr_states):
        # Get next state and check if the action leads to a terminal state
        # (either by eating an apple or by hitting a wall)
        s_primes, terminal_apples, terminal_deaths = get_next_state(states[i, :], N)

        # Iterate over the three possible actions
        for action in range(nbr_actions):
            # Extract stuff corresponding to action
            next_state = s_primes[action, :]
            terminal_apple = terminal_apples[action]
            terminal_death = terminal_deaths[action]

            # Get index corresponding to the next state, except for terminal
            # actions which have certain terminal symbols (-1 or 0)
            if terminal_apple:
                next_state_idx = -1  # Terminal symbol for apple
            elif terminal_death:
                next_state_idx = -2   # Terminal symbol for death
            else:
                next_state_idx = np.where((states[:, 0] == next_state[0]) &
                                          (states[:, 1] == next_state[1]) &
                                          (states[:, 2] == next_state[2]) &
                                          (states[:, 3] == next_state[3]))[0][0]

            # Insert next state index (or terminal symbol) into next_state_idxs
            next_state_idxs[i, action] = next_state_idx

    return states, next_state_idxs

def policy_iteration(pol_eval_tol, next_state_idxs, rewards, gamm):
    # Get number of non-terminal states and actions.
    nbr_states, nbr_actions = next_state_idxs.shape

    # Arbitrary initialization of values and policy.
    values = np.random.randn(1, nbr_states)
    policy = np.random.randint(0, nbr_actions, size=(nbr_states,))  # policy is size 1-by-nbr_states
                                                           # the entries of policy are 0, 1, or 2
                                                           # selected uniformly at random

    # Counters over the number of policy iterations and policy evaluations,
    # for possible diagnostic purposes.
    nbr_pol_iter = 0
    nbr_pol_eval = 0

    # This while-loop runs the policy iteration.
    while True:
        # Policy evaluation.
        while True:
            Delta = 0
            for state_idx in range(nbr_states):
                V_old = values[0,state_idx] # Gets the old V(s)

                action = policy[state_idx] # Takes the action pi(s) from policy

                if next_state_idxs[state_idx, action] > 0: # Non-terminal state
                    reward = rewards['default']
                    next_state_value = values[0,int(next_state_idxs[state_idx, action])]
                elif next_state_idxs[state_idx, action] == -2: # id = -2 for death
                    reward = rewards['death']
                    next_state_value = 0
                else: # id = -1 for apple
                    reward = rewards['apple']
                    next_state_value = 0

                V_new = reward + gamm*next_state_value # Deterministic problem, no probabilities

                Delta = max(Delta, abs(V_old - V_new)) # Update the error

                values[0,state_idx] = V_new # Update to the new V(s)

            # Increase nbr_pol_eval counter.
            nbr_pol_eval += 1

            # Check for policy evaluation termination.
            if Delta < pol_eval_tol:
                break
            else:
                print('Delta:', Delta)

        # Policy improvement.
        policy_stable = True
        for state_idx in range(nbr_states):
            a_old = policy[state_idx]
            V_list = []
            for a in range(nbr_actions):
                action = a # Test all posible actions
                if next_state_idxs[state_idx, action] > 0: # Non-terminal state
                    reward = rewards['default']
                    next_state_value = values[0,int(next_state_idxs[state_idx, action])]
                elif next_state_idxs[state_idx, action] == -2: # id = -2 for death
                    reward = rewards['death']
                    next_state_value = 0
                else: # id = -1 for apple
                    reward = rewards['apple']
                    next_state_value = 0

                V_a = reward + gamm*next_state_value

                V_list.append(V_a)
            
            a_best = np.argmax(V_list) # Get the best action
            policy[state_idx] = a_best # Update pi(s) to the best choice of action
            if a_best != a_old:
                policy_stable = False

        # Increase the number of policy iterations.
        nbr_pol_iter += 1
        
        # Check for policy iteration termination
        if policy_stable:
            break

    return values, policy, nbr_pol_iter, nbr_pol_eval

