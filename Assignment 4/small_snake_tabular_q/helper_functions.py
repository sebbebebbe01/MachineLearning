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
        action = [1, 2, 3]

    # Get the number of actions we want to compute next_state_4_tuple for
    nbr_actions = len(action)

    # Default next_state_4_tuple (will be irrelevant if termination occurs)
    next_state_4_tuple = np.full((nbr_actions, 4), np.nan)

    # Default terminal logicals (assume non-terminal by default)
    terminal_apple = np.zeros((1, nbr_actions), dtype=bool)
    terminal_death = np.zeros((1, nbr_actions), dtype=bool)

    # Extract relevant things (note: linear indexes in column-major order)
    apple_loc, head_loc, body1_loc = state_4_tuple

    # Transform linear indexing to matrix indexing
    # Note: N-2, since the states are considered only in the interior grid
    apple_loc_m, apple_loc_n = np.unravel_index(apple_loc, (N - 2, N - 2), order='F')
    head_loc_m, head_loc_n = np.unravel_index(head_loc, (N - 2, N - 2), order='F')
    body1_loc_m, body1_loc_n = np.unravel_index(body1_loc, (N - 2, N - 2), order='F')

    # Compute where the next head location ends up being (check collisions or apple eating)
    for i in range(len(action)):
        # Get the current action
        act = action[i]

        # Get the next head location based on the chosen action
        next_head_loc = get_next_head_loc((head_loc_m, head_loc_n), (body1_loc_m, body1_loc_n), act)
        next_head_loc_m, next_head_loc_n = next_head_loc

        # Check if the action yields an apple-terminal state
        if next_head_loc_m == apple_loc_m and next_head_loc_n == apple_loc_n:
            terminal_apple[0, i] = True
            continue

        # Check if the action yields a death-terminal state
        if next_head_loc_m == 0 or next_head_loc_m == N - 1 or \
                next_head_loc_n == 0 or next_head_loc_n == N - 1:
            terminal_death[0, i] = True
            continue

        # Now we know that the next state is not terminal - let's find out what the next state is!
        next_state_4_tuple[i, 0] = apple_loc
        next_state_4_tuple[i, 1] = np.ravel_multi_index((next_head_loc_m, next_head_loc_n), (N - 2, N - 2), order='F')
        next_state_4_tuple[i, 2] = head_loc
        next_state_4_tuple[i, 3] = body1_loc

    return next_state_4_tuple, terminal_apple, terminal_death

def get_states(nbr_states, N):
    """
    Function to generate state representation for quick access during policy
    iteration in the small Snake game.

    Parameters:
    - nbr_states: Number of non-terminal states
    - N: The size of the game grid (an N-by-N matrix)

    Returns:
    - states: nbr_states-by-4 matrix representing all the non-terminal game states.
              Each row has the form [apple_loc, head_loc, body1_loc, body2_loc],
              where the entries are linear indexes in column-major order.
    """

    states = np.full((nbr_states, 4), np.nan)  # Will be filled with meaningful stuff
    ctr = 0  # Keeps track of position in states

    for apple_loc in range((N - 2)**2):  # Iterate over apple locations
        for head_loc in range((N - 2)**2):  # Iterate over head locations
            for body1_loc in range((N - 2)**2):  # Iterate over body-part 1 locations
                for body2_loc in range((N - 2)**2):  # Iterate over body-part 2 locations

                    # Current state-4-tuple candidate
                    state = [apple_loc, head_loc, body1_loc, body2_loc]

                    # If it is a valid state, insert into state representation
                    if is_valid_state(state, N):
                        states[ctr, :] = state
                        ctr += 1

    return states

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
    apple_loc = apple_loc[0]*(N-2)+apple_loc[1]
    head_loc = np.where(grid_inner == 4)  # Inner representation of head
    head_loc = head_loc[0]*(N-2)+head_loc[1]
    body1_loc = np.where(grid_inner == 3)  # Inner representation of body part 1
    body1_loc = body1_loc[0]*(N-2)+body1_loc[1]
    body2_loc = np.where(grid_inner == 2)  # Inner representation of body part 2
    body2_loc = body2_loc[0]*(N-2)+body2_loc[1]

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

def update_snake_grid(grid, snake_len, score, rewards, action=2):
    """
    Function to update the state of the game and give appropriate reward
    to the learning agent.

    Parameters:
    - grid: The game screen (N-by-N matrix)
    - snake_len: The length of the snake (positive integer)
    - score: The current score of the game (nonnegative integer)
    - rewards: Struct of the form {'default': x, 'apple': y, 'death': z}
               Here x refers to the default reward, which is received when
               the snake only moves without dying and without eating an apple;
               y refers to the reward obtained when eating an apple;
               and z refers to the reward obtained when dying.
    - action: Integer in {1, 2, 3} representing the action taken by the
              agent, where 1: left, 2: forward, 3: right. Default: 2

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
        reward = rewards['death']     # Give death reward
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
            reward = rewards['apple']  # Give apple reward
        else:
            reward = rewards['default']  # Give default reward

    return grid, score, reward, terminate

