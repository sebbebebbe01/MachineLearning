import numpy as np

def update_snake_grid(grid, head_loc, prev_head_loc, snake_len, score, rewards, action=2):
    # Default settings
    if action is None:
        action = 2  # Move forward by default

    # Will not terminate by default
    terminate = False

    # Compute where next head location ends up being
    next_head_loc = get_next_head_loc(head_loc, prev_head_loc, action)

    # Update game state based on chosen action
    if grid[next_head_loc[0], next_head_loc[1]] > 0:  # If snake collides, terminate the game
        terminate = True
        reward = rewards['death']  # Give death reward
    elif grid[next_head_loc[0], next_head_loc[1]] == -1:  # If snake eats apple, update accordingly and increase score by 1
        snake_len += 1  # Increase snake length
        grid[next_head_loc[0], next_head_loc[1]] = snake_len + 1  # Internal representation
        prev_head_loc = head_loc  # The next prev_head_loc is this head_loc
        head_loc = next_head_loc  # The next head_loc is next_head_loc
        N = grid.shape[0]  # Size of the grid (N-by-N)
        occ_locs = np.argwhere(grid > 0)  # Find occupied pixels, so that the new apple ends up in an unoccupied pixel
        while True:
            new_apple_loc = np.random.randint(0, N - 1, size=(1, 2))  # N-1, since it cannot be at the wall
            if not np.any(np.all(occ_locs == new_apple_loc, axis=1)):  # This happens when we sampled an unoccupied pixel location
                break
        grid[new_apple_loc[0, 0], new_apple_loc[0, 1]] = -1  # Apple represented by -1
        score += 1  # Increase score by 1
        reward = rewards['apple']  # Give apple reward
    else:  # In this case, the snake simply moved in a direction, not eating any apple nor dying
        snake = np.argwhere(grid > 1)  # Find snake
        tail_idx = np.argmin(grid[snake[:,0], snake[:,1]])  # The tail is found due to internal representation
        grid[snake[:,0], snake[:,1]] -= 1  # Internal representation
        grid[snake[tail_idx, 0], snake[tail_idx, 1]] = 0  # The previous tail location becomes unoccupied
        grid[next_head_loc[0], next_head_loc[1]] = snake_len + 1  # The new head location
        prev_head_loc = head_loc  # The next prev_head_loc is this head_loc
        head_loc = next_head_loc  # The next head_loc is next_head_loc
        reward = rewards['default']  # Give default reward

    return grid, head_loc, prev_head_loc, snake_len, score, reward, terminate

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

def Q_fun(weights, state_action_feats, action=0):

    if action:  # Q-value for a particular action (left / forward / right)
        Q_val = np.dot(np.transpose(weights), state_action_feats[:, (action -1)])
    else:  # Q-value for all possible actions computed
        Q_val = np.dot(np.transpose(weights), state_action_feats)
    return Q_val

def gen_snake_grid(N=30, snake_len=None, nbr_apples=1):
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
    # head_loc = [14, 14]
    snake_rot = np.random.randint(1, 5)  # 1: north, 2: east, 3: south, 4: west
    # snake_rot = 1
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
            # apple_loc = np.array([[5,5]])
            if not np.any(np.all(occ_locs == apple_loc, axis=1)):
                break

        # Update occ_locs and insert apple into snake_grid
        occ_locs[i + snake_len, :] = apple_loc
        grid[apple_loc[0][0], apple_loc[0][1]] = -1

    return grid, head_loc

def extract_state_action_features(prev_grid, grid, prev_head_loc, nbr_feats):
    # Extract grid size
    N = grid.shape[0]

    # Initialize state_action_feats to nbr_feats-by-3 matrix
    state_action_feats = np.empty((nbr_feats, 3))
    state_action_feats[:] = np.nan

    # Based on how grid looks now and at the previous time step, infer head location
    change_grid = grid - prev_grid
    prev_grid = grid  # Used in later calls to "extract_state_action_features.m"

    # Find head location (initially known that it is in the center of the grid)
    if np.count_nonzero(change_grid) > 0:  # True, except in the initial time step
        head_loc = np.argwhere(change_grid > 0)[0]
    else:  # True only in the initial time step
        head_loc = np.array([round(N / 2) -1, round(N / 2) -1])

    # Previous head location
    prev_head_loc_m = prev_head_loc[0]
    prev_head_loc_n = prev_head_loc[1]

    # Infer current movement directory (N/E/S/W) by looking at how current and previous
    # head locations are related
    if prev_head_loc_m == head_loc[0] + 1 and prev_head_loc_n == head_loc[1]:  # NORTH
        movement_dir = 1
    elif prev_head_loc_m == head_loc[0] and prev_head_loc_n == head_loc[1] - 1:  # EAST
        movement_dir = 2
    elif prev_head_loc_m == head_loc[0] - 1 and prev_head_loc_n == head_loc[1]:  # SOUTH
        movement_dir = 3
    else:  # WEST
        movement_dir = 4

    # The current head_loc will at the next time-step be prev_head_loc
    prev_head_loc = head_loc

    # HERE BEGINS YOUR STATE-ACTION FEATURE ENGINEERING
    # Replace this to fit the number of state-action features per features
    # you choose (3 are used below), and of course replace the randn()
    # by something more sensible.

    snake_len = np.sum(grid[1:N-1,1:N-1] > 0)
    apple_locs = np.argwhere(grid == -1)

    for action in range(1, 4):  # Evaluate all the different actions (left, forward, right).
        # Feel free to uncomment below line of code if you find it useful.
        next_head_loc, next_move_dir = get_next_info(action, movement_dir, head_loc)

        # ## Distance to closest wall after moving
        # state_action_feats[0, action-1] = np.min( (abs(next_head_loc - np.array([N-1, N-1])), abs(next_head_loc)) ) / N
        
        ## Manhattan norm to (closest) apple after moving, normalised and divided by the length of the snake
        manhattan_apple_distances = np.linalg.norm(next_head_loc - apple_locs, ord = 1, axis = 1)
        state_action_feats[0, action-1] = np.min(manhattan_apple_distances) / (2*(N-3))
        
        ## Is next tile body or border?
        is_death = int(grid[next_head_loc[0], next_head_loc[1]] == 1)
        state_action_feats[1, action-1] = is_death

        ## What's infront of it and to the sides?
        ## Looks at all tiles in front of the snake and to the sides (excluding border)
        line_north = np.flip(grid[1 : next_head_loc[0], next_head_loc[1]])
        line_east = grid[next_head_loc[0], next_head_loc[1]+1 : N-1]
        line_south = grid[next_head_loc[0]+1 : N-1, next_head_loc[1]]
        line_west = np.flip(grid[next_head_loc[0], 1 : next_head_loc[1]])
        if next_move_dir==1: # North
            line_of_sight, line_right, line_left = line_north, line_east, line_west
        elif next_move_dir==2: # East
            line_of_sight, line_right, line_left = line_east, line_south, line_north
        elif next_move_dir==3: # South
            line_of_sight, line_right, line_left = line_south, line_west, line_east
        elif next_move_dir==4: # West
            line_of_sight, line_right, line_left = line_west, line_north, line_south

        forw_occupied = np.argwhere(line_of_sight == 1)
        if len(forw_occupied)==0:
            dist_to_forw_occupied = 1
        else:
            dist_to_forw_occupied = forw_occupied[0] / (N-3) # Normalised w.r.t. max distance
        # if len(line_right)==1:
        #     dist_to_right_occupied = 0
        # else:
        #     right_occupied = np.argwhere(line_right[1:] == 1) # Starting from the tile to the right of the snake
        #     dist_to_right_occupied = (right_occupied[0]+1) / (N-1) # Normalised w.r.t. max distance
        # if len(line_left)==1:
        #     dist_to_left_occupied = 0
        # else:
        #     left_occupied = np.argwhere(line_left[1:] == 1) # Starting from the tile to the left of the snake
        #     dist_to_left_occupied = (left_occupied[0]+1) / (N-1) # Normalised w.r.t. max distance
        state_action_feats[2, action-1] = dist_to_forw_occupied
        # state_action_feats[3, action-1] = dist_to_right_occupied
        # state_action_feats[4, action-1] = dist_to_left_occupied

        # ## Euclidean distance to the centre of mass
        # grid_no_borders = grid[1:N-1, 1:N-1]
        # # Get indices of the borderless matrix
        # rows, cols = np.indices(grid_no_borders.shape)
        # # Compute centre of mass
        # total_mass = grid_no_borders.sum()
        # com = np.array([(rows * grid_no_borders).sum() / total_mass, (cols * grid_no_borders).sum() / total_mass])

        # centre_of_mass_dist = np.linalg.norm(com - (next_head_loc-1), ord=1) / np.sqrt(2) / (N-2)
        # state_action_feats[2, action-1] = centre_of_mass_dist
        


    return state_action_feats, prev_grid, prev_head_loc

def get_next_info(action, movement_dir, head_loc):
    # Function to infer next head location and movement direction

    # Extract relevant stuff
    head_loc_m = head_loc[0]
    head_loc_n = head_loc[1]

    if movement_dir == 1:  # NORTH
        if action == 1:  # left
            next_head_loc_m = head_loc_m
            next_head_loc_n = head_loc_n - 1
            next_move_dir = 4
        elif action == 2:  # forward
            next_head_loc_m = head_loc_m - 1
            next_head_loc_n = head_loc_n
            next_move_dir = 1
        else:  # right
            next_head_loc_m = head_loc_m
            next_head_loc_n = head_loc_n + 1
            next_move_dir = 2
    elif movement_dir == 2:  # EAST
        if action == 1:
            next_head_loc_m = head_loc_m - 1
            next_head_loc_n = head_loc_n
            next_move_dir = 1
        elif action == 2:
            next_head_loc_m = head_loc_m
            next_head_loc_n = head_loc_n + 1
            next_move_dir = 2
        else:
            next_head_loc_m = head_loc_m + 1
            next_head_loc_n = head_loc_n
            next_move_dir = 3
    elif movement_dir == 3:  # SOUTH
        if action == 1:
            next_head_loc_m = head_loc_m
            next_head_loc_n = head_loc_n + 1
            next_move_dir = 2
        elif action == 2:
            next_head_loc_m = head_loc_m + 1
            next_head_loc_n = head_loc_n
            next_move_dir = 3
        else:
            next_head_loc_m = head_loc_m
            next_head_loc_n = head_loc_n - 1
            next_move_dir = 4
    else:  # WEST
        if action == 1:
            next_head_loc_m = head_loc_m + 1
            next_head_loc_n = head_loc_n
            next_move_dir = 3
        elif action == 2:
            next_head_loc_m = head_loc_m
            next_head_loc_n = head_loc_n - 1
            next_move_dir = 4
        else:
            next_head_loc_m = head_loc_m - 1
            next_head_loc_n = head_loc_n
            next_move_dir = 1

    next_head_loc = np.array([next_head_loc_m, next_head_loc_n])
    return next_head_loc, next_move_dir

