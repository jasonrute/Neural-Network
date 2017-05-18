import neural_network as nn
import random
import time

#
# Automata functions
#
def shift_left_rule(c,e,ne,n,nw,w,sw,s,se):
    if w == 1:
        return 1
    else:
        return 0

def serpinski_rule(c,e,ne,n,nw,w,sw,s,se):
    if (nw, n, ne) == (1, 0, 0):
        return 1
    elif (nw, n, ne) == (0, 0, 1):
        return 1
    elif (nw, n, ne) == (1, 1, 0):
        return 1
    elif (nw, n, ne) == (0, 1, 1):
        return 1
    elif c == 1 and (nw, n, ne) == (0, 0, 0):
        return 1
    else:
        return 0

def serpinski_rule_plus_shift(c,e,ne,n,nw,w,sw,s,se):
    if (w, c, e) == (1, 0, 0):
        return 1
    elif (w, c, e) == (0, 0, 1):
        return 1
    elif (w, c, e) == (1, 1, 0):
        return 1
    elif (w, c, e) == (0, 1, 1):
        return 1
    elif s == 1 and (w, c, e) == (0, 0, 0):
        return 1
    else:
        return 0

def rule_30(c,e,ne,n,nw,w,sw,s,se):
    if (nw, n, ne) == (1, 0, 0):
        return 1
    elif (nw, n, ne) == (0, 0, 1):
        return 1
    elif (nw, n, ne) == (0, 1, 0):
        return 1
    elif (nw, n, ne) == (0, 1, 1):
        return 1
    elif c == 1 and (nw, n, ne) == (0, 0, 0):
        return 1
    else:
        return 0

def conway_game_of_life(c,e,ne,n,nw,w,sw,s,se):
    if c == 1:
        if sum([e,ne,n,nw,w,sw,s,se]) in (2,3):
            return 1
        else:
            return 0
    elif sum([e,ne,n,nw,w,sw,s,se]) == 3:
        return 1
    else:
        return 0

def high_life(c,e,ne,n,nw,w,sw,s,se):
    if c == 1:
        if sum([e,ne,n,nw,w,sw,s,se]) in (2,3):
            return 1
        else:
            return 0
    elif sum([e,ne,n,nw,w,sw,s,se]) in (3, 6):
        return 1
    else:
        return 0

def blob(c,e,ne,n,nw,w,sw,s,se):
    if sum([c,e,n,w,s]):
        return 1
    else:
        return 0

def blob_unblob(c,e,ne,n,nw,w,sw,s,se):
    if 0 < sum([c,e,n,w,s]) < 5:
        return 1
    else:
        return 0

def checkers(c,e,ne,n,nw,w,sw,s,se):
    if sum([c,ne,nw,sw,se]):
        return 1
    else:
        return 0

#
# Grid functions
#
def neighbor_bits(i,j,grid):
    # assume grid is counter so it defaults as zero for points outside the grid
    return [(grid[i+di,j+dj] if (i+di,j+dj) in grid else 0) 
                for di,dj in ((0,0),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1))]

def next_grid(grid, rule):
    new_grid = {(i,j): rule(*neighbor_bits(i,j,grid)) for i,j in grid}
    return new_grid

def all_nbhr_bits(grid):
    return (neighbor_bits(i,j,grid) for i,j in grid)

def input_data(grid):
    point_neighbor_pairs = (((i,j), neighbor_bits(i,j,grid)) for i,j in grid)
    points, neighbors = zip(*point_neighbor_pairs) # this unzips it
    return points, neighbors

def next_training_data(grid, rule):
    new_training_data = ((input_bits, [rule(*input_bits)]) 
                            for input_bits in all_nbhr_bits(grid))
    return new_training_data

def random_grid(h,w):
    new_grid = {(i,j): random.randrange(2) for i in range(height) for j in range(width)}
    return new_grid

def print_grid(grid, h, w):
    for i in range(h):
        for j in range(w):
            print("#" if grid[i,j] else ".", end=" ")
        print("")
    print("")


#
# Various starting configurations
#
height = 15
width = 15
frame_count = 6

blank_grid = {(i,j):0 for i in range(height) for j in range(width)}

serpinski_start = blank_grid
serpinski_start[0,7] = 1

blob_start = blank_grid
blob_start[7,7] = 1

serpinski_shift_start = blank_grid
serpinski_shift_start[13,7] = 1

conway_glider_start = blank_grid
conway_glider_start[2,0] = 1
conway_glider_start[2,1] = 1
conway_glider_start[2,2] = 1
conway_glider_start[1,2] = 1
conway_glider_start[0,1] = 1
conway_glider_start[8,6] = 1
conway_glider_start[8,7] = 1
conway_glider_start[8,8] = 1
conway_glider_start[7,8] = 1
conway_glider_start[6,7] = 1

random_start_grid = random_grid(height, width)


rules = \
    [(shift_left_rule, random_start_grid, "Left shift"),
     (serpinski_rule, serpinski_start, "Serpinski"),
     (rule_30, serpinski_start, "Rule 30"),
     (serpinski_rule_plus_shift, serpinski_shift_start, "Serpinski plus shift"),
     (blob, blob_start, "Serpinski plus shift"),
     (blob_unblob, blob_start, "Blob/Unblob"),
     (checkers, blob_start, "Checkers"),
     (conway_game_of_life, conway_glider_start, "Conway glider"),
     (conway_game_of_life, random_start_grid, "Conway game of life"),
     (high_life, random_start_grid, "High Life")]

#
# Set parameters
#
num_inputs = 9
num_outputs = 1
hidden_layer_sizes = [20, 20]
num_training_iterations = 1000
learning_rate = .5

for my_rule, init_grid, description in rules:
    print()
    print(description)
    
    # Build training data
    grids = []
    training_data = []
    grids.append(init_grid)
    for i in range(frame_count):
        grids.append(next_grid(grids[i], my_rule))
        training_data.extend(next_training_data(grids[i], my_rule))
        print_grid(grids[i+1], height, width)

    test_points, test_inputs = input_data(grids[frame_count])
    grids.append(next_grid(grids[frame_count], my_rule))

    my_neural_network = nn.Neural_Network(num_inputs, num_outputs, hidden_layer_sizes)

    #
    # Train network
    #

    start_time = time.perf_counter()
    my_neural_network.train_in_batches(training_data, 10, 10000, learning_rate)
    print("Training took: ", time.perf_counter() - start_time)

    #
    # Print answer
    #

    prediction_grid = {}
    for point, test_input in zip(test_points, test_inputs):
        out = my_neural_network.compute_output(test_input)
        prediction_grid[point] = round(out[0])

    print("Prediction:")
    print_grid(prediction_grid, height, width)
    is_good = (prediction_grid == grids[frame_count+1])
    if is_good:
        print("Success!")
    else:
        print("FAILED!")
