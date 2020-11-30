from z3 import *
import json

def get_neighbour_indexes (index, size):
    #working but not readable
    #vectors = list(set([(i*k,j*l) for i in range(2) for j in range(2) for k in [-1, 1] for l in [-1,1]]))
    vectors = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    neighbour_locations = list(map(lambda x: (index[0] + x[0], index[1] + x[1]), vectors))
    legal_neighbour_locations = list(filter(lambda x: (x[0] in range(size[0]) and x[1] in range(size[1])), neighbour_locations))
    return legal_neighbour_locations

def matrix_to_str(matrix):
    return ("".join([cell for row in matrix for cell in (["\n"] + row)])).lstrip()

def print_pretty_formatted_math_matrix(matrix):
    print(matrix_to_str(matrix))
    
def print_pretty_formatted_cartesian_matrix(matrix):
    math_conversion = get_math_matrix(from_cartesian_matrix=matrix)
    print_pretty_formatted_math_matrix(math_conversion)

def get_cartesian_matrix(*, from_math_matrix):
    """ 
    inversion of the reference system of the matrix to go from 
    math notation M[i, j] to cartesian notation (x, y)
    """
    math_matrix = from_math_matrix
    #invert rows cause we use cartesian (x,y) rappresentation for gridmap and not matrix m_i,j notation
    math_matrix_inverted_rows = math_matrix[::-1]
    #invert (x, y) to (y, x) cause we use cartesian notation accessing first columns then rowa
    return [[math_matrix_inverted_rows[y][x] for y in range(len(math_matrix_inverted_rows))] for x in range(len(math_matrix_inverted_rows[0]))] 

def get_math_matrix(*, from_cartesian_matrix):
    """ 
    inversion of the reference system of the matrix to go from 
    cartesian notation (x, y) to math notation M[i, j]
    """
    cartesian_matrix = from_cartesian_matrix
    #invert (x, y) to (y, x) cause we use math notation accessing first rows and then columns
    inverted_indexes_cartesian_matrix = [[cartesian_matrix[j][i] for j in range(len(cartesian_matrix))] for i in range(len(cartesian_matrix[0]))] 
    #invert rows cause we use cartesian (x,y) rappresentation for gridmap and not matrix m_i,j notation
    return inverted_indexes_cartesian_matrix[::-1]

def get_all_models(*, with_solver, for_m_xy_list):
    s = with_solver
    m_xy_list = for_m_xy_list
    
    models = []
    while s.check() == sat:
        models.append([s.model().eval(el) for el in m_xy_list])
        s.add(Or([v() != s.model()[v] for v in s.model()]))
    return models

def get_conclusions(*, from_sat_models):
    sat_models = from_sat_models

    sat_results =[]
    #cartesian matrix
    inverted_indexes_models = [[sat_models[j][i] for j in range(len(sat_models))] for i in range(len(sat_models[0]))] 
    for value in inverted_indexes_models:
        if all([(x == 0) for x in value]):
            sat_results.append("s")
        elif all([(x == 1) for x in value]):
            sat_results.append("X")
        else: 
            sat_results.append("?")
    return sat_results

def get_updated_cartesian_matrix(*, from_mine_cartesian_matrix, with_sat_conclusions):
    mine_cartesian_matrix = from_mine_cartesian_matrix
    sat_conclusions = with_sat_conclusions
    
    result_cartesian_matrix = []
    for x in range(game_width):
        column = []
        for y in range(game_height):
            if mine_cartesian_matrix[x][y] != "?":
                column.append(mine_cartesian_matrix[x][y])
            else:
                column.append(sat_conclusions[x * game_height + y])
        result_cartesian_matrix.append(column)

    return result_cartesian_matrix

def update(*, json_object, with_result_math_matrix):
    result_math_matrix = with_result_math_matrix
    json_object["configuration"] = matrix_to_str(result_math_matrix)
    return json_object
    
# read configuration from file
with open(sys.argv[1]) as json_file:
    game_info = json.load(json_file)

s = Solver()

# get all parameters from json
number_of_mines = game_info["numberOfMines"]
game_width = game_info["size"][0]
game_height = game_info["size"][1]
mine_math_matrix = [[x for x in line] for line in game_info["configuration"].split("\n")]
mine_cartesian_matrix = get_cartesian_matrix(from_math_matrix=mine_math_matrix)

# generate all variables and indexes for Z3 solver
m_xy_list = [Int(f"m_{i}{j}") for i in range(game_width) for j in range (game_height)]
index_list = [(i, j) for i in range(game_width) for j in range (game_height)]
index_m_xy_list = list(zip(index_list, m_xy_list))

# each of the cell m_xy of the minesweeper can be a mine (1) or not (0)
for m_xy in m_xy_list:
    s.add(m_xy <= 1)
    s.add(m_xy >= 0)

#  the total number of mines in the world is equal to number_of_mines
s.add(Sum(m_xy_list) == number_of_mines)

# for each cell check if it is known (numeric) and in case of known cells assert that the sum all neighbors is equal to its value
for index_m_xy in index_m_xy_list:
    cell_value = mine_cartesian_matrix[index_m_xy[0][0]][index_m_xy[0][1]] #take care of different indexing for matrix (top to bottom) and game indexing (x,y) -cartesian
    
    if cell_value not in {"?"}:
        if cell_value in {"X"}:
            s.add(index_m_xy[1] == 1)
        elif cell_value.isnumeric():
            neighbour_indexes = get_neighbour_indexes(index_m_xy[0], (game_width, game_height))
            neighbour_vars = list(map(lambda x: x[1], filter(lambda x: x[0] in neighbour_indexes, index_m_xy_list)))
            s.add(Sum(neighbour_vars) == int(cell_value))

models = get_all_models(with_solver=s, for_m_xy_list=m_xy_list)

if not models:
    print("The game is not deterministic")
    sys.exit(0)

sat_conclusions = get_conclusions(from_sat_models=models)
result_cartesian_matrix = get_updated_cartesian_matrix(from_mine_cartesian_matrix=mine_cartesian_matrix, with_sat_conclusions=sat_conclusions)
result_math_matrix = get_math_matrix(from_cartesian_matrix=result_cartesian_matrix)
updated_json = update(json_object=game_info, with_result_math_matrix=result_math_matrix)

#write to json-file
with open(sys.argv[2], "w") as file_json:
    json.dump(updated_json, file_json)

# write to terminal
print("Input matrix:", end="\n\n")
print_pretty_formatted_math_matrix(mine_math_matrix)
print("\nResulting matrix:", end="\n\n")
print_pretty_formatted_math_matrix(result_math_matrix)
