from z3 import *
from collections import Counter
import json

class Cell:
    def __init__(self, x, y, has_hunter=" ", has_wumpus="?", has_stence="?", has_pit="?", has_breeze="?"):
        self.has_hunter = has_hunter
        self.has_wumpus = has_wumpus
        self.has_stence = has_stence
        self.has_pit = has_pit
        self.has_breeze = has_breeze

    def __str__(self):
        return f"|{self.has_hunter}{self.has_stence}{self.has_wumpus}{self.has_breeze}{self.has_pit}|"

    def __repr__(self):
        return f"|{self.has_hunter}{self.has_stence}{self.has_wumpus}{self.has_breeze}{self.has_pit}|"

    def get_string(self, for_fields=None):
        if for_fields:
            return f"|{self.has_hunter if 'A' in for_fields else ''}{self.has_stence if 'S' in for_fields else ''}{self.has_wumpus if 'W' in for_fields else ''}{self.has_breeze if 'B' in for_fields else ''}{self.has_pit if 'P' in for_fields else ''}|"
        else:
            return self.__str__()

def get_neighbour_indexes (index, size):
    vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbour_locations = list(map(lambda x: (index[0] + x[0], index[1] + x[1]), vectors))
    legal_neighbour_locations = list(filter(lambda x: (x[0] in range(size[0]) and x[1] in range(size[1])), neighbour_locations))
    return legal_neighbour_locations

def matrix_to_str(matrix, for_fields=None):
    return ("".join([cell if isinstance(cell, str) else cell.get_string(for_fields=for_fields) for row in matrix for cell in (["\n"] + row)])).lstrip()

def print_pretty_formatted_math_matrix(matrix, for_fields=None):
    print(matrix_to_str(matrix, for_fields))
    
def print_pretty_formatted_cartesian_matrix(matrix, for_fields=None):
    math_conversion = get_math_matrix(from_cartesian_matrix=matrix)
    print_pretty_formatted_math_matrix(math_conversion, for_fields)

def get_cartesian_matrix(*, with_objects):
    """ 
    inversion of the reference system of the matrix to go from 
    math notation M[i, j] to cartesian notation (x, y)
    """
    objects = with_objects
    width = with_objects["size"][0]
    height = with_objects["size"][1]

    cartesian_matrix = []
    for x in range(width):
        column = []
        for y in range(height):
            cell = Cell(x, y)
            column.append(cell)
        cartesian_matrix.append(column)

    for hunter in objects["hunters"]:
        cartesian_matrix[hunter[0]][hunter[1]].has_hunter = "A"
    for pit in objects["pits"]:
        cartesian_matrix[pit[0]][pit[1]].has_pit = "P"
    for no_pit in objects["noPits"]:
        cartesian_matrix[no_pit[0]][no_pit[1]].has_pit = " "
    for breeze in objects["breezes"]:
        cartesian_matrix[breeze[0]][breeze[1]].has_breeze = "B"
    for no_breeze in objects["noBreezes"]:
        cartesian_matrix[no_breeze[0]][no_breeze[1]].has_breeze = " "
    for wumpus in objects["wumpuses"]:
        cartesian_matrix[wumpus[0]][wumpus[1]].has_wumpus = "W"
    for no_wumpus in objects["noWumpuses"]:
        cartesian_matrix[no_wumpus[0]][no_wumpus[1]].has_wumpus = " "
    for stence in objects["stences"]:
        cartesian_matrix[stence[0]][stence[1]].has_stence = "S"
    for no_stence in objects["noStences"]:
        cartesian_matrix[no_stence[0]][no_stence[1]].has_stence = " "
    return cartesian_matrix

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

def get_all_models(*, with_solver, for_c_xy_list):
    solver = with_solver
    c_xy_list = for_c_xy_list
    
    models = []
    while solver.check() == sat:
        models.append([solver.model().eval(el) for el in c_xy_list])
        solver.add(Or([v() != solver.model()[v] for v in solver.model()]))
    return models

def get_conclusions(*, from_sat_models):
    sat_models = from_sat_models

    sat_results =[]
    #cartesian matrix
    inverted_indexes_models = [[sat_models[j][i] for j in range(len(sat_models))] for i in range(len(sat_models[0]))] 
    for value in inverted_indexes_models:
        if all([(x == 0) for x in value]):
            sat_results.append("0")
        elif all([(x == 1) for x in value]):
            sat_results.append("1")
        else: 
            sat_results.append("?")
    return sat_results

def get_updated_cartesian_matrix(*, from_cartesian_matrix, with_sat_conclusions, on_field):
    cartesian_matrix = from_cartesian_matrix
    sat_conclusions = with_sat_conclusions
    field = on_field
    
    result_cartesian_matrix = []
    for x in range(len(cartesian_matrix)):
        column = []
        for y in range(len(cartesian_matrix[0])):
            if field in {"P"}:
                has_pit = sat_conclusions[x * game_height + y]
                if has_pit in {"1"}:
                    has_pit = "P"
                elif has_pit in {"0"}:
                    has_pit = " "
                cell = Cell(x, y, 
                                    has_hunter=cartesian_matrix[x][y].has_hunter,
                                    has_wumpus=cartesian_matrix[x][y].has_wumpus,
                                    has_stence=cartesian_matrix[x][y].has_stence, 
                                    has_pit=has_pit,
                                    has_breeze=cartesian_matrix[x][y].has_breeze
                                    )
            elif field in {"W"}:
                has_wumpus = sat_conclusions[x * game_height + y]
                if has_wumpus in {"1"}:
                    has_wumpus = "W"
                elif has_wumpus in {"0"}:
                    has_wumpus = " "
                cell = Cell(x, y, 
                                    has_hunter=cartesian_matrix[x][y].has_hunter,
                                    has_wumpus=has_wumpus,
                                    has_stence=cartesian_matrix[x][y].has_stence, 
                                    has_pit=cartesian_matrix[x][y].has_pit,
                                    has_breeze=cartesian_matrix[x][y].has_breeze
                                    )
            
            column.append(cell)
        result_cartesian_matrix.append(column)
    return result_cartesian_matrix

def update(*, json_object, with_result_cartesian_matrix):
    cartesian_matrix = with_result_cartesian_matrix
    wumpuses = []
    no_wumpuses = []
    stences = []
    no_stences = []
    pits = []
    no_pits = []
    breezes = []
    no_breezes = []

    for x in range(len(cartesian_matrix)):
        for y in range(len(cartesian_matrix[0])):
            cell = cartesian_matrix[x][y]
            if cell.has_wumpus in {"W"}:
                wumpuses.append([x, y])
            elif cell.has_wumpus in {" "}:
                no_wumpuses.append([x, y])
            if cell.has_stence in {"S"}:
                stences.append([x, y])
            elif cell.has_stence in {" "}:
                no_stences.append([x, y])
            if cell.has_pit in {"P"}:
                pits.append([x, y])
            elif cell.has_pit in {" "}:
                no_pits.append([x, y])
            if cell.has_breeze in {"B"}:
                breezes.append([x, y])
            elif cell.has_breeze in {" "}:
                no_breezes.append([x, y])
                
    json_object["wumpuses"] = wumpuses
    json_object["noWumpuses"] = no_wumpuses
    json_object["stences"] = stences
    json_object["noStences"] = no_stences
    json_object["pits"] = pits
    json_object["noPits"] = no_pits
    json_object["breezes"] = breezes
    json_object["noBreezes"] = no_breezes
    return json_object

def get_index_c_xy_list(*, prefix, game_width, game_height):
    """
    Generates all variables and indexes for Z3 solver
    """
    c_xy_list = [Int(f"{prefix}_{i}{j}") for i in range(game_width) for j in range(game_height)]
    index_list = [(i, j) for i in range(game_width) for j in range(game_height)]
    return list(zip(index_list, c_xy_list))

def add_value_range_clauses(*, to_solver, for_index_c_xy_list, with_min_value, with_max_value):
    """
    defines a max and a min value for each cell c_xy of the hunt the wumpus and adds them to the solver
    """
    solver = to_solver
    index_c_xy_list = for_index_c_xy_list
    min_value = with_min_value
    max_value = with_max_value

    for index_c_xy in index_c_xy_list:
        solver.add(index_c_xy[1] >= min_value)
        solver.add(index_c_xy[1] <= max_value)
    return
    
def add_sum_clause(*, to_solver, for_index_c_xy_list, with_overall_sum):
    """
    adds a clause that the total sum of for_index_c_xy_listin is with_total_value
    """
    solver = to_solver
    index_c_xy_list = for_index_c_xy_list
    overall_sum = with_overall_sum

    if overall_sum.isnumeric():
        solver.add(Sum(list(map(lambda x: x[1], index_c_xy_list))) == int(overall_sum))

def add_property_clauses(*, to_solver, for_index_c_xy_list, when_cartesian_matrix, has_value1, with_value1, has_value2, with_value2):
    """
    for each p_xy check if there has_value a pit or breeze and add the assertions to solver
    """
    solver = to_solver
    index_c_xy_list = for_index_c_xy_list
    cartesian_matrix = when_cartesian_matrix
    value1 = (has_value1, with_value1)
    value2 = (has_value2, with_value2)

    for index_c_xy in index_c_xy_list:
        cell = cartesian_matrix[index_c_xy[0][0]][index_c_xy[0][1]] #take care of different indexing for matrix (top to bottom) and game indexing (x,y) -cartesian
        if value1[0] in {"P"}:
            if cell.has_pit in {value1[0]}:
                solver.add(index_c_xy[1] == value1[1])
            elif cell.has_pit in {value2[0]}:
                solver.add(index_c_xy[1] == value2[1])
        elif value1[0] in {"W"}:
            if cell.has_wumpus in {value1[0]}:
                solver.add(index_c_xy[1] == value1[1])
            elif cell.has_wumpus in {value2[0]}:
                solver.add(index_c_xy[1] == value2[1])

def add_sum_of_neighbour_clauses(*, to_solver, for_index_c_xy_list, when_cartesian_matrix, has_value1, with_value1, has_value2, with_value2):
    solver = to_solver
    index_c_xy_list = for_index_c_xy_list
    cartesian_matrix = when_cartesian_matrix
    value1 = (has_value1, with_value1)
    value2 = (has_value2, with_value2)

    for index_c_xy in index_c_xy_list:
        cell = cartesian_matrix[index_c_xy[0][0]][index_c_xy[0][1]]
        if value1[0] in {"B"}:
            if cell.has_breeze not in {"?"}:
                neighbour_indexes = get_neighbour_indexes(index_c_xy[0], (game_width, game_height))
                neighbour_vars = list(map(lambda x: x[1], filter(lambda x: x[0] in neighbour_indexes, index_c_xy_list)))
                if cell.has_breeze in {value1[0]}:
                    solver.add(Sum(neighbour_vars) >= value1[1])
                    solver.add(Sum(neighbour_vars) <= len(neighbour_vars) * value1[1])
                elif cell.has_breeze in {value2[0]}:
                    solver.add(Sum(neighbour_vars) == value2[1])
        elif value1[0] in {"S"}:
            if cell.has_stence not in {"?"}:
                neighbour_indexes = get_neighbour_indexes(index_c_xy[0], (game_width, game_height))
                neighbour_vars = list(map(lambda x: x[1], filter(lambda x: x[0] in neighbour_indexes, index_c_xy_list)))
                if cell.has_stence in {value1[0]}:
                    solver.add(Sum(neighbour_vars) >= value1[1])
                    solver.add(Sum(neighbour_vars) <= len(neighbour_vars) * value1[1])
                elif cell.has_stence in {value2[0]}:
                    solver.add(Sum(neighbour_vars) == value2[1])

# read configuration from file
with open(sys.argv[1]) as json_file:
    world_info = json.load(json_file)

# get all parameters from json
game_width = world_info["size"][0]
game_height = world_info["size"][1]
number_of_pits = world_info["numberOfPits"]
number_of_wumpuses = world_info["numberOfWumpuses"]

world_cartesian_matrix = get_cartesian_matrix(with_objects=world_info)

pit_solver = Solver()
index_p_xy_list = get_index_c_xy_list(prefix="p", game_width=game_width, game_height=game_height)
add_value_range_clauses(to_solver=pit_solver, for_index_c_xy_list=index_p_xy_list, with_min_value=0, with_max_value=1)
add_sum_clause(to_solver=pit_solver, for_index_c_xy_list=index_p_xy_list, with_overall_sum=number_of_pits)
add_property_clauses(to_solver=pit_solver, for_index_c_xy_list=index_p_xy_list, when_cartesian_matrix=world_cartesian_matrix, has_value1="P", with_value1=1, has_value2=" ", with_value2=0)
add_sum_of_neighbour_clauses(to_solver=pit_solver, for_index_c_xy_list=index_p_xy_list, when_cartesian_matrix=world_cartesian_matrix, has_value1="B", with_value1=1, has_value2=" ", with_value2=0 )
pit_models = get_all_models(with_solver=pit_solver, for_c_xy_list=list(map(lambda x: x[1], index_p_xy_list)))

if not pit_models:
    print("The game is not deterministic. There is no model for pits.")
    sys.exit(0)

pit_sat_conclusions = get_conclusions(from_sat_models=pit_models)
pit_result_cartesian_matrix = get_updated_cartesian_matrix(from_cartesian_matrix=world_cartesian_matrix, with_sat_conclusions=pit_sat_conclusions, on_field="P")

wumpus_solver = Solver()
index_w_xy_list = get_index_c_xy_list(prefix="w", game_width=game_width, game_height=game_height)
add_value_range_clauses(to_solver=wumpus_solver, for_index_c_xy_list=index_w_xy_list, with_min_value=0, with_max_value=1)
add_sum_clause(to_solver=wumpus_solver, for_index_c_xy_list=index_w_xy_list, with_overall_sum=number_of_wumpuses)
add_property_clauses(to_solver=wumpus_solver, for_index_c_xy_list=index_w_xy_list, when_cartesian_matrix=world_cartesian_matrix, has_value1="W", with_value1=1, has_value2=" ", with_value2=0)
add_sum_of_neighbour_clauses(to_solver=wumpus_solver, for_index_c_xy_list=index_w_xy_list, when_cartesian_matrix=world_cartesian_matrix, has_value1="S", with_value1=1, has_value2=" ", with_value2=0 )
wumpus_models = get_all_models(with_solver=wumpus_solver, for_c_xy_list=list(map(lambda x: x[1], index_w_xy_list)))

if not wumpus_models:
    print("The game is not deterministic. There is no model for wumpus.")
    sys.exit(0)

wumpus_sat_conclusions = get_conclusions(from_sat_models=wumpus_models)

result_cartesian_matrix = get_updated_cartesian_matrix(from_cartesian_matrix=pit_result_cartesian_matrix, with_sat_conclusions=wumpus_sat_conclusions, on_field="W")
updated_json = update(json_object=world_info, with_result_cartesian_matrix=result_cartesian_matrix)

#write to json-file
with open(sys.argv[2], "w") as file_json:
    json.dump(updated_json, file_json)

# write to terminal
print("Input:", end="\n\n")
if "numberOfPits" in world_info:
    print(f"Total number of pits: {world_info['numberOfPits']}")
else:
    print(f"Total number of pits: ?")
if "numberOfWumpuses" in world_info:
    print(f"Total number of wumpuses: {world_info['numberOfWumpuses']}\n")
else:
    print(f"Total number of wumpuses: ?\n")
print("Input matrix:", end="\n\n")
print_pretty_formatted_cartesian_matrix(matrix=world_cartesian_matrix)
print("\nResulting matrix:", end="\n\n")
print_pretty_formatted_cartesian_matrix(matrix=result_cartesian_matrix)

print("\n")
print(f"Total number of pit models: {len(pit_models)}")
print(f"Total number of wumpus models: {len(wumpus_models)}")


def zz3_to_string(x):
    if x == 0: 
        return "0" 
    elif x == 1:
        return "1" 
    else:
        return "?"

if "printModels" in world_info and world_info["printModels"] == True:
    print("\n*****************************************************")
    print(f"Total number of pit models: {len(pit_models)}")
    for key, value in sorted(Counter([sum(map(lambda x: x.as_long(), pit_model)) for pit_model in pit_models]).items()):
        print(f"Number of models with {key} pits: {value}")
    for index, pit_model in enumerate(pit_models):
        print(f"\nPit model n. {index+1}:\nNumber of pits: {sum(map(lambda x: x.as_long(), pit_model))}")
        print_pretty_formatted_cartesian_matrix(get_updated_cartesian_matrix(from_cartesian_matrix=world_cartesian_matrix, with_sat_conclusions=list(map(lambda x: zz3_to_string(x), pit_model)), on_field="P"), for_fields={"P","B"})
    print("\n*****************************************************")
    print(f"Total number of wumpus models: {len(wumpus_models)}")
    for key, value in sorted(Counter([sum(map(lambda x: x.as_long(), wumpus_model)) for wumpus_model in wumpus_models]).items()):
        print(f"Number of models with {key} wumpus: {value}")
    for index, wumpus_model in enumerate(wumpus_models):
        print(f"\nWumpus Model n. {index+1}:\nNumber of wumpusses: {sum(map(lambda x: x.as_long(), wumpus_model))}")
        print_pretty_formatted_cartesian_matrix(get_updated_cartesian_matrix(from_cartesian_matrix=world_cartesian_matrix, with_sat_conclusions=list(map(lambda x: zz3_to_string(x), wumpus_model)), on_field="W"), for_fields={"W","S"})