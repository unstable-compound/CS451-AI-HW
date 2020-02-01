

from sys import argv
import sys

'''
Abstract formulation of problem instance.

• States: A state description specifies the location of each of the eight tiles and the blank
in one of the nine squares.
• Initial state: Any state can be designated as the initial state. Note that any given goal
can be reached from exactly half of the possible initial states (Exercise 3.4).
• Actions: The simplest formulation defines the actions as movements of the blank space
Left, Right, Up, or Down. Different subsets of these are possible depending on where
the blank is.
• Transition model: Given a state and action, this returns the resulting state; for example,
if we apply Left to the start state in Figure 3.4, the resulting state has the 5 and the blank
switched.
• Goal test: This checks whether the state matches the goal configuration shown in Figure
3.4. (Other goal configurations are possible.)
• Path cost: Each step costs 1, so the path cost is the number of steps in the path.
'''

allowable_moves_map = {
    0: ('D', 'R'),
    1: ('D', 'L', 'R'),
    2: ('D', 'L'),
    3: ('U', 'D', 'R'),
    4: ('U', 'D', 'L', 'R'),
    5: ('U', 'D', 'R'),
    6: ('U', 'R'),
    7: ('U', 'L', 'R'),
    8: ('U', 'L')
}

move_map = {
    'D': 3,
    'U': -3,
    'L': -1,
    'R': 1
}


def step_cost(state, action):
    return 1;
    # I think this might be the depth in this case


def transition_model(state, action, index):
    index_to_swap = index + move_map[action]
    new_state = list(state)
    new_state[index] = state[index_to_swap]
    new_state[index_to_swap] = state[index]
    return new_state


class Problem:
    '''
    THis func needs to be re-examined: VV
    I dont think Problem.current_index is a valid thing cause it never gets updated.
    '''

    def get_blank_space_index(self, state):
        current = 0
        for i in range(0, 9):
            if state[i] == 'b':
                current = i
        return current

    def __init__(self, initial_state, goal_state):
        self.path_cost = 0
        self.initial_state = initial_state
        self.current_state = initial_state
        self.goal = goal_state

    def actions(self, state):
        blank_index = self.get_blank_space_index(state)
        allowed_moves = allowable_moves_map[blank_index]
        return allowed_moves

    def goal_test(self, state):
        for i in range(0, len(state)):
            if state[i] is not self.goal[i]:
                return False
        return True

    def RESULT(self, state, action):
        index = self.get_blank_space_index(state)
        return transition_model(state, action, index)


def check_parity(s, g):
    # replace 'b' with 9
    start = list(s)
    goal = list(g)

    inversions_start = 0
    inversions_goal = 0
    for i in range(0, 9):
        x = start[i]
        if x is not 'b':
            for j in range(i + 1, 9):
                if start[j] is 'b':
                    continue
                if x > start[j]:
                    inversions_start += 1
        y = goal[i]
        if y is not 'b':
            for j in range(i + 1, 9):
                if goal[j] is 'b':
                    continue
                if y > goal[j]:
                    inversions_goal += 1

    if inversions_start % 2 == inversions_goal % 2:
        return True
    return False


def getCurrent(openSet, fScore):
    lowest = p_infinity
    retval = None
    for state in openSet:
        f = fScore[state]
        if f < lowest:
            lowest = f
            retval = state
    return retval


def reconstruct_path(cameFrom, current):
    path_list = list()
    path_list.append(current)
    while current in cameFrom.keys():
        to_add = cameFrom[current]
        path_list.append(to_add)
        current = to_add
    return path_list


def best_first(problem, hueristic):
    openSet = set()  # frontier
    closedSet = set()  # explored
    cameFrom = dict()  # path

    fScore = dict()

    start = tuple(problem.initial_state)
    goal = tuple(problem.goal)
    openSet.add(start)

    fScore[start] = hueristic(start, goal)

    while len(openSet) != 0:
        current = getCurrent(openSet, fScore)
        if problem.goal_test(current):
            solution = reconstruct_path(cameFrom, current)
            solution = solution[::-1]
            return solution

        openSet.remove(current)
        closedSet.add(current)

        """*****************"""
        for action in problem.actions(current):
            child_state = tuple(problem.RESULT(current, action))
            f_keys = fScore.keys()
            if child_state not in f_keys:  # fscore are supposed to be default infinity
                fScore[child_state] = p_infinity

            if child_state not in closedSet and child_state not in openSet:  # discovered a new state
                openSet.add(child_state)
            if child_state in closedSet:
                continue

            ## This path is best so far, so record it
            cameFrom[child_state] = current

            fScore[child_state] = hueristic(child_state, goal)

    # if end while loop, then openset is empty and we have no solution
    return None


def a_star(problem, hueristic):
    openSet = set()  # frontier
    closedSet = set()  # explored
    cameFrom = dict()  # path

    gScore = dict()
    fScore = dict()

    start = tuple(problem.initial_state)
    goal = tuple(problem.goal)
    openSet.add(start)

    gScore[start] = 0
    fScore[start] = hueristic(start, goal)

    while len(openSet) != 0:
        current = getCurrent(openSet, fScore)
        if problem.goal_test(current):
            solution = reconstruct_path(cameFrom, current)
            solution = solution[::-1]
            return solution

        openSet.remove(current)
        closedSet.add(current)

        """*****************"""
        for action in problem.actions(current):
            child_state = tuple(problem.RESULT(current, action))
            f_keys = fScore.keys()
            g_keys = gScore.keys()
            if child_state not in f_keys:  # gscore and fscore are supposed to be default infinity
                fScore[child_state] = p_infinity
            if child_state not in g_keys:
                gScore[child_state] = p_infinity

            if child_state not in closedSet and child_state not in openSet:  # discovered a new state
                openSet.add(child_state)
            if child_state in closedSet:
                continue

            tent_g_score = gScore[current] + 1  # is 1 for all children cause the distance is same for each move
            if tent_g_score >= gScore[child_state]:
                continue  # this is not a better path

            ## This path is best so far, so record it
            cameFrom[child_state] = current
            gScore[child_state] = tent_g_score
            fScore[child_state] = tent_g_score + hueristic(child_state, goal)

    # if end while loop, then openset is empty and we have no solution
    return None


def puzzle_solve(initial_state_array, goal_state_array, search_algorithm, hueristic):
    if not check_parity(initial_state_array, goal_state_array):
        print("This puzzle is not solvable")
        return 0
    problem = Problem(initial_state_array, goal_state_array)
    solution = search_algorithm(problem, hueristic)
    if solution is None:
        return 0
    print("The solution, beginning with initial state and progressing to")
    print("the final (goal) state in order of moves taken is:")

    retval = -1
    for item in solution:  # (state, action)
        item_state = item
        sys.stdout.write(str(item_state))
        sys.stdout.write("-->")
        retval += 1
        if retval % 4 == 0:
            print("\n")

    return retval

    '''
    function TREE-SEARCH(problem) returns a solution, or failure
        initialize the frontier using the initial state of problem
        loop do
        if the frontier is empty then return failure
            choose a leaf node and remove it from the frontier
        if the node contains a goal state then return the corresponding solution
            expand the chosen node, adding the resulting nodes to the frontier
    '''
    # for greedy best first search, the frontier should be a priority queue sorted by lowest cost estimate
    # according to the evaluation function.


def number_misplaced_nodes(state, goal_state):
    counter = 0
    for i in range(0, 9):
        if state[i] != goal_state[i]:
            counter += 1
    return counter


def make_child_node(problem, parent, action):
    new_state = problem.RESULT(parent.state, action)
    path_cost = parent.cost + 1
    return SearchNode(new_state, parent, action, path_cost, parent.depth + 1)


'''
solutions() should return a set of states and their moves, beginning with the initial state
and progressing to the final (goal) state
'''


def solutions(node):
    moves_list = list()
    node_to_add = node
    while node_to_add is not None:
        moves = (node_to_add.state, node_to_add.action)
        moves_list.insert(0, moves)  # push

        node_to_add = node_to_add.parent
    return moves_list


p_infinity = float('Inf')


def get_current(known_items, fScore):  # known_items
    lowest = p_infinity
    retval = None
    for item in known_items:
        fscore = fScore[tuple(item.state)]
        if fscore < lowest:
            lowest = fscore
            retval = item  # the node
    return retval


# a node of the search tree needs to have at least:
# 1. STATE -> the state that this node represents in the
#       state space.
# 2. PARENT -> the node in the search tree that generated
#       this node.
# 3. ACTION -> the action that was applied to the parent
#       in order to generate this node
# 4. PATH-COST -> the cost, traditionally denoted g(n)
#       of the path from the initial state to the node,
#       as indicated by the parent "pointers"


class SearchNode:
    def __init__(self, state, parent, action, path_cost, depth):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = path_cost
        self.depth = depth


def h_custom(state, goal):
    return man(state, goal) - 1


def man(state, goal_s):
    goal = list(goal_s)
    board = list(state)
    for i in range(0, 9):
        if goal[i] == 'b':
            goal[i] = 0
        if board[i] == 'b':
            board[i] = 0
    return sum(abs(b % 3 - g % 3) + abs(b // 3 - g // 3)
               for b, g in ((board.index(i), goal.index(i)) for i in range(1, 9)))


test_puzzles = [
    # [1, 2, 3, 4, 5, 6, 7, 'b', 8],
    [1, 2, 3, 4, 5, 6, 'b', 7, 8],
    [1, 2, 3, 4, 5, 'b', 6, 7, 8],
    [1, 2, 3, 4, 5, 'b', 7, 8, 6],
    [3, 4, 1, 5, 2, 'b', 8, 7, 6],
    [3, 2, 1, 4, 5, 8, 7, 'b', 6],
    # [8, 4, 3, 5, 1, 2, 6, 7, 'b']

]

if __name__ == '__main__':
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 'b']

    for start_state in test_puzzles:
        print("Puzzle to solve:")
        print("Start State:" + str(start_state))
        print("Goal State:" + str(goal_state))

        print("\nHere is the Best First Search Path:")
        num_moves = puzzle_solve(start_state, goal_state, best_first, number_misplaced_nodes)
        print("\nNumber of moves in path: " + str(num_moves) + "\n")
        print("\nHere is the A* Search Path:")
        num_moves = puzzle_solve(start_state, goal_state, a_star, number_misplaced_nodes)
        print("\nNumber of moves in path: " + str(num_moves) + "\n")
        print("\n********************************************************\n")

    # run greedy_best_first with num_misplaced_tiles_hueristic

"""
def a_star_search(problem, hueristic):
    map_states_to_nodes = dict()
    # Set of nodes already evaluated
    closedSet = set()  # closedSet
    evaluated_states = set()
    def_map_val = p_infinity
    # set of known nodes which have yet to be evaluated:
    openSet = set()  # openSet
    known_states = set()
    goal = problem.goal
    # node ←a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    node = SearchNode(problem.initial_state, None, None, 0, 0)
    map_states_to_nodes[tuple(node.state)] = node
    openSet.add(node)
    known_states.add(tuple(node.state))

    came_from = dict()

    # initialize this map where each vertex has a cost associated with it
    # specifically the cost of getting from start to the key vertex.
    # it should have a default value of infinity for each vertex.
    gScore = dict()

    # The cost of going from start to start is 0
    gScore[tuple(node.state)] = 0

    # for each node the total cost of getting from start to the goal by passing by that node.
    # That value is partly known and partly heuristic.
    # Represented by map with default value of infinity
    fScore = dict()

    # For the first node, (start), that value is completely heuristic
    fScore[tuple(node.state)] = hueristic(node.state, goal_state)

    while len(openSet) != 0:
        current = get_current(openSet, fScore)
        if current is None:
            print("Failure to find valid node in known set")
            return None
        if problem.goal_test(current.state):
            return solutions(current)
        openSet.remove(current)  # openSet
        known_states.remove(tuple(current.state))
        closedSet.add(current)  # closedSet
        evaluated_states.add(tuple(current.state))

        # grab neighbors
        for action in problem.actions(node.state):
            # check if we need new node or have the node already
            new_state = problem.RESULT(node.state, action)
            if tuple(new_state) in map_states_to_nodes.keys():
                child = map_states_to_nodes[tuple(new_state)]
            else:
                # generate child node
                child = make_child_node(problem, node, action)
                f_keys = fScore.keys()
                g_keys = gScore.keys()
                if tuple(child.state) not in f_keys:  # gscore and fscore are supposed to be default infinity
                    fScore[tuple(child.state)] = p_infinity
                if tuple(child.state) not in g_keys:
                    gScore[tuple(child.state)] = p_infinity
                # add to state node map
                map_states_to_nodes[tuple(child.state)] = child

            if tuple(child.state) in evaluated_states:
                continue
                # this node was already evaluated
            if tuple(child.state) not in known_states:  # new node discovered
                openSet.add(child)
                known_states.add(tuple(child.state))
                # set default g and f scores
            tentative_gscore = gScore[tuple(current.state)] + 1
            if tentative_gscore >= gScore[tuple(child.state)]:
                continue

            # so far this is best path so record it
            came_from[child] = current
            gScore[tuple(child.state)] = tentative_gscore
            fScore[tuple(child.state)] = gScore[tuple(child.state)] + hueristic(child.state, goal_state)

    # if end while loop then fail
    return None


def greedy_best_first_search(problem, hueristic):
    goal = problem.goal
    # node ←a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    node = SearchNode(problem.initial_state, None, None, 0, 0)

    # initialize the frontier using the initial state of problem
    # frontier is a prio-queue sorted by lowest cost_value, and store a tuple:
    # (prio, node)
    priority = hueristic(problem.initial_state, goal)
    frontier = PriorityQueue()
    frontier.put((priority, node))

    # make path of nodes:

    # set up explored set to avoid circles and what not
    explored = list()

    while not frontier.empty():

        # difference between node state and this tuple dog!!!!

        tup = frontier.get()  # get lowest cost state
        node = tup[1]
        if problem.goal_test(node.state):
            return solutions(node)
        explored.append(node)

        for action in problem.actions(node.state):
            child = make_child_node(problem, node, action)
            if child not in frontier.queue and child not in explored:
                priority = hueristic(child.state, goal)
                tup = (priority, child)
                frontier.put(tup)
            elif child.state in frontier:
                priority_one = hueristic(child.state, goal)
                for item in frontier:
                    if item[1] is child.state:
                        if item[0] > priority_one:
                            item[0] = priority_one

    # if we exited loop with no solution
    return None
"""

"""
def a_star(problem, hueristic):
    openSet = set()
    closedSet = set()
    evaluated_states = set()

    gScore = dict()
    fScore = dict()

    goal = problem.goal
    # node ←a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    node = SearchNode(problem.initial_state, None, None, 0, 0)
    openSet.add(node)
    gScore[tuple(node.state)] = 0
    fScore[tuple(node.state)] = hueristic(node.state, goal)

    cameFrom = dict()

    while len(openSet) != 0:
        current = get_current(openSet, fScore)
        if current is None:
            print("Failure to find valid node in known set")
            return None
        if problem.goal_test(current.state):
            return solutions(current)
        openSet.remove(current)
        closedSet.add(current)
        evaluated_states.add(tuple(current.state))
        ## generate successors
        for action in problem.actions(current.state):

            child: SearchNode = make_child_node(problem, current, action)
            f_keys = fScore.keys()
            g_keys = gScore.keys()
            if tuple(child.state) not in f_keys:  # gscore and fscore are supposed to be default infinity
                fScore[tuple(child.state)] = p_infinity
            if tuple(child.state) not in g_keys:
                gScore[tuple(child.state)] = p_infinity

            if tuple(child.state) in evaluated_states:
                # then this node was already evaluated so leave it alone.
                continue
            if child not in openSet:
                # new node discovered
                openSet.add(child)
            tent_g_score = gScore[tuple(current.state)] + 1
            if tent_g_score < gScore[tuple(child.state)]:
                # then this is best path so record it
                cameFrom[child] = current
                gScore[tuple(child.state)] = tent_g_score
                fScore[tuple(child.state)] = tent_g_score + hueristic(current.state, child.state)

    return None
"""
