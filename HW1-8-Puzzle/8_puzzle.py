import itertools
from queue import PriorityQueue
from sys import argv

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
            if self.current_state[i] == 'b':
                current = i
        return current

    def __init__(self, initial_state, goal_state):
        self.path_cost = 0
        self.initial_state = initial_state
        self.current_state = initial_state
        self.goal = goal_state

    def actions(self, state):
        blank_index = self.get_blank_space_index(state)
        return allowable_moves_map[blank_index]

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
    for i in range(0, 9):
        if start[i] == 'b':
            start[i] = 9
        if goal[i] == 'b':
            goal[i] = 9

    inversions_start = 0
    inversions_goal = 0
    for i in range(0, 9):
        x = start[i]
        for j in range(i + 1, 9):
            if x > start[j]:
                inversions_start += 1
        y = goal[i]
        for j in range(i + 1, 9):
            if y > goal[j]:
                inversions_goal += 1

    if inversions_start % 2 == inversions_goal % 2:
        return True
    return False


def puzzle_solve(initial_state_array, goal_state_array, search_algorithm, hueristic):
    if not check_parity(initial_state_array, goal_state_array):
        print("This puzzle is not solvable")
    problem = Problem(initial_state_array, goal_state_array)
    solution = search_algorithm(problem, hueristic)
    if solution is None:
        return
    print("The solution, beginning with initial state and progressing to")
    print("the final (goal) state in order of moves taken is:")
    for item in solution:  # (state, action)
        print("\nThe blank space move to the next state is:")
        print(item[1])
        print("State: \t")
        row_one = item[0][0:3]
        row_two = item[0][3:6]
        row_three = item[0][6:9]
        print(row_one)
        print(row_two)
        print(row_three)

    # TODO: Finish

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
    path_cost = parent.cost + step_cost(parent.state, action)  # TODO implement stepcost
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


def a_star_search(problem, hueristic):
    # Set of nodes already evaluated
    visited_items = set()  # closedSet
    def_map_val = p_infinity
    # set of known nodes which have yet to be evaluated:
    known_items = set()  # openSet
    known_states = set()
    # known_items is set of tuples:

    goal = problem.goal
    # node ←a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    node = SearchNode(problem.initial_state, None, None, 0, 0)
    known_items.add(node)
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

    while len(known_items) != 0:
        current = get_current(known_items, fScore)
        if current is None:
            print("Failure to find valid node in known set")
            return None
        if problem.goal_test(current.state):
            return solutions(current)
        known_items.remove(current)  # openSet
        known_states.remove(tuple(current.state))
        visited_items.add(current)  # closedSet

        # grab neighbors
        for action in problem.actions(node.state):
            child = make_child_node(problem, node, action)
            for item in visited_items:
                if item.state == child.state:
                    continue
            if tuple(child.state) not in known_states:  # new node discovered
                known_items.add(child)
                known_states.add(tuple(child.state))
                # set default g and f scores
                fScore[tuple(child.state)] = p_infinity
                gScore[tuple(child.state)] = p_infinity

            tentative_gscore = gScore[tuple(current.state)] + (child.depth - current.depth)
            if tentative_gscore < gScore[tuple(child.state)]:
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


class PuzzleSolver:
    def __init__(self, puzzle_array):
        self.goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 'b']
        self.puzzle_start = puzzle_array
        # allowable_moves_map maps each index of the list to its corresponding
        # allowable moves
        # INDEX:    TUPLE of ALLOWABLE DIRECTIONS
        self.allowable_moves_map = {
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
        # move_map maps the moves (Left, Right, Up, Down) to the corresponding
        # action that needs to be taken. [-3 means move the blank space by -3 in the list]
        self.move_map = {
            'D': +3,
            'U': -3,
            'L': -1,
            'R': +1
        }

        '''
        Greedy-best-first-algorithm:  Modify for tree search.
        
        
        function UNIFORM-COST-SEARCH(problem) returns a solution, or failure
        node ←a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
        frontier ←a priority queue ordered by f(n), with node as the only element
        
        loop do
        if EMPTY?( frontier) then return failure
        node←POP( frontier ) /* chooses the lowest-cost node in frontier */
        if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
        
        for each action in problem.ACTIONS(node.STATE) do
        child ←CHILD-NODE(problem, node, action)
        if child .STATE is not in explored or frontier then
        frontier ←INSERT(child , frontier )
        else if child .STATE is in frontier with higher PATH-COST then
        replace that frontier node with child 
        
        
        '''


if __name__ == '__main__':
    if len(argv) > 1:
        start_state = argv[1]
    else:
        start_state = [1, 3, 'b', 4, 2, 5, 7, 8, 6]
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 'b']
    puzzle_solve(start_state, goal_state, a_star_search, number_misplaced_nodes)

    # run greedy_best_first with num_misplaced_tiles_hueristic
