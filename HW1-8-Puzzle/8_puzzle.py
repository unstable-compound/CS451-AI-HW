from queue import PriorityQueue

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

class Problem:
    def get_blank_space_index(self):
        current = 0
        for i in range (0,9):
            if self.current_state[i] == 'b':
               current = i
        return current


    def __init__(self, initial_state, goal_state):
        self.path_cost = 0
        self.initial_state = initial_state
        self.current_state = initial_state
        self.goal = goal_state
        self.allowable_moves_map = {
            0:  ('D', 'R'),
            1:  ('D', 'L', 'R'),
            2:  ('D', 'L'),
            3:  ('U', 'D', 'R'),
            4:  ('U', 'D', 'L', 'R'),
            5:  ('U', 'D', 'R'),
            6:  ('U', 'R'),
            7:  ('U', 'L', 'R'),
            8:  ('U', 'L')
        }
        self.move_map = {
            'D': 3,
            'U': -3,
            'L': -1,
            'R': 1
        }
    def actions(self, state, ): ## TODO: Left off Here
    def goal_test(self, state):
        for i in range(0, len(state)):
            if state[i] is not self.goal[i]:
                return False
        return True


    def step_cost(self, state, action):
        # TODO Figure out how to define this BS


    def RESULT(self, state, action):
        index = self.get_blank_space_index()
        return self.transition_model(state, action, index)

    def transition_model(self, state, action, index):
        index_to_swap = index + self.move_map(action)
        new_state = list(state)
        new_state[index] = state[index_to_swap]
        new_state[index_to_swap] = state[index]
        return new_state



def puzzle_solve(initial_state_array, goal_state_array):
    problem = Problem(initial_state_array, goal_state_array)


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
    #for greedy best first search, the frontier should be a priority queue sorted by lowest cost estimate
    #according to the evaluation function.



def make_child_node(problem, parent, action):
    new_state = problem.RESULT(parent.state, action)
    path_cost = parent.cost + problem.step_cost(parent.state, action) #TODO implement stepcost
    return SearchNode(new_state, parent, action, path_cost)





def greedy_best_first_search(problem):
    goal_state = problem.goal
    # node ←a node with STATE = problem.INITIAL-STATE, PATH-COST = 0
    node = SearchNode(problem.initial_state, None, None, 0, 0)

    # initialize the frontier using the initial state of problem
    # frontier is a prio-queue sorted by lowest cost_value, and store a tuple:
    # (prio, state)
    priority = evaluation_function(problem.initial_state, goal_state)
    frontier = PriorityQueue()
    frontier.put((priority, problem.initial_state))

    # set up explored set to avoid circles and what not
    explored = set()


    while not frontier.empty():
        node = frontier.get() #get lowest cost node
        if problem.goal_test(node.state)
            return solutions(node)
        explored.add(node.state)

        for action in problem.actions(node.state):
            child = make_child_node(problem, node, action)
            if child.state is not in frontier and child.state is not in explored:
                priority = evaluation_function(child.state, goal_state)
                frontier.put((priority, child.state))
            elif child.state is in frontier:
                priority_one = evaluation_function(child.state, goal_state)
                for item in frontier:
                    if item[1] is child.state:
                        if item[0] > priority_one:
                            item[0] = priority_one

    #if we exited loop with no solution
    return None





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
            0:  ('D', 'R'),
            1:  ('D', 'L', 'R'),
            2:  ('D', 'L'),
            3:  ('U', 'D', 'R'),
            4:  ('U', 'D', 'L', 'R'),
            5:  ('U', 'D', 'R'),
            6:  ('U', 'R'),
            7:  ('U', 'L', 'R'),
            8:  ('U', 'L')
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
