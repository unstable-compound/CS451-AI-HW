
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
