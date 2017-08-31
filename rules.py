
import pdb
from valid_rules import VALID_RULES
import numpy as np
from collections import namedtuple, defaultdict, deque
from itertools import product
import random
import logging
import sys

ADJACENCY_OPT=["T","C","G","A","r","g","b","y"]
ATTRACT_OPT=["T","C","G","A","r","g","b","y"]
ATTRIBUTION_OPT=["rT", "gT", "bT", "yT", "rC", "gC", "bC", "yC", "rG", "gG", "bG", "yG", "rA", "gA", "bA", "yA"]
EQUILIBRIUM_OPT=["T","C","G","A","r","g","b","y"]
PRIORITY_OPT=["rT", "gT", "bT", "yT", "rC", "gC", "bC", "yC", "rG", "gG", "bG", "yG", "rA", "gA", "bA", "yA"]
REPULSION_OPT=["TC", "TG", "TA", "Te", "CG", "CA", "Ce", "GA", "Ge", "Ae"]
UNIFORM_OPT=["T", "C", "G", "A", "r", "g", "b", "y"]
ZONING_OPT=["TL", "TM", "TR", "ML", "MR", "LL", "LM", "LR", "COL0", "COL1", "COL2", "COL3", "ROW0", "ROW1", "ROW2", "ROW3"]


class GameState:
    """ --------------------------------------------------
    The rule states are:
        adjacency, attraction, attribution, equilibrium, priority, repulsion, uniformity, zoning
    These variables will be set to one of the corresponding <RULE>_OPT strings
    There are 16 letter/color combinations that can be placed on the 4x4 game board.
    The "board" variable is a 4x4 np.array that contains a string indicating the state of a square
    Squares can be either <color><letter> or 'e' which indicates empty.
        Examples: rT, yA, e
    """
    def __init__(self, adjacency = "", attraction = "", attribution = "", equilibrium = "", priority = "",
                 repulsion = "", uniformity = "", zoning = ""):
        self.adjacency = ""
        self.attraction = ""
        self.attribution = ""
        self.equilibrium = ""
        self.priority = ""
        self.repulsion = ""
        self.uniformity = ""
        self.zoning = ""
        self.board=np.chararray((4,4),itemsize=2)
        self.board[:]='e'

    def initialize_rules(self, adjacency, attraction, attribution, equilibrium, priority, repulsion,
                         uniformity, zoning):
        self.adjacency   = adjacency
        self.attraction  = attraction
        self.attribution = attribution
        self.equilibrium = equilibrium
        self.priority    = priority
        self.repulsion   = repulsion
        self.uniformity  = uniformity
        self.zoning      = zoning

def get_adjacent_squares(board, square):
    vals=[]
    left = get_left(board, square)
    right = get_right(board, square)
    up = get_up(board, square)
    down = get_down(board, square)
    vals.append(left) if left else None
    vals.append(right) if right else None
    vals.append(up) if up else None
    vals.append(down) if down else None
    return vals

def get_val(board,loc):
    return board[loc[0],loc[1]].decode('ascii')

def check_single_adjacency(board, square, adjacency_value):
    """ --------------------------------------------------
    All adjacency_value must be next to eachother, minimum of 2
    adjacency_value can be one of: [T,C,G,A,r,g,b,y]
    square is a 2d array indicating (row,col)
    """
    val = get_val(board, square)
    if adjacency_value not in val:
        return True

    for a_sq in get_adjacent_squares(board, square):
        if adjacency_value in a_sq:
            return True
    return False

def check_single_attraction(board, square, attraction_val):
    """ --------------------------------------------------
    All attraction_val must have 3 non-empty squares next to it, minimum 1
    attraction_val can be one of [T,C,G,A,r,g,b,y]
    square is a 2d array indicating (row,col)
    """
    val = get_val(board, square)
    if attraction_val not in val:
        return True
    adjacent_squares = get_adjacent_squares(board, square)
    if adjacent_squares.count('e') <= 1 and len(adjacent_squares) > 2:
        return True
    return False

def check_single_attribution(board, square, attribution_val):
    """ --------------------------------------------------
    All attribution_val[0] must also be attribution_val[1], minimum 1
    attribution_val can be one of [rT,gT,bT,yT,rC,gC,bC,yC,rG,gG,bG,yG,rA,gA,bA,yA]
    square is a 2d array indicating (row,col)
    """
    val = get_val(board, square)
    if attribution_val[0] in val and not attribution_val[1] in val:
        return False
    return True

# --------------------------------------------------

def check_adjacency(board, adjacency_value):
    if not _check_rule(board, check_single_adjacency, adjacency_value):
        return False
    # check that minimum case exists
    count=0
    for item in get_all_vals_as_list(board):
        if adjacency_value in item:
            count += 1
    return count >= 2

def check_attraction(board, attraction_value):
    if not _check_rule(board, check_single_attraction, attraction_value):
        return False
    # check that minimum case exists
    return any([attraction_value in x for x in get_all_vals_as_list(board)])

def check_attribution(board, attribution_value):
    if not _check_rule(board, check_single_attribution, attribution_value):
        return False
    # check that miniimum case exists
    return attribution_value in get_all_vals_as_list(board)

def check_equilibrium(board, equilibrium_val):
    """ --------------------------------------------------
    All equilibrium_val must be the same above and below middle.
    equilibrium_val can be one of [T,C,G,A,r,g,b,y]
    """
    # get totals
    above_total, below_total = get_above_and_below_totals(board, equilibrium_val)
    if above_total != below_total:
        return False
    if above_total == 0:
        return False
    return True

def get_above_and_below_totals(board, val):
    above_total=0
    below_total=0
    sq = np.nditer(board, flags=['multi_index'])
    while not sq.finished:
        if val in get_iterval(sq):
            if sq.multi_index[0] <= 1:
                above_total += 1
            else:
                below_total += 1
        sq.iternext()
    return (above_total, below_total)

def check_priority(board, priority_val):
    """ --------------------------------------------------
    There must be a greater number of priority_val[0] than priority_val[1]
    priority_val can be one of [rT,gT,bT,yT,rC,gC,bC,yC,rG,gG,bG,yG,rA,gA,bA,yA]
    """
    all_vals=get_all_vals_as_list(board)
    
    count1 = [x[0] if len(x) > 1 else None for x in all_vals].count(priority_val[0])
    count2 = [x[1] if len(x) > 1 else None for x in all_vals].count(priority_val[1])
    return count1 > count2

def check_priority_old(board, priority_val):
    """ --------------------------------------------------
    All priority_val[0] must be above all priority_val[1]
    priority_val can be one of [rT,gT,bT,yT,rC,gC,bC,yC,rG,gG,bG,yG,rA,gA,bA,yA]
    """
    highest_row_color_found = -1

    # first pass: get highest row for color
    sq = np.nditer(board, flags=['multi_index'])
    while not sq.finished:
        if priority_val[0] in get_iterval(sq):
            highest_row_color_found=max(highest_row_color_found,sq.multi_index[0])
        sq.iternext()

    if highest_row_color_found == -1:
        return False

    found_letter=False
    # second pass: check if any letter is above or equal to highest_row_found
    sq = np.nditer(board, flags=['multi_index'])
    while not sq.finished:
        if priority_val[1] in get_iterval(sq):
            if sq.multi_index[0] <= highest_row_color_found:
                return False
            else:
                found_letter = True
        sq.iternext()
    return found_letter

def check_single_repulsion(board, square, repulsion_val):
    """ --------------------------------------------------
    No repulsion_val[0] next to any repulsion_val[1]
    repulsion_val cal be one of [TC,TG,TA,Te,CG,CA,Ce,GA,Ge,Ae]
    """
    val = get_val(board, square)
    adjacent_squares=get_adjacent_squares(board,square)
    if repulsion_val[0] in val:
        # check for repulsion_val[1] in adjacent squares
        if any([repulsion_val[1] in x for x in adjacent_squares]):
            return False
    if repulsion_val[1] in val:
        # check for repulsion_val[0] in adjacent squares
        if any([repulsion_val[0] in x for x in adjacent_squares]):
            return False
    return True

def check_repulsion(board, repulsion_val):
    if not _check_rule(board, check_single_repulsion, repulsion_val):
        return False
    # check that minimum case exists
    all_vals=get_all_vals_as_list(board)
    if not any([repulsion_val[0] in x for x in all_vals]):
        return False
    if not any([repulsion_val[1] in x for x in all_vals]):
        return False
    return True

def check_uniformity(board, uniformity_val):
    sq = np.nditer(board, flags=['multi_index'])
    col=-1
    #row=-1
    while not sq.finished:
        if uniformity_val in get_iterval(sq):
            #if row == -1 and uniformity_val[0]=='R':
            #    row=sq.multi_index[0]
            #if col == -1 and uniformity_val[0]=='C':
            if col == -1:
                col=sq.multi_index[1]
            #if row != -1:
            #    if sq.multi_index[0] != row:
            #        return False
            if col != -1:
                if sq.multi_index[1] != col:
                    return False
        sq.iternext()

    # check that the minimum case exists
    total = 0
    for val in get_all_vals_as_list(board):
        if uniformity_val in val:
            total += 1
            if total >= 2:
                return True
    return False

def get_iterval(it):
    return it[0].tolist().decode('ascii')

def check_zoning(board, zoning_val):
    """ --------------------------------------------------
    All squares specified by zoning_val must be empty
    zoning_val can be one of [TL,TM,TR,ML,MR,LL,LM,LR,COL0,COL1,COL2,COL3,ROW0ROW1,ROW2,ROW3]

    square is a 2d array indicating (row,col)
    """
    if get_vals_in_zone(board, zoning_val) == 'eeee':
        return True
    return False

# --------------------------------------------------

def get_all_vals_as_list(board):
    return [x.decode('ascii') for x in board.reshape(16).tolist()]

def get_vals_in_zone(board, zone):
    if zone=="TL":
        return ''.join(board[0:2,0:2].reshape(4).decode('ascii'))
    elif zone=="TM":
        return ''.join(board[0:2,1:3].reshape(4).decode('ascii'))
    elif zone=="TR":
        return ''.join(board[0:2,2:].reshape(4).decode('ascii'))
    elif zone=="ML":
        return ''.join(board[1:3,0:2].reshape(4).decode('ascii'))
    elif zone=="MR":
        return ''.join(board[1:3,2:].reshape(4).decode('ascii'))
    elif zone=="LL":
        return ''.join(board[2:,0:2].reshape(4).decode('ascii'))
    elif zone=="LM":
        return ''.join(board[2:,1:3].reshape(4).decode('ascii'))
    elif zone=="LR":
        return ''.join(board[2:,2:].reshape(4).decode('ascii'))
    elif zone[:3]=="COL":
        return ''.join(board[:,int(zone[3])].reshape(4).decode('ascii'))
    elif zone[:3]=="ROW":
        return ''.join(board[int(zone[3]),:].reshape(4).decode('ascii'))

def check_in_zone(square, zone):
    if zone=="TL":
        return square[0] < 2 and square[1] < 2
    elif zone=="TM":
        return square[0] < 2 and (1 <= square[1] < 3)
    elif zone=="TR":
        return square[0] < 2 and square[1] >= 2
    elif zone=="ML":
        return (1 <= square[0] < 3) and square[1] < 2
    elif zone=="MR":
        return (1 <= square[0] < 3) and square[1] >= 2
    elif zone=="LL":
        return square[0] >= 2 and square[1] < 2
    elif zone=="LM":
        return square[0] >= 2 and (1 <= square[1] < 3)
    elif zone=="LR":
        return square[0] >= 2 and square[1] >= 2
    elif zone[:3]=="COL":
        return square[1] == int(zone[3])
    elif zone[:3]=="ROW":
        return square[0] == int(zone[3])

def _check_rule(board, rule, value):
    sq=np.nditer(board, flags=['multi_index'])
    while not sq.finished:
        result=rule(board, sq.multi_index, value)
        if not result:
            return False
        sq.iternext()
    return True

def print_board(board):
    print ("Board:\n")
    out_line=""
    for row in range(0,board.shape[0]):
        for col in range(0,board.shape[1]):
            out_line += "{:3}".format(board[row][col].decode('ascii'))
        out_line += "\n"
    print (out_line)

def get_left(board, square):
    if square[1] == 0:
        return None
    return board[square[0],square[1]-1].decode()

def get_left_square(board, square):
    if square[1] == 0:
        return None
    return (square[0],square[1]-1)

def get_right(board, square):
    if square[1] == board.shape[1]-1:
        return None
    return board[square[0],square[1]+1].decode()

def get_right_square(board, square):
    if square[1] == board.shape[1]-1:
        return None
    return (square[0],square[1]+1)

def get_up(board, square):
    if square[0] == 0:
        return None
    return board[square[0]-1,square[1]].decode()

def get_up_square(board, square):
    if square[0] == 0:
        return None
    return (square[0]-1,square[1])

def get_down(board, square):
    if square[0] == board.shape[0]-1:
        return None
    return board[square[0]+1,square[1]].decode()

def get_down_square(board, square):
    if square[0] == board.shape[0]-1:
        return None
    return (square[0]+1,square[1])

def get_left_and_up_neighbors(board, square):
    ReturnVal = namedtuple("ReturnVal", "left_square up_square values squares")
    left_square = get_left_square(board,square)
    up_square = get_up_square(board,square)
    values=[]
    squares=[]
    if left_square:
        values.append(get_val(board,left_square))
        squares.append(left_square)
    if up_square:
        values.append(get_val(board,up_square))
        squares.append(up_square)
    return ReturnVal(left_square, up_square, values, squares)

def check_mustbe_cantbe(must_be, cant_be):
    if len(must_be.intersection(cant_be)) > 0:
        return False
    if 'e' in must_be and len(must_be) > 1:
        return False
    complete_cards=[]
    # check if we need a specific card
    for item in must_be:
        if len(item) > 1:
            complete_cards.append(item)
    if len(complete_cards) > 1:
        # We need two different cards in the same space
        return False
    if len(complete_cards) > 0:
        if any([x in complete_cards[0] for x in cant_be]):
            return False
    # count colors
    total_colors=0
    total_letters=0
    for item in must_be:
        if item in ['r','g','b','y']:
            total_colors += 1
        if item in ['T','C','G','A']:
            total_letters += 1
    if total_colors > 1 or total_letters > 1:
        return False

    cantbe_colors=0
    cantbe_letters=0
    for item in cant_be:
        if item in ['r','g','b','y']:
            cantbe_colors += 1
        if item in ['T','C','G','A']:
            cantbe_letters += 1
    if cantbe_colors == 4 or cantbe_letters == 4:
        return False
    return True

def get_possible_vals(game, square, last, prelim_cant_be):
    adjacent_squares=get_adjacent_squares(game.board, square)
    must_be=set()
    cant_be=set(prelim_cant_be)
    valid_colors=set(['r','g','b','y'])
    valid_letters=set(['T','C','G','A'])
    valid_combos=set('e')

    # adjacency
    adjacency_vals=get_possible_adjacency(game, square, last, adjacent_squares)
    if len(adjacency_vals.must_be) > 0:
        must_be=must_be.union(adjacency_vals.must_be)
    if len(adjacency_vals.cant_be) > 0:
        cant_be=cant_be.union(adjacency_vals.cant_be)
    if adjacency_vals.broken or not check_mustbe_cantbe(must_be, cant_be):
        logging.debug ("Adjacency: found conflicting criteria for square " + str(square))
        logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
        return None
    logging.debug ("ADJACENCY: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # attraction
    attraction_vals=get_possible_attraction(game, square, last)
    if len(attraction_vals.must_be) > 0:
        must_be=must_be.union(attraction_vals.must_be)
    if len(attraction_vals.cant_be) > 0:
        cant_be=cant_be.union(attraction_vals.cant_be)
    if attraction_vals.broken or not check_mustbe_cantbe(must_be, cant_be):
        logging.debug ("Attraction: found conflicting criteria for square " + str(square))
        logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
        return None
    logging.debug ("ATTRACTION: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # attribution
    # if attribution is rT, add rG, rC, rA to cant_be list
    [cant_be.add(game.attribution[0] + x) for x in valid_letters.difference(set([game.attribution[1]]))]
    if last and not check_attribution(game.board, game.attribution):
        must_be.add(game.attribution)
    
    logging.debug ("ATTRIBUTION: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # equilibrium
    above_total, below_total = get_above_and_below_totals(game.board, game.equilibrium)
    if square[0] == 1 and square[1] == 3 and above_total == 0:
        must_be.add(game.equilibrium)
    diff = above_total - below_total
    if diff > 2 and square[0] == 3 and square[1] == diff:
        must_be.add(game.equilibrium)
    if last:
        if abs(above_total - below_total) > 1:
            logging.debug ("Equilibrium: unable to fix with last square")
            return None
        if above_total > below_total and square[0] >= 2:
            must_be.add(game.equilibrium)
        elif above_total < below_total and square[0] <= 1:
            must_be.add(game.equilibrium)
        elif above_total == below_total:
            cant_be.add(game.equilibrium)
        else:
            logging.debug ("Equilibrium: unable to fix with last square " + str(square))
            logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
            return None
    logging.debug ("EQUILIBRIUM: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # priority
    priority_vals=get_possible_priority(game, square, last)
    if len(priority_vals.must_be) > 0:
        must_be=must_be.union(priority_vals.must_be)
    if len(priority_vals.cant_be) > 0:
        cant_be=cant_be.union(priority_vals.cant_be)
    if priority_vals.broken or not check_mustbe_cantbe(must_be, cant_be):
        logging.debug ("Priority: found conflicting criteria for square " + str(square))
        logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
        return None
    logging.debug ("PRIORITY: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # repulsion
    neighbor_vals=get_left_and_up_neighbors(game.board, square).values
    if any([game.repulsion[0] in x for x in neighbor_vals]):
        cant_be.add(game.repulsion[1])
    if any([game.repulsion[1] in x for x in neighbor_vals]):
        cant_be.add(game.repulsion[0])
    if last and not check_repulsion(game.board, game.repulsion):
        all_vals=get_all_vals_as_list(game.board)
        if not any([game.repulsion[0] in x for x in all_vals]):
            must_be.add(game.repulsion[0])
        if not any([game.repulsion[1] in x for x in all_vals]):
            must_be.add(game.repulsion[1])
    if not check_mustbe_cantbe(must_be, cant_be):
        logging.debug ("Repulsion: found conflicting criteria for square " + str(square))
        logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
        return None
    logging.debug ("REPULSION: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # uniformity
    uniformity_vals=get_possible_uniformity(game, square, last)
    if len(uniformity_vals.must_be) > 0:
        must_be=must_be.union(uniformity_vals.must_be)
    if len(uniformity_vals.cant_be) > 0:
        cant_be=cant_be.union(uniformity_vals.cant_be)
    if uniformity_vals.broken or not check_mustbe_cantbe(must_be, cant_be):
        logging.debug ("Uniformity: found conflicting criteria for square " + str(square))
        logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
        return None
    logging.debug ("UNIFORMITY: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    # zoning
    if check_in_zone(square, game.zoning):
        must_be.add('e')
    else:
        cant_be.add('e')
    if not check_mustbe_cantbe(must_be, cant_be):
        logging.debug ("Zoning: found conflicting criteria for square " + str(square))
        logging.debug ("Must be: " + str(must_be) + " Cant be: " + str(cant_be))
        return None
    logging.debug ("ZONING: Must be: " + str(must_be) + " Cant be: " + str(cant_be))

    if 'e' in must_be:
        return 'e'

    for item in must_be:
        if len(item) > 1:
            return [item]
        else:
            if item in valid_letters:
                valid_letters=[item]
            if item in valid_colors:
                valid_colors=[item]

    for item in cant_be:
        if item in valid_colors:
            valid_colors.remove(item)
        if item in valid_letters:
            valid_letters.remove(item)
        if item in valid_combos:
            valid_combos.remove(item)

    for combo in product(valid_colors, valid_letters):
        valid_combos.add(combo[0] + combo[1])

    for item in cant_be:
        if item in valid_combos:
            valid_combos.remove(item)

    return valid_combos

#def check_rule_combos(game, square):
#    # attribution / adjacency
#    if 

def safe_remove(item, list_):
    if item in list_:
        return

def get_possible_uniformity(game, square, last):
    ReturnVals = namedtuple("ReturnVals","must_be cant_be broken")
    must_be=[]
    cant_be=[]
    broken=False

    col=-1
    #row=-1
    fulfilled = check_uniformity(game.board, game.uniformity)
    sq = np.nditer(game.board, flags=['multi_index'])
    while not sq.finished:
        if game.uniformity in get_iterval(sq):
            #if row == -1 and game.uniformity[0]=='R':
            #    row=sq.multi_index[0]
            if col == -1:
                col=sq.multi_index[1]
            #if row != -1:
            #    if square[0] == row:
            #        if last and not fulfilled:
            #            must_be.append(game.uniformity[1])
            #    else:
            #        cant_be.append(game.uniformity[1])
            if col != -1:
                if square[1] == col:
                    if last and not fulfilled:
                        must_be.append(game.uniformity)
                else:
                    cant_be.append(game.uniformity)
        sq.iternext()
    if last and not fulfilled and len(must_be) == 0:
        broken=True
    return ReturnVals(must_be, cant_be, broken)

def get_possible_adjacency(game, square, last, adjacent_squares):
    ReturnVals = namedtuple("ReturnVals","must_be cant_be broken")
    must_be=[]
    cant_be=[]
    broken=False
    # get possibilities for adjacency
    all_vals=get_all_vals_as_list(game.board)
    found_one=False
    if any([game.adjacency in x for x in all_vals]):
        found_one=True
        # a game.adjacency exists
        if all([game.adjacency not in x for x in adjacent_squares]):
            cant_be.append(game.adjacency)

    if get_up(game.board, square) and game.adjacency in get_up(game.board, square):
        # check that this is the last match for the top row row and cur column
        up_row_vals=game.board[square[0]-1][:].decode().tolist()
        cur_row_vals=game.board[square[0]][:square[1]].decode().tolist()
        if ([game.adjacency in x for x in up_row_vals].count(True) == 1) and \
           ([game.adjacency in x for x in cur_row_vals].count(True) == 0):
            if not found_one:
                # we're about to move away from the only square that we can attach to
                must_be.append(game.adjacency)
            else:
                # we're about to move away from the last square that can be adjacent
                if game.adjacency in game.attribution and game.attribution not in all_vals:
                    must_be.append(game.attribution)
                if game.equilibrium == game.adjacency:
                    above_total, below_total = get_above_and_below_totals(game.board, game.equilibrium)
                    if above_total > below_total:
                        must_be.append(game.equilibrium)

    #if last:
    #    if not check_adjacency(game.board, game.adjacency):
    #        cur_val=game.board[square[0],square[1]]
    #        # temporarily change the board
    #        game.board[square[0],square[1]] = game.adjacency
    #        if check_adjacency(game.board, game.adjacency):
    #            # change the board back
    #            game.board[square[0],square[1]]=cur_val
    #            must_be.append(game.adjacency)
    #        else:
    #            # change the board back
    #            game.board[square[0],square[1]]=cur_val
    #            logging.debug ("Broken adjacency " + game.adjacency + " at " + str(square))
    #            broken = True
    #    else:
    #        cant_be.add(game.adjacency)
    #
    return ReturnVals(must_be, cant_be, broken)

def get_possible_attraction(game, square, last):
    #pdb.set_trace()
    ReturnVals = namedtuple("ReturnVals","must_be cant_be broken")
    must_be=[]
    cant_be=[]
    broken=False

    # add corners to cant_be
    if square in [(0,0),(3,0),(0,3),(3,3)]:
        cant_be.append(game.attraction)

    # check if any neighboring squares have the attraction value
    # if so, we need to determine if it's ok for this square to be empty
    left_square=get_left_square(game.board, square)
    #right_square=get_right_square(game.board, square)
    up_square=get_up_square(game.board, square)
    #down_square=get_down_square(game.board, square)
    
    check_squares=[]
    
    if left_square and game.attraction in get_val(game.board, left_square):
        num_e=0
        cur_square=left_square
        if get_left(game.board, cur_square) in {'e', None}:
            num_e += 1
        if get_up(game.board, cur_square) in {'e', None}:
            num_e += 1
        if not get_down(game.board, cur_square):
            num_e += 1
        check_squares.append((left_square, num_e))
    #if right_square and game.attraction in get_val(game.board, right_square):
    #    num_out=0
    #    num_e=0
    #    cur_square=right_square
    #    if get_up(game.board, cur_square) in {'e', None}:
    #        num_e += 1
    #    if get_right(game.board, cur_square):
    #        num_out += 1
    #    if get_down(game.board, cur_square):
    #        num_out += 1
    #    check_squares.append((right_square, num_e, num_out))
    if up_square and game.attraction in get_val(game.board, up_square):
        num_e = 0
        cur_square=up_square
        if get_left(game.board, cur_square) in {'e', None}:
            num_e += 1
        if get_up(game.board, cur_square) in {'e',None}:
            num_e += 1
        if get_right(game.board, cur_square) in {'e',None}:
            num_e += 1
        check_squares.append((up_square, num_e))
    #if down_square and game.attraction in get_val(game.board, down_square):
    #    num_out = 0
    #    num_e = 0
    #    cur_square=down_square
    #    if get_left(game.board, cur_square):
    #        num_out += 1
    #    if get_right(game.board, cur_square):
    #        num_out += 1
    #    if get_down(game.board, cur_square):
    #        num_out += 1
    #    check_squares.append((down_square, num_e, num_out))

    for check in check_squares:
        cur_square, num_e = check
        if num_e >= 2:
            broken=True
            return ReturnVals(must_be, cant_be, broken)
        elif num_e == 1:
            cant_be.append('e')
        elif num_e == 0:
            continue
        #if abs(num_out-num_e) == 1:
        #    cant_be.append('e')
        #elif num_e > 0 and num_e == num_out:
        #    broken=True
        #    return ReturnVals(must_be, cant_be, broken)
        #elif num_e - num_out > 1:
        #    broken=True
        #    return ReturnVals(must_be, cant_be, broken)
        
    ret_vals = ReturnVals(must_be, cant_be, broken)

    return ret_vals

def get_possible_priority(game, square, last):
    ReturnVals = namedtuple("ReturnVals","must_be cant_be broken")
    must_be=[]
    cant_be=[]
    broken=False
    
    num_sq_left = len(SQ_INDEX_LIST) - SQ_INDEX_LIST.index(square)
    all_vals=get_all_vals_as_list(game.board)
    count1 = [x[0] if len(x) > 1 else None for x in all_vals].count(game.priority[0])
    count2 = [x[1] if len(x) > 1 else None for x in all_vals].count(game.priority[1])

    diff = count2 - count1 + 1
    if diff == num_sq_left:
        #logging.debug("APPENDING " + game.priority[0])
        must_be.append(game.priority[0])
    elif diff > num_sq_left:
        broken = True
        return ReturnVals(must_be, cant_be, broken)
    elif diff == num_sq_left + 1:
        cant_be.append(game.priority[1])
    return ReturnVals(must_be, cant_be, broken)

def get_possible_priority_old(game, square, last):
    ReturnVals = namedtuple("ReturnVals","must_be cant_be broken")
    must_be=[]
    cant_be=[]
    broken=False

    cant_be.append(game.priority)
    already_good=check_priority(game.board, game.priority)
    lowest_row_letter_found=5
    highest_row_color_found=-1
    # find highest occurence of letter
    sq = np.nditer(game.board, flags=['multi_index'])
    while not sq.finished:
        if game.priority[1] in get_iterval(sq):
            lowest_row_letter_found=min(lowest_row_letter_found, sq.multi_index[0])
        if game.priority[0] in get_iterval(sq):
            highest_row_color_found=max(highest_row_color_found, sq.multi_index[0])
        sq.iternext()
    if lowest_row_letter_found <= square[0]:
        if not already_good:
            # need to put the color down, but the letter is already above
            broken=True
            return ReturnVals(must_be,cant_be,broken)
    if highest_row_color_found == -1 or highest_row_color_found == square[0]:
        # don't have a color yet, or it's on the current line:
        # add letter to cant_be
        #logging.debug ("HIGHEST ROW COLOR: " + str(highest_row_color_found))
        cant_be.append(game.priority[1])
    if lowest_row_letter_found <= square[0]:
        # letter exists and is lower or equal to current row
        # add color ot cant_be
        cant_be.append(game.priority[0])
    if last:
        if not already_good:
            if highest_row_color_found != -1 and highest_row_color_found < 3:
                must_be.append(game.priority[1])
            else:
                broken=True
                return ReturnVals(must_be,cant_be,broken)
    return ReturnVals(must_be,cant_be,broken)

SQ_INDEX_LIST=[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3)]

def get_comma_string(board):
    return ','.join(board.reshape(16).decode().tolist())

def get_hash_board(board):
    return get_comma_string(board).__hash__()

def undo_last_step(board, i, cards, cant_be_dict):
    """ --------------------------------------------------
    undoes the previous square attempt and adds to the cant_be_dict
    returns a new i value
    """
    i -= 1
    ival = SQ_INDEX_LIST[i]
    last_val=board[ival].decode()
    cards[last_val] += 1
    board[ival]='e'
    cant_be_dict[get_hash_board(board)].append(last_val)
    return i

def multi_try_find_valid_state(game):
    for i in range(0,200):
        if find_valid_state(game, 50):
            #logging.info("MULTI TRY: {}".format(i))
            return True
        if i % 10 == 0:
            logging.debug ("Try #{}".format(i))
    return False

def find_valid_state(game, cutoff=0):
    cards=defaultdict(lambda:4)

    game.board[:]='e'
    sq=np.nditer(game.board, flags=['multi_index'])
    sq_deque = deque()
    sq_deque.appendleft(sq)
    cant_be_dict = defaultdict(list)
    i=0
    num_end_states=0
    while i < len(SQ_INDEX_LIST):
        if i < 0:
            logging.info ("Impossible set")
            print_board(game.board)
            return False
        ival=SQ_INDEX_LIST[i]
        possible_vals = get_possible_vals(game, ival, False, cant_be_dict[get_hash_board(game.board)])
        if not possible_vals:
            # we got into a situation where there are no possible cards to try
            # go back to previous square, add the current value of it to the cant_be list
            # and try again
            i = undo_last_step(game.board, i, cards, cant_be_dict)
            continue
        adjusted_vals=[]
        for item in possible_vals:
            if cards[item] > 0:
                adjusted_vals.append(item)
        if len(adjusted_vals) == 0:
            i = undo_last_step(game.board, i, cards, cant_be_dict)
            continue
        # made it through
        card = random.choice(adjusted_vals)
        #card = adjusted_vals[0]
        game.board[ival] = card
        cards[card] -= 1
        i += 1
        logging.debug ("Trying: " + get_comma_string(game.board))
        num_end_states += 1
        if cutoff != 0 and num_end_states > cutoff:
            return False
        if num_end_states % 1000 == 0:
            logging.info ("Tried: " + str(num_end_states))
        if i == len(SQ_INDEX_LIST):
            if check_win(game):
                #logging.info ("FIND TRY: {}".format(num_end_states))
                return True
            else:
                i=undo_last_step(game.board, i, cards, cant_be_dict)
                continue
    logging.error ("Shouldn't be able to get here")
    return False

def check_win(game):
    adjacency   = check_adjacency   (game.board, game.adjacency   )
    attraction  = check_attraction  (game.board, game.attraction  )
    attribution = check_attribution (game.board, game.attribution )
    equilibrium = check_equilibrium (game.board, game.equilibrium )
    priority    = check_priority    (game.board, game.priority    )
    repulsion   = check_repulsion   (game.board, game.repulsion   )
    uniformity  = check_uniformity  (game.board, game.uniformity  )
    zoning      = check_zoning      (game.board, game.zoning      )

    if not adjacency:
        logging.debug ("Adjacency failed")
    if not attraction:
        logging.debug ("attraction failed")
    if not attribution:
        logging.debug ("attribution failed")
    if not equilibrium:
        logging.debug ("equilibrium failed")
    if not priority:
        logging.debug ("priority failed")
    if not repulsion:
        logging.debug ("repulsion failed")
    if not uniformity:
        logging.debug ("uniformity failed")
    if not zoning:
        logging.debug ("zoning failed")
    
    if not all([adjacency, attraction, attribution, equilibrium, priority, repulsion, uniformity, zoning]):
        return False
    return True

def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#game.board.reshape(16)[:]=['e','e','yC','yC','e','e','bC','bA','bA','bC','bC','rG','gC','bC','bT','rG']
#print_board(game.board)

def validate_rules(game):
    # 1. attribution of red / T with priority of red over T
    # 2. Row uniformity of T with equilibrium of T

    # 1. Attribution of rT with priority of rT
    if (game.attribution == game.priority):
        logging.info ("Invalid rules(1): " + ", ".join(rules))
        return False

    # 2. Row1 / Row2 zoning with Column Uniformity T, Adjacency T, Equilibrium T
    if (game.zoning == "ROW1" or game.zoning == "ROW2") and game.uniformity == game.adjacency == game.equilibrium:
        logging.info ("Invalid rules(2): " + ", ".join(rules))
        return False

    # 3. Row1 / Row2 zoning with Attraction T, Equilibrium T
    if (game.zoning == "ROW1" or game.zoning == "ROW2") and game.attraction == game.equilibrium:
        logging.info ("Invalid rules(3): " + ", ".join(rules))
        return False

    # 4. Top/Bottom Middle zone, Equilibrium of T, Attraction with T
    if (game.zoning == 'TM' or game.zoning == 'LM') and game.equilibrium == game.attraction:
        logging.info ("Invalid rules(4): " + ', '.join(rules))
        return False

    # 5. Repulsion of Te with equilibrium of T, and any row
    if (game.zoning[:3] == "ROW" and 'e' in game.repulsion):
        logging.info ("Invalid rules(5): " + ', '.join(rules))
        return False
    
    return True

def choose_new_rule(rule_name):
    if rule_name == 'adjacency':
        return random.choice(ADJACENCY_OPT)
    elif rule_name == 'attraction':
        return random.choice(ATTRACT_OPT)
    elif rule_name == 'attribution':
        return random.choice(ATTRIBUTION_OPT)
    elif rule_name == 'equilibrium':
        return random.choice(EQUILIBRIUM_OPT)
    elif rule_name == 'priority':
        return random.choice(PRIORITY_OPT)
    elif rule_name == 'repulsion':
        return random.choice(REPULSION_OPT)
    elif rule_name == 'uniformity':
        return random.choice(UNIFORM_OPT)
    elif rule_name == 'zoning':
        return random.choice(ZONING_OPT)
    return None

def adjust_rules(game):
    # two player, change rules
    rule_options=['adjacency', 'attraction', 'attribution', 'equilibrium', 'priority', 'repulsion', 'uniformity', 'zoning']
    num_groups=2
    random.shuffle(rule_options)
    slice_size = int(len(rule_options) / num_groups)
    for i in range(0,num_groups):
        options = rule_options[i*slice_size:(i+1)*slice_size]
        is_bad = True
        val_dict=defaultdict(int)
        while is_bad:
            print ("RULES: " + ', '.join([game.adjacency, game.attraction,
                                          game.attribution, game.equilibrium,
                                          game.priority, game.repulsion,
                                          game.uniformity, game.zoning]))
            for option in options:
                for val in getattr(game,option):
                    val_dict[val] += 1
            for key in val_dict:
                found_bad=False
                if val_dict[key] >= len(options)/2:
                    found_bad=True
                    for opt in options:
                        if key in game.__getattribute__(opt):
                            setattr(game, opt, choose_new_rule(opt))
                if not found_bad:
                    is_bad = False

                    
if __name__ == "__main__":
    RANDOM = 0
    ADJUST_RULES = 0  # the "players" adjust the rules they see to reduce repeat letters / colors
    CHECK_IF_ALREADY_VALID = 0
    setup_logging()
    game = GameState()
    i=0
    rules_prod = product(ADJACENCY_OPT, ATTRACT_OPT, ATTRIBUTION_OPT, EQUILIBRIUM_OPT, PRIORITY_OPT, REPULSION_OPT, UNIFORM_OPT, ZONING_OPT)
    while True:
        if RANDOM:
            rules=(random.choice(ADJACENCY_OPT),
                   random.choice(ATTRACT_OPT),
                   random.choice(ATTRIBUTION_OPT),
                   random.choice(EQUILIBRIUM_OPT),
                   random.choice(PRIORITY_OPT),
                   random.choice(REPULSION_OPT),
                   random.choice(UNIFORM_OPT),
                   random.choice(ZONING_OPT))

            game.initialize_rules(*rules)
            if ADJUST_RULES:
                adjust_rules(game)
        else:
            rules = rules_prod.__next__()
            game.initialize_rules(*rules)

        if CHECK_IF_ALREADY_VALID:
            if tuple(rules) in VALID_RULES:
                print ("Already validated: " + str(rules))
                i += 1
                continue
        
        if not validate_rules(game):
            i += 1
            continue
        print ("Trying: " + ', '.join(rules))

        if multi_try_find_valid_state(game):
            print (str(i) + ". TRUE: " + str(rules))
        else:
            print (str(i) + ". FALSE: " + str(rules))
        i += 1
