# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    p,x,y = problem.getSuccessors(problem.getStartState())[1]
    print x
    from game import Directions
    print Directions.x
    """
    from util import Stack
    start = problem.getStartState()
    closed = []
    fringe = Stack()
    tup = (start, [])
    fringe.push(tup)
    while not fringe.isEmpty():
        spot, path  = fringe.pop()
        if problem.isGoalState(spot):
            return path
        if not spot in closed:
            closed += [spot]
            for successor in problem.getSuccessors(spot):
                nspot, direction, val = successor
                ntup = (nspot, path + [direction])
                fringe.push(ntup)
    return "No Path Found"
                
        
    
        
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    start = problem.getStartState()
    closed = []
    fringe = Queue()
    tup = (start, [])
    fringe.push(tup)
    while not fringe.isEmpty():
        spot, path  = fringe.pop()
        if problem.isGoalState(spot):
            return path
        if not spot in closed:
            closed += [spot]
            for successor in problem.getSuccessors(spot):
                nspot, direction, val = successor
                ntup = (nspot, path + [direction])
                fringe.push(ntup)
    return "No Path Found"

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    start = problem.getStartState()
    '''for successor in problem.getSuccessors(start):
        print successor
        x,y,z = successor
        for successor in problem.getSuccessors(x):
            print "successor of successor",successor'''
    closed = []
    fringe = PriorityQueue()
    tup = (start, [], 0)
    fringe.push(tup,1)
    while not fringe.isEmpty():
        spot, path, pprior  = fringe.pop()
        if problem.isGoalState(spot):
            return path
        if not spot in closed:
            closed += [spot]
            for successor in problem.getSuccessors(spot):
                nspot, direction, priority = successor
                ntup = (nspot, path + [direction], pprior + priority)
                fringe.push(ntup, priority + pprior)

    return "No Path Found"

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    start = problem.getStartState()
    closed = []
    fringe = PriorityQueue()
    tup = (start, [], 0)
    fringe.push(tup,1 + heuristic(start, problem))
    while not fringe.isEmpty():
        spot, path, pprior  = fringe.pop()
        if problem.isGoalState(spot):
            return path
        if not spot in closed:
            closed += [spot]
            for successor in problem.getSuccessors(spot):
                nspot, direction, priority = successor
                ntup = (nspot, path + [direction], priority + pprior)
                oldhcost = heuristic(spot, problem) 
                newhcost =heuristic(nspot, problem)
                #if oldhcost - newhcost > priority:
                #print "Consistancy problem", oldhcost, newhcost, spot, nspot
                fringe.update(ntup, priority + pprior + newhcost)
    return "No Path Found"


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
