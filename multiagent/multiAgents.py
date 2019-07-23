# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        ghostPositions = successorGameState.getGhostPositions()
        score = successorGameState.getScore()
        foodlist = newFood.asList()
        """Finds the average distance to all food remaining"""
        averageDist = 0
        foodleft = len(foodlist)
        for food in foodlist:
          averageDist += manhattanDistance(newPos, food)
        if foodleft != 0:
          averageDist = averageDist/foodleft
        """Sums the distance to the closest x food"""
        foodMult = [1,.75,.5, .25, .15, .1, .05, .01]
        n = 0
        minFoodDist = 0
        closestfood = newPos
        while len(foodMult) > n and foodlist:
          dist, closestfood = min([(manhattanDistance(closestfood, food),food) for food in foodlist])
          minFoodDist += (foodMult[n])*dist
          n += 1
          foodlist.remove(closestfood)
        """finds distance to the closest Ghost,
         and scales according to the scared timer"""
        ghostdist = 0
        minGhostDist = min(manhattanDistance(newPos, ghost) for ghost in ghostPositions)
        if minGhostDist <= 3:
          if minGhostDist != 0: 
            ghostdist = 1/minGhostDist
          else: 
            ghostdist = 0
        minscaredtime = min(newScaredTimes)
        timermult = -1
        if minscaredtime > 2:
          timermult = 1
          if minGhostDist != 0: 
            ghostdist = 1/minGhostDist
          else: 
            ghostdist = 0
        """ Finaly I noticed that pacman would ocasionaly stop moving,
        so i'm implementin a stop penalty"""
        stoppen = 0
        if action == "Stop":
          stoppen = minFoodDist + foodleft + (timermult)*ghostdist
        """ Now we return a linear combo of these major values"""
        expectedScore = score - 3*foodleft - 2*minFoodDist + 6*(timermult)*ghostdist - 4/5*averageDist - stoppen
        return expectedScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        """After trying for some time to solve it in one go I 
        implemented this recursive function that finds the necesarry value for each state"""
        def searchVal(gameState, depth, agent, numghosts):
          if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
          legalActions = gameState.getLegalActions(agent)
          if agent == 0:
            val = max(searchVal(gameState.generateSuccessor(agent, action), depth, 1, numghosts) for action in legalActions)
          elif agent == numghosts:
            val = min(searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0, numghosts) for action in legalActions)
          else:
            val = min(searchVal(gameState.generateSuccessor(agent, action), depth , agent + 1, numghosts) for action in legalActions)
          return val

        """ Finaly using search functions we will use the recursion to find the solution"""
        legalActions = gameState.getLegalActions(self.index) 
        numghosts = gameState.getNumAgents() - 1
        maxval = -(float("inf"))
        retaction = Directions.STOP
        depth = self.depth
        for action in legalActions:
          successor = gameState.generateSuccessor(0, action)
          newval = searchVal(successor, depth, 1, numghosts)
          #print (newval, action)
          if newval > maxval:
            maxval = newval
            retaction = action
        return retaction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """To start we fallow the slides and create the min/max value functions"""
        def maxValue(gameState, depth, agent, numghosts, alpha, beta):
          if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
          v = -float("inf")
          legalActions = gameState.getLegalActions(agent)
          for action in legalActions:
            child = gameState.generateSuccessor(agent, action)
            v = max(v, minValue(child, depth, agent + 1, numghosts, alpha, beta))
            if v > beta:
              return v
            alpha = max(v, alpha)
          return v 
        def minValue(gameState, depth, agent, numghosts, alpha, beta):
          if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
          v = float("inf")
          legalActions = gameState.getLegalActions(agent)
          for action in legalActions:
            child = gameState.generateSuccessor(agent, action)
            if agent == numghosts:
              v = min(v, maxValue(child, depth - 1, 0, numghosts, alpha, beta))
            else:
              v = min(v, minValue(child, depth, agent + 1, numghosts, alpha, beta))
            if v < alpha:
              return v
            beta = min(v, beta)
          return v 
        """Finaly we set up the values to use these functions and run a version 
        of max val outside of the fucntion to corectly return the action we want to take"""
        v = -float("inf")
        agent = self.index
        depth = self.depth
        numghosts = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions(agent)
        retaction = Directions.STOP
        alpha = -float("inf")
        beta =  float("inf")
        for action in legalActions:
          prevV = v
          child = gameState.generateSuccessor(agent, action)
          v = max(v, minValue(child, depth, agent + 1, numghosts, alpha, beta))
          alpha = max(v, alpha)
          if v > prevV:
            retaction = action
          if v >= beta:
            return retaction
        return retaction 
          
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def searchVal(gameState, depth, agent, numghosts):
          if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState)
          legalActions = gameState.getLegalActions(agent)
          numActions = len(legalActions)
          if agent == 0:
            val = max(searchVal(gameState.generateSuccessor(agent, action), depth, 1, numghosts) for action in legalActions)
          elif agent == numghosts:
            val = 0
            for action in legalActions:
              val += searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0, numghosts)
            if numActions != 0:
              val = val/numActions
          else:
            val = 0
            for action in legalActions:
              val += searchVal(gameState.generateSuccessor(agent, action), depth , agent + 1, numghosts)
            if numActions != 0:
              val = val/numActions
          return val

        """ Finaly using search functions we will use the recursion to find the solution"""
        legalActions = gameState.getLegalActions(self.index) 
        numghosts = gameState.getNumAgents() - 1
        maxval = -(float("inf"))
        retaction = Directions.STOP
        depth = self.depth
        for action in legalActions:
          successor = gameState.generateSuccessor(0, action)
          newval = searchVal(successor, depth, 1, numghosts)
          #print (newval, action)
          if newval > maxval:
            maxval = newval
            retaction = action
        return retaction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPosition = currentGameState.getFood()
    pelletPosition = currentGameState.getCapsules()
    ghostPositions = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    evaluationScore = currentGameState.getScore()
    foodlist = foodPosition.asList()
    foodleft = len(foodlist)
    """First lets find the average of all food placement from pacman,
     since dont want pacman to get stuck anywhere, realized this wasnt very helpfull"""
    averageDist = 0
    for food in foodlist:
      averageDist += manhattanDistance(pacmanPosition, food)
    if foodleft != 0:
      averageDist = averageDist/foodleft
    if averageDist != 0:
      averageDist = 1/averageDist
    else:
      averageDist = 0
    """second lets find a value for the closest food to pacman """
    foodMult = 16
    positionMult = [1, .5]
    num = 0
    totaldist = 0
    start = pacmanPosition
    minfood = 1
    while foodlist and num < len(positionMult):
      minfood, food = min([(manhattanDistance(start, food), food) for food in foodlist])
      foodlist.remove(food)
      totaldist += positionMult[num]*minfood
      num += 1
    if totaldist != 0:
      Ffood = 1/totaldist
    else:
     Ffood = 1
    """third we find the value for the closest Ghost"""
    ghostMulti = 0
    Fghost = min([manhattanDistance(pacmanPosition, ghost) for ghost in ghostPositions])
    if Fghost <= 5:
      ghostMulti = 6
    """Use scared time to maximize the hunting of ghost""" 
    scrdTime = min(scaredTimes)
    if scrdTime > 3:
      if Fghost != 0:
        Fghost = 1/Fghost + 1
      else:
        Fghost = 1
      ghostMulti = 16
    """We use ghost position and pellet position to see if grabing 
    a pellet is an amazig idea"""
    pelletMulti = 0
    Fpellet = 0
    if pelletPosition:
      Fpellet = min([manhattanDistance(pacmanPosition, pellet) for pellet in pelletPosition])
    if Fpellet != 0:
      Fpellet = 1/Fpellet
    if Fpellet < Fghost and ghostMulti > 0:
        pelletMulti = 1

    evaluationScore += (foodMult)*Ffood + (ghostMulti)*Fghost + (0)*averageDist + pelletMulti*Fpellet - 1*foodleft
    return evaluationScore
# Abbreviation
better = betterEvaluationFunction

