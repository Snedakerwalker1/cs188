# myAgentP3.py
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#########
# Agent #
#########
class myAgentP3(CaptureAgent):
  """Students' Names: Walker Snedaker
      Phase Number: 3
      Description of Bot: uses a minimax search a couple of layers
       deep to opptomise the score bassed on the teammates broadcasted actions
       and avoid the ghosts if they are nearby.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line. 
    # If you would like to use Manhattan distances instead 
    # of maze distances in order to save on initialization 
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.distancer.getMazeDistances() 

  def chooseAction(self, gameState):
    """
    takes the given state knowlege, like ghosts positions, teammate's planne action and such
    and uses these to run a short depth minimax search, will be updated to expecti max if 
    need be, with approximate q values found in the same sensce as the previous 2 parts
    """
    def searchVal(gameState, depth, agent, teammateIndex, teammateActions, numagents,alpha, beta,  actionlist, nextact, actionNotBroken):
      if depth == 0 or self.isWin(gameState):
        "update for this thingy, need something that checks win conditions an lose conditions as well"
        evaluationVal, evaluationAction = self.evaluationFunction(gameState)
        retval = (evaluationVal, actionlist + [evaluationAction])
        #print retval
        return retval
      legalActions = gameState.getLegalActions(agent)
      legalActions = actionsWithoutStop(legalActions)
      if agent == self.index:
        val = (-float('inf'), actionlist)
        for action in legalActions:
          child = gameState.generateSuccessor(agent, action)
          if agent == numagents:
            val = max(val, searchVal(child, depth - 1, 0, teammateIndex, teammateActions, numagents, alpha, beta, actionlist + [action], nextact, actionNotBroken))
          else: 
            val = max(val, searchVal(child, depth, agent + 1, teammateIndex, teammateActions, numagents, alpha, beta, actionlist + [action], nextact, actionNotBroken))
          if val[0] > beta:
            return val
          alpha = max(val[0], alpha)
      elif agent == teammateIndex and agent != numagents:
        "looks at the agent list if it is up to date in the current chain of actions, else maximizes on expected value"
        if teammateActions:
          if nextact < len(teammateActions) and actionNotBroken:
            predepaction = teammateActions[nextact]    
            if predepaction in legalActions:     
              val = searchVal(gameState.generateSuccessor(agent, predepaction), depth, agent + 1, teammateIndex, teammateActions, numagents,alpha, beta, actionlist, nextact + 1, actionNotBroken)
              return val
            else:
              actionNotBroken = False
        randaction = random.choice(legalActions)
        val = searchVal(gameState.generateSuccessor(agent, randaction), depth, agent + 1, teammateIndex, teammateActions, numagents, alpha, beta, actionlist, nextact + 1, actionNotBroken)
      elif agent == teammateIndex and agent == numagents:
        "the case where the teammate is the last agent and we have to reset the agent counter"
        if teammateActions:
          if nextact < len(teammateActions) and actionNotBroken:
            predepaction = teammateActions[nextact]    
            if predepaction in legalActions:     
              val = searchVal(gameState.generateSuccessor(agent, predepaction), depth - 1, 0, teammateIndex, teammateActions, numagents,alpha, beta, actionlist, nextact + 1, actionNotBroken)
              return val
            else:
              actionNotBroken = False
        randaction = random.choice(legalActions)
        val = searchVal(gameState.generateSuccessor(agent, randaction), depth - 1, 0, teammateIndex, teammateActions, numagents, alpha, beta, actionlist, nextact + 1, actionNotBroken)
      else:
        "ghost actions currently minimized over them updated with alpha beta pruining"
        #legalActions = actionsWithoutReverse(legalActions, gameState, agent)
        ghostIndices = self.getOpponents(gameState)
        ghostPositions = [gameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
        ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
        ghostDistances.append( 1000 )
        closestGhost = min( ghostDistances ) + 1.0
        if closestGhost >= 2*depth:
          randaction =  random.choice(legalActions)
          if agent == numagents:
            val = searchVal(gameState.generateSuccessor(agent, randaction), depth - 1, 0, teammateIndex, teammateActions, numagents,alpha, beta, actionlist, nextact, actionNotBroken)
          else:
            val = searchVal(gameState.generateSuccessor(agent, randaction), depth, agent + 1, teammateIndex, teammateActions, numagents,alpha, beta, actionlist, nextact, actionNotBroken)
        else:
          val = (float('inf'), actionlist)
          for action in legalActions:
            child = gameState.generateSuccessor(agent, action)
            if agent == numagents:
              val = min(val, searchVal(child, depth - 1, 0,teammateIndex, teammateActions, numagents,alpha, beta, actionlist, nextact, actionNotBroken))
            else:
              val = min(val, searchVal(child, depth, agent + 1,teammateIndex, teammateActions, numagents,alpha, beta, actionlist, nextact, actionNotBroken))
            if val[0] < alpha:
              return val
            beta = min(val[0], beta)
      return val

    """ Finaly using search functions we will use the recursion to find the solution"""
    newPos = gameState.getAgentPosition(self.index)
    ghostIndices = self.getOpponents(gameState)
    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    numagents = len(ghostIndices) + 1
    teammateActions = self.receivedBroadcast
    legalActions = gameState.getLegalActions(self.index)
    legalActions = actionsWithoutStop(legalActions)
    depth = 2
    retaction = Directions.STOP
    alpha = -float("inf")
    beta =  float("inf")
    val = (-float("inf"), [])
    toBroadcast = None
    """Went in and added alpha beta pruining"""
    for action in legalActions:
      value, oldaction = val
      prevV = value
      child = gameState.generateSuccessor(self.index, action)
      if self.index == numagents:
        val = max(val, searchVal(child, depth, 0, teammateIndex, teammateActions, numagents,alpha, beta, [], 0, True))
      else:
        val = max(val, searchVal(child, depth, self.index + 1, teammateIndex, teammateActions, numagents,alpha, beta, [], 0, True))
      value, actionlist = val
      if value > prevV:
        retaction = action
        toBroadcast = actionlist
      if value >= beta:
        self.toBroadcast = actionlist
        return action
      alpha = max(value, alpha)
    self.toBroadcast = toBroadcast
    return retaction

  def isWin(self, gameState):
    """
    Returns if the current game state is a winning game state,
    needs to be updated based on the specificnes of the problem
    """
    food = gameState.getFood()
    foodList = food.asList()
    if not foodList:
      return True
    return False

  def evaluationFunction(self, gameState):
    """
    Returns maximum q value over all legalActions for the current gameState
    """
    legalActions = gameState.getLegalActions(self.index) 
    retaction = Directions.STOP
    qval = (-float("inf"), retaction)
    for action in legalActions:
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      qval = max(qval, (features * weights, action))
    return qval

  def getFeatures(self, gameState, action):
    """
    Uses given values in a state to come up with a heuristic on how "good" the state is
    """
    features = util.Counter()

    ### Useful information you can extract from a GameState (pacman.py) ###
    successorGameState = gameState.generateSuccessor(self.index, action)
    newPos = successorGameState.getAgentPosition(self.index)
    oldFood = gameState.getFood()
    newFood = successorGameState.getFood()
    ghostIndices = self.getOpponents(successorGameState)
    
    # Determines how many times the agent has already been in the newPosition in the last 20 moves
    numRepeats = sum([1 for x in self.observationHistory[-5:] if x.getAgentPosition(self.index) == newPos])

    foodPositions = oldFood.asList()
    numFood = len(foodPositions)
    foodDistances, foodclosest = min([(self.getMazeDistance(newPos, foodPosition), foodPosition) for foodPosition in foodPositions])
    foodPosition = foodclosest
    #lst = []
    #for foodpos in foodPositions:
    #    numClosest = 0
    #    for food in foodPositions:
    #        if self.getMazeDistance(food, foodpos) <= 3:
    #            numClosest += 1
    #    lst = lst + [(numClosest, foodpos)]
    #clusterclosest = min([(self.getMazeDistance(newPos, food) -1.9*numClosest) for numClosest, food in lst]) + 1
    closestFood = foodDistances +1

    ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
    ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append( 1000 )
    closestGhost = min( ghostDistances ) + 1.0
    ghostdist = 0
    if closestGhost < 8:
      if closestGhost != 0:
        ghostdist = 1/closestGhost

    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    teammatePos = successorGameState.getAgentPosition(teammateIndex)
    teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

    pacmanDeath = successorGameState.data.num_deaths
    stoppenalty = 0
    if action == 'Stop':
      stoppenalty = 1

    features['successorScore'] = self.getScore(successorGameState)

    # CHANGE YOUR FEATURES HERE
    features['closestFood'] = closestFood
    features['numFood'] = numFood
    features['closestGhost'] = ghostdist
    features['numRepeats'] = numRepeats
    features['teammateDistance'] = teammateDistance
    #features['numClosest'] = clusterclosest
    features['pacmanDeath'] = pacmanDeath
    features['stoppenalty'] = stoppenalty

    return features

  def getWeights(self, gameState, action):
    """
    Returns the weights for the features used to calcualte how good a state is 
    """
    return {'successorScore': 100,'stoppenalty': -100000, 'pacmanDeath':-1, 'closestGhost': -4, 'numClosest': 0, 'closestFood' : -1, 'numFood': -100, 'numRepeats': 0}



def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions