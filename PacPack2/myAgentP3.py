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
      Description of Bot: Main first idea is to build a minimax or expecti max search 
      a couple of layers deep at each spot and use this to work with the teammate and 
      atempte to avoid the ghosts.
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

  def chooseAction(self, gameState):
    """
    takes the given state knowlege, like ghosts positions, teammate's planne action and such
    and uses these to run a short depth minimax search, will be updated to expecti max if 
    need be, with approximate q values found in the same sensce as the previous 2 parts
    """
    def searchVal(gameState, depth, agent, teammateIndex, teammateActions, numagents, actionlist = [], nextact = 0, actionNotBroken = True):
      if depth == 0 or self.isWin(gameState):
      "update for this thingy, need something that checks win conditions an lose conditions as well"
        return (self.evaluationFunction(gameState), actionlist)
      legalActions = gameState.getLegalActions(agent)
      if agent == self.index:
        if agent == numagents:
          val = max(searchVal(gameState.generateSuccessor(agent, action), depth, agent + 1, teammateIndex, teammateActions, numagents, actionlist + [action], nextact, actionNotBroken) for action in legalActions)
        else:
          val = max(searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0, teammateIndex, teammateActions, numagents, actionlist + [action], nextact, actionNotBroken) for action in legalActions)
      elif agent == teammateIndex and agent != numagents:
        "looks at the agent list if it is up to date in the current chain of actions, else maximizes on expected value"
        if nextact <= teammateActions and actionNotBroken:
          action = teammateActions[nextact]    
          if action in legalActions:     
            val = searchVal(gameState.generateSuccessor(agent, action), depth, agent + 1, teammateIndex, teammateActions, numagents, actionlist, nextact + 1, actionNotBroken)
          else:
            val = max(searchVal(gameState.generateSuccessor(agent, action), depth, agent + 1, teammateIndex, teammateActions, numagents, actionlist, nextact + 1, False) for action in legalActions)
        else:
          val = max(searchVal(gameState.generateSuccessor(agent, action), depth, agent + 1, teammateIndex, teammateActions, numagents, actionlist, nextact + 1, actionNotBroken) for action in legalActions)
      elif agent == teammateIndex and agent == numagents:
        "the case where the teammate is the last agent and we have to reset the agent counter"
        if nextact <= teammateActions and actionNotBroken:
          action = teammateActions[nextact]    
          if action in legalActions:     
            val = searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0, teammateIndex, teammateActions, numagents, actionlist, nextact + 1, actionNotBroken)
          else:
            val = max(searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0, teammateIndex, teammateActions, numagents, actionlist, nextact + 1, False) for action in legalActions)
        else:
          val = max(searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0, teammateIndex, teammateActions, numagents, actionlist, nextact + 1, actionNotBroken) for action in legalActions)
      elif agent == numagents:
        "ghost actions currently minimized over them may be updated in future versions"
        val = min(searchVal(gameState.generateSuccessor(agent, action), depth - 1, 0,teammateIndex, teammateActions, numagents, actionlist, nextact, actionNotBroken) for action in legalActions)
      else:
        val = min(searchVal(gameState.generateSuccessor(agent, action), depth , agent + 1, teammateIndex, teammateActions,  numagents, actionlist, nextact, actionNotBroken) for action in legalActions)
      return val

    """ Finaly using search functions we will use the recursion to find the solution"""
    ghostIndices = self.getOpponents(gameState)
    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    numagents = len(ghostIndices) + 2
    teammateActions = self.receivedBroadcast
    legalActions = gameState.getLegalActions(self.index)
    depth = max(3, len(teammateIndex))
    if self.index == numagents:
      val, actionlist = max(searchVal(gameState.generateSuccessor(self.index, action), depth, 0, teammateIndex, teammateActions, numagents, [], 0, True) for action in legalActions)
    else:
      val, actionlist = max(searchVal(gameState.generateSuccessor(self.index, action), depth, self.index + 1, teammateIndex, teammateActions, numagents, [], 0, True) for action in legalActions)
    self.toBroadcast = actionlist[1:]
    return actionlist[0]

  def isWin(self, gameState):
    """
    Returns if the current game state is a winning game state,
    needs to be updated based on the specificnes of the problem
    """
    food = gameState.getFood()
    foodList = food..asList()
    if not foodList:
      return True
    return False

  def evaluationFunction(self, gameState):
    """
    Returns maximum q value over all legalActions for the current gameState
    """
    legalActions = gameState.getLegalActions(self.index) 
    qval = -float("inf")
    for action in legalActions:
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      qval = max(qval, features * weights)
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
    numRepeats = sum([1 for x in self.observationHistory[-20:] if x.getAgentPosition(self.index) == newPos])

    foodPositions = oldFood.asList()
    numFood = len(foodPositions)
    foodDistances, foodclosest = min([(self.getMazeDistance(newPos, foodPosition), foodPosition) for foodPosition in foodPositions])
    foodPosition = foodclosest
    lst = []
    for foodpos in foodPositions:
        numClosest = 0
        for food in foodPositions:
            if self.getMazeDistance(food, foodpos) <= 3:
                numClosest += 1
        lst = lst + [(numClosest, foodpos)]
    clusterclosest = min([(self.getMazeDistance(newPos, food) -1.9*numClosest) for numClosest, food in lst]) + 1
    closestFood = foodDistances +1

    ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
    ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append( 1000 )
    closestGhost = min( ghostDistances ) + 1.0
    ghostdist = 0
    if closestGhost < 5:
      ghostdist = closestGhost

    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    teammatePos = successorGameState.getAgentPosition(teammateIndex)
    teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

    pacmanDeath = successorGameState.data.num_deaths

    features['successorScore'] = self.getScore(successorGameState)

    # CHANGE YOUR FEATURES HERE
    features['closestFood'] = closestFood
    features['numFood'] = numFood
    features['closestGhost'] = ghostdist
    features['numRepeats'] = numRepeats
    features['teammateDistance'] = teammateDistance
    features['numClosest'] = clusterclosest
    features['pacmanDeath'] = pacmanDeath

    return features

  def getWeights(self, gameState, action):
    """
    Returns the weights for the features used to calcualte how good a state is 
    """
    return {'successorScore': 100,'teammateDistance': 0,'pacmanDeath':-1, 'closestGhost': 4000, 'numClosest': -1000, 'closestFood' : -1000, 'numFood': -1000, 'numRepeats': -100}



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