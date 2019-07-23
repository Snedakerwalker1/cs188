# myAgentP1.py
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


class myAgentP1(CaptureAgent):
  """
  Students' Names: Walker Snedaker 
            Phase Number: 1  
            Description of Bot: 
            After about four hours of work on 7/19/2018 I had my bot being hevily penalized
            for going away form the closest food especialy if that food was part of a large 
            cluster of food. I encouraged the bot to eat food by adding a large negative factor 
            based on the number of food remaining. And finaly I penalized the bot for repeatedly 
            going over the same spot. This yeiled an 8/10 with one of these being 

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
    Picks among the actions with the highest Q(s,a).
    """

    """ Idea is to head toward the closest food to
     me that is not the closest food to 
    the other agent i am "working" with"""
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values, action = max([(self.evaluate(gameState, a), a) for a in actions])
    return action
    print values
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    # INSERT YOUR LOGIC HERE

    return random.choice(actions)

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
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
    #closestFood = min(closestFood, clusterclosest)
    #closestPairlst = []
    #for food in foodPositions:
    #    dist = self.getMazeDistance(food, newPos)
    #    arr = []
    #    for food2 in foodPositions:
    #        if food2 != food:
    #            arr = arr + [self.getMazeDistance(food, food2)]
    #    if arr:
    #        dist += min(arr)
    #    closestPairlst = closestPairlst + [dist]
    #closestPair = 0
    #if closestPairlst:
    #    closestPair = min(closestPairlst)
    #closestFood = max(closestPair, closestFood)

    #foodPositions.remove(foodclosest)
    #mult = [1,1, 1, .9,.8,.7, .6, .5,.3,.1]
    #dist = 0
    #count = 0
    #while len(mult) > count and foodPositions:
    #    distance, food = min([(self.getMazeDistance(foodPosition, food), food) for food in foodPositions])
    #    foodPositions.remove(food)
    #    foodPosition = food
    #    dist += mult[count]*distance
    #    count += 1
    #foodDistances += dist


    ghostPositions = [successorGameState.getAgentPosition(ghostIndex) for ghostIndex in ghostIndices]
    ghostDistances = [self.getMazeDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append( 1000 )
    closestGhost = min( ghostDistances ) + 1.0

    teammateIndices = [index for index in self.getTeam(gameState) if index != self.index]
    assert len(teammateIndices) == 1, "Teammate indices: {}".format(self.getTeam(gameState))
    teammateIndex = teammateIndices[0]
    teammatePos = successorGameState.getAgentPosition(teammateIndex)
    teammateDistance = self.getMazeDistance(newPos, teammatePos) + 1.0

    pacmanDeath = successorGameState.data.num_deaths

    features['successorScore'] = self.getScore(successorGameState)

    # CHANGE YOUR FEATURES HERE
    features['numClosest'] = clusterclosest
    features['closestFood'] = closestFood
    features['numFood'] = numFood

    features['numRepeats'] = numRepeats

    features['teammateDistance'] = teammateDistance

    return features

  def getWeights(self, gameState, action):
    # CHANGE YOUR WEIGHTS HERE
    return {'successorScore': 100,'teammateDistance': 0,'numClosest': -1000, 'closestFood' : -1000, 'numFood': -1000, 'numRepeats': -100}