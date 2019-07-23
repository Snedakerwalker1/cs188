import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.variables = []
        """try1:  vals lr = .01, h = 10: loss = .041577
        try2: vals lr = .01, h = 50: loss = .026561
        try3: vals lr = .005, h = 50: loss = not good
        try4: vals lr = .01, h = 75: loss = .015910 lit
        """
        i = 1
        h = 150
        self.variables.append(nn.Variable(i,h))
        self.variables.append(nn.Variable(h))
        self.variables.append(nn.Variable(h, i))
        self.variables.append(nn.Variable(i))

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph(self.variables)
        input_x = nn.Input(graph, x)
        xw1 = nn.MatrixMultiply(graph, input_x, self.variables[0])
        sumxw1b1 = nn.MatrixVectorAdd(graph, xw1, self.variables[1])
        relu = nn.ReLU(graph, sumxw1b1)
        reluW2 = nn.MatrixMultiply(graph, relu, self.variables[2])
        sumRW2b2 = nn.MatrixVectorAdd(graph, reluW2, self.variables[3])

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, sumRW2b2, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            nodes = graph.get_nodes()
            lastnode = nodes[-1]
            out = graph.get_output(lastnode)
            return out

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.variables = []
        """for now just trying last times f(X) - f(-x) with the vals that yeilded a 
        sucsess on part 4"""
        """try1: vals lr = .01, h = 75: loss = was real bad
        try2: vals lr = .001, h = 10: loss = .17, started going down but then incresed
        try3: vals lr = .001, h = 50: loss = still started goin back up
        try4: vals lr = .0001, h = 5: loss = .119032 never strarted going up except at very begining 
        found the error: had a small typo in the add loss node seciton that was causing this whole problem
        try5: vals lr = .001, h = 75, loss: .057773
        try6: vals lr = .01, h = 75, loss: .003225 but for the last 3000 tests 
        the solution became slightly unstable and started oscilating back and fourth between 
        a loss of .008 and .001 range.       
        """
        i = 1
        h = 150
        self.variables.append(nn.Variable(i, h))
        self.variables.append(nn.Variable(h))
        self.variables.append(nn.Variable(h, i))
        self.variables.append(nn.Variable(i))

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph(self.variables)
        negative1 = -1*np.ones((1,1))
        input_x = nn.Input(graph, x)
        neg_1 = nn.Input(graph, negative1)
        """First we do the positives"""
        xw1 = nn.MatrixMultiply(graph, input_x, self.variables[0])
        sumxw1b1 = nn.MatrixVectorAdd(graph, xw1, self.variables[1])
        relu = nn.ReLU(graph, sumxw1b1)
        reluW2 = nn.MatrixMultiply(graph, relu, self.variables[2])
        """Now we do the negatives"""
        negx = nn.MatrixMultiply(graph, input_x, neg_1)
        nxw1 = nn.MatrixMultiply(graph, negx, self.variables[0])
        sumnxw1 = nn.MatrixVectorAdd(graph, nxw1, self.variables[1])
        nrelu = nn.ReLU(graph, sumnxw1)
        nreluW2 = nn.MatrixMultiply(graph, nrelu, self.variables[2])
        """Set the negative value of negative x to negative"""
        nsumNRW2b2 = nn.MatrixMultiply(graph, nreluW2, neg_1)
        """Add the two sums together"""
        totalSum = nn.Add(graph, reluW2, nsumNRW2b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, totalSum, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            nodes = graph.get_nodes()
            lastnode = nodes[-1]
            out = graph.get_output(lastnode)
            return out

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .8
        self.variables = []
        """Im a bit lazy so im jsut going to implement a version of question 4 see what
         happens, not working cant get above 93, so ima try the odd version see it thats better
         """
        """using initial values 
        try1: lr: .01, h = 75 got percent 75%
        try2: lr: .01, h = 100 got percent 74%
        try3: lr: .1, h = 75 got percent 90.24%
        try4: lr: .3, h = 75 got percent 92.69%
        try5: lr: .5, h = 75 got percent 93.61%
        try6: lr: .9, h = 75 got percent 93.79%
        try7: lr: 2, h= 75 got percent 92.92%
        try8: lr: .95, h = 50 got percent 93.4%
        now with no b2 added: 
        try9: lr: .9, h = 75 got percent 94.14%
        remove b1 see what we get
        try10: lr: .9, h = 75 get percent 93.98/94.20 at time step before. 
        ima try a 3 layer network maybe but im pretty tired so maybe tommorow. 
        try11: lr: .1, h - 75 using 3 layers now 75.19%
        try12: lr: .5, h = 75 gets 92.50%
        try13: lr: .4, h = 80 gets 93.28%
        try14: lr: .4, h = 85  gets 92.44
        try15: lr: .5, h = 80 gets 92.96
        changed the variables to allow more depenence on h in the last values
        try16: lr: .4, h =75  gets 93.67%
        lr:.4, h = 80 93.72%
        lr:.35, h=80, 93.69   
        lr:3 h = 80 93.18
        lr3 h100 93.75
        lr.325 h100 93.27
        lr.3 h150 93.81%
        lr.3 h200 94.56
        lr.3 h400 95.19
        lr .35 h400 95.42
        lr .35 h500 95.68
        lr.4 h500 96.04
        lr.4 h600 96.02
        lr.45 h600 96.08%
        lr.7 h400 96.25%
        lr.9 h400 96.63
        lr1, h400 96.55
        lr.9, h600 96.73
           
        """
        h = 200
        self.variables.append(nn.Variable(784, h))
        self.variables.append(nn.Variable(h))
        self.variables.append(nn.Variable(h, 10))
        self.variables.append(nn.Variable(10))
        #self.variables.append(nn.Variable(h,10))
        #self.variables.append(nn.Variable(10))


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph(self.variables)
        input_x = nn.Input(graph, x)
        xw1 = nn.MatrixMultiply(graph, input_x, self.variables[0])
        sumxw1b1 = nn.MatrixVectorAdd(graph, xw1, self.variables[1])
        relu = nn.ReLU(graph, sumxw1b1)
        reluW2 = nn.MatrixMultiply(graph, relu, self.variables[2])
        finalSum = nn.MatrixVectorAdd(graph, reluW2, self.variables[3])
        #relu2 = nn.ReLU(graph, sumRW2b2)
        #mul3 = nn.MatrixMultiply(graph, relu2, self.variables[4])
        #finalSum = nn.MatrixVectorAdd(graph, mul3, self.variables[5])

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, finalSum, input_y)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            nodes = graph.get_nodes()
            lastnode = nodes[-1]
            out = graph.get_output(lastnode)
            return out


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .01
        self.variables = []
        h = 20
        self.variables.append(nn.Variable(4, h))
        self.variables.append(nn.Variable(h))
        self.variables.append(nn.Variable(h, h))
        self.variables.append(nn.Variable(h))
        self.variables.append(nn.Variable(h,2))
        self.variables.append(nn.Variable(2))
        #self.variables.append(nn.Variable(1,2))
        #self.weightR = self.weightL


    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        """ After much considerartion I came up with a scheme to multiply the S node by a matrix that would split 
        appart its two commponents and onece doint the multiples with w and such """
        """identR = np.identity(4)
        identL = np.zeros_like(identR)
        identL = np.subtract(identL, identR)
        vec10 = np.ones((1,2))
        vec10[0,1] = 0
        vec01 = np.subtract(np.ones((1,2)), vec10)
        graph = nn.Graph([self.weightL])
        input_S = nn.Input(graph, states)
        input_idL = nn.Input(graph, identL)
        input_idR = nn.Input(graph, identR)
        input_vec10 = nn.Input(graph, vec10)
        input_vec01 = nn.Input(graph, vec01)"""
        """Now that we have the stuff set up lets deal with the left side"""
        """
        mulSL = nn.MatrixMultiply(graph, input_S, input_idL)
        mulSLW = nn.MatrixMultiply(graph, mulSL, self.weightL)
        Lmul = nn.MatrixMultiply(graph, mulSLW, input_vec10)"""
        """Now we do the same with the right """
        """mulSR = nn.MatrixMultiply(graph, input_S, input_idR)
        mulSRW = nn.MatrixMultiply(graph, mulSR, self.weightL)
        Rmul = nn.MatrixMultiply(graph, mulSRW, input_vec01)
        totalMul = nn.Add(graph, Lmul, Rmul)"""
        graph = nn.Graph(self.variables)
        input_x = nn.Input(graph, states)
        xw1 = nn.MatrixMultiply(graph, input_x, self.variables[0])
        sumxw1b1 = nn.MatrixVectorAdd(graph, xw1, self.variables[1])
        relu = nn.ReLU(graph, sumxw1b1)
        reluW2 = nn.MatrixMultiply(graph, relu, self.variables[2])
        sum2 = nn.MatrixVectorAdd(graph, reluW2, self.variables[3])
        relu2 = nn.ReLU(graph, sum2)
        relusW3 = nn.MatrixMultiply(graph, relu2, self.variables[4])
        totalSum = nn.MatrixVectorAdd(graph, relusW3, self.variables[5])

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            qt = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, qt, totalSum)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            nodes = graph.get_nodes()
            lastnode = nodes[-1]
            out = graph.get_output(lastnode)
            return out


    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .01
        self.variables = []
        self.h = 100
        self.variables.append(nn.Variable(self.h,self.h))
        self.variables.append(nn.Variable(self.h))
        self.variables.append(nn.Variable(self.h,self.h))
        self.variables.append(nn.Variable(self.num_chars, self.h))
        self.variables.append(nn.Variable(self.h))
        self.variables.append(nn.Variable(self.h, self.h))
        self.variables.append(nn.Variable(self.h, len(self.languages)))
        self.variables.append(nn.Variable(len(self.languages)))

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """

        "*** YOUR CODE HERE ***"
        batch_size = xs[0].shape[0]
        h = np.zeros((batch_size, self.h))
        graph = nn.Graph(self.variables)
        hnode = nn.Input(graph, h)
        for x in xs:
            input_x = nn.Input(graph, x)
            hmul = nn.MatrixMultiply(graph, hnode, self.variables[0])
            hsum = nn.MatrixVectorAdd(graph, hmul, self.variables[1])
            reluH = nn.ReLU(graph, hsum)
            reluHM = nn.MatrixMultiply(graph, reluH, self.variables[2])
            xmul = nn.MatrixMultiply(graph, input_x, self.variables[3])
            xsum = nn.MatrixVectorAdd(graph, xmul, self.variables[4])
            relux = nn.ReLU(graph, xsum)
            reluxm = nn.MatrixMultiply(graph, relux, self.variables[5])
            sumsum = nn.Add(graph, reluHM, reluxm)
            #mulmul = nn.MatrixMultiply(graph, sumsum, self.variables[5])
            hnode = nn.ReLU(graph, sumsum)

        finMul = nn.MatrixMultiply(graph, hnode, self.variables[6])
        totalSum = nn.MatrixVectorAdd(graph, finMul, self.variables[7])
        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, totalSum, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            nodes = graph.get_nodes()
            lastnode = nodes[-1]
            out = graph.get_output(lastnode)
            return out
