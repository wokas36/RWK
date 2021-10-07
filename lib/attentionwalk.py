import torch
import numpy as np
from attention_utils import feature_calculator, adjacency_opposite_calculator
from IPython.core.debugger import Tracer

class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see the paper.
    """
    def __init__(self, dimensions, epochs, window_size, num_of_walks, beta, gamma, learning_rate, shapes):
        """
        Setting up the layer.
        :param args: Arguments object.
        :param shapes: Shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        #Number of dimensions.
        self.dimensions = dimensions
        #Number of gradient descent iterations.
        self.epochs = epochs
        #Skip-gram window size.
        self.window_size = window_size
        #Number of random walks.
        self.num_of_walks = num_of_walks
        #Regularization parameter.
        self.beta = beta
        #Regularization parameter.
        self.gamma = gamma
        #Gradient descent learning rate.
        self.learning_rate = learning_rate
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        half_dim = int(self.dimensions/2)
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], half_dim))
        self.right_factors = torch.nn.Parameter(torch.Tensor(half_dim, self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0], 1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.uniform_(self.left_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.right_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.attention, -0.01, 0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor factorized.
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim=0)
        probs = self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_tensor = weighted_target_tensor * probs
        weighted_tar_mat = torch.sum(weighted_target_tensor, dim=0)
        weighted_tar_mat = weighted_tar_mat.view(self.shapes[1], self.shapes[2])
        estimate = torch.mm(self.left_factors, self.right_factors)
        #Tracer()()
        loss_on_target = - weighted_tar_mat* torch.log(torch.sigmoid(estimate))
        loss_opposite = -adjacency_opposite * torch.log(1-torch.sigmoid(estimate))
        loss_on_mat = self.num_of_walks*weighted_tar_mat.shape[0]*loss_on_target+loss_opposite
        abs_loss_on_mat = torch.abs(loss_on_mat)
        average_loss_on_mat = torch.mean(abs_loss_on_mat)
        norms = torch.mean(torch.abs(self.left_factors))+torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.beta * (self.attention.norm(2)**2)
        loss = average_loss_on_mat + loss_on_regularization + self.gamma*norms
        return loss

class AttentionWalkTrainer(object):
    """
    Class for training the AttentionWalk model.
    """
    def __init__(self, G, dimensions, epochs, window_size, num_of_walks, beta, gamma, learning_rate):
        """
        Initializing the training object.
        :param args: Arguments object.
        """
        #Number of dimensions.
        self.dimensions = dimensions
        #Number of gradient descent iterations.
        self.epochs = epochs
        #Skip-gram window size.
        self.window_size = window_size
        #Number of random walks.
        self.num_of_walks = num_of_walks
        #Regularization parameter.
        self.beta = beta
        #Regularization parameter.
        self.gamma = gamma
        #Gradient descent learning rate.
        self.learning_rate = learning_rate
        self.graph = G
        self.initialize_model_and_features()

    def initialize_model_and_features(self):
        """
        Creating data tensors and factroization model.
        """
        self.target_tensor = feature_calculator(self.window_size, self.graph)
        self.target_tensor = torch.FloatTensor(self.target_tensor)
        self.adjacency_opposite = adjacency_opposite_calculator(self.graph)
        self.adjacency_opposite = torch.FloatTensor(self.adjacency_opposite)
        self.model = AttentionWalkLayer(self.dimensions, self.epochs, self.window_size, self.num_of_walks, self.beta, 
                                        self.gamma, self.learning_rate, self.target_tensor.shape)
        
    def fit(self):
        """
        Fitting the model
        """
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.epochs = np.arange(self.epochs)
        for _ in self.epochs:
            self.optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            self.optimizer.step()

    def create_embedding(self):
        """
        Returning the embedding matrices as one unified embedding.
        """
        left = self.model.left_factors.detach().numpy()
        right = self.model.right_factors.detach().numpy().T
        embedding = np.concatenate([left, right], axis=1)
        return embedding