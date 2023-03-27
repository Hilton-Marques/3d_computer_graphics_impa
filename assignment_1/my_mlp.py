import torch
import math
import networkx as nx # to plot networks
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand
from torch import nn


class Node:
	def __init__(self, parents=[], value = None, is_updatable = False, label = None, color = '#5d9b9b'):
		self.parents = parents #node parents
		self.value = value #save function evaluates through the graph
		self.grad = 0 #save gradients through the graph
		self.visited = False #for DFS
		self.is_updatable = is_updatable #for update the parameters
		self.label = label # label for graph plot
		self.color = color # color for graph plot
	def eval(self):
		raise NotImplementedError
	def update_derivatives(self):
		raise NotImplementedError
class Const(Node):
	def __init__(self, parents = [], value = None, is_updatable = False, label = None, color = '#5d9b9b'):
		Node.__init__(self, parents, value, is_updatable, label, color)
	def eval(self):
		pass
	def update_derivatives(self):
		pass
class Linear(Node):
	def __init__(self, in_features, out_features):        
		k = 1/math.sqrt(in_features)       
		torch.manual_seed(1)
		weights = torch.FloatTensor(out_features, in_features).uniform_(-k, k)
		torch.manual_seed(1)
		bias = torch.FloatTensor(out_features).uniform_(-k,k)
		w = Const(value = weights, is_updatable = True, label = "$W")
		b = Const(value = bias,  is_updatable = True, label = "$b")
		Node.__init__(self, [w, b], label="$L", color="#FFC296")
	def eval(self):
		input = self.parents[2].value
		w = self.parents[0].value
		b = self.parents[1].value
		self.value =  torch.mm(input, torch.transpose(w,0,1)) + b
	def update_derivatives(self):
		input = self.parents[2].value
		w = self.parents[0].value
		b = self.parents[1].value
		grad = self.grad		
		self.parents[0].grad += torch.mm(torch.transpose(grad,0,1), input) #update grad weights
		self.parents[1].grad += torch.sum(grad, axis = 0) # update grad bias
		self.parents[2].grad += torch.mm(grad, w) #update grad inputs

class Sigmoid(Node):
	def __init__(self):
		Node.__init__(self,[],label="$\sigma",color="#C48DF3")
	def eval(self):
		input = self.parents[0].value
		self.value = 1 / (1 + torch.exp(-input))
	def update_derivatives(self):        
		self.eval()
		s = self.value
		d = s * (1 - s)
		self.parents[0].grad += d * self.grad

class MSE(Node):
	def __init__(self, model):
		Y = Const(label = "$Y$", color='#FF6961')
		Node.__init__(self, [model, Y])
		self.label = "$C(\\theta)$"
		self.color = "#F49AC2"
	def eval(self, pred, y): 
		self.parents[1].value = y
		n = pred.size(0)
		diff = pred - y
		return torch.sum(torch.pow(diff, 2))/n
	def update_derivatives(self):
		n = self.parents[0].value.size(0)
		diff = self.parents[0].value - self.parents[1].value
		self.parents[0].grad =  2*diff/n		
	def backward(self):
		self.update_derivatives()
		for node in self.parents[0].nodes_ordered:
			node.update_derivatives()
	def show(self):
		nodes = self.parents[0].nodes_ordered.copy()
		nodes.insert(0,self)
		nodes.insert(0,self.parents[1])
		n_nodes = len(nodes)
		layers = (n_nodes - 2)//4
		adj_list = [[] for i in range(n_nodes)]
		labels = {}
		labels_dict = {"$L": layers,"$\sigma":layers - 1}
		colors = []
		for i in range(n_nodes):
			colors.append(nodes[i].color)
			name = nodes[i].label
			if name in labels_dict:
				labels_dict[name] -= 1
				name = name + "_{" + str(labels_dict[name]) + "}" + "$"
			if name[1] == 'L':
				nodes[i].parents[0].label = nodes[i].parents[0].label + "_{" + name[4] + "}" + "$"
				nodes[i].parents[1].label = nodes[i].parents[1].label + "_{" + name[4] + "}" + "$"
			labels[i] = name
			for parent in nodes[i].parents:
				for j in range(n_nodes):
					if (nodes[j] == parent):
						break
				adj_list[j].append(i)
		adj_dict = {i: adj_list[i] for i in range(len(adj_list))}
		G = nx.DiGraph(adj_dict)
		for layer, nodes in enumerate(nx.topological_generations(G)):    
			for node in nodes:
				G.nodes[node]["layer"] = layer
		pos = nx.multipartite_layout(G, subset_key="layer")
		nx.draw(G,pos,labels=labels, node_color = colors, edge_color="tab:green", node_size=450)
		plt.savefig('computational_graph.png', bbox_inches='tight')
	
class SGD:
	def __init__(self, trainable_nodes, learning_rate = 0.001):
		self.trainable_nodes = trainable_nodes
		self.learning_rate = learning_rate
	def step(self):
		for node in self.trainable_nodes:
			node.value += -self.learning_rate * node.grad

class GraphNetwork:
	def __init__(self, *args):
		n_nodes = len(args)
		self.layers = args
		self.last_node = args[-1]
  		#update graph adjacency list
		for i in range(1 , n_nodes):
			args[i].parents.append(args[i-1])
		#create a X layer and a Y layer
		self.X = Const(label="$X$", color='#FF6961')
		self.layers[0].parents.append(self.X) # create edge to first layer
		#sort the nodes in topological order for backpropagation
		nodes_ordered = []
		self.topological_sort(self.last_node, nodes_ordered)
		self.last_node.nodes_ordered = nodes_ordered
	def topological_sort(self, node, list):
		node.visited = True
		if node.parents:
			for parent in node.parents:
				if not parent.visited:
					self.topological_sort(parent, list)
		list.insert(0,node)
	def eval(self, X):
		self.X.value = X
		for node in self.last_node.nodes_ordered:
			node.is_visited = False
		self.forward(self.last_node)
		return self.last_node.value
	def backward(self):
		self.zero_grad()
		for node in self.last_node.nodes_ordered:
			self.update_derivatives()
	def zero_grad(self):
		for node in self.last_node.nodes_ordered:
			node.grad = 0
	def parameters(self):
		parameters = []
		for node in self.last_node.nodes_ordered:
			if node.is_updatable:
				parameters.append(node)
		return parameters
	def forward(self, node):
		if node.is_visited:
			return
		node.is_visited = True
		if node.parents:
			for parent in node.parents:
				self.forward(parent)
		node.eval()
  
# This is a Dataset class to work with PyTorch
class AlgebraicDataset(Dataset):
  '''Abstraction for a dataset of a 1D function'''

  def __init__(self, f, interval, nsamples):
    X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
    self.data = [(x, f(x)) for x in X]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

class MultiLayerNetwork(nn.Module):
  def __init__(self):
    super().__init__()    
    self.layers = nn.Sequential(
        nn.Linear(1, 128),
        nn.Sigmoid(),
        nn.Linear(128,64),
        nn.Sigmoid(),
        nn.Linear(64, 32),
    	nn.Sigmoid(),
        nn.Linear(32,8),
        nn.Sigmoid(),
        nn.Linear(8,1),
    )

  def forward(self, x):
    return self.layers(x)

#build computational graph
model = GraphNetwork(
 	Linear(1, 128),
    Sigmoid(),
    Linear(128,64),
    Sigmoid(),
    Linear(64, 32),
    Sigmoid(),
    Linear(32,8),
    Sigmoid(),
    Linear(8,1),
)

#build loss function
loss = MSE(model.last_node)
#loss.show()
#build optmizer 
optimizer = SGD(model.parameters(), learning_rate=1e-3)
#Load dataset
model2 = MultiLayerNetwork()
lossfunc2 = nn.MSELoss()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)

line = lambda x: 2*x + 3

interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100
train_dataset = AlgebraicDataset(line, interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_nsamples, shuffle=True)

#train 

def train(model, dataloader, lossfunc, optimizer, model2, lossfunc2, optimizer2):

  cumloss = 0.0
  for X, y in dataloader:
    X = X.unsqueeze(1).float()
    y = y.unsqueeze(1).float()

    pred = model.eval(X)
    pred2 = model2.forward(X)
    loss = lossfunc.eval(pred, y)
    loss2 = lossfunc2(pred2, y)

    # we need to "clean" the accumulated gradients
    optimizer2.zero_grad()
    model.zero_grad()
    # computes gradients
    lossfunc.backward()
    loss2.backward()
    #check the gradient value
    check = model2.layers[-3].bias.grad - model.layers[-3].parents[1].grad
    # updates parameters going in the direction that decreases the local error
    optimizer.step()
    optimizer2.step()

    # loss is a tensor so we use *item* to get the underlying float value
    cumloss += loss.item() 
  
  return cumloss / len(dataloader)


epochs = 1001

# Let''s make a Gif of the training
filename_output = "./line_approximation.gif"

for t in range(epochs):
  train_loss = train(model, train_dataloader, loss, optimizer, model2, lossfunc2, optimizer2)