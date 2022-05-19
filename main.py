import networkx as nx
import numpy as np

from trainer import Trainer

if __name__ == "__main__":
    adjlist = nx.read_adjlist("karate_club.adjlist", nodetype=int)
    adj = nx.to_numpy_array(adjlist)
    labels = np.loadtxt("karate_label.txt")

    dim = 4
    epoch = 100
    n_node = adj.shape[0]

    trainer = Trainer(dim, n_node, adj, epoch)
    trainer.train()
