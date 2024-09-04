import numpy as np

class Graph:
    def __init__(self, filename):
        self.adj_matrix = self.load_from_file(filename)
        self.num_nodes = len(self.adj_matrix)
    
    def load_from_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            matrix = [list(map(int, line.split())) for line in lines]
        return np.array(matrix)
    
    def get_distance(self, i, j):
        return self.adj_matrix[i][j]
