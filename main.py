from graph import Graph
from genetic_algorithm import GeneticAlgorithm

def main():
    filename = 'graph.txt'
    graph = Graph(filename)
    
    ga = GeneticAlgorithm(graph)
    best_path, best_distance = ga.run()
    
    print("Melhor caminho encontrado:", best_path)
    print("Distância do melhor caminho:", best_distance)

if __name__ == "__main__":
    main()
