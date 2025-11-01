Task-1

from collections import deque

def create_graph():
    graph = {}
    n = int(input("Enter number of nodes: "))
    for i in range(n):
        node = input(f"Enter name of node {i + 1}: ")
        graph[node] = []
    
    e = int(input("Enter number of edges (paths): "))
    for i in range(e):
        src = input(f"Enter source node for edge {i + 1}: ")
        dest = input(f"Enter destination node for edge {i + 1}: ")
        graph[src].append(dest)
    
    return graph

def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    if node not in visited:
        print(f"Visited: {node}")
        visited.add(node)
        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(f"Visited: {node}")
            visited.add(node)
            queue.extend(graph[node])

def main():
    print("===== Graph Creation =====")
    graph = create_graph()

    print("\n===== Graph Structure =====")
    for node, neighbors in graph.items():
        print(f"{node} -> {neighbors}")

    start = input("\nEnter starting node for traversal: ")

    print("\n===== DFS Traversal =====")
    dfs(graph, start)

    print("\n===== BFS Traversal =====")
    bfs(graph, start)

if __name__ == "__main__":
    main()


Task-3

def aStarAlgo(start_node, stop_node):
    open_set = set([start_node])
    closed_set = set()
    g = {}
    parents = {}

    g[start_node] = 0
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n is None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v

        if n is None:
            print('Path does not exist!')
            return None

        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found:', path)
            return path

        for (m, weight) in get_neighbors(n):
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

        open_set.remove(n)
        closed_set.add(n)

    print('Path does not exist!')
    return None

def get_neighbors(v):
    return Graph_nodes.get(v, [])

def heuristic(n):
    h_dist = {
        'A': 11, 'B': 6, 'C': 5, 'D': 7, 'E': 3,
        'F': 6, 'G': 5, 'H': 3, 'I': 1, 'J': 0
    }
    return h_dist[n]

Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('A', 6), ('C', 3), ('D', 2)],
    'C': [('B', 3), ('D', 1), ('E', 5)],
    'D': [('B', 2), ('C', 1), ('E', 8)],
    'E': [('C', 5), ('D', 8), ('I', 5), ('J', 5)],
    'F': [('A', 3), ('G', 1), ('H', 7)],
    'G': [('F', 1), ('I', 3)],
    'H': [('F', 7), ('I', 2)],
    'I': [('E', 5), ('G', 3), ('H', 2), ('J', 3)],
}

print("Following is the A* Algorithm:")
aStarAlgo('A', 'J')


Task-4

def alpha_beta_pruning(depth, node_index, maximizing_player, values, alpha, beta):

    if depth == 0 or node_index >= len(values):
        return values[node_index]

    if maximizing_player:
        max_eval = float('-inf')
        for i in range(2):
            eval = alpha_beta_pruning(depth - 1, node_index * 2 + i, False, values, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break 
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(2):
            eval = alpha_beta_pruning(depth - 1, node_index * 2 + i, True, values, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  
        return min_eval


if __name__ == "__main__":
    values = [3, 5, 6, 9, 1, 2, 0, -1]
    depth = 3
    alpha = float('-inf')
    beta = float('inf')
    optimal_value = alpha_beta_pruning(depth, 0, True, values, alpha, beta)
    print(f"The optimal value is: {optimal_value}")
