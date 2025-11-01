
graph = {
    "a":[["b",2]],
    "b":[["c",3]],
    "c":[["d",1]],
    "d":[["a",4]],
    }   
total_cost = 0
visited_edges=set()

print("costs of all edges: ")
for node, edges in graph.items():
    for neighbor,cost in edges:
            print(f"Edge from {node} to {neighbor} has cost: {cost}")

for node, edges in graph.items():
    for neighbor, cost in edges:
        # by sorting the nodes in the edge to create a unique identifier
        edge_identifier = tuple(sorted((node, neighbor)))
        if edge_identifier not in visited_edges:
            total_cost += cost
            visited_edges.add(edge_identifier)

print(f"Total cost of all edges: {total_cost}")
