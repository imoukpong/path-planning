import numpy as np
import heapq
from collections import deque
from .graph import Cell
from .utils import trace_path


def depth_first_search(graph, start, goal):
    graph.init_graph()
    stack = [start]
    seen = set([(start.i, start.j)])
    graph.parent[(start.i, start.j)] = None

    while stack:
        node = stack.pop()
        graph.visited_cells.append(Cell(node.i, node.j))

        if node.i == goal.i and node.j == goal.j:
            return trace_path(goal, graph)

        for nxt in graph.find_neighbors(node.i, node.j):
            key = (nxt.i, nxt.j)
            if key not in seen:
                seen.add(key)
                graph.parent[key] = Cell(node.i, node.j)
                stack.append(nxt)

    return []


def breadth_first_search(graph, start, goal):
    graph.init_graph()
    q = deque([start])
    seen = {(start.i, start.j)}
    graph.parent[(start.i, start.j)] = None

    while q:
        node = q.popleft()
        graph.visited_cells.append(Cell(node.i, node.j))

        if node.i == goal.i and node.j == goal.j:
            return trace_path(goal, graph)

        for nxt in graph.find_neighbors(node.i, node.j):
            key = (nxt.i, nxt.j)
            if key not in seen:
                seen.add(key)
                graph.parent[key] = Cell(node.i, node.j)
                q.append(nxt)

    return []


def a_star_search(graph, start, goal):
    graph.init_graph()

    def heuristic(c):
        return ((c.i - goal.i) ** 2 + (c.j - goal.j) ** 2) ** 0.5

    graph.parent[(start.i, start.j)] = None
    graph.distance[(start.i, start.j)] = 0

    idx = 0
    pq = [(heuristic(start), 0, idx, start)]
    visited = set()

    while pq:
        f, g, _, node = heapq.heappop(pq)
        key = (node.i, node.j)

        if key in visited:
            continue

        visited.add(key)
        graph.visited_cells.append(Cell(node.i, node.j))

        if node.i == goal.i and node.j == goal.j:
            return trace_path(goal, graph)

        for nxt in graph.find_neighbors(node.i, node.j):
            nk = (nxt.i, nxt.j)
            cost = g + 1

            if nk not in visited and (nk not in graph.distance or cost < graph.distance[nk]):
                graph.distance[nk] = cost
                graph.parent[nk] = Cell(node.i, node.j)
                idx += 1
                heapq.heappush(pq, (cost + heuristic(nxt), cost, idx, nxt))

    return []
