from networkx import DiGraph
from collections import deque
import time


FILE = 'test.txt'


def generate_max_graph(n):
    G = DiGraph()
    G.add_nodes_from([i+1 for i in range(n)])
    k = int((n - 2) / 2)
    for i in range(1, k+1):
        G.add_edge(i, i + 1, capacity=k-i+2, flow=0)
    for i in range(1, k+2):
        G.add_edge(i, k+2, capacity=1, flow=0)
    for i in range(k+2, n):
        G.add_edge(i, i+1, capacity=k+1, flow=0)
    G.add_edge(k+1, n, capacity=1, flow=0)
    # for i in range(2, n):
    #     for j in range(2, n):
    #         if i != j:
    #             G.add_edge(i, j, capacity=100, flow=0)
    # for i in range(2, n):
    #     G.add_edge(1, i, capacity=10000, flow=0)
    # for i in range(3, n):
    #     G.add_edge(i, n, capacity=100, flow=0)
    # G.add_edge(2, n, capacity=10000, flow=0)
    return G


class ResidualNetwork(object):
    def __init__(self, Gc):
        self.G = Gc.copy()
        for node in Gc.nodes():
            self.G.add_node(node)
        for node, nbrsdict in self.G.adjacency():
            for nbr, dict in nbrsdict.items():
                dict.update({'flow': 0, 'forward': True})

    def neighbours(self, u):
        return self.G.adj[u]

    def edge(self, u, v):
        return self.G.edges[u, v]

    def has_edge(self, u, v):
        return self.G.has_edge(u, v)

    def add_edge(self, u, v, **kwargs):
        self.G.add_edge(u, v, **kwargs)

    def remove_edge(self, u, v):
        self.G.remove_edge(u, v)

    def reversed_copy(self):
        Gc = DiGraph()
        for node in self.G.nodes():
            Gc.add_node(node)
        for edge in self.G.edges():
            info = self.G.edges[edge[0], edge[1]]
            if info.get('forward') and info.get('flow') < info.get('capacity'):
                Gc.add_edge(edge[1], edge[0])
        return Gc


def parse_file(file=FILE):
    with open(file) as f:
        n, m = map(int, f.readline().split())
        G = DiGraph()
        G.add_nodes_from([i+1 for i in range(n)])
        for line in f:
            node1, node2, capacity = map(int, line.split())
            G.add_edge(node1, node2, capacity=capacity)
        return n, m, G


def relabel(G, h, u, m):
    neighbours = G.neighbours(u).keys()
    min = m
    for node in neighbours:
        if h[node-1] < min:
            min = h[node-1]
    h[u-1] = min+1


def push(G, e, u, v):
    info = G.edge(u, v)
    if info.get('forward'):  # Проверяем направление ребра в остаточной сети
        delta = min(e[u-1], info.get('capacity')-info.get('flow'))
    else:
        delta = min(e[u - 1], info.get('flow'))

    # Изменяем значения избытков, удаляем или добавляем прямые/обратные ребра в остаточной сеи при необходимости
    e[u - 1] -= delta
    e[v - 1] += delta
    if G.has_edge(v, u):
        G.edge(v, u)['flow'] += delta
    else:
        G.add_edge(v, u, capacity=info.get('capacity'), flow=delta, forward=False)
    if delta == info.get('capacity') - info.get('flow'):
        G.remove_edge(u, v)
    else:
        G.edge(u, v)['flow'] -= delta


def max_flow(G, n, m):  # Алгоритм макс потока v0.02
    e = [0 for i in range(n)]  # Начальные условия
    h = [0 for i in range(n)]
    Gf = ResidualNetwork(G)
    h[0] = n
    queue = deque()

    neighbours = Gf.neighbours(1).keys()
    for node in neighbours:
        capacity = Gf.edge(1, node)['capacity']
        e[node-1] = capacity
        Gf.add_edge(node, 1, capacity=capacity, flow=capacity, forward=False)
        queue.append(node)
    list_neighbours = []
    for node in neighbours:
        list_neighbours.append(node)
    for node in list_neighbours:
        Gf.remove_edge(1, node)

    Gc = Gf.reversed_copy()
    d = bfs(Gc, n, n)
    for i in range(1, n):
        if d[i] != 0:
            h[i] = d[i]
    gr_counter = 0  # Счетчик для запуска bfs

    while len(queue) != 0:
        cur = queue.popleft()
        has_neighbour_with_h = False
        while e[cur - 1] != 0:
            if gr_counter >= m:
                Gc = Gf.reversed_copy()
                d = bfs(Gc, 1, n)
                for i in range(1, n):
                    if d[i] != 0:
                        h[i] = h[0] + d[i]
                d = bfs(Gc, n, n)
                for i in range(1, n):
                    if d[i] != 0:
                        h[i] = d[i]
                gr_counter = 0
            neighbours = Gf.neighbours(cur)  # Переменные для сохранения соседа, у которого h(u) = h(v) + 1
            has_neighbour_with_h = False
            neighbour_with_h = 0
            e_in_neighbour = 0
            for node in neighbours.keys():  # Поиск такого соседа
                if h[cur-1] == h[node-1] + 1:
                    has_neighbour_with_h = True
                    neighbour_with_h = node
                    e_in_neighbour = e[node-1]
                    break
            if has_neighbour_with_h:  # Если сосед найден, делаем push, иначе relabel
                push(Gf, e, cur, neighbour_with_h)
                gr_counter += 1
                if e_in_neighbour == 0 and neighbour_with_h not in (1, n):  # Добавляем соседва в очередь,
                    queue.append(neighbour_with_h)  # если его избыток был равен 0
            else:
                break
        if not has_neighbour_with_h:
            relabel(Gf, h, cur, m)
            queue.append(cur)  # Добавляем текущую вершину в очередь
            gr_counter += 1
    return e[n-1]


def bfs(G, u, n):
    d = [0 for i in range(n)]
    visited = [False for i in range(n)]
    queue = deque()
    queue.append(u)
    visited[u-1] = True
    while len(queue) != 0:
        cur = queue.popleft()
        for node in G.adj[cur].keys():
            if not visited[node-1]:
                d[node-1] = d[cur-1] + 1
                visited[node-1] = True
                queue.append(node)
    return d


def dfs(G, u, n):
    visited = [False for i in range(n)]
    reachable = []
    stack = deque()
    stack.append(u)
    visited[u-1] = True
    while len(stack) != 0:
        cur = stack.pop()
        for node in G.adj[cur].keys():
            if not visited[node-1]:
                reachable.append(node)
                visited[node-1] = True
                stack.append(node)
    return reachable


if __name__ == '__main__':
    # n, m, G = parse_file()
    n = 10000
    m = (n-1)**2 - 1
    G = generate_max_graph(n)
    print(G.size())
    t0 = time.time()
    mf = max_flow(G, n, m)
    t1 = time.time()
    print(mf)
    print(t1-t0)

