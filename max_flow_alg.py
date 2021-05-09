from networkx import DiGraph, MultiDiGraph
from collections import deque
import time

FILE = 'test2.txt'


def generate_max_graph(n):
    G = DiGraph()
    G.add_nodes_from([i for i in range(n)])
    k = int((n - 2) / 2)
    for i in range(0, k):
        G.add_edge(i, i + 1, capacity=k-i+1, flow=0)
    for i in range(0, k+1):
        G.add_edge(i, k+1, capacity=1, flow=0)
    for i in range(k+1, n-1):
        G.add_edge(i, i+1, capacity=k+1, flow=0)
    G.add_edge(k, n-1, capacity=1, flow=0)
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
        self.G = MultiDiGraph()
        for node in Gc.nodes():
            self.G.add_node(node)
        for node, nbrsdict in Gc.adjacency():
            for nbr, dict in nbrsdict.items():
                capacity = dict.get('capacity')
                # dict.update({'flow': capacity, 'forward': True})
                self.G.add_edge(node, nbr, 'forward', capacity=capacity, flow=capacity, forward=True)
                self.G.add_edge(nbr, node, 'backward', capacity=capacity, flow=0, forward=False)
        # edges = tuple(G.edges())
        # for edge in edges:
        #     capacity = self.edge(edge[0], edge[1]).get('capacity')
        #     self.G.add_edge(edge[1], edge[0], capacity=capacity, flow=0, forward=False)

    def neighbours(self, u):
        return self.G.adj[u]

    def edge(self, u, v, d):
        return self.G.edges[u, v, d]

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
            info = self.G.get_edge_data(edge[0], edge[1])
            info = info.get('forward')
            if info and info.get('flow') != 0:
                Gc.add_edge(edge[1], edge[0])
        return Gc


def parse_file(file=FILE):  # Обработка файла, создание графа
    with open(file) as f:
        n, m = map(int, f.readline().split())
        G = DiGraph()
        G.add_nodes_from([i for i in range(n)])
        for line in f:
            node1, node2, capacity = map(int, line.split())
            G.add_edge(node1 - 1, node2 - 1, capacity=capacity)
        return G, n, m


def relabel(G, h, u, m):
    neighbours = []
    for node, dict in G.neighbours(u).items():
        for info in dict.values():
            if info.get('flow') != 0:
                neighbours.append(node)
    min = m
    for node in neighbours:
        if h[node] < min:
            min = h[node]
    h[u] = min+1


def push(G, e, u, v, direction):
    info = G.edge(u, v, direction)
    # Проверяем направление ребра в остаточной сети
    # if info.get('forward'):
    #     delta = min(e[u], info.get('capacity')-info.get('flow'))
    # else:
    delta = min(e[u], info.get('flow'))

    # Изменяем значения избытков, удаляем или добавляем прямые/обратные ребра в остаточной сеи при необходимости
    e[u] -= delta
    e[v] += delta
    G.edge(u, v, direction)['flow'] -= delta
    if direction == 'forward':
        G.edge(v, u, 'backward')['flow'] += delta
    else:
        G.edge(v, u, 'forward')['flow'] += delta
    # if G.has_edge(v, u):
    #     G.edge(v, u)['flow'] += delta
    # else:
    #     if info.get('forward'):
    #         G.add_edge(v, u, capacity=info.get('capacity'), flow=delta, forward=False)
    #     else:
    #         G.add_edge(v, u, capacity=info.get('capacity'), flow=delta, forward=True)
    # if delta == info.get('flow'):
    #     G.remove_edge(u, v)
    # else:
    #     G.edge(u, v)['flow'] -= delta


def max_flow(G, n, m):  # Алгоритм макс потока v0.02
    e = [0 for i in range(n)]  # Инициализация избытка, высоты и очереди
    h = [0 for i in range(n)]
    Gf = ResidualNetwork(G)
    queue = deque()

    neighbours = Gf.neighbours(0).keys()  # Проталкиваем поток по всем ребрам истока
    for node in neighbours:
        capacity = Gf.edge(0, node, 'forward')['capacity']
        Gf.edge(0, node, 'forward')['flow'] = 0
        Gf.edge(node, 0, 'backward')['flow'] = capacity
        e[node] = capacity
        e[0] -= capacity
        # Gf.add_edge(node, 0, capacity=capacity, flow=capacity, forward=False)
        queue.append(node)
    # list_neighbours = []
    # for node in neighbours:
    #     list_neighbours.append(node)
    # for node in list_neighbours:
    #     Gf.remove_edge(0, node)

    Gc = Gf.reversed_copy()
    d = bfs(Gc, n-1, n)
    h[0] = n
    for i in range(1, n):
        if d[i] != 0:
            h[i] = d[i]
    gr_counter = 0  # Счетчик для запуска bfs для global relabeling

    while len(queue) != 0:  # Алгоритм проталкивания предпотока
        cur = queue.popleft()
        has_neighbour_with_h = False
        while e[cur] != 0:
            if gr_counter >= m:  # Если выполнили m операций push\relabel выполняем global relabeling
                Gc = Gf.reversed_copy()
                d = bfs(Gc, 0, n)
                for i in range(1, n):
                    if d[i] != 0:
                        h[i] = h[0] + d[i]
                d = bfs(Gc, n-1, n)
                for i in range(1, n):
                    if d[i] != 0:
                        h[i] = d[i]
                gr_counter = 0
            neighbours = Gf.neighbours(cur)  # Переменные для сохранения соседа, у которого h(u) = h(v) + 1
            has_neighbour_with_h = False
            neighbour_with_h = 0
            e_in_neighbour = 0
            direction_to_neighbour = None
            for node, dict in neighbours.items():  # Поиск такого соседа
                for direction, info in dict.items():
                    if h[cur] == h[node] + 1 and info.get('flow') != 0:
                        has_neighbour_with_h = True
                        neighbour_with_h = node
                        e_in_neighbour = e[node]
                        direction_to_neighbour = direction
                        break
            if has_neighbour_with_h:  # Если сосед найден, делаем push
                push(Gf, e, cur, neighbour_with_h, direction_to_neighbour)
                gr_counter += 1
                if e_in_neighbour == 0 and neighbour_with_h not in (0, n-1):  # Добавляем соседва в очередь,
                    queue.append(neighbour_with_h)  # если его избыток был равен 0
            else:
                break
        if not has_neighbour_with_h:  # Если текущая вершина, не имеет соседей с h(v) = h(u) - 1, тогда
            relabel(Gf, h, cur, m)  # текущую вершину нужно поднять
            queue.append(cur)  # Добавляем текущую вершину в конец очереди
            gr_counter += 1
    return Gf, e[n-1]


def bfs(G, u, n):  # Поиск в ширину
    d = [0 for i in range(n)]
    visited = [False for i in range(n)]
    queue = deque()
    queue.append(u)
    visited[u] = True
    while len(queue) != 0:
        cur = queue.popleft()
        for node in G.adj[cur].keys():
            if not visited[node]:
                d[node] = d[cur] + 1
                try:
                    visited[node] = True
                except IndexError:
                    print(node, cur)
                queue.append(node)
    return d


def dfs(G, u, n):  # Поиск в глубину
    visited = [False for i in range(n)]
    reachable = [u]
    stack = deque()
    stack.append(u)
    visited[u] = True
    while len(stack) != 0:
        cur = stack.pop()
        for node, dict in G.neighbours(cur).items():
            for info in dict.values():
                if info.get('flow') != 0 and not visited[node]:
                    reachable.append(node)
                    visited[node] = True
                    stack.append(node)
    return reachable


if __name__ == '__main__':
    G, n, m = parse_file()
    # n = 10000
    # m = (n-1)**2 - 1
    # G = generate_max_graph(n)
    # print(G.size())
    # t0 = time.time()
    Gf, mf = max_flow(G, n, m)
    # print(time.time()-t0)
    print(dfs(Gf, 0, n))
    print(mf)

