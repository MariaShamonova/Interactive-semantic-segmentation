from networkx import DiGraph
from collections import deque


FILE = 'test1.txt'


def parse_file(file=FILE):
    with open(file) as f:
        n, m = map(int, f.readline().split())
        G = DiGraph()
        G.add_nodes_from([i+1 for i in range(n)])
        for line in f:
            node1, node2, capacity = map(int, line.split())
            G.add_edge(node1, node2, capacity=capacity, flow=0, forward=True)
        return n, m, G


def relabel(G, h, u, m):
    neighbours = G.adj[u].keys()
    min = m
    for node in neighbours:
        if h[node-1] < min:
            min = h[node-1]
    h[u-1] = min+1


def push(G, e, u, v):
    info = G.edges[u, v]
    if info.get('forward'):  # Проверяем направление ребра в остаточной сети
        delta = min(e[u-1], info.get('capacity')-info.get('flow'))
    else:
        delta = min(e[u - 1], info.get('flow'))

    # Изменяем значения избытков, удаляем или добавляем прямые/обратные ребра в остаточной сеи при необходимости
    e[u - 1] -= delta
    e[v - 1] += delta
    if G.has_edge(v, u):
        G.edges[v, u]['flow'] += delta
    else:
        G.add_edge(v, u, capacity=info.get('capacity'), flow=delta, forward=False)
    if delta == info.get('capacity') - info.get('flow'):
        G.remove_edge(u, v)
    else:
        G.edges[u, v]['flow'] -= delta


def max_flow(G, n, m):  # Алгоритм макс потока v0.01
    e = [0 for i in range(n)]  # Начальные условия
    h = [0 for i in range(n)]
    h[0] = n
    queue = deque()
    neighbours = G.adj[1].keys()
    for node in neighbours:
        capacity = G.edges[1, node]['capacity']
        e[node-1] = capacity
        G.add_edge(node, 1, capacity=capacity, flow=capacity, forward=False)
        queue.append(node)
    list_neighbours = []
    for node in neighbours:
        list_neighbours.append(node)
    for node in list_neighbours:
        G.remove_edge(1, node)
    prev = 0  # Переменная для хранения предыдущей вершины
    while len(queue) != 0 or prev != 0:
        if prev == 0:  # Если избыток в предыдущей вершине равен 0, то рассматриваем новую из очереди
            cur = queue.popleft()
        else:
            cur = prev
        neighbours = G.adj[cur]  # Переменные для сохранения соседа, у которого h(u) = h(v) + 1
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
            push(G, e, cur, neighbour_with_h)
            if e[cur-1] == 0:  # Обнуляем значение в prev, если избыток в текущем узле равен 0
                prev = 0
            else:
                prev = cur
            if e_in_neighbour == 0 and neighbour_with_h not in (1, n):  # Добавляем соседва в очередь,
                queue.append(neighbour_with_h)  # если его избыток был равен 0
        else:
            relabel(G, h, cur, m)
            prev = 0  # Обнуляем значение в prev, если для текущего узла выполняем relabel
            queue.append(cur)  # Добавляем текущую вершину в очередь
    return e[n-1]


if __name__ == '__main__':
    n, m, G = parse_file()
    Gf = DiGraph(G)
    print(max_flow(G, n, m))

