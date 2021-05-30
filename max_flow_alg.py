from networkx import DiGraph
from collections import deque

FILE = 'test2.txt'


class ResidualNetwork(object):
    def __init__(self, Gc):
        self.capacity = []
        self.flow = []
        n = Gc.number_of_nodes()
        self.n = n
        for node in range(n):
            self.capacity.append({})
            self.flow.append({})
        for node, nbrsdict in Gc.adjacency():
            for nbr, dict in nbrsdict.items():
                capacity = dict.get('capacity')
                self.capacity[node].update({nbr: capacity})
                if self.flow[node].get(nbr):
                    self.flow[node][nbr].update({'forward': capacity})
                else:
                    self.flow[node].update({nbr: {'forward': capacity}})
        self.s_nbrs = []
        for nbr in self.neighbours(0):
            self.s_nbrs.append(nbr)

    def add_edge(self, u, v, d, flow):
        if self.flow[u].get(v) is None:
            self.flow[u].update({v: {d: flow}})
        else:
            self.flow[u].get(v).update({d: flow})

    def remove_edge(self, u, v, d):
        if len(self.flow[u].get(v)) > 1:
            self.flow[u].get(v).pop(d)
        else:
            self.flow[u].pop(v)

    def number_of_nodes(self):
        return self.n

    def neighbours(self, u):
        return self.flow[u]

    # def edge(self, u, v, d):
    #     return self.G.edges[u, v, d]

    def change_capacity(self, u, v, c):
        old_capacity = self.get_capacity(u, v, 'forward')
        self.capacity[u].update({v: c})
        add_flow = c - old_capacity
        if self.has_edge(u, v, 'forward'):
            self.update_flow(u, v, 'forward', add_flow, True)
        else:
            self.add_edge(u, v, 'forward', add_flow)

    def get_capacity(self, u, v, d):
        if d == 'forward':
            return self.capacity[u].get(v)
        else:
            return self.capacity[v].get(u)

    def get_flow(self, u, v, d):
        return self.flow[u].get(v).get(d)

    def update_flow(self, u, v, d, delta, plus):
        flow = self.get_flow(u, v, d)
        if plus:
            self.flow[u].get(v).update({d: flow + delta})
        else:
            self.flow[u].get(v).update({d: flow - delta})

    def has_edge(self, u, v, d):
        return True if self.flow[u].get(v) and self.flow[u].get(v).get(d) else False

    def direct_bfs_for_s_fast(self):  # Используетс в fast
        graph = [{} for i in range(self.n)]
        for node in range(self.n):
            for nbr, dict in self.neighbours(node).items():
                for direction, flow in dict.items():
                    capacity = self.get_capacity(node, nbr, direction)
                    # if direction == 'forward' and nbr != 0:
                    #     graph[node].update({nbr: 0})
                    if direction == 'backward' and nbr != 0:
                        graph[nbr].update({node: 0})
        d = [0 for i in range(self.n)]
        visited = [False for i in range(self.n)]
        visited_queue = deque()
        queue = deque()
        visited[self.n-1] = True
        for nbr in graph[self.n-1].keys():
            if nbr != 0:
                d[nbr] = 1
                visited[nbr] = True
                queue.append(nbr)
                visited_queue.append(nbr)
        while len(queue) != 0:
            cur = queue.popleft()
            for nbr in graph[cur].keys():
                if not visited[nbr] and nbr != 0:
                    d[nbr] = d[cur] + 1
                    visited[nbr] = True
                    queue.append(nbr)
                    visited_queue.append(nbr)
        return d, visited_queue, visited

    def direct_bfs_for_s(self):  # Используется
        d = [0 for i in range(self.n)]
        visited = [False for i in range(self.n)]
        visited_queue = deque()
        queue = deque()
        visited[0] = True
        for nbr in self.s_nbrs:
            if self.flow[nbr].get(0) is not None and nbr != self.n-1:
                d[nbr] = 1
                visited[nbr] = True
                queue.append(nbr)
                visited_queue.append(nbr)
        while len(queue) != 0:
            cur = queue.popleft()
            for nbr, dict in self.neighbours(cur).items():
                for direction, flow in dict.items():
                    # capacity = self.get_capacity(cur, nbr, direction)
                    if direction == 'forward' and nbr != self.n-1 \
                            and not visited[nbr]:
                        d[nbr] = d[cur] + 1
                        visited[nbr] = True
                        queue.append(nbr)
                        visited_queue.append(nbr)
        return d, visited_queue, visited

    def reversed_bfs_for_s(self):  # Не используется
        d = [0 for i in range(self.n)]
        visited = [False for i in range(self.n)]
        visited_queue = deque()
        queue = deque()
        visited[0] = True
        for nbr, dict in self.neighbours(0).items():
            for direction, flow in dict.items():
                capacity = self.get_capacity(0, nbr, direction)
                if direction == 'backward' and flow != capacity and nbr != self.n-1:
                    d[nbr] = 1
                    visited[nbr] = True
                    queue.append(nbr)
                    visited_queue.append(nbr)
        while len(queue) != 0:
            cur = queue.popleft()
            for nbr, dict in self.neighbours(cur).items():
                for direction, flow in dict.items():
                    capacity = self.get_capacity(cur, nbr, direction)
                    if direction == 'backward' and flow != capacity and nbr != self.n-1 \
                            and not visited[nbr]:
                        d[nbr] = d[cur] + 1
                        visited[nbr] = True
                        queue.append(nbr)
                        visited_queue.append(nbr)
        return d, visited_queue, visited

    def reversed_bfs_for_t(self):
        graph = [{} for i in range(self.n)]
        for node in range(self.n):
            for nbr, dict in self.neighbours(node).items():
                for direction, flow in dict.items():
                    if direction == 'forward' and nbr != 0:
                        graph[nbr].update({node: 0})
        d = [0 for i in range(self.n)]
        visited = [False for i in range(self.n)]
        visited_queue = deque()
        queue = deque()
        visited[self.n-1] = True
        for nbr in graph[self.n-1].keys():
            if nbr != 0:
                d[nbr] = 1
                visited[nbr] = True
                queue.append(nbr)
                visited_queue.append(nbr)
        while len(queue) != 0:
            cur = queue.popleft()
            for nbr in graph[cur].keys():
                if not visited[nbr] and nbr != 0:
                    d[nbr] = d[cur] + 1
                    visited[nbr] = True
                    queue.append(nbr)
                    visited_queue.append(nbr)
        # for nbr, dict in self.neighbours(n-1).items():
        #     for direction, flow in dict.items():
        #         capacity = self.get_capacity(n-1, nbr, direction)
        #         if direction == 'backward' and flow != capacity and nbr != 0:
        #             d[nbr] = 1
        #             visited[nbr] = True
        #             queue.append(nbr)
        #             visited_queue.append(nbr)
        # while len(queue) != 0:
        #     cur = queue.popleft()
        #     for nbr, dict in self.neighbours(cur).items():
        #         for direction, flow in dict.items():
        #             capacity = self.get_capacity(cur, nbr, direction)
        #             if direction == 'backward' and flow != capacity and nbr != 0 \
        #                     and not visited[nbr]:
        #                 d[nbr] = d[cur] + 1
        #                 visited[nbr] = True
        #                 queue.append(nbr)
        #                 visited_queue.append(nbr)
        return d, visited_queue, visited


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


def parse_file(file=FILE):  # Обработка файла, создание графа
    with open(file) as f:
        n, m = map(int, f.readline().split())
        G = DiGraph()
        G.add_nodes_from([i for i in range(n)])
        for line in f:
            node1, node2, capacity = map(int, line.split())
            G.add_edge(node1 - 1, node2 - 1, capacity=capacity)
        return G, n, m


def relabel(G, h, u, n):
    neighbours = []
    for node, dict in G.neighbours(u).items():
        for flow in dict.values():
            if flow != 0:
                neighbours.append(node)
    min = n**2
    for node in neighbours:
        if h[node] < min:
            min = h[node]
    h[u] = min+1


def push(G, e, u, v, direction):
    flow = G.get_flow(u, v, direction)
    delta = min(e[u], flow)
    opposite_direction = {'forward': 'backward', 'backward': 'forward'}
    # Изменяем значения избытков, удаляем или добавляем прямые/обратные ребра в остаточной сеи при необходимости
    e[u] -= delta
    e[v] += delta
    # G.update_flow(u, v, direction, delta, False)
    # if direction == 'forward':
    #     G.update_flow(v, u, 'backward', delta, True)
    # else:
    #     G.update_flow(v, u, 'forward', delta, True)
    if G.has_edge(v, u, opposite_direction.get(direction)):
        G.update_flow(v, u, opposite_direction.get(direction), delta, True)
    else:
        G.add_edge(v, u, opposite_direction.get(direction), delta)
    if delta == flow:
        G.remove_edge(u, v, direction)
    else:
        G.update_flow(u, v, direction, delta, False)


def max_flow(G, n, m):  # Алгоритм макс потока v0.02
    e = [0 for i in range(n)]  # Инициализация избытка, высоты и очереди
    h = [0 for i in range(n)]
    Gf = ResidualNetwork(G)
    queue = deque()

    neighbours = list(Gf.neighbours(0).keys())  # Проталкиваем поток по всем ребрам истока
    for node in neighbours:
        if Gf.has_edge(0, node, 'forward'):
            capacity = Gf.get_capacity(0, node, 'forward')
            Gf.add_edge(node, 0, 'backward', capacity)
            e[node] = capacity
            e[0] -= capacity
    for node in neighbours:
        Gf.remove_edge(0, node, 'forward')

    d, visited_queue, visited = Gf.reversed_bfs_for_t()
    h[0] = n
    for node in visited_queue:
        h[node] = d[node]
    while len(visited_queue) != 0:
        queue.append(visited_queue.pop())
    for i in range(1, n - 1):
        if not visited[i]:
            queue.append(i)
    gr_counter = 0  # Счетчик для запуска bfs для global relabeling

    while len(queue) != 0:  # Алгоритм проталкивания предпотока
        cur = queue.popleft()
        has_neighbour_with_h = False
        while e[cur] != 0:
            if gr_counter >= m:  # Если выполнили m операций relabel выполняем global relabeling
                d, visited_queue_s, visited_s = Gf.direct_bfs_for_s()  # Разница здесь
                for node in visited_queue_s:
                    h[node] = h[0] + d[node]
                d, visited_queue_t, visited_t = Gf.reversed_bfs_for_t()
                for node in visited_queue_t:
                    h[node] = d[node]
                gr_counter = 0
            neighbours = Gf.neighbours(cur)  # Переменные для сохранения соседа, у которого h(u) = h(v) + 1
            has_neighbour_with_h = False
            neighbour_with_h = 0
            e_in_neighbour = 0
            direction_to_neighbour = None
            for node, dict in neighbours.items():  # Поиск такого соседа
                for direction, flow in dict.items():
                    if h[cur] == h[node] + 1:
                        has_neighbour_with_h = True
                        neighbour_with_h = node
                        e_in_neighbour = e[node]
                        direction_to_neighbour = direction
                        break
            if has_neighbour_with_h:  # Если сосед найден, делаем push
                push(Gf, e, cur, neighbour_with_h, direction_to_neighbour)
                # gr_counter += 1
                if e_in_neighbour == 0 and neighbour_with_h not in (0, n - 1):  # Добавляем соседва в очередь,
                    queue.append(neighbour_with_h)  # если его избыток был равен 0
            else:
                break
        if not has_neighbour_with_h and e[cur] != 0:  # Если текущая вершина, не имеет соседей с h(v) = h(u) - 1, тогда
            relabel(Gf, h, cur, n)  # текущую вершину нужно поднять
            queue.append(cur)  # Добавляем текущую вершину в конец очереди
            gr_counter += 1
    return Gf, e, h


def redo_graph(Gf, n, new_edges):
    for elem in new_edges:
        Gf.change_capacity(0, elem[0], elem[1])
        Gf.change_capacity(elem[0], n-1, elem[2])


def max_flow_additional(Gf, n, m):  # Алгоритм макс потока v0.02
    e = [0 for i in range(n)]  # Инициализация избытка, высоты и очереди
    h = [0 for i in range(n)]
    queue = deque()

    neighbours = list(Gf.neighbours(0).keys())  # Проталкиваем поток по всем ребрам истока
    for node in neighbours:
        if Gf.has_edge(0, node, 'forward'):
            capacity = Gf.get_capacity(0, node, 'forward')
            flow = Gf.get_flow(0, node, 'forward')
            if Gf.has_edge(node, 0, 'backward'):
                Gf.update_flow(node, 0, 'backward', flow, True)
            else:
                Gf.add_edge(node, 0, 'backward', capacity)
            e[node] = flow
            queue.append(node)
            e[0] -= flow
    for node in neighbours:
        Gf.remove_edge(0, node, 'forward')

    d, visited_queue, visited = Gf.reversed_bfs_for_t()
    h[0] = n
    for node in visited_queue:
        h[node] = d[node]
    gr_counter = 0  # Счетчик для запуска bfs для global relabeling

    while len(queue) != 0:  # Алгоритм проталкивания предпотока
        cur = queue.popleft()
        has_neighbour_with_h = False
        while e[cur] != 0:
            if gr_counter >= m:  # Если выполнили m операций relabel выполняем global relabeling
                d, visited_queue_s, visited_s = Gf.direct_bfs_for_s()  # Разница здесь
                for node in visited_queue_s:
                    h[node] = h[0] + d[node]
                d, visited_queue_t, visited_t = Gf.reversed_bfs_for_t()
                for node in visited_queue_t:
                    h[node] = d[node]
                gr_counter = 0
            neighbours = Gf.neighbours(cur)  # Переменные для сохранения соседа, у которого h(u) = h(v) + 1
            has_neighbour_with_h = False
            neighbour_with_h = 0
            e_in_neighbour = 0
            direction_to_neighbour = None
            for node, dict in neighbours.items():  # Поиск такого соседа
                for direction, flow in dict.items():
                    if h[cur] == h[node] + 1:
                        has_neighbour_with_h = True
                        neighbour_with_h = node
                        e_in_neighbour = e[node]
                        direction_to_neighbour = direction
                        break
            if has_neighbour_with_h:  # Если сосед найден, делаем push
                push(Gf, e, cur, neighbour_with_h, direction_to_neighbour)
                # gr_counter += 1
                if e_in_neighbour == 0 and neighbour_with_h not in (0, n - 1):  # Добавляем соседва в очередь,
                    queue.append(neighbour_with_h)  # если его избыток был равен 0
            else:
                break
        if not has_neighbour_with_h and e[cur] != 0:  # Если текущая вершина, не имеет соседей с h(v) = h(u) - 1, тогда
            relabel(Gf, h, cur, n)  # текущую вершину нужно поднять
            queue.append(cur)  # Добавляем текущую вершину в конец очереди
            gr_counter += 1
    return Gf, e, h


def bfs(G, u, n):  # Поиск в ширину
    d = [0 for i in range(n)]
    visited = [False for i in range(n)]
    visited_queue = deque()
    queue = deque()
    queue.append(u)
    visited[u] = True
    while len(queue) != 0:
        cur = queue.popleft()
        for node in G.adj[cur].keys():
            if not visited[node]:
                d[node] = d[cur] + 1
                visited[node] = True
                queue.append(node)
                visited_queue.append(node)
    return d, visited_queue, visited


def min_cut(G, n, m):
    Gf, e, h = max_flow(G, n, m)
    r, v, c = dfs(Gf, 0, n)
    mc = [[], []]
    for i in range(n):
        if v[i]:
            mc[0].append(i)
        else:
            mc[1].append(i)
    cut = []
    for i in range(n):
        for j in Gf.neighbours(i):
            if v[i] != v[j]:
                cut.append((i, j))
    return Gf, mc, cut


def min_cut_additional(G, n, m):
    Gf, e, h = max_flow_additional(G, n, m)
    r, v, c = dfs(Gf, 0, n)
    mc = [[], []]
    for i in range(n):
        if v[i]:
            mc[0].append(i)
        else:
            mc[1].append(i)
    cut = []
    for i in range(n):
        for j in Gf.neighbours(i):
            if v[i] != v[j]:
                cut.append((i, j))
    return Gf, mc, cut


def dfs(G, u, n):  # Поиск в глубину
    visited = [False for i in range(n)]
    reachable = [u]
    stack = deque()
    stack.append(u)
    count = 0
    visited[u] = True
    while len(stack) != 0:
        cur = stack.pop()
        for node, dict in G.neighbours(cur).items():
            for flow in dict.values():
                if flow != 0 and not visited[node]:
                    count += 1
                    reachable.append(node)
                    visited[node] = True
                    stack.append(node)
    return reachable, visited, count


def find_path(G, n, start, stop):
    prev = [-1 for i in range(n)]
    visited = [False for i in range(n)]
    queue = deque()
    queue.append(start)
    cur = None
    while len(queue) != 0 and cur != stop:
        cur = queue.popleft()
        for nbr in G.neighbours(cur).keys():
            if not visited[nbr]:
                queue.append(nbr)
                visited[nbr] = True
                prev[nbr] = cur
    list = [stop]
    while prev[stop] != -1 and prev[stop] != start:
        list.append(prev[stop])
        stop = prev[stop]
    return list

