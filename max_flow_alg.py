from networkx import DiGraph
from collections import deque
import os
import time

import numpy as np
from math import exp, pow
import networkx as nx
import cv2
from PIL import Image, ImageOps

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
                if self.flow[nbr].get(node):
                    self.flow[nbr][node].update({'backward': 0})
                else:
                    self.flow[nbr].update({node: {'backward': 0}})

    def neighbours(self, u):
        return self.flow[u]

    # def edge(self, u, v, d):
    #     return self.G.edges[u, v, d]

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
        return True if self.flow[u].get(v).get(d) else False

    def reversed_copy_bfs_for_s(self):
        d = [0 for i in range(self.n)]
        visited = [False for i in range(self.n)]
        visited_queue = deque()
        queue = deque()
        visited[0] = True
        for nbr, dict in self.neighbours(0).items():
            for direction, flow in dict.items():
                capacity = self.get_capacity(0, nbr, direction)
                if direction == 'backward' and flow != capacity and nbr != n-1:
                    d[nbr] = 1
                    visited[nbr] = True
                    queue.append(nbr)
                    visited_queue.append(nbr)
        while len(queue) != 0:
            cur = queue.popleft()
            for nbr, dict in self.neighbours(cur).items():
                for direction, flow in dict.items():
                    capacity = self.get_capacity(cur, nbr, direction)
                    if direction == 'backward' and flow != capacity and nbr != n-1 \
                            and not visited[nbr]:
                        d[nbr] = d[cur] + 1
                        visited[nbr] = True
                        queue.append(nbr)
                        visited_queue.append(nbr)
        return d, visited_queue, visited

    def reversed_copy_bfs_for_t(self):
        d = [0 for i in range(self.n)]
        visited = [False for i in range(self.n)]
        visited_queue = deque()
        queue = deque()
        visited[n-1] = True
        for nbr, dict in self.neighbours(n-1).items():
            for direction, flow in dict.items():
                capacity = self.get_capacity(n-1, nbr, direction)
                if direction == 'backward' and flow != capacity and nbr != 0:
                    d[nbr] = 1
                    visited[nbr] = True
                    queue.append(nbr)
                    visited_queue.append(nbr)
        while len(queue) != 0:
            cur = queue.popleft()
            for nbr, dict in self.neighbours(cur).items():
                for direction, flow in dict.items():
                    capacity = self.get_capacity(cur, nbr, direction)
                    if direction == 'backward' and flow != capacity and nbr != 0 \
                            and not visited[nbr]:
                        d[nbr] = d[cur] + 1
                        visited[nbr] = True
                        queue.append(nbr)
                        visited_queue.append(nbr)
        return d, visited_queue, visited

    # def add_edge(self, u, v, **kwargs):
    #     self.G.add_edge(u, v, **kwargs)
    #
    # def remove_edge(self, u, v):
    #     self.G.remove_edge(u, v)


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

    # Изменяем значения избытков, удаляем или добавляем прямые/обратные ребра в остаточной сеи при необходимости
    e[u] -= delta
    e[v] += delta
    G.update_flow(u, v, direction, delta, False)
    if direction == 'forward':
        G.update_flow(v, u, 'backward', delta, True)
    else:
        G.update_flow(v, u, 'forward', delta, True)
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
        if Gf.has_edge(0, node, 'forward'):
            capacity = Gf.get_capacity(0, node, 'forward')
            Gf.update_flow(0, node, 'forward', capacity, False)
            Gf.update_flow(node, 0, 'backward', capacity, True)
            e[node] = capacity
            e[0] -= capacity

    d, visited_queue, visited = Gf.reversed_copy_bfs_for_t()
    h[0] = n
    for i in range(1, n):
        if d[i] != 0:
            h[i] = d[i]
    while len(visited_queue) != 0:
        queue.append(visited_queue.pop())
    for i in range(1, n-1):
        if not visited[i]:
            queue.append(i)
    gr_counter = 0  # Счетчик для запуска bfs для global relabeling

    while len(queue) != 0:  # Алгоритм проталкивания предпотока
        cur = queue.popleft()
        has_neighbour_with_h = False
        while e[cur] != 0:
            if gr_counter >= n:  # Если выполнили m операций push\relabel выполняем global relabeling
                d, visited_queue_s, visited_s = Gf.reversed_copy_bfs_for_s()
                for node in visited_queue:
                    h[node] = h[0] + d[node]
                d, visited_queue_t, visited_t = Gf.reversed_copy_bfs_for_t()
                for node in visited_queue:
                    h[node] = d[node]
                gr_counter = 0
            neighbours = Gf.neighbours(cur)  # Переменные для сохранения соседа, у которого h(u) = h(v) + 1
            has_neighbour_with_h = False
            neighbour_with_h = 0
            e_in_neighbour = 0
            direction_to_neighbour = None
            for node, dict in neighbours.items():  # Поиск такого соседа
                for direction, flow in dict.items():
                    if h[cur] == h[node] + 1 and flow != 0:
                        has_neighbour_with_h = True
                        neighbour_with_h = node
                        e_in_neighbour = e[node]
                        direction_to_neighbour = direction
                        break
            if has_neighbour_with_h:  # Если сосед найден, делаем push
                push(Gf, e, cur, neighbour_with_h, direction_to_neighbour)
                # gr_counter += 1
                if e_in_neighbour == 0 and neighbour_with_h not in (0, n-1):  # Добавляем соседва в очередь,
                    queue.append(neighbour_with_h)  # если его избыток был равен 0
            else:
                break
        if not has_neighbour_with_h and e[cur] != 0:  # Если текущая вершина, не имеет соседей с h(v) = h(u) - 1, тогда
            relabel(Gf, h, cur, n)  # текущую вершину нужно поднять
            queue.append(cur)  # Добавляем текущую вершину в конец очереди
            # gr_counter += 1
    return Gf, e[n-1]


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

SIGMA = 30
OBJCOLOR, BKGCOLOR = (152, 94, 56), (255,255,255)
OBJ, BKG = "OBJ", "BKG"
OBJCODE, BKGCODE = 1, 2

SOURCE, SINK = -2, -1
SF = 10
CUTCOLOR = (0, 0, 255)

def plantSeed(image, r, c):

    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
            seeds[x][y] = OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
            seeds[x][y] = BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
    seeds = np.zeros((r, c), dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    radius = 10
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False
    
    paintSeeds(OBJ)
    paintSeeds(BKG)
    
    return seeds, image


def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def makeNLinks(graph, image, r, c):
    K = -float("inf")
    count = 0
    for i in range(r):
        for j in range(c):
            graph.add_node(i * c + j)
            count = count + 1
            x = i * c + j
            if i + 1 < r: # Нижний пиксель
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
               
            if i + 1 < r: # Верхний пиксель
                y = (i - 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i - 1][j])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
            if j + 1 < c: # Пиксель справа
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
            if j + 1 < c: # Пиксель слева
                y = i * c + j - 1
                bp = boundaryPenalty(image[i][j], image[i][j - 1])
                graph.add_edge(x, y,capacity= bp)
                graph.add_edge(y, x,capacity= bp)
                K = max(K, bp)
    print(count)
    return K


def makeTLinks(graph, seeds, K, r, c):
    
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                graph.add_edge(SOURCE,x, capacity=K)
            elif seeds[i][j] == BKGCODE:
                graph.add_edge(x,SINK, capacity=K)
                
    return graph
                
                
def buildGraph(image, image_rgb, rows, columns):
  
    graph = nx.DiGraph()
    
    #Добавляем исток и сток
    graph.add_node(0)
    graph.add_node(rows*columns+2)
    
    #n-links - список соседних ребер между пикселями
    K = makeNLinks(graph, image,  rows, columns)
    
    seeds, seededImage = plantSeed(image, rows, columns)
    
    #n-links -Добавление вершин между вершинами и  стоком, стоком
    makeTLinks(graph, seeds, K,  rows, columns)
    
    return graph, seededImage

def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image

if __name__ == '__main__':
    # n = 10000
    # m = (n-1)**2 - 1
    # G = generate_max_graph(n)
    # print(G.size())
    # G, n, m = parse_file()
    # Gf, mf = max_flow(G, n, m)
    # print(mf)

    #dir = "MaxFlow-tests"
    #file = 'test_rd07.txt'
    #for file in os.listdir(dir):
    #    G, n, m = parse_file(os.path.join(dir, file))
    #    print(file)
    #    t0 = time.time()
    #    Gf, mf = max_flow(G, n, m)
    #    t1 = time.time()
    #    print('Time: {}'.format(t1-t0))
    #    print('Max flow: {}'.format(mf))

    # print(dfs(Gf, 0, n))
    
    #Сегментация изображения
    
    img = Image.open('image.jpg')
    arr = np.asarray(img, dtype='uint8')
    
 
    
    
    gray_image = ImageOps.grayscale(img)
    arr_gray = np.asarray(gray_image, dtype='uint8')
    arr_rgb = np.asarray(arr, dtype='uint8')
   
    
    rows = len(arr)
    columns = len(arr[0])
    
    graph, seededImage = buildGraph(arr_gray, arr_rgb, rows, columns)
   
    G = graph
    m = graph.number_of_edges() 
    n = graph.number_of_nodes()
    print(m)
    print(n)
    t0 = time.time()
    Gf, mf = max_flow(G, n, m)
    t1 = time.time()
    print('Time: {}'.format(t1-t0))
    print('Max flow: {}'.format(mf))