# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:29:43 2021

@author: honey
"""
from networkx import DiGraph
from collections import deque

import numpy as np
from math import exp, pow
import networkx as nx
import cv2
from PIL import Image, ImageOps

from max_flow_alg import max_flow
from max_flow_alg import dfs


FILE = 'test2.txt'
SIGMA = 30
OBJCOLOR, BKGCOLOR = (152, 94, 56), (255,15,255)
OBJ, BKG = "OBJ", "BKG"
OBJCODE, BKGCODE = 1, 2


SF = 10
CUTCOLOR = (0, 0, 255)




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
    for i in range(1, r + 1):
        for j in range(1, c  + 1):
            graph.add_node((i - 1) * c + j)
            count = count + 1
         
            x = (i - 1) * c + j
            if i < r: # Нижний пиксель
                y = i * c + j
                bp = boundaryPenalty(image[i - 2][j - 1], image[i - 1][j - 1])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
               
            if i - 2 > 0: # Верхний пиксель
                y = (i - 2) * c + j
                bp = boundaryPenalty(image[i - 2][j - 1], image[i - 3][j - 1])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
            if j + 1 < c: # Пиксель справа
                y = (i - 1)  * c + j + 1
                bp = boundaryPenalty(image[i - 2][j - 1], image[i - 2][j ])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
            if j - 1 > 0: # Пиксель слева
                y = (i - 1) * c + j - 1
                bp = boundaryPenalty(image[i - 2][j - 1], image[i - 2][j - 2])
                graph.add_edge(x, y,capacity= bp)
                graph.add_edge(y, x,capacity= bp)
                K = max(K, bp)

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
                
def buildGraph(image, image_rgb, r, c):
  
    graph = nx.DiGraph()
    
    #Добавляем исток и сток
    graph.add_node(0)
    graph.add_node(rows*columns+1)
    
    global SOURCE 
    SOURCE = 0
    global SINK
    SINK = rows*columns+1
    
    #n-links - список соседних ребер между пикселями
    K = makeNLinks(graph, image,  rows, columns)
    
    seeds, seededImage = plantSeed(image, rows, columns)
    
    #n-links -Добавление вершин между вершинами и  стоком, стоком
    makeTLinks(graph, seeds, K,  rows, columns)
    
    return graph, seededImage

def displayCut(image, visited, r, c):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print(cuts)
    for c in cuts:
        
        if c[0] != True and c[0] != False and c[1] != True and c[1] != False:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    
            
    return image

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
   
    img = Image.open('image.jpg')
    image_1 = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
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
    
    Gf, mf, h = max_flow(G, n, m)

  
    r, visited = dfs(Gf, SOURCE, Gf.number_of_nodes())
    print(visited)
    
    cuts = []

    for i in range(rows):
    
        for j in range(columns):
            x = (i - 1) * columns + j
        
            if i < rows: # Нижний пиксель
                y = i * columns + j
                exist_edge = G.get_edge_data(x, y)
                if visited[x] and not visited[y] and exist_edge != None:
                    cuts.append((x , y))
            if i - 2 > 0: # Верхний пиксель
                y = (i - 2) * columns + j
                exist_edge = G.get_edge_data(x, y)
                if visited[x] and not visited[y] and exist_edge != None:
                    cuts.append((x , y))
            if j + 1 < columns: # Пиксель справа
                y = (i - 1)  * columns + j + 1
                exist_edge = G.get_edge_data(x, y)
                if visited[x] and not visited[y] and exist_edge != None:
                    cuts.append((x , y))
            if j - 1 > 0: # Пиксель слева
                y = (i - 1) * columns + j - 1
                exist_edge = G.get_edge_data(x, y)
                if visited[x] and not visited[y] and exist_edge != None:
                    cuts.append((x , y))
                
   
    #visited = np.zeros(Gf.number_of_nodes(), dtype=bool)
    #dfs_1(Gf, Gf.number_of_nodes(), SOURCE, visited)
  

   
    im = displayCut(image_1, cuts, rows, columns)
    show_image(im)
    
    ###pathname = os.path.splitext('image.jpg')[0]
    #savename = pathname + "cut.jpg"
    #cv2.imwrite(savename, im)
    #print("Saved image as", savename)
    
   