# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:20:51 2021

@author: honey
"""

from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
from max_flow_alg import min_cut, min_cut_additional, redo_graph
from math import exp, pow
import networkx as nx
import time

SIGMA = 30
LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 1
LOADSEEDS = False

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    
def getInterval(image,  i, j):
   
    
    p = image[i][j]

    step = 256//25
    count = 0
   
    for i in range(0, 255, step):
        if (i <  (256 // step * step)):
            if(step * count <= p and p < step*(count + 1)):
                return count 
        else:
            if (step * count <= p):
                return count
        count += 1
    
def plantSeed(image):
    
    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
            intervals[getInterval(image_gray, y , x), 1] += 1
        else:
            color, code = BKGCOLOR, BKGCODE
            intervals[getInterval(image_gray, y , x), 2] += 1
    
        intervals[getInterval(image_gray, y, x), 0] += 1
        cv2.circle(image, (x , y), radius, color, thickness)
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
    
    intervals = np.zeros((26, 3), dtype="uint8")
    seeds = np.zeros((len(image), len(image[0])), dtype="uint8")
   
    image_gray = image.copy()
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    radius = 10
    thickness = -1
    global drawing
    drawing = False
    
    paintSeeds(OBJ)
    paintSeeds(BKG)
    
    count_of_seeds = sum((int(intervals[i, 0]) for i in range(0, int(len(intervals)))))

    return seeds, image, intervals, count_of_seeds
    
# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def regionalPenalty(favorableValue, allValue):
    if (favorableValue != 0 and allValue != 0):
        rp = LAMBDA * (favorableValue / allValue)
    else:
        rp = 0  
    return rp

def buildGraph(image):
    
    graph = nx.DiGraph()
    graph.add_node(0)
    
    rows = len(image)
    columns = len(image[0])
    
    global SOURCE, SINK
    SOURCE = 0 
    SINK   = rows*columns+1

    K = makeNLinks(graph, image)
    
    seeds, seededImage, intervals, count_of_seeds = plantSeed(image)
    makeTLinks(graph, seeds, K, intervals,count_of_seeds,  image)
  
    graph.add_node(rows*columns+1)
    
    return graph, seededImage, K, seeds, intervals
    

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j + 1
            if i + 1 < r:
                y = (i + 1) * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp) 
                K = max(K, bp)
            if j + 1 < c: 
                y = i * c + j + 2
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)           
                K = max(K, bp)
    return K

def makeTLinks(graph, seeds, K, intervals, count_of_seeds, image):
    r, c = len(seeds), len(seeds[0])
   
    for i in range(r):
        for j in range(c):
            x = i * c + j + 1
    
            if seeds[i][j] == OBJCODE:
                graph.add_edge(SOURCE, x, capacity=K)
            elif seeds[i][j] == BKGCODE:
                graph.add_edge(x, SINK, capacity=K)   
            else:
                if (intervals[getInterval(image, i, j), 2] != 0 and count_of_seeds):
                    graph.add_edge(
                        SOURCE,
                        x,
                        capacity=regionalPenalty(
                            intervals[getInterval(image, i, j), 2],
                            count_of_seeds
                        )
                    )
                if (intervals[getInterval(image, i, j), 1] != 0 and \
                    count_of_seeds):
                    graph.add_edge(
                        x,
                        SINK,
                        capacity=regionalPenalty(
                            intervals[getInterval(image, i, j), 1],
                            count_of_seeds
                        )
                    )
              

def displayCut(image, cuts):
    print('cuts')
    def colorPixel(i, j):
        try:
            image[i][j] = CUTCOLOR
        except:   
            print(image[i][j])

    col = len(image[0])
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(((c[0] - 1) // col) , ((c[0] - 1) % col))
            colorPixel((c[1] - 1) // col, (c[1] - 1) % col)
    return image

def createHistogram(imagefile):
    import cv2
    # импорт библиотеки для черчения
    from matplotlib import pyplot as plt
     
    # читает входное изображение
    img = cv2.imread(imagefile,0)
       
    # найти частоту пикселей в диапазоне 0-255
    
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    
    # показать графическое изображение изображения
    plt.plot(histr)
    plt.show()
    
# def addedSeeds(image, graph, K, intervals, imagefile):
#     image_gray =  cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
#     def drawLines(x, y, pixelType):
#         vertex = y * len(image[0]) + x
#         allValue = intervals[getInterval(image_gray, y, x), 0]
#         if pixelType == OBJ:
#             if (graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK)):
#
#                 color, code = OBJCOLOR, OBJCODE
#
#                 graph.edges[SOURCE, vertex]["capacity"] +=  K + regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
#                 graph.edges[vertex, SINK]["capacity"] += regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
#
#                 cv2.circle(image, (x, y), radius, color, thickness)
#         else:
#             if (graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK)):
#
#                 color, code = BKGCOLOR, BKGCODE
#
#                 graph.edges[SOURCE, vertex]["capacity"] +=  K + regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
#                 graph.edges[vertex, SINK]["capacity"] += regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
#
#                 cv2.circle(image, (x, y), radius, color, thickness)
#
#
#     def onMouse(event, x, y, flags, pixelType):
#         global drawing
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             drawLines(x, y, pixelType)
#         elif event == cv2.EVENT_MOUSEMOVE and drawing:
#             drawLines(x, y, pixelType)
#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#
#     def paintSeeds(pixelType):
#         print("Planting", pixelType, "seeds")
#         global drawing
#         drawing = False
#         windowname = "Plant " + pixelType + " seeds"
#         cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
#         cv2.setMouseCallback(windowname, onMouse, pixelType)
#         while (1):
#             cv2.imshow(windowname, image)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#         cv2.destroyAllWindows()
#
#     radius = 10
#     thickness = -1 # fill the whole circle
#     global drawing
#     drawing = False
#
#     paintSeeds(OBJ)
#     paintSeeds(BKG)
#
#     return graph


def addedSeeds(image, graph, K, intervals, imagefile):
    image_gray = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    lst_vertex = []

    def drawLines(x, y, pixelType):
        vertex = y * len(image[0]) + x + 1
        allValue = intervals[getInterval(image_gray, y, x), 0]

        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
            cv2.circle(image, (x, y), radius, color, thickness)
            if not (graph.has_edge(SOURCE, vertex) and not graph.has_edge(vertex, SINK)
                    and graph.edges[SOURCE, vertex]["capacity"] == K)\
                    or not (not graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK)
                            and graph.edges[vertex, SINK]["capacity"] == K):

                if graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK):
                    to_source = graph.edges[SOURCE, vertex]["capacity"] + K + \
                                regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
                    to_sink = graph.edges[vertex, SINK]["capacity"] + \
                              regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
                elif not graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK):
                    to_source = K + \
                                regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
                    to_sink = graph.edges[vertex, SINK]["capacity"] + \
                              regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
                elif graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK):
                    to_source = graph.edges[SOURCE, vertex]["capacity"] + K + \
                                regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
                    to_sink = regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
                else:
                    to_source = K
                    to_sink = 0

                lst_vertex.append((vertex, to_source, to_sink))
        else:
            color, code = BKGCOLOR, BKGCODE
            cv2.circle(image, (x, y), radius, color, thickness)
            if not (graph.has_edge(SOURCE, vertex) and not graph.has_edge(vertex, SINK)
                    and graph.edges[SOURCE, vertex]["capacity"] == K)\
                    or not (not graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK)
                            and graph.edges[vertex, SINK]["capacity"] == K):

                if graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK):
                    to_source = graph.edges[SOURCE, vertex]["capacity"] + \
                                regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
                    to_sink = graph.edges[vertex, SINK]["capacity"] + K + \
                              regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
                elif not graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK):
                    to_source = regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
                    to_sink = graph.edges[vertex, SINK]["capacity"] + K + \
                              regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
                elif graph.has_edge(SOURCE, vertex) and graph.has_edge(vertex, SINK):
                    to_source = graph.edges[SOURCE, vertex]["capacity"] + \
                                regionalPenalty(intervals[getInterval(image_gray, y, x), 1], allValue)
                    to_sink = K + regionalPenalty(intervals[getInterval(image_gray, y, x), 2], allValue)
                else:
                    to_source = 0
                    to_sink = K

                lst_vertex.append((vertex, to_source, to_sink))

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

    radius = 10
    thickness = -1  # fill the whole circle
    global drawing
    drawing = False

    paintSeeds(OBJ)
    paintSeeds(BKG)

    return graph, lst_vertex

    
def drawContur(image, reachable, non_reachable):
    def colorPixel(i, j, color):
        try:
            image[i][j] = color
        except:   
            print(image[i][j])

    col = len(image[0])
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for c in reachable:
        if (c != SOURCE  and c != SINK):  
            colorPixel(((c - 1) // col) , ((c - 1) % col), 255)
       
    for c in non_reachable:
        if (c != SOURCE and c != SINK ):  
            colorPixel(((c - 1 )// col) , ((c - 1 ) % col), 0)
    
    return image


def compareImages(image, image_compare):
    
    r, c = len(image_compare), len(image_compare[0])
    size = (c, r)
    image = cv2.resize(image, size)

    # b_obj = np.ones((r, c))
    # for node in reachable:
    #     if node != 0 :
    #         rows = (node - 1) // c
    #         columns = (node - 1) % c
    #         b_obj[rows][columns] = 0

    correctly = 0
    intersection = 0
    union = 0
    for i in range(r):
        for j in range(c):
            if image[i][j] == image_compare[i][j]:
                correctly += 1
            if image[i][j] == 0 and image_compare[i][j] == 0:
                intersection += 1
            if image[i][j] == 0 or image_compare[i][j] == 0:
                union += 1

    relationship = correctly / (r * c)
    measure_zhakkar = intersection / union
    return relationship, measure_zhakkar


def imageSegmentation( ):

    imagefile = 'book-gr.jpg'
    imagefile_compare = 'book.bmp'
    
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image_compare = cv2.imread(imagefile_compare, cv2.IMREAD_GRAYSCALE)
    print(len(image))
    print(len(image[0]))
    graph, seededImage, K, seeds, intervals = buildGraph(image)
    # createHistogram(imagefile)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    t0 = time.time()
    Gf, partition, cutset, cut_value = min_cut(graph, n, m)
    t1 = time.time()
    reachable, non_reachable = partition

    print('Time: {}'.format(t1-t0))
    print(cut_value)
    print('reachable: ', len(reachable))
  
    #Разбиение
    image_res = displayCut(image, cutset)
    
    #Метрика разбиения
    image_contur = drawContur(image,reachable, non_reachable)

    relationship, measure_zhakkar = compareImages(image_contur, image_compare)
    print('1 metrica: ', relationship)
    print('2 metrica: ', measure_zhakkar)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image_res)
    savename = pathname + "contur.jpg"
    cv2.imwrite(savename, image_contur)

    show_image(image_contur)
    show_image(image_res)

    answer = input("If you want to improve segmentation inter Y else N:")
    while answer == "Y":

        graph, lst_vertex = addedSeeds(image_res, graph, K, intervals, imagefile)

        Gf, partition, cutset, cut_value = min_cut_additional(Gf, n, m, lst_vertex)
        # cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK)
        reachable, non_reachable = partition
        print(cut_value)
        print('reachable: ', len(reachable))

        image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)

        # Разбиение
        image_res = displayCut(image, cutset)

        # Метрика разбиения
        image_contur = drawContur(image, reachable, non_reachable)

        relationship, measure_zhakkar = compareImages(image_contur, image_compare)
        print('1 metrica: ', relationship)
        print('2 metrica: ', measure_zhakkar)
        savename = pathname + "cut_improve.jpg"
        cv2.imwrite(savename, image_res)
        savename = pathname + "contur_improve.jpg"
        cv2.imwrite(savename, image_contur)

        show_image(image_contur)
        show_image(image_res)

        answer = input("If you want to improve segmentation inter Y else N:")
       

    print("Segmentation completed")
    
if __name__ == "__main__":

    imageSegmentation()  
