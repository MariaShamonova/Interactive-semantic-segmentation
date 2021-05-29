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
from math import exp, pow
import networkx as nx

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
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
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
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False
    
    paintSeeds(OBJ)
    paintSeeds(BKG)
    
    count_of_seeds = sum((int(intervals[i, 0]) for i in range(0, int(len(intervals)))))

    return seeds, image, intervals, count_of_seeds

def addedSeed(seeds, image):
    
    def drawLines(x, y, pixelType):
    
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE

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

    radius = 10
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False

    paintSeeds(OBJ)
    paintSeeds(BKG)
    
    return seeds
    
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
    print(intervals)
    makeTLinks(graph, seeds, K, intervals,count_of_seeds,  image)
  
    graph.add_node(rows*columns+1)
    
    return graph, seededImage, K, seeds
    

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j + 1
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                
                K = max(K, bp)
            #if i + 1 < r: # pixel top
            #    y = (i - 1) * c + j + 1
            #    bp = boundaryPenalty(image[i][j], image[i - 1][j])
            #    graph.add_edge(x, y, capacity=bp)
            #    graph.add_edge(y, x, capacity=bp)
                
                K = max(K, bp)
            if j + 1 < c: # pixel to the right
                y = i * c + j + 2
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph.add_edge(x, y, capacity=bp)
                graph.add_edge(y, x, capacity=bp)
                K = max(K, bp)
            #if i + 1 < r: # pixel left
            #    y = i * c + j 
            #    bp = boundaryPenalty(image[i][j], image[i - 1][j])
            #    graph.add_edge(x, y, capacity=bp)
            #    graph.add_edge(y, x, capacity=bp)
                
                K = max(K, bp)
    return K

def makeTLinks(graph, seeds, K, intervals, count_of_seeds, image):
    r, c = len(seeds), len(seeds[0])
   
    for i in range(r):
        for j in range(c):
            x = i * c + j
    
            if seeds[i][j] == OBJCODE:
                graph.add_edge(SOURCE, x, capacity=K)
            elif seeds[i][j] == BKGCODE:
                graph.add_edge(x, SINK, capacity=K)   
            else:
                graph.add_edge(
                    SOURCE, 
                    x, 
                    capacity=regionalPenalty(
                        intervals[getInterval(image, i, j), 2],
                        intervals[getInterval(image, i, j), 0]
                    )
                )  
                graph.add_edge(
                    x, 
                    SINK, 
                    capacity=regionalPenalty(
                        intervals[getInterval(image, i, j), 1],
                        intervals[getInterval(image, i, j), 0]
                    )
                ) 
              

def displayCut(image, cuts):
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
    

def imageSegmentation( ):
    #size=(30, 30)
    imagefile= 'cross-gr.jpg'
    
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    
    #image = cv2.resize(image, size)
    
    graph, seededImage, K, seeds = buildGraph(image)
    createHistogram(imagefile)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    
    cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK)
 
    reachable, non_reachable = partition
  
    cutset = set()
    
    for u, nbrs in ((n, graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    [('c', 'y'), ('x', 'b')]
    cut_value == sum(graph.edges[u, v]["capacity"] for (u, v) in cutset)
  
    image = displayCut(image, cutset)
    
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    
    answer = input("If you want to improve segmentation inter Y else N:")
    while answer == "Y":

        seeds = addedSeed(seeds,image)
        makeTLinks(graph, seeds, K)
        cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK)
 
        reachable, non_reachable = partition
        cutset = set()
    
        for u, nbrs in ((n, graph[n]) for n in reachable):
            cutset.update((u, v) for v in nbrs if v in non_reachable)
            #print(sorted(cutset))
            [('c', 'y'), ('x', 'b')]
            cut_value == sum(graph.edges[u, v]["capacity"] for (u, v) in cutset)
            #print(cut_value)
        image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
        image = displayCut(image, cutset)
        show_image(image)
        answer = input("If you want to improve segmentation inter Y else N:")
       

    print("Segmentation completed")
    
if __name__ == "__main__":

    imageSegmentation()  
