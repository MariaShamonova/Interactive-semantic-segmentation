Interactive-semantic-segmentation

Requirements:
- python 3.7
- networkx

max_flow - алгоритм нахождения макс потока:
- G - граф(объект класса DiGraph модуля networkx), 
должен содержать узлы, нумерация которых начинается с 1. 
Также должен хранить информацию о пропускной способности 
в атрибуте ребер capacity.
- n - количество вершин в графе
- m - количество ребер в графе

parse_file - используется для создания объекта DiGraph, сгенерированного из файла с именем FILE
