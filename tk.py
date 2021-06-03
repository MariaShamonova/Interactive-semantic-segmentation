from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from PIL import Image, ImageDraw, ImageTk
import os
from image_segmentation_1 import *


class App(object):
    def __init__(self):
        self.image = None
        self.window = Tk()
        self.window.geometry('750x550')
        self.open_btn = Button(self.window, text='Открыть', command=self.open_file)
        self.open_btn.grid(column=0, row=0)
        self.w_btn = Button(self.window, text='Начать работу', command=self.segmentation)
        self.w_btn.grid(column=1, row=0)

        self.window.mainloop()

    def open_file(self):
        file = askopenfile(mode='r',
                           filetypes=[('Изображения', ['.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif', '.bmp'])])
        if file is not None:
            self.image_name = file.name
            self.image = Image.open(self.image_name)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas = Canvas(self.window, height=self.image.size[1], width=self.image.size[0])
            self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
            self.canvas.grid(row=1, column=1)

    def segmentation(self):
        image = cv2.imread(self.image_name, cv2.IMREAD_GRAYSCALE)
        self.graph, seededImage, self.K, seeds, intervals = buildGraph(image)
        self.intervals = intervals

        self.n = self.graph.number_of_nodes()
        self.m = self.graph.number_of_edges()
        self.Gf, partition, cutset, cut_value = min_cut(self.graph, self.n, self.m)

        image = displayCut(image, cutset)

        self.window2 = Toplevel()
        self.window2.title = 'Segmentation'
        self.image_for_alg = image
        out_im = Image.fromarray(image)
        self.size = (out_im.size[1], out_im.size[0])
        self.photo2 = ImageTk.PhotoImage(out_im)
        self.canvas2 = Canvas(self.window2, height=out_im.size[1], width=out_im.size[0])
        self.c_image2 = self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
        self.canvas2.grid(row=0, column=0)
        self.w_btn = Button(self.window, text='Продолжить', command=self.additional_segmentation)
        self.w_btn.grid(column=1, row=0)

    def additional_segmentation(self):
        self.photo = self.photo2
        self.canvas.destroy()
        self.canvas = Canvas(self.window, height=self.size[0], width=self.size[1])
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(rows=1, columns=2)

        image = self.image_for_alg
        graph, lst_vertex = addedSeeds(image, self.graph, self.K, self.intervals, self.image_name)

        self.Gf, partition, cutset, cut_value = min_cut_additional(self.Gf, self.n, self.m, lst_vertex)

        image_start = cv2.imread(self.image_name, cv2.IMREAD_GRAYSCALE)
        image = displayCut(image_start, cutset)

        self.window2 = Toplevel()
        self.window2.title = 'Segmentation'
        self.image_for_alg = image
        out_im = Image.fromarray(image)
        self.photo2 = ImageTk.PhotoImage(out_im)
        self.canvas2 = Canvas(self.window2, height=out_im.size[1], width=out_im.size[0])
        self.c_image2 = self.canvas2.create_image(0, 0, anchor='nw', image=self.photo2)
        self.canvas2.grid(row=0, column=0)


if __name__ == '__main__':
    app = App()
