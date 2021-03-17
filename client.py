import socket
import time, threading
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import struct
from queue import Queue
from PIL import ImageTk, Image
import robot
import psutil
import io
import sys
import math
import cv2 as cv
np.set_printoptions(threshold=sys.maxsize)
from threading import Thread


def rgb2gray(rgb):#rgb转换灰度

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gfunc(x,y,sigma):#高斯函数
    return (math.exp(-(x**2 + y**2)/(2*(sigma**2))))/(2*3.14*(sigma**2))

def gaussFilter(size, sigma):#高斯滤波器
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = gfunc(i-size[0]//2,j-size[1]//2, sigma )
    return out/np.sum(out)

#FLANN模板匹配算法
def template_match(img,query):
    sift = cv.SIFT_create()
    # numpy数组到opencv矩阵转换
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    query = cv.cvtColor(np.array(query), cv.COLOR_RGB2BGR)


    # SIFT查找关键特征点和距离
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(query, None)

    # FLANN 参数
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # FLANN 匹配相似像素点
    matches = flann.knnMatch(des1, des2, k=2)

    # 判断是否满足相似条件的相似矩阵
    matchesMask = [[0, 0] for i in range(len(matches))]


    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(img, kp1, query, kp2, matches, None, **draw_params)
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
    return img3

def covFilter(img,filter): #滤波器卷积函数
    img_arr = np.array(img)
    #re_arr = np.zeros_like(img_arr)
    re_arr = np.copy(img_arr)
    center_x = filter.shape[0]//2 #滤波器中心点
    center_y = filter.shape[1]//2 #滤波器中心点
    img_arr = np.pad(img_arr, ((center_x,center_x), (center_y,center_y), (0, 0)))#padding操作
    re_arr = re_arr.astype(np.float64)
    img_arr = img_arr.astype(np.float64)
    re_arr =re_arr/255.0 #像素值转换0~255到0~1
    img_arr =img_arr/255.0 #像素值转换0~255到0~1
    re_arr = filter[center_x,center_y] * re_arr
    for x in range(0,filter.shape[0]):
        for y in range(0,filter.shape[1]):
            if x!=center_x and y!=center_y:
                re_arr = re_arr + filter[x,y] * img_arr[x:x+re_arr.shape[0],y:y+re_arr.shape[1],:]

    re_arr = re_arr.astype(np.float64)
    re_arr =re_arr*255.0
    re_arr = re_arr.astype(np.uint8)
    """
    for x in range(re_arr.shape[0]):
        for y in range(re_arr.shape[1]):
            re_arr[x,y,:] = np.mean(img_arr[x:x+9,y:y+9,:],axis=(0, 1))

    print(re_arr.dtype)
    """
    return re_arr

#均值滤波器
def MeanFilter(img):
    img_arr = np.array(img)
    #re_arr = np.zeros_like(img_arr)
    re_arr = np.copy(img_arr)
    img_arr = np.pad(img_arr, ((4,4), (4,4), (0, 0)))
    re_arr = re_arr.astype(np.float64)
    img_arr = img_arr.astype(np.float64)
    re_arr =re_arr/255.0
    img_arr =img_arr/255.0  #像素值转换0~255到0~1
    for x in range(0,9):
        for y in range(0,9):
            if x!=4 and y!=4:
                re_arr = re_arr + img_arr[x:x+re_arr.shape[0],y:y+re_arr.shape[1],:]

    re_arr = re_arr.astype(np.float64)
    re_arr =re_arr/81.0*255.0 #平均值操作
    re_arr = re_arr.astype(np.uint8)
    """
    for x in range(re_arr.shape[0]):
        for y in range(re_arr.shape[1]):
            re_arr[x,y,:] = np.mean(img_arr[x:x+9,y:y+9,:],axis=(0, 1))

    print(re_arr.dtype)
    """
    return re_arr

#Harris角点
def find_harris_corners(input_img, k=0.04, window_size=5, threshold = 10000.00):

    output_img = np.copy(input_img)
    input_img = MeanFilter(input_img)
    offset = int(window_size / 2)
    input_img = rgb2gray(input_img)
    x_range = input_img.shape[0] - offset
    y_range = input_img.shape[1] - offset
    blue = np.array([0, 0, 255],dtype=np.uint8)

    dy, dx = np.gradient(input_img) #梯度计算
    Ixx = dx ** 2 #梯度平方，x与x协方差
    Ixy = dy * dx #梯度平方，x与y协方差
    Iyy = dy ** 2 #梯度平方，y与y协方差

    Sxx = Ixx[offset:x_range,offset:y_range]
    Sxy = Ixy[offset:x_range,offset:y_range]
    Syy = Iyy[offset:x_range,offset:y_range]

    for x in range(window_size):
        for y in range(window_size):
            if x != offset and y != offset:
                Sxx = Sxx + Ixx[x:x+Sxx.shape[0],y:y+Sxx.shape[1]]
                Sxy = Sxy + Ixy[x:x+Sxy.shape[0],y:y+Sxy.shape[1]]
                Syy = Syy + Iyy[x:x+Syy.shape[0],y:y+Syy.shape[1]]

    # Calculate determinant and trace of the matrix
    det = (Sxx * Syy) - (Sxy ** 2) #行列式计算
    trace = Sxx + Syy #矩阵迹运算

    # Calculate r for Harris Corner equation
    r = det - k * (trace ** 2)
    r = np.pad(r, ((offset, offset), (offset, offset)))
    output_img[r > threshold,:] = blue #条件判断，大于threshold像素值为蓝色

    return output_img


class Client(tk.Frame):
    def __init__(self, window,r):
        tk.Frame.__init__(self, window)
        self.r = r
        self.url = "127.0.0.1"
        self.port = 9000
        self.photo = None
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mode = "normal"
        self.start = "START"
        self.stop = "STOP"
        self.run = False
        self.have_started = False
        self.content = None
        self.buffer_size = 1024
        self.listener = threading.Thread(target=self.listen, name='ListenThread')
        self.displayer = threading.Thread(target=self.display, name='DisplayThread')

        self.window = window
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.wm_withdraw()
        self.queue = Queue()
        self.create_ui()
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        self.window.wm_deiconify()




    def create_ui(self): #控制客户端UI组件
        self.window.title("控制客户端")

        self.window.rowconfigure(0, minsize=528, weight=1)
        self.window.columnconfigure(1, minsize=800, weight=1)

        self.fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
        self.btn_start = tk.Button(self.fr_buttons, text="Start", command=self.run_all)
        self.mean_filter = tk.Button(self.fr_buttons, text="均值滤波器", command=self.mean_func)
        self.high_pass = tk.Button(self.fr_buttons, text="高通滤波器", command=self.high_pass_func)
        self.low_pass = tk.Button(self.fr_buttons, text="低通滤波器", command=self.low_pass_func)
        self.guassian = tk.Button(self.fr_buttons, text="高斯滤波器", command=self.guassian_func)
        self.harris_corners = tk.Button(self.fr_buttons, text="Harris角点", command=self.harris_func)
        self.template = tk.Button(self.fr_buttons, text="模板匹配", command=self.template_func)
        self.btn_stop = tk.Button(self.fr_buttons, text="Stop", command=self.stop_all)

        self.btn_start.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_stop.grid(row=1, column=0, sticky="ew", padx=5)
        self.mean_filter.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.high_pass.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        self.low_pass.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.guassian.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        self.harris_corners.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        self.template.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

        self.fr_buttons.grid(row=0, column=0, sticky="ns")

        if self.content is None:
            img = np.zeros([528, 800, 3], dtype=np.uint8)
            img = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(img)
        else:
            self.photo = ImageTk.PhotoImage(self.content)
        self.panel = ttk.Label(self.window, image=self.photo)
        self.panel.grid(row=0, column=1)


    def run_all(self): #开始操作
        if not self.run:
            """
            img = np.zeros([528,800, 3], dtype=np.uint8)
            img.fill(255)
            self.content = Image.fromarray(img)
            img = ImageTk.PhotoImage(self.content)
            self.panel.configure(image=img)
            self.panel.image = img
            """
            try:
                print(self.start)
                self.s.sendto(bytes(self.start, "utf-8"),(self.url, self.port))
            except Exception as e:
                print(e)
                self.s.close()
                return
            self.run = True
            if not self.have_started:
                self.displayer.start()
                self.listener.start()
                self.have_started = True

    def mean_func(self):
        self.mode = "mean"

    def harris_func(self):
        self.mode = "harris"

    def template_func(self):
        self.mode = "template"

    def high_pass_func(self):
        self.mode = "high_pass"

    def low_pass_func(self):
        self.mode = "low_pass"

    def guassian_func(self):
        self.mode = "guassian"

    def display(self):
        while True:
            if self.content is not None:
                if self.mode == "normal":
                    self.queue.put(self.content)
                elif self.mode == "mean":
                    arr = MeanFilter(self.content)
                    img = Image.fromarray(arr)
                    self.queue.put(img)
                elif self.mode == "harris":
                    arr = find_harris_corners(self.content)
                    img = Image.fromarray(arr)
                    self.queue.put(img)
                elif self.mode == "template":
                    path = "appendix/2.1_Images/template/000.png"
                    query = Image.open(path, mode='r')
                    query = query.convert('RGB')
                    arr = template_match(self.content,query)
                    img = Image.fromarray(arr)
                    self.queue.put(img)
                elif self.mode == "high_pass":
                    (hpfw, hpfh) = (3, 3)
                    highPassFilter = -1 * np.ones((hpfw, hpfh))
                    highPassFilter[hpfw // 2, hpfh // 2] = -np.sum(highPassFilter) - 1
                    arr = covFilter(self.content,highPassFilter)
                    img = Image.fromarray(arr)
                    self.queue.put(img)
                elif self.mode == "low_pass":
                    (lpfw, lpfh) = (3, 3)
                    lowPassFilter = np.ones((lpfw, lpfh)) * 1 / (lpfw * lpfh)
                    arr = covFilter(self.content,lowPassFilter)
                    img = Image.fromarray(arr)
                    self.queue.put(img)
                elif self.mode == "guassian":
                    (gfw, gfh) = (3, 3)
                    gaussianFilter = gaussFilter((gfw, gfh), 1)
                    arr = covFilter(self.content,gaussianFilter)
                    img = Image.fromarray(arr)
                    self.queue.put(img)
                self.event_generate('<<MessageGenerated>>')

    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            img = self.queue.get()
            self.photo = ImageTk.PhotoImage(img)
            self.panel.configure(image=self.photo)

    def listen(self): #监听服务器线程
        while True:
            try:
                
                fhead_size = struct.calcsize('l')
                buf, addr = self.s.recvfrom(fhead_size)
                if buf:
                    data_size = struct.unpack('l', buf)[0]
                    print("recv:",data_size)
                recvd_size = 0
                data_total = b''
                while not recvd_size == data_size:  #分包读取字节流
                    if data_size - recvd_size > self.buffer_size:
                        data, server = self.s.recvfrom(self.buffer_size)
                        recvd_size += len(data)
                    else:
                        data, addr = self.s.recvfrom(self.buffer_size)
                        recvd_size = data_size
                    data_total += data

                print("recv:",data_total)
                imageStream = io.BytesIO(data_total) #读取字节流
                self.content = Image.open(imageStream).convert("RGB") #转为RGB格式
                imageStream.close()
                print(self.content.size)

                
            except Exception as e:
                print(e)
                continue


    def stop_all(self): #停止操作
        try:
            if self.run:
                self.s.sendto(bytes(self.stop, "utf-8"),(self.url, self.port))
                self.run = False
        except Exception as e:
            print(e)
            self.s.close()

    def quit(self):
        self.s.close()
        self.r.quit()
        self.window.destroy()
        sys.exit()
r = robot.Robot()
r.start()
window = tk.Tk()
client = Client(window,r)
window.mainloop()