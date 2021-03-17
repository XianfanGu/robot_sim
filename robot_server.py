import socket
from PIL import Image
import io
import tkinter as tk
import struct
from threading import Thread
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
import time
import threading

class RobotServer(tk.Frame):
    def __init__(self, window):
        tk.Frame.__init__(self, window)
        self.window = window
        self.address = None
        self.listen_port = 6789
        self.buffer_size = 19
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(('127.0.0.1', self.listen_port))
        self.txt = ""
        self.continuePlotting = False
        self.data_point = []
        self.data_point1 = []
        self.data_point2 = []
        self.data_point3 = []
        self.data_point4 = []
        self.data_point5 = []
        self.data_point6 = []
        self.data_point7 = []

        self.fig = plt.figure(figsize=(14, 5), dpi=100)
        self.fig.subplots_adjust(hspace=.5)

        self.ax = self.fig.add_subplot(2, 4, 1)
        self.ax.set_ylim(0, 10)
        self.ax.set_xlim(0, 50)
        self.ax.set_title("channel 1")

        self.ax1 = self.fig.add_subplot(2, 4, 2)
        self.ax1.set_ylim(0, 10)
        self.ax1.set_xlim(0, 50)
        self.ax1.set_title("channel 2")

        self.ax2 = self.fig.add_subplot(2, 4, 3)
        self.ax2.set_ylim(0, 10)
        self.ax2.set_xlim(0, 50)
        self.ax2.set_title("channel 3")

        self.ax3 = self.fig.add_subplot(2, 4, 4)
        self.ax3.set_ylim(0, 10)
        self.ax3.set_xlim(0, 50)
        self.ax3.set_title("channel 4")

        self.ax4 = self.fig.add_subplot(2, 4, 5)
        self.ax4.set_ylim(0, 10)
        self.ax4.set_xlim(0, 50)
        self.ax4.set_title("channel 5")

        self.ax5 = self.fig.add_subplot(2, 4, 6)
        self.ax5.set_ylim(0, 10)
        self.ax5.set_xlim(0, 50)
        self.ax5.set_title("channel 6")

        self.ax6 = self.fig.add_subplot(2, 4, 7)
        self.ax6.set_ylim(0, 10)
        self.ax6.set_xlim(0, 50)
        self.ax6.set_title("channel 7")

        self.ax7 = self.fig.add_subplot(2, 4, 8)
        self.ax7.set_ylim(0, 10)
        self.ax7.set_xlim(0, 50)
        self.ax7.set_title("channel 8")


        self.line1, = self.ax.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line2, = self.ax1.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line3, = self.ax2.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line4, = self.ax3.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line5, = self.ax4.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line6, = self.ax5.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line7, = self.ax6.plot([i for i in range(len(self.data_point))], self.data_point)
        self.line8, = self.ax7.plot([i for i in range(len(self.data_point))], self.data_point)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        #ani = animation.FuncAnimation(self.fig, self.update, interval=10, blit=False)
        self.run()
        self.window.protocol("WM_DELETE_WINDOW", self.quit)


    def listen_method(self): #监听客户端接口
        while True:
            try:
                data, self.address = self.s.recvfrom(self.buffer_size)
                arr = [data[i:i+1].hex() for i in range(len(data))]
                arr = arr[1:17]
                hex_list = [arr[i+1]+arr[i]+"0000" for i in range(0,len(arr),2)]
                #print(hex_list)
                if(len(self.data_point)>=50):
                    self.data_point.pop(0)
                self.data_point.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[0]))[0]))
                if(len(self.data_point1)>=50):
                    self.data_point1.pop(0)
                self.data_point1.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[1]))[0]))
                if(len(self.data_point2)>=50):
                    self.data_point2.pop(0)
                self.data_point2.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[2]))[0]))
                if(len(self.data_point3)>=50):
                    self.data_point3.pop(0)
                self.data_point3.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[3]))[0]))
                if(len(self.data_point4)>=50):
                    self.data_point4.pop(0)
                self.data_point4.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[4]))[0]))
                if(len(self.data_point5)>=50):
                    self.data_point5.pop(0)
                self.data_point5.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[5]))[0]))
                if(len(self.data_point6)>=50):
                    self.data_point6.pop(0)
                self.data_point6.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[6]))[0]))
                if(len(self.data_point7)>=50):
                    self.data_point7.pop(0)
                self.data_point7.append(10**41*float(struct.unpack('<f', bytes.fromhex(hex_list[7]))[0]))
                #print(float_list)
            except Exception as e:
                print(e)
                self.s.close()
                return

    def run(self):
        t1 = Thread(target=self.listen_method)#启动监听线程
        t1.start()

    def update(self,i):
        print(self.data_point)
        self.line1.set_data([i for i in range(len(self.data_point))], self.data_point)
        self.line2.set_data([i for i in range(len(self.data_point1))], self.data_point1)
        self.line3.set_data([i for i in range(len(self.data_point2))], self.data_point2)
        self.line4.set_data([i for i in range(len(self.data_point3))], self.data_point3)
        self.line5.set_data([i for i in range(len(self.data_point4))], self.data_point4)
        self.line6.set_data([i for i in range(len(self.data_point5))], self.data_point5)
        self.line7.set_data([i for i in range(len(self.data_point6))], self.data_point6)
        self.line8.set_data([i for i in range(len(self.data_point7))], self.data_point7)

    def add_point(self, line, y):
        coords = self.canvas.coords(line)
        x = coords[-2] + 5
        coords.append(x)
        coords.append(y)
        coords = coords[-1000:] # keep # of points to a manageable size
        self.canvas.coords(line, *coords)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def quit(self):
        self.s.close()
        self.window.destroy()
        sys.exit()
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotServer(root)
    app.pack(side="top", fill="both", expand=True)
    ani = animation.FuncAnimation(app.fig, app.update, interval=1000, blit=False)
    root.mainloop()
