import socket
from PIL import Image
import io
import struct
from threading import Thread
import time
import sys
class Robot(Thread):
    def __init__(self):
        super(Robot,self).__init__()
        self.address=None
        self.listen_port = 9000
        self.buffer_size = 1024
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(('127.0.0.1', self.listen_port))
        self.START = 'START'
        self.STOP = 'STOP'
        self.txt = ""

    def listen_method(self): #监听客户端接口
        while True:
            try:
                data, self.address = self.s.recvfrom(self.buffer_size)
                txt = str(data.decode("utf-8"))
                print(txt)
                self.txt = txt
            except Exception as e:
                print(e)
                self.s.close()
                return

    def send_method(self): #发送给客户端接口
        while True:
            try:
                # send image to server
                if self.txt.startswith(self.START):#如果接收到start指令开始发送
                    for i in range(1,5):#图片标签
                        path = "appendix/2.1_Images/source/00" + str(i) + ".png"
                        img = Image.open(path, mode='r')
                        img = img.convert('RGB')
                        print(img.size)
                        imgByteArr = io.BytesIO()
                        img.save(imgByteArr, format='jpeg')
                        imgByteArr = imgByteArr.getvalue()
                        print(len(imgByteArr))
                        print(imgByteArr)
                        # send image size to server
                        if self.address is not None:
                            fhead = struct.pack('l', len(imgByteArr))#包头发送图片文件大小
                            self.s.sendto(fhead, self.address)
                            for i in range(len(imgByteArr) // self.buffer_size + 1): #分包发送
                                if 1024 * (i + 1) > len(imgByteArr):
                                    self.s.sendto(imgByteArr[self.buffer_size * i:], self.address)
                                else:
                                    self.s.sendto(imgByteArr[self.buffer_size * i:self.buffer_size * (i + 1)]
                                                  , self.address)
                        # check what server send
                        if self.txt.startswith(self.STOP):
                            break
                        time.sleep(0.5)
                    #self.s.close()
                    self.txt = ""
                    time.sleep(1)

            except Exception as e:
                print(e)
                self.s.close()
                return
    def quit(self):
        self.s.close()

    def run(self):
        t1 = Thread(target=self.listen_method)#启动监听线程
        t2 = Thread(target=self.send_method)#启动发送线程
        t1.start()
        t2.start()