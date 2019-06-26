# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:10:31 2019

@author: lee
"""
import socket
import time

uplink = "exit"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost", 9201))
time.sleep(1)
uplink = uplink.encode()
sock.send(uplink)

print("上联:"+uplink.decode())
print("下联:"+sock.recv(1024).decode())

sock.close() 