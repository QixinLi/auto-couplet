# -*- coding: utf-8 -*-
# @Author  : Qixin Li
# @Time    : 2021/4/26 下午5:12

import socket

uplink = "gogogo"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost", 9201))
while uplink != "exit":
    print("=" * 30)
    uplink = input("输入:")
    if not uplink:
        continue

    uplink = uplink.encode()
    sock.send(uplink)

    print("上联:" + uplink.decode())
    print("下联:" + sock.recv(1024).decode())

sock.close()