# -*- coding = utf-8 -*-
# @Time : 2020/12/14 16:34
# @Author : 王浩
# @File : demo.py
# @Software : PyCharm

from Sat_IoT_env import Sat_IoT
import matplotlib.pyplot as plt
# import pylab
import numpy as np

if __name__ == "__main__":
    # print(plt.__version__)
    plt.switch_backend('TkAgg')
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
'''  
if __name__ == "__main__":
    sat_iot = Sat_IoT()
    # sat_iot.__init__()
    # sat_iot.show_system()
    sat_iot.step(51283)
    sat_iot.show_system()
    sat_iot.step(512)
    sat_iot.show_system()
'''