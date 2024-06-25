"""
@Create Time : 2023/11/27 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 加载并预处理待处理的图像
image_path = '01.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cm = plt.cm.get_cmap('jet')
plt.imshow(image, cmap=cm)
plt.colorbar()
# plt.savefig("_heatmap.png")
plt.show()

#
