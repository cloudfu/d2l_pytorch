import matplotlib.pyplot as plt
import numpy as np

# figsize 指定图标尺寸
fig=plt.figure(figsize=(7,8),facecolor='blue')
x = np.arange(0, 100)  
#作图1
plt.subplot(221)  
plt.plot(x, x)  
#作图2
plt.subplot(222)  
plt.plot(x, -x)  
 #作图3
plt.subplot(223)  
plt.plot(x, x ** 2)  
plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
#作图4
plt.subplot(224)  
plt.plot(x, np.log(x))  
plt.show()  