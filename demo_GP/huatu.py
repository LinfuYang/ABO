import numpy as np

import matplotlib.pyplot as plt

np.random.seed(2000)
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_2 = plt.subplot(122)

plt.sca(ax_1)
y = np.random.standard_normal((10, 1))
plt.plot([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], y[:,0], lw = 1.5,label = '%s-st'%str(2))

plt.sca(ax_2)
y = np.random.standard_normal((10, 1))
plt.plot([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], y[:,0], lw = 1.5,label = '%s-st'%str(2))

plt.grid(True)
plt.legend(loc=0) #图例位置自动
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A simple plot')
plt.show()