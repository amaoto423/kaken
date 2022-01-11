"""
(4) 3次元ベクトル場の図示
原点においた点電荷が作る静電場
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D #3次元プロットのためのimport

import numpy as np

fig = plt.figure()
ax = Axes3D(fig)


LX, LY, LZ = 2,2,2  # xyzメッシュのパラメータ
gridwidth=0.9 # 
X, Y, Z= np.meshgrid(np.arange(-LX, LX, gridwidth), np.arange(-LY, LY,gridwidth),np.arange(-LZ, LZ, gridwidth) ) #メッシュ生成

lamda,d,beta,delta,p,c=0,0,0.00001,3.4,3.381,3.3

U = lamda-d*X-beta*X*Z
V = beta*X*Z-delta*Y
W= p*Y-c*Z


ax.quiver(X, Y, Z, U, V, W, color='red',length=1, normalize=False)

ax.set_xlim([-LX, LX])
ax.set_ylim([-LY, LY])
ax.set_zlim([-LZ, LZ])
plt.show()
