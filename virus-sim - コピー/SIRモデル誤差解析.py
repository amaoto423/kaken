import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化

x=np.empty(3)
y=np.empty(3)

x[0]=0.05
x[1]=0.1
x[2]=0.2

y[0]=15.28903168
y[1]=15.26409551
y[2]=15.21429203
plt.plot(x,y)
plt.scatter(x,y)
plt.title('R0=2')  # グラフタイトル パラメータmとTの値表示
plt.xlabel('dt')                              # 横軸ラベル
plt.ylabel('｜解析解-数値解｜')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()
