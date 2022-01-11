# （1）拡張モジュールのインポート
import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化

# （2）時間変数tの導入
Time = 20                # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
n = 10*Time              # 変数tの範囲をn等分   n=T/h=T/0.1=10*T （T=250のときはn=2500）
h = 0.1                  # 等差数列の公差:0.1 固定
t = np.arange(0,Time,h)  # 0から公差dtでTを超えない範囲で等差数列を生成 t[0],...,t[n-1] 要素数n個

# （3）SIRモデル
# 3-1パラメータ
lamda = 0                #標的細胞の増加率
d=0                      #標的細胞の死亡率
beta=0.00001157          #ウィルス粒子の侵入率
delta=3.412              #感染細胞の死亡率
p=0.020099               #ウィルス粒子を吐き出す数
c=3.381                  #ウィルス粒子の死亡率
# 3-2初期値
I_0 = 1                  # 初期感染細胞
V_0 = 0.961              # 初期ウィルス粒子数
T_0 = 400000000          #初期標的細胞
# 3-3微分方程式
dTdt = lambda T, I, V, t : lamda-d*T-beta*T*V      # dTdt ≡ dT/dt  dSdt(T, I, V, t)
dIdt = lambda T, I, V, t : beta*T*V-delta*I        # dIdt ≡ dI/dt  dIdt(T, I, V, t)
dVdt = lambda T, I, V, t : p*I-c*V                 # dVdt ≡ dV/dt  dRdt(T ,I ,V, t)
# 3-4数値積分変数S,I,Rをリストとして生成
T = np.empty(n)          # T[0],...,T[n-1] 要素数n個
I = np.empty(n)          # I[0],...,I[n-1] 要素数n個
V = np.empty(n)          # V[0],...,R[n-1] 要素数n個
# 3-5初期値代入
T[0] = T_0
I[0] = I_0
V[0] = V_0

# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個
  
  
  kT1 = h * dTdt( T[j] ,I[j] ,V[j] ,t[j] )
  kI1 = h * dIdt( T[j] ,I[j] ,V[j] ,t[j] )
  kV1 = h * dVdt( T[j] ,I[j] ,V[j] ,t[j] )

  kT2 = h * dTdt( T[j] + kT1/2 ,I[j] + kI1/2 ,V[j] + kV1/2 ,t[j] + h/2 )
  kI2 = h * dIdt( T[j] + kT1/2 ,I[j] + kI1/2 ,V[j] + kV1/2 ,t[j] + h/2 )
  kV2 = h * dVdt( T[j] + kT1/2 ,I[j] + kI1/2 ,V[j] + kV1/2 ,t[j] + h/2 )

  kT3 = h * dTdt( T[j] + kT2/2 ,I[j] + kI2/2 ,V[j] + kV2/2, t[j] + h/2 )
  kI3 = h * dIdt( T[j] + kT2/2 ,I[j] + kI2/2 ,V[j] + kV2/2, t[j] + h/2 )
  kV3 = h * dVdt( T[j] + kT2/2 ,I[j] + kI2/2 ,V[j] + kV2/2, t[j] + h/2 )

  kT4 = h * dTdt( T[j] + kT3 ,I[j] + kI3 ,V[j] + kV3 ,t[j] + h )
  kI4 = h * dIdt( T[j] + kT3 ,I[j] + kI3 ,V[j] + kV3 ,t[j] + h )
  kV4 = h * dVdt( T[j] + kT3 ,I[j] + kI3 ,V[j] + kV3 ,t[j] + h )

  T[j+1] = T[j] + 1/6 * ( kT1 + 2*kT2 + 2*kT3 + kT4 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  I[j+1] = I[j] + 1/6 * ( kI1 + 2*kI2 + 2*kI3 + kI4 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  V[j+1] = V[j] + 1/6 * ( kV1 + 2*kV2 + 2*kV3 + kV4 )   # 末項 j=n-2 -> R[j+1]=R[n-1]

# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
plt.plot(t, T, color = "green", label = "T:標的細胞", linewidth = 1.0)
plt.plot(t, I, color = "red", label = "I:感染細胞", linewidth = 1.0)
plt.plot(t, V, color= "blue", label = "V:ウィルス粒子数", linewidth = 1.0)
# グラフの見た目設定
plt.xlabel('時間')                         # 横軸ラベル
plt.ylabel('数（総数に対する割合）')        # 縦軸ラベル
plt.grid(True)                           # グリッド表示
#plt.legend()                             # 凡例表示
# 設定反映しプロット描画
plt.yscale('log')
plt.show()
