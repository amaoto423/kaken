# （1）拡張モジュールのインポート
import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化
import math
# （2）時間変数tの導入
T = 700                  # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
n = 10*T                 # 変数tの範囲をn等分   n=T/h=T/0.1=10*T （T=250のときはn=2500）
h = 0.1                  # 等差数列の公差:0.1 固定
t = np.arange(0,T,h)     # 0から公差dtでTを超えない範囲で等差数列を生成 t[0],...,t[n-1] 要素数n個

# （3）SIRモデル
# 3-1パラメータ
N = 10000000             # モデルエリアの人口（人）（東京都1400万人に匹敵するエリアを想定） N=S+I+R＝一定
m = 20                  # 1日1人あたり接触する人数（人）（10,50,100,5と値を変えてシミュレーションを行う）
p = 0.02               #5接触ごとに感染が生じる1日あたりの確率
d = 10                   # 感染者の回復平均日数（日）
nu=0                 #ワクチン接種率
alpha=0              #重症化率
beta = m*p / N           # 接触あたりの感染率
gamma = 1/d              # 回復率（隔離率）
# 3-2初期値
I_0 = 100                # 初期感染者数（人）100人
R_0 = 0                  # 初期回復者数（人）0人
S_0 = N - I_0 - R_0      # 初期未感染者数（人）S_0 = N - Im_0- Is_0 - R_0

# 3-3微分方程式
dSdt = lambda S, I, R, t : - beta*S*I               # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIdt = lambda S, I, R, t : beta*S*I - gamma*I       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dRdt = lambda S, I, R, t : gamma*I                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)
# 3-4数値積分変数S,I,Rをリストとして生成
S = np.empty(n)          # S[0],...,S[n-1] 要素数n個
I = np.empty(n)          # I[0],...,I[n-1] 要素数n個
R = np.empty(n)          # R[0],...,R[n-1] 要素数n個
energy=np.empty(n)
# 3-5初期値代入
S[0] = S_0
I[0] = I_0
R[0] = R_0
energy[0]=I[0]+S[0]-(gamma/beta)*math.log(S[0])
# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

  kS1 = h * dSdt( S[j] ,I[j] ,R[j] ,t[j] )
  kI1 = h * dIdt( S[j] ,I[j] ,R[j] ,t[j] )
  kR1 = h * dRdt( S[j] ,I[j] ,R[j] ,t[j] )

  kS2 = h * dSdt( S[j] + kS1/2 ,I[j] + kI1/2 ,R[j] + kR1/2 ,t[j] + h/2 )
  kI2 = h * dIdt( S[j] + kS1/2 ,I[j] + kI1/2 ,R[j] + kR1/2 ,t[j] + h/2 )
  kR2 = h * dRdt( S[j] + kS1/2 ,I[j] + kI1/2 ,R[j] + kR1/2 ,t[j] + h/2 )

  kS3 = h * dSdt( S[j] + kS2/2 ,I[j] + kI2/2 ,R[j] + kR2/2, t[j] + h/2 )
  kI3 = h * dIdt( S[j] + kS2/2 ,I[j] + kI2/2 ,R[j] + kR2/2, t[j] + h/2 )
  kR3 = h * dRdt( S[j] + kS2/2 ,I[j] + kI2/2 ,R[j] + kR2/2, t[j] + h/2 )

  kS4 = h * dSdt( S[j] + kS3 ,I[j] + kI3 ,R[j] + kR3 ,t[j] + h )
  kI4 = h * dIdt( S[j] + kS3 ,I[j] + kI3 ,R[j] + kR3 ,t[j] + h )
  kR4 = h * dRdt( S[j] + kS3 ,I[j] + kI3 ,R[j] + kR3 ,t[j] + h )

  S[j+1] = S[j] + 1/6 * ( kS1 + 2*kS2 + 2*kS3 + kS4 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  I[j+1] = I[j] + 1/6 * ( kI1 + 2*kI2 + 2*kI3 + kI4 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  R[j+1] = R[j] + 1/6 * ( kR1 + 2*kR2 + 2*kR3 + kR4 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  energy[j+1]=S[j+1]+I[j+1]-(gamma/beta)*math.log(S[j+1])
  
print(R[n-1])
# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
#plt.plot(t, S, color = "green", label = "S:未感染者", linewidth = 1.0)
plt.plot(t, I, color = "red", label = "I:感染者", linewidth = 1.0)
#plt.plot(t, R, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
#plt.title('SIRモデル RK4によるシミュレーション)')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
#plt.xlabel('時間（日）')                              # 横軸ラベル
#plt.ylabel('人数（総人口1000万人に対する割合）')      # 縦軸ラベル
#plt.grid(True)                                        # グリッド表示
#plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()


plt.plot(t,energy, color= "black", label = "保存量変化", linewidth = 1.0)
plt.xlabel('時間')                              # 横軸ラベル
plt.ylabel('S+I-(gamma/nu)log S')      
plt.legend()                                          # 凡例表示

plt.show()
