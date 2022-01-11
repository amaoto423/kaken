
# （1）拡張モジュールのインポート
import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化
import math
import random
# （2）時間変数tの導入
T = 700                  # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
n = 10*T                 # 変数tの範囲をn等分   n=T/h=T/0.1=10*T （T=250のときはn=2500）
h = 0.1                  # 等差数列の公差:0.1 固定
time = np.arange(0,T,h)     # 0から公差dtでTを超えない範囲で等差数列を生成 t[0],...,t[n-1] 要素数n個
c=np.zeros((10,10,10,10))
# （3）SIRモデル
# 3-1パラメータ
N=np.zeros((10,10))
m=np.zeros((10,10))                # 1日1人あたり接触する人数（人）（10,50,100,5と値を変えてシミュレーションを行う）
p = 0.02               #5接触ごとに感染が生じる1日あたりの確率
d = 14                   # 感染者の回復平均日数（日）
beta = np.zeros((10,10))           # 接触あたりの感染率
beta_p=np.zeros((10,10,10,10))
gamma = 1/d              # 回復率（隔離率）
alpha=0.01                #重症化率
# 3-2初期値

for i in range(10):
  for j in range(10):
    N[i][j]=random.randrange(10000,100000)
    
    m[i][j]=random.randrange(10,100)
    beta[i][j]=m[i][j]*p/N[i][j]

for i in range(10):
  for j in range(10):
    for s in range(10):
      for t in range(10):
        if i!=s or j!=t:
          beta_p[i][j][s][t]=(beta[i][j]+beta[s][t])/(100*math.sqrt((i-s)**2+(t-j)**2))


# 3-4数値積分変数S,I,Rをリストとして生成
S = np.zeros((10,10,n))          # S[0],...,S[n-1] 要素数n個
Im = np.zeros((10,10,n))          # I[0],...,I[n-1] 要素数n個
Is = np.zeros((10,10,n))          # I[0],...,I[n-1] 要素数n個
R = np.zeros((10,10,n))          # R[0],...,R[n-1] 要素数n個
Is_sum=np.zeros(n)
Im_sum=np.zeros(n)
# 3-5初期値代入
for i in range(10):
  for j in range(10):
      #Im [i][j][0]=10*random.random()
      Im[i][j][0]=0
      Is[i][j][0]=0
      R[i][j][0]=0
      S[i][j][0]=N[i][j]-Im[i][j][0]
Im[5][5][0]=100
S[5][5][0]=N[5][5]-Im[5][5][0]


for i in range(10):
  for j in range(10):
    for s in range(10):
      for t in range(10):
        if i==s and j==t:
          c[i][j][s][t]=0
        elif (i==s and abs(j-t)==1) or (abs(i-s)==1 and j==t):
          c[i][j][s][t]=1
        elif 1-math.exp(-(math.sqrt((i-s)**2+(j-t)**2)))<random.random():
          c[i][j][s][t]=1
        
       # print("c[{}][{}][{}][{}]=".format(i,j,s,t))
        #print(c[5][5][5][6])


for t in range(n-1): 
  for i in range(10):
    for j in range(10):
      Su=0
      for k in range(10):
        for s in range(10):
            Su=Su+c[i][j][k][s]*beta_p[i][j][k][s]*S[i][j][t]*Im[k][s][t]
      S[i][j][t+1]=S[i][j][t]-h*beta[i][j]*S[i][j][t]*Im[i][j][t]-h*Su
      Im[i][j][t+1]=Im[i][j][t]+h*Su+h*beta[i][j]*S[i][j][t]*Im[i][j][t]-h*alpha*Im[i][j][t]
      Is[i][j][t+1]=Is[i][j][t]+h*alpha*Im[i][j][t]-h*gamma*Is[i][j][t]
      R[i][j][t+1]=R[i][j][t]+h*gamma*(Im[i][j][t]+Is[i][j][t])
for t in range(n):
  for i in range(10):
    for j in range(10):
      Im_sum[t]=Im_sum[t]+Im[i][j][t]
      Is_sum[t]=Is_sum[t]+Is[i][j][t]


# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods


# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
for i in range(10):
  for j in range(10):
    plt.plot(time, Is[i][j], color = "red", linewidth = 1.0)
# グラフの見た目設定
plt.title('各ノードIs')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()



for i in range(10):
  for j in range(10):
    plt.plot(time, Im[i][j], color = "blue", linewidth = 1.0)
# グラフの見た目設定
plt.title('各ノードのIm')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()


plt.plot(time,Im_sum,color="green",linewidth=1.0)

plt.title('Imの和')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()



plt.plot(time,Is_sum,color="black",linewidth=1.0)

plt.title('Isの和')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()