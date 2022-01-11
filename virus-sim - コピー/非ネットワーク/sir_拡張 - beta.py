# （1）拡張モジュールのインポート
import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化

# （2）時間変数tの導入
T = 700                  # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
n = 10*T                 # 変数tの範囲をn等分   n=T/h=T/0.1=10*T （T=250のときはn=2500）
h = 0.1                  # 等差数列の公差:0.1 固定
t = np.arange(0,T,h)     # 0から公差dtでTを超えない範囲で等差数列を生成 t[0],...,t[n-1] 要素数n個

# （3）SIRモデル
# 3-1パラメータ
N = 100000             # モデルエリアの人口（人）（東京都1400万人に匹敵するエリアを想定） N=S+I+R＝一定
m1 = 10
m2 = 20                  # 1日1人あたり接触する人数（人）（10,50,100,5と値を変えてシミュレーションを行う）
m3 = 30
p = 0.02               #5接触ごとに感染が生じる1日あたりの確率
d = 14                   # 感染者の回復平均日数（日）
nu1 = 0                 #ワクチン接種率
nu2 = 0.005
nu3 = 0.01
alpha1 = 0.01               #重症化率
alpha2 = 0.01
alpha3 = 0.01
beta1 = m1*p / N           # 接触あたりの感染率
beta2 = m2*p / N           # 接触あたりの感染率
beta3 = m3*p / N           # 接触あたりの感染率

gamma = 1/d              # 回復率（隔離率）
# 3-2初期値
Im_0 = 100                # 初期感染者数（人）100人
Is_0=0
R_0 = 0                  # 初期回復者数（人）0人
S_0 = N - Im_0 - Im_0 - R_0      # 初期未感染者数（人）S_0 = N - Im_0- Is_0 - R_0

# 3-3微分方程式
dSdt = lambda S, Im,Is, R,alpha,beta,nu, t : - beta*S*Im - nu*S              # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dImdt = lambda S, Im,Is, R,alpha,beta,nu, t : beta*S*Im - alpha*Im - gamma*Im       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dIsdt = lambda S, Im,Is, R,alpha,beta,nu, t : alpha*Im-gamma*Is
dRdt = lambda S, Im,Is, R,alpha,beta,nu, t : gamma*(Im+Is)+nu*S                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)
# 3-4数値積分変数S,I,Rをリストとして生成
S1 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im1 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is1=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R1 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S3  = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im3 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is3 = np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R3  = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S2 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im2 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is2=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R2 = np.empty(n)          # R[0],...,R[n-1] 要素数n個
sum1=np.empty(n)
sum2=np.empty(n)
sum3=np.empty(n)
# 3-5初期値代入
S1[0] = S_0
Im1[0] = Im_0
Is1[0]=Is_0
R1[0] = R_0

S2[0] = S_0
Im2[0] = Im_0
Is2[0]=Is_0
R2[0] = R_0

S3[0] = S_0
Im3[0] = Im_0
Is3[0]=Is_0
R3[0] = R_0
sum1[0]=0
sum2[0]=0
sum3[0]=0

# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

  kS1 = h * dSdt( S1[j] ,Im1[j],Is1[j],R1[j] ,alpha1,beta1,nu1 ,t[j] )
  kIm1 = h * dImdt( S1[j] ,Im1[j],Is1[j],R1[j] ,alpha1,beta1,nu1 ,t[j] )
  kIs1 = h * dIsdt( S1[j] ,Im1[j],Is1[j],R1[j] ,alpha1,beta1,nu1 ,t[j] )
  kR1 = h * dRdt( S1[j] ,Im1[j],Is1[j],R1[j] ,alpha1,beta1,nu1 ,t[j] )

  kS2 = h * dSdt( S1[j] + kS1/2 ,Im1[j] + kIm1/2 ,Is1[j]+kIs1/2,R1[j] + kR1/2 ,alpha1,beta1,nu1,t[j] + h/2 )
  kIm2 = h * dImdt( S1[j] + kS1/2 ,Im1[j] + kIm1/2 ,Is1[j]+kIs1/2,R1[j] + kR1/2 ,alpha1,beta1,nu1,t[j] + h/2 )
  kIs2=h * dIsdt( S1[j] + kS1/2 ,Im1[j] + kIm1/2 ,Is1[j]+kIs1/2,R1[j] + kR1/2 ,alpha1,beta1,nu1,t[j] + h/2 )
  kR2 = h * dRdt( S1[j] + kS1/2 ,Im1[j] + kIm1/2 ,Is1[j]+kIs1/2,R1[j] + kR1/2 ,alpha1,beta1,nu1,t[j] + h/2 )

  kS3 = h * dSdt( S1[j] + kS2/2 ,Im1[j] + kIm2/2 ,Is1[j]+kIs2/2,R1[j] + kR2/2, alpha1,beta1,nu1,t[j] + h/2 )
  kIm3 = h * dImdt( S1[j] + kS2/2 ,Im1[j] + kIm2/2 ,Is1[j]+kIs2/2,R1[j] + kR2/2, alpha1,beta1,nu1,t[j] + h/2)
  kIs3 = h * dIsdt( S1[j] + kS2/2 ,Im1[j] + kIm2/2 ,Is1[j]+kIs2/2,R1[j] + kR2/2, alpha1,beta1,nu1,t[j] + h/2)
  
  kR3 = h * dRdt( S1[j] + kS2/2 ,Im1[j] + kIm2/2 ,Is1[j]+kIs2/2,R1[j] + kR2/2, alpha1,beta1,nu1,t[j] + h/2 )

  kS4 = h * dSdt( S1[j] + kS3 ,Im1[j] + kIm3 ,Is1[j]+kIs3,R1[j] + kR3 ,alpha1,beta1,nu1,t[j] + h )
  kIm4 = h * dImdt(S1[j] + kS3 ,Im1[j] + kIm3 ,Is1[j]+kIs3,R1[j] + kR3 ,alpha1,beta1,nu1,t[j] + h )
  kIs4 = h * dIsdt(S1[j] + kS3 ,Im1[j] + kIm3 ,Is1[j]+kIs3,R1[j] + kR3 ,alpha1,beta1,nu1,t[j] + h )

  kR4 = h * dRdt( S1[j] + kS3 ,Im1[j] + kIm3 ,Is1[j]+kIs3,R1[j] + kR3 ,alpha1,beta1,nu1,t[j] + h )

  S1[j+1] = S1[j] + 1/6 * ( kS1 + 2*kS2 + 2*kS3 + kS4 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im1[j+1] = Im1[j] + 1/6 * ( kIm1 + 2*kIm2 + 2*kIm3 + kIm4 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is1[j+1] = Is1[j] + 1/6 * ( kIs1 + 2*kIs2 + 2*kIs3 + kIs4 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R1[j+1] = R1[j] + 1/6 * ( kR1 + 2*kR2 + 2*kR3 + kR4 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  sum1[j+1]=sum1[j]+alpha1*Im1[j]*h
  
  #2個目の計算
  kS1 = h * dSdt( S2[j] ,Im2[j],Is2[j],R2[j] ,alpha1,beta2,nu1 ,t[j] )
  kIm1 = h * dImdt( S2[j] ,Im2[j],Is2[j],R2[j] ,alpha1,beta2,nu1 ,t[j] )
  kIs1 = h * dIsdt( S2[j] ,Im2[j],Is2[j],R2[j] ,alpha1,beta2,nu1 ,t[j] )
  kR1 = h * dRdt( S2[j] ,Im2[j],Is2[j],R2[j] ,alpha1,beta2,nu1 ,t[j] )

  kS2 = h * dSdt( S2[j] + kS1/2 ,Im2[j] + kIm1/2 ,Is2[j]+kIs1/2,R2[j] + kR1/2 ,alpha1,beta2,nu1,t[j] + h/2 )
  kIm2 = h * dImdt( S2[j] + kS1/2 ,Im2[j] + kIm1/2 ,Is2[j]+kIs1/2,R2[j] + kR1/2 ,alpha1,beta2,nu1,t[j] + h/2 )
  kIs2=h * dIsdt( S2[j] + kS1/2 ,Im2[j] + kIm1/2 ,Is2[j]+kIs1/2,R2[j] + kR1/2 ,alpha1,beta2,nu1,t[j] + h/2 )
  kR2 = h * dRdt( S2[j] + kS1/2 ,Im2[j] + kIm1/2 ,Is2[j]+kIs1/2,R2[j] + kR1/2 ,alpha1,beta2,nu1,t[j] + h/2 )

  kS3 = h * dSdt( S2[j] + kS2/2 ,Im2[j] + kIm2/2 ,Is2[j]+kIs2/2,R2[j] + kR2/2, alpha1,beta2,nu1,t[j] + h/2 )
  kIm3 = h * dImdt( S2[j] + kS2/2 ,Im2[j] + kIm2/2 ,Is2[j]+kIs2/2,R2[j] + kR2/2, alpha1,beta2,nu1,t[j] + h/2)
  kIs3 = h * dIsdt( S2[j] + kS2/2 ,Im2[j] + kIm2/2 ,Is2[j]+kIs2/2,R2[j] + kR2/2, alpha1,beta2,nu1,t[j] + h/2)
  
  kR3 = h * dRdt( S2[j] + kS2/2 ,Im2[j] + kIm2/2 ,Is2[j]+kIs2/2,R2[j] + kR2/2, alpha1,beta2,nu1,t[j] + h/2 )

  kS4 = h * dSdt( S2[j] + kS3 ,Im2[j] + kIm3 ,Is2[j]+kIs3,R2[j] + kR3 ,alpha1,beta2,nu1,t[j] + h )
  kIm4 = h * dImdt(S2[j] + kS3 ,Im2[j] + kIm3 ,Is2[j]+kIs3,R2[j] + kR3 ,alpha1,beta2,nu1,t[j] + h )
  kIs4 = h * dIsdt(S2[j] + kS3 ,Im2[j] + kIm3 ,Is2[j]+kIs3,R2[j] + kR3 ,alpha1,beta2,nu1,t[j] + h )
  kR4 = h * dRdt( S2[j] + kS3 ,Im2[j] + kIm3 ,Is2[j]+kIs3,R2[j] + kR3 ,alpha1,beta2,nu1,t[j] + h )

  S2[j+1] = S2[j] + 1/6 * ( kS1 + 2*kS2 + 2*kS3 + kS4 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im2[j+1] = Im2[j] + 1/6 * ( kIm1 + 2*kIm2 + 2*kIm3 + kIm4 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is2[j+1] = Is2[j] + 1/6 * ( kIs1 + 2*kIs2 + 2*kIs3 + kIs4 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R2[j+1] = R2[j] + 1/6 * ( kR1 + 2*kR2 + 2*kR3 + kR4 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  sum2[j+1]=sum2[j]+alpha1*Im2[j]*h
  
  #3つめの計算
  kS1 = h * dSdt( S3[j] ,Im3[j],Is3[j],R3[j] ,alpha1,beta3,nu1 ,t[j] )
  kIm1 = h * dImdt( S3[j] ,Im3[j],Is3[j],R3[j] ,alpha1,beta3,nu1 ,t[j] )
  kIs1 = h * dIsdt( S3[j] ,Im3[j],Is3[j],R3[j] ,alpha1,beta3,nu1 ,t[j] )
  kR1 = h * dRdt( S3[j] ,Im3[j],Is3[j],R3[j] ,alpha1,beta3,nu1 ,t[j] )

  kS2 = h * dSdt( S3[j] + kS1/2 ,Im3[j] + kIm1/2 ,Is3[j]+kIs1/2,R3[j] + kR1/2 ,alpha1,beta3,nu1,t[j] + h/2 )
  kIm2 = h * dImdt( S3[j] + kS1/2 ,Im3[j] + kIm1/2 ,Is3[j]+kIs1/2,R3[j] + kR1/2 ,alpha1,beta3,nu1,t[j] + h/2 )
  kIs2=h * dIsdt( S3[j] + kS1/2 ,Im3[j] + kIm1/2 ,Is3[j]+kIs1/2,R3[j] + kR1/2 ,alpha1,beta3,nu1,t[j] + h/2 )
  kR2 = h * dRdt( S3[j] + kS1/2 ,Im3[j] + kIm1/2 ,Is3[j]+kIs1/2,R3[j] + kR1/2 ,alpha1,beta3,nu1,t[j] + h/2 )

  kS3 = h * dSdt( S3[j] + kS2/2 ,Im3[j] + kIm2/2 ,Is3[j]+kIs2/2,R3[j] + kR2/2, alpha1,beta3,nu1,t[j] + h/2 )
  kIm3 = h * dImdt( S3[j] + kS2/2 ,Im3[j] + kIm2/2 ,Is3[j]+kIs2/2,R3[j] + kR2/2, alpha1,beta3,nu1,t[j] + h/2)
  kIs3 = h * dIsdt( S3[j] + kS2/2 ,Im3[j] + kIm2/2 ,Is3[j]+kIs2/2,R3[j] + kR2/2, alpha1,beta3,nu1,t[j] + h/2)
  kR3 = h * dRdt( S3[j] + kS2/2 ,Im3[j] + kIm2/2 ,Is3[j]+kIs2/2,R3[j] + kR2/2, alpha1,beta3,nu1,t[j] + h/2 )

  kS4 = h * dSdt( S3[j] + kS3 ,Im3[j] + kIm3 ,Is3[j]+kIs3,R3[j] + kR3 ,alpha1,beta3,nu1,t[j] + h )
  kIm4 = h * dImdt(S3[j] + kS3 ,Im3[j] + kIm3 ,Is3[j]+kIs3,R3[j] + kR3 ,alpha1,beta3,nu1,t[j] + h )
  kIs4 = h * dIsdt(S3[j] + kS3 ,Im3[j] + kIm3 ,Is3[j]+kIs3,R3[j] + kR3 ,alpha1,beta3,nu1,t[j] + h )
  kR4 = h * dRdt( S3[j] + kS3 ,Im3[j] + kIm3 ,Is3[j]+kIs3,R3[j] + kR3 ,alpha1,beta3,nu1,t[j] + h )

  S3[j+1] = S3[j] + 1/6 * ( kS1 + 2*kS2 + 2*kS3 + kS4 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im3[j+1] = Im3[j] + 1/6 * ( kIm1 + 2*kIm2 + 2*kIm3 + kIm4 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is3[j+1] = Is3[j] + 1/6 * ( kIs1 + 2*kIs2 + 2*kIs3 + kIs4 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R3[j+1] = R3[j] + 1/6 * ( kR1 + 2*kR2 + 2*kR3 + kR4 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  sum3[j+1]=sum3[j]+alpha1*Im3[j]*h
  

print(max(Is1))
print(max(Is2))
print(max(Is3))
print(sum1[n-1])

print(sum2[n-1])
print(sum3[n-1])
# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
#plt.plot(t, S1, color = "green", label = "S:未感染者", linewidth = 1.0)
#plt.plot(t, Is1, color = "red", label = "Is:重症", linewidth = 1.0)
plt.plot(t, Is1, color = "blue", label = "beta=2.0*10^-8 max={:.3f}".format(max(Is1)/N), linewidth = 1.0)
plt.plot(t, Is2, color = "red", label = "beta=4.0*10^-8 max={:.3f}".format(max(Is2)/N), linewidth = 1.0)
plt.plot(t, Is3, color = "green", label = "beta=6.0*10^-8 max={:.3f}".format(max(Is3)/N), linewidth = 1.0)

#plt.plot(t, R1, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('小さい街:βを変更:時間変化')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()


plt.plot(t, sum1, color = "blue", label = "beta=2.0*10^-8 sum={:.3f}".format(sum1[n-1]/N), linewidth = 1.0)
plt.plot(t, sum2, color = "red", label = "beta=4.0*10^-8 sum={:.3f}".format(sum2[n-1]/N), linewidth = 1.0)
plt.plot(t, sum3, color = "green", label = "beta=6.0*10^-8 sum={:.3f}".format(sum3[n-1]/N), linewidth = 1.0)

#plt.plot(t, R1, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('小さい街:βを変更:累積')  # グラフタイトル パラメータmとTの値表示
plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()
