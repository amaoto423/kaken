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
N1 = 10000000             # モデルエリアの人口（人）（東京都1400万人に匹敵するエリアを想定） N=S+I+R＝一定
N2 = 100000
N3 = 1000000
m1 = 10
m2 = 10                  # 1日1人あたり接触する人数（人）（10,50,100,5と値を変えてシミュレーションを行う）
m3 = 10
p = 0.02               #5接触ごとに感染が生じる1日あたりの確率
d = 14                   # 感染者の回復平均日数（日）
nu1 = 0                 #ワクチン接種率
nu2 = 0
nu3 = 0
alpha1 = 0.01               #重症化率
alpha2 = 0.01
alpha3 = 0.01
beta1 = m1*p / N1           # 接触あたりの感染率
beta2 = m2*p / N2           # 接触あたりの感染率
beta3 = m3*p / N3           # 接触あたりの感染率
beta12=(beta1+beta2)/100
beta23=(beta2+beta3)/100
beta31=(beta3+beta1)/100

c12=1
c23=1
c31=1

gamma = 1/d              # 回復率（隔離率）
# 3-2初期値
Im1_0 = 100                # 初期感染者数（人）100人
Is1_0=0
R1_0 = 0                  # 初期回復者数（人）0人
S1_0 = N1 - Im1_0 - Is1_0 - R1_0      # 初期未感染者数（人）S_0 = N - Im_0- Is_0 - R_0

Im2_0 = 0                # 初期感染者数（人）100人
Is2_0=0
R2_0 = 0                  # 初期回復者数（人）0人
S2_0 = N2 - Im2_0 - Is2_0 - R2_0      # 初期未感染者数（人）S_0 = N - Im_0- Is_0 - R_0

Im3_0 = 0                # 初期感染者数（人）100人
Is3_0=0
R3_0 = 0                  # 初期回復者数（人）0人
S3_0 = N3 - Im3_0 - Is3_0 - R3_0      # 初期未感染者数（人）S_0 = N - Im_0- Is_0 - R_0

# 3-3微分方程式
dS1dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t : - beta1*S1*Im1 - nu1*S1-c12*beta12* S1*Im2-c31*beta31* S1*Im3             # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIm1dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t: beta1*S1*Im1 - alpha*Im1 - gamma*Im1+c12*beta12* S1*Im2+c31*beta31* S1*Im3       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dIs1dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t : alpha*Im1-gamma*Is1
dR1dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t: gamma*(Im1+Is1)+nu1*S1                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)

dS2dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t : - beta2*S2*Im2 - nu2*S2-c12*beta12* S2*Im1-c23*beta23* S2*Im3             # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIm2dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t: beta2*S2*Im2 - alpha*Im2 - gamma*Im2+c12*beta12* S2*Im1+c23*beta23* S2*Im3       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dIs2dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t : alpha*Im2-gamma*Is2
dR2dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t: gamma*(Im2+Is2)+nu2*S2                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)

dS3dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t : - beta3*S3*Im3 - nu3*S3 - c31*beta31* S3*Im1-c23*beta23* S3*Im2             # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIm3dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t: beta3*S3*Im3 - alpha*Im3 - gamma*Im3+c31*beta31* S3*Im1+c23*beta23* S3*Im2       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dIs3dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t : alpha*Im3-gamma*Is3
dR3dt = lambda S1,S2,S3, Im1,Im2,Im3,Is1,Is2,Is3, R1,R2,R3,alpha,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3,c12,c23,c31, t: gamma*(Im3+Is3)+nu3*S3                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)

# 3-4数値積分変数S,I,Rをリストとして生成
S1 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im1 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is1=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R1 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S2 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im2 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is2=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R2 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S3  = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im3 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is3 = np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R3  = np.empty(n)          # R[0],...,R[n-1] 要素数n個

sum1=np.empty(n)
sum2=np.empty(n)
sum3=np.empty(n)
# 3-5初期値代入
S1[0] = S1_0
Im1[0] = Im1_0
Is1[0]=Is1_0
R1[0] = R1_0

S2[0] = S2_0
Im2[0] = Im2_0
Is2[0]=Is2_0
R2[0] = R2_0

S3[0] = S3_0
Im3[0] = Im3_0
Is3[0]=Is3_0
R3[0] = R3_0
sum1[0]=0
sum2[0]=0
sum3[0]=0

# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

  kS11 = h * dS1dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIm11 = h * dIm1dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs11 = h * dIs1dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR11 = h * dR1dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kS21 = h * dS2dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIm21 = h * dIm2dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs21 = h * dIs2dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR21 = h * dR2dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kS31 = h * dS3dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIm31 = h * dIm3dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs31 = h * dIs3dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR31 = h * dR3dt( S1[j],S2[j],S3[j] ,Im1[j],Im2[j],Im3[j],Is1[j],Is2[j],Is3[j],R1[j] ,R2[j],R3[j],alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])

  kS12 = h * dS1dt(  S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIm12 =h * dIm1dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs12 =h * dIs1dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR12 = h * dR1dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kS22 = h * dS2dt(  S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIm22 =h * dIm2dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs22 =h * dIs2dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR22 = h * dR2dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kS32 = h * dS3dt(  S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIm32 =h * dIm3dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs32 =h * dIs3dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR32 = h * dR3dt(S1[j]+kS11/2,S2[j]+kS21/2,S3[j]+kS31/2 ,Im1[j]+kIm11/2,Im2[j]+kIm21/2,Im3[j]+kIm31/2,Is1[j]+kIs11/2,Is2[j]+kIs21/2,Is3[j]+kIs31/2,R1[j]+kR11/2 ,R2[j]+kR21/2,R3[j]+kR31/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])



  kS13 = h * dS1dt( S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIm13 = h * dIm1dt( S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIs13 = h * dIs1dt(  S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR13 = h * dR1dt(  S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kS23 = h * dS2dt( S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIm23 = h * dIm2dt( S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIs23 = h * dIs2dt(  S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR23 = h * dR2dt(  S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kS33 = h * dS3dt( S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIm33 = h * dIm3dt( S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIs33 = h * dIs3dt(  S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR33 = h * dR3dt(  S1[j]+kS12/2,S2[j]+kS22/2,S3[j]+kS32/2 ,Im1[j]+kIm12/2,Im2[j]+kIm22/2,Im3[j]+kIm32/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,Is3[j]+kIs32/2,R1[j]+kR12/2 ,R2[j]+kR22/2,R3[j]+kR32/2,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])




  kS14 = h * dS1dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIm14 = h * dIm1dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs14 = h * dIs1dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR14 = h * dR1dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kS24 = h * dS2dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIm24 = h * dIm2dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs24 = h * dIs2dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR24 = h * dR2dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kS34 = h * dS3dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kIm34 = h * dIm3dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )
  kIs34 = h * dIs3dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j])
  kR34 = h * dR3dt(S1[j]+kS13,S2[j]+kS23,S3[j]+kS33 ,Im1[j]+kIm13,Im2[j]+kIm23,Im3[j]+kIm33,Is1[j]+kIs13,Is2[j]+kIs23,Is3[j]+kIs33,R1[j]+kR13 ,R2[j]+kR23,R3[j]+kR33,alpha1,beta1,beta2,beta3,beta12,beta23,beta31,nu1,nu2,nu3 ,c12,c23,c31,t[j] )



  S1[j+1] = S1[j] + 1/6 * ( kS11 + 2*kS12 + 2*kS13 + kS14 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im1[j+1] = Im1[j] + 1/6 * ( kIm11 + 2*kIm12 + 2*kIm13 + kIm14 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is1[j+1] = Is1[j] + 1/6 * ( kIs11 + 2*kIs12 + 2*kIs13 + kIs14 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R1[j+1] = R1[j] + 1/6 * ( kR11 + 2*kR12 + 2*kR13 + kR14 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  sum1[j+1]=sum1[j]+alpha1*Im1[j]*h
  
  S2[j+1] = S2[j] + 1/6 * ( kS21 + 2*kS22 + 2*kS23 + kS24 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im2[j+1] = Im2[j] + 1/6 * ( kIm21 + 2*kIm22 + 2*kIm23 + kIm24 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is2[j+1] = Is2[j] + 1/6 * ( kIs21 + 2*kIs22 + 2*kIs23 + kIs24 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R2[j+1] = R2[j] + 1/6 * ( kR21 + 2*kR22 + 2*kR23 + kR24 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  sum2[j+1]=sum2[j]+alpha2*Im2[j]*h
  
  S3[j+1] = S3[j] + 1/6 * ( kS31 + 2*kS32 + 2*kS33 + kS34 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im3[j+1] = Im3[j] + 1/6 * ( kIm31 + 2*kIm32 + 2*kIm33 + kIm34 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is3[j+1] = Is3[j] + 1/6 * ( kIs31 + 2*kIs32 + 2*kIs33 + kIs34 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R3[j+1] = R3[j] + 1/6 * ( kR31 + 2*kR32 + 2*kR33 + kR34 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  sum3[j+1]=sum3[j]+alpha3*Im3[j]*h
  
  

# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
#plt.plot(t, S1, color = "green", label = "S:未感染者", linewidth = 1.0)
#plt.plot(t, Is1, color = "red", label = "Is:重症", linewidth = 1.0)
plt.plot(t, Is1, color = "blue", label = " max={}".format(max(Is1)), linewidth = 1.0)
plt.plot(t, Is2, color = "red", label = " max={}".format(max(Is2)), linewidth = 1.0)
plt.plot(t, Is3, color = "green", label = " max={}".format(max(Is3)), linewidth = 1.0)

#plt.plot(t, R1, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('SIRモデル RK4によるシミュレーション')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()


plt.plot(t, sum1, color = "blue", label = "sum={}".format(sum1[n-1]), linewidth = 1.0)
plt.plot(t, sum2, color = "red", label = "sum={}".format(sum2[n-1]), linewidth = 1.0)
plt.plot(t, sum3, color = "green", label = "sum={}".format(sum3[n-1]), linewidth = 1.0)

#plt.plot(t, R1, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('SIRモデル RK4によるシミュレーション')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()
