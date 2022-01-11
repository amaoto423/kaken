# （1）拡張モジュールのインポート
import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化



# （2）時間変数tの導入
T = 700                  # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
n = 10*T                 # 変数tの範囲をn等分   n=T/h=T/0.1=10*T （T=250のときはn=2500）
h = 0.1                  # 等差数列の公差:0.1 固定
t = np.arange(0,T,h)     # 0から公差dtでTを超えない範囲で等差数列を生成 t[0],...,t[n-1] 要素数n個

kekka1=np.empty(n)
kekka2=np.empty(n)
kekka3=np.empty(n)

sumkekka1=np.empty(n)
sumkekka2=np.empty(n)
sumkekka3=np.empty(n)
# （3）SIRモデル
# 3-1パラメータ
N1 = 8000000             # 街１の人口
N2 = 2000000               #街２の人口
m1 = 10                   #街１の接触数
m2 = 10                  #街２の接触数
p = 0.02               #5接触ごとに感染が生じる1日あたりの確率
d = 14                   # 感染者の回復平均日数（日）
nu1 = 0.01                 #ワクチン接種率
nu2 = 0   
alpha = 0.01               #重症化率
beta1 = m1*p / N1           # 接触あたりの感染率
beta2 = m2*p / N2           # 接触あたりの感染率
beta_p=(beta1+beta2)/10

gamma = 1/d              # 回復率（隔離率）
# 3-2初期値
Im1_0 = 100                # 初期感染者数（人）100人
Is1_0=0
R1_0 = 0                  # 初期回復者数（人）0人
S1_0 = N1 - Im1_0 - Is1_0 - R1_0      # 初期未感染者数（人）S_0 = N - Im_0- Is_0 - R_0

Im2_0=0
Is2_0=0
R2_0=0
S2_0=N2 - Im2_0 - Is2_0 - R2_0 
# 3-3微分方程式
dS1dt =  lambda S1,S2,Im1,Im2,Is1,Is2  ,R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t : - beta1*S1*Im1 - nu1*S1 -beta_p*S1*Im2             # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIm1dt = lambda S1,S2, Im1,Im2,Is1,Is2, R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t :  beta1*S1*Im1 + beta_p*S1*Im2 - alpha*Im1 - gamma*Im1       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dIs1dt = lambda S1,S2, Im1,Im2,Is1,Is2, R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t : alpha*Im1-gamma*Is1
dR1dt =  lambda S1,S2,Im1,Im2,Is1,Is2,  R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t : gamma*(Im1+Is1)+nu1*S1                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)

dS2dt = lambda S1,S2,Im1,Im2,Is1,Is2 ,R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t : - beta2*S2*Im2 - nu2*S2 -beta_p*S2*Im1             # dSdt ≡ dS/dt  dSdt(S, I, R, t)
dIm2dt = lambda S1,S2,Im1,Im2,Is1,Is2 ,R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t : beta2*S2*Im2 + beta_p*S2*Im1 - alpha*Im2 - gamma*Im2       # dIdt ≡ dI/dt  dIdt(S, I, R, t)
dIs2dt = lambda S1,S2,Im1,Im2,Is1,Is2 ,R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t: alpha*Im2-gamma*Is2
dR2dt = lambda S1,S2,Im1,Im2,Is1,Is2 ,R1,R2,alpha,beta1,beta2,beta_p,nu1,nu2, t: gamma*(Im2+Is2)+nu2*S2                  # dRdt ≡ dR/dt  dRdt(S ,I ,R, t)

# 3-4数値積分変数S,I,Rをリストとして生成
S1 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im1 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is1=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R1 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S2 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im2 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is2=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R2 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

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

sum1[0]=0
sum2[0]=0
sum3[0]=0

# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

  kS11 = h * dS1dt(   S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j] )
  kIm11 = h * dIm1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIs11 = h * dIs1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kR11 = h * dR1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])

  kS21 =  h * dS2dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIm21 = h * dIm2dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIs21 = h * dIs2dt( S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kR21 =  h * dR2dt(  S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])


  kS12 = h * dS1dt( S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm12 = h * dIm1dt( S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIs12=h * dIs1dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR12 = h * dR1dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)

  kS22 = h * dS2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIm22 = h * dIm2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIs22=h * dIs2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kR22 = h * dR2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)


  kS13 = h * dS1dt( S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm13 = h * dIm1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIs13 = h * dIs1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR13 = h * dR1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )

  kS23 = h * dS2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm23 = h * dIm2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIs23 = h * dIs2dt( S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR23 = h * dR2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )


  kS14 =  h * dS1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIm14 = h * dIm1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIs14 = h * dIs1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kR14 =  h * dR1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )


  kS24 =  h * dS2dt(  S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIm24 = h * dIm2dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIs24 = h * dIs2dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h  )
  kR24 =  h * dR2dt(  S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )


  S1[j+1]  = S1[j]  + 1/6 * ( kS11  + 2*kS12  + 2*kS13  + kS14 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im1[j+1] = Im1[j] + 1/6 * ( kIm11 + 2*kIm12 + 2*kIm13 + kIm14 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is1[j+1] = Is1[j] + 1/6 * ( kIs11 + 2*kIs12 + 2*kIs13 + kIs14 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R1[j+1]  = R1[j]  + 1/6 * ( kR11  + 2*kR12  + 2*kR13  + kR14 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  
  S2[j+1]  = S2[j]  + 1/6 * ( kS21  + 2*kS22  + 2*kS23  + kS24 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im2[j+1] = Im2[j] + 1/6 * ( kIm21 + 2*kIm22 + 2*kIm23 + kIm24 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is2[j+1] = Is2[j] + 1/6 * ( kIs21 + 2*kIs22 + 2*kIs23 + kIs24 ) 
  R2[j+1]  = R2[j]  + 1/6 * ( kR21  + 2*kR22  + 2*kR23  + kR24 ) 
  
  sum1[j+1]=sum1[j]+alpha*Im1[j]*h
  sum2[j+1]=sum2[j]+alpha*Im2[j]*h
  
kekka1=Is1+Is2
sumkekka1=sum1+sum2

S1 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im1 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is1=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R1 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S2 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im2 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is2=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R2 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

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

sum1[0]=0
sum2[0]=0
sum3[0]=0

nu1=0
nu2=0.01
# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

  kS11 = h * dS1dt(   S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j] )
  kIm11 = h * dIm1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIs11 = h * dIs1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kR11 = h * dR1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])

  kS21 =  h * dS2dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIm21 = h * dIm2dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIs21 = h * dIs2dt( S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kR21 =  h * dR2dt(  S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])


  kS12 = h * dS1dt( S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm12 = h * dIm1dt( S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIs12=h * dIs1dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR12 = h * dR1dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)

  kS22 = h * dS2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIm22 = h * dIm2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIs22=h * dIs2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kR22 = h * dR2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)


  kS13 = h * dS1dt( S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm13 = h * dIm1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIs13 = h * dIs1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR13 = h * dR1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )

  kS23 = h * dS2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm23 = h * dIm2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIs23 = h * dIs2dt( S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR23 = h * dR2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )


  kS14 =  h * dS1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIm14 = h * dIm1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIs14 = h * dIs1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kR14 =  h * dR1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )


  kS24 =  h * dS2dt(  S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIm24 = h * dIm2dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIs24 = h * dIs2dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h  )
  kR24 =  h * dR2dt(  S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )


  S1[j+1]  = S1[j]  + 1/6 * ( kS11  + 2*kS12  + 2*kS13  + kS14 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im1[j+1] = Im1[j] + 1/6 * ( kIm11 + 2*kIm12 + 2*kIm13 + kIm14 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is1[j+1] = Is1[j] + 1/6 * ( kIs11 + 2*kIs12 + 2*kIs13 + kIs14 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R1[j+1]  = R1[j]  + 1/6 * ( kR11  + 2*kR12  + 2*kR13  + kR14 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  S2[j+1]  = S2[j]  + 1/6 * ( kS21  + 2*kS22  + 2*kS23  + kS24 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im2[j+1] = Im2[j] + 1/6 * ( kIm21 + 2*kIm22 + 2*kIm23 + kIm24 )   # 末項 j=n-2 -> I[j+1]=I[n-1]

  Is2[j+1] = Is2[j] + 1/6 * ( kIs21 + 2*kIs22 + 2*kIs23 + kIs24 ) 
  R2[j+1]  = R2[j]  + 1/6 * ( kR21  + 2*kR22  + 2*kR23  + kR24 ) 
  
  sum1[j+1]=sum1[j]+alpha*Im1[j]*h
  sum2[j+1]=sum2[j]+alpha*Im2[j]*h
 
kekka2=Is1+Is2
sumkekka2=sum1+sum2
S1 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im1 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is1=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R1 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

S2 = np.empty(n)          # S[0],...,S[n-1] 要素数n個
Im2 = np.empty(n)          # Im[0],...,Im[n-1] 要素数n個
Is2=np.empty(n)            # Is[0],...,Is[n-1] 要素数n個
R2 = np.empty(n)          # R[0],...,R[n-1] 要素数n個

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

sum1[0]=0
sum2[0]=0
sum3[0]=0


nu1=0.007
nu2=0.003
# （4）数値積分 4次ルンゲ-クッタ法 4th-Order Runge–Kutta Methods
for j in range(n-1):     # j=0,...,n-2 -> S[j]=S[0],...,S[n-1]（I[j],R[j]も同じ） 要素数n個

  kS11 = h * dS1dt(   S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j] )
  kIm11 = h * dIm1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIs11 = h * dIs1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kR11 = h * dR1dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])

  kS21 =  h * dS2dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIm21 = h * dIm2dt(S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kIs21 = h * dIs2dt( S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])
  kR21 =  h * dR2dt(  S1[j],S2[j] ,Im1[j],Im2[j],Is1[j],Is2[j],R1[j],R2[j] ,alpha,beta1,beta2,beta_p,nu1,nu2 ,t[j])


  kS12 = h * dS1dt( S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm12 = h * dIm1dt( S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIs12=h * dIs1dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR12 = h * dR1dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)

  kS22 = h * dS2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIm22 = h * dIm2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kIs22=h * dIs2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)
  kR22 = h * dR2dt(S1[j] + kS11/2 ,S2[j]+kS21/2,Im1[j] + kIm11/2,Im2[j] + kIm21/2 ,Is1[j]+kIs11/2,Is2[j]+kIs21/2,R1[j] + kR11/2 ,R2[j]+kR21/2,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2)


  kS13 = h * dS1dt( S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm13 = h * dIm1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIs13 = h * dIs1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR13 = h * dR1dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22/2,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )

  kS23 = h * dS2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIm23 = h * dIm2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kIs23 = h * dIs2dt( S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )
  kR23 = h * dR2dt(S1[j] + kS12/2 ,S2[j]+kS22/2,Im1[j] + kIm12/2 ,Im2[j] + kIm22/2,Is1[j]+kIs12/2,Is2[j]+kIs22,R1[j] + kR12/2,R2[j]+kR22/2, alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h/2 )


  kS14 =  h * dS1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIm14 = h * dIm1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIs14 = h * dIs1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kR14 =  h * dR1dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )


  kS24 =  h * dS2dt(  S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIm24 = h * dIm2dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )
  kIs24 = h * dIs2dt( S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h  )
  kR24 =  h * dR2dt(  S1[j] + kS13,S2[j]+kS23,Im1[j] + kIm13,Im2[j] + kIm23 ,Is1[j]+kIs13,Is2[j]+kIs23,R1[j] + kR13 ,R2[j]+kR23,alpha,beta1,beta2,beta_p,nu1,nu2,t[j] + h )


  S1[j+1]  = S1[j]  + 1/6 * ( kS11  + 2*kS12  + 2*kS13  + kS14 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im1[j+1] = Im1[j] + 1/6 * ( kIm11 + 2*kIm12 + 2*kIm13 + kIm14 )   # 末項 j=n-2 -> I[j+1]=I[n-1]
  Is1[j+1] = Is1[j] + 1/6 * ( kIs11 + 2*kIs12 + 2*kIs13 + kIs14 )   # 末項 j=n-2 -> I[j+1]=I[n-1] 
  R1[j+1]  = R1[j]  + 1/6 * ( kR11  + 2*kR12  + 2*kR13  + kR14 )   # 末項 j=n-2 -> R[j+1]=R[n-1]
  S2[j+1]  = S2[j]  + 1/6 * ( kS21  + 2*kS22  + 2*kS23  + kS24 )   # 末項 j=n-2 -> S[j+1]=S[n-1]
  Im2[j+1] = Im2[j] + 1/6 * ( kIm21 + 2*kIm22 + 2*kIm23 + kIm24 )   # 末項 j=n-2 -> I[j+1]=I[n-1]

  Is2[j+1] = Is2[j] + 1/6 * ( kIs21 + 2*kIs22 + 2*kIs23 + kIs24 ) 
  R2[j+1]  = R2[j]  + 1/6 * ( kR21  + 2*kR22  + 2*kR23  + kR24 ) 
  
  sum1[j+1]=sum1[j]+alpha*Im1[j]*h
  sum2[j+1]=sum2[j]+alpha*Im2[j]*h
 
kekka3=Is1+Is2
sumkekka3=sum1+sum2
# （5）結果表示 データプロットによるグラフ表示
# 点(t,S),点(t,I),点(t,R) それぞれ要素数n個のプロット
#plt.plot(t, S1, color = "green", label = "S:未感染者", linewidth = 1.0)
#plt.plot(t, Is1, color = "red", label = "Is:重症", linewidth = 1.0)

plt.plot(t, kekka1, color = "blue", label = "nu1=0.0001 nu2=0 max={}".format(max(kekka1)), linewidth = 1.0)
plt.plot(t, kekka2, color = "red", label = "nu1=0 nu2=0.01 max={}".format(max(kekka2)), linewidth = 1.0)
plt.plot(t, kekka3, color = "green", label = "nu1=0.005 nu2=0.005 max={}".format(max(kekka3)), linewidth = 1.0)

#plt.plot(t, R1, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('ワクチン接種率別時間変化')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()


plt.plot(t, sumkekka1, color = "blue", label = "nu1=0.0001 nu2=0 max={}".format(max(kekka1)), linewidth = 1.0)
plt.plot(t, sumkekka2, color = "red", label = "nu1=0 nu2=0.01 max={}".format(max(kekka2)), linewidth = 1.0)
plt.plot(t, sumkekka3, color = "green", label = "nu1=0.05 nu2=0.05 max={}".format(max(kekka3)), linewidth = 1.0)

#plt.plot(t, R1, color= "blue", label = "R:免疫獲得者", linewidth = 1.0)
# グラフの見た目設定
plt.title('ワクチン接種率別累積')  # グラフタイトル パラメータmとTの値表示
#plt.yticks(np.arange(0,N1+N2+0.1,(N1+N2)/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
#plt.gca().set_yticklabels(['{:.0f}%'.format(y/((N1+N2)/100)) for y in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()
