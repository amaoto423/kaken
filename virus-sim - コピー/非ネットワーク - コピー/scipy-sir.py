# （1）拡張モジュールのインポート
import numpy as np                  # 配列を扱う数値計算ライブラリNumPy
from scipy.integrate import odeint  # 数値積分scipy.integrate 常微分方程式（ODE）の積分器odeint
import matplotlib.pyplot as plt     # グラフ描画ライブラリmatplotlib
import japanize_matplotlib          # matplotlibの日本語化

# （2）時間変数tのリスト生成 時間変数t t=[0, dt, 2dt,..., T-dt]
T = 250                  # 変数tの範囲 0≦t<T（日）（250,150,150,700と値を変えてシミュレーションを行う）
dt = 0.1                 # 等差数列の公差:0.1 固定
t = np.arange(0, T, dt)  # 0から公差dtでTを超えない範囲で等差数列（リストt[j]）を生成

# （3）SIRモデル
# パラメータ
N = 10000000             # モデルエリアの人口（人）（東京都1400万人に匹敵するエリアを想定） N=S+I+R＝一定
m = 10                   # 1日1人あたり接触する人数（人）（10,50,100,5と値を変えてシミュレーションを行う）
p = 0.02                 # 接触ごとに感染が生じる1日あたりの確率
d = 14                   # 感染者の回復平均日数（日）
beta = m*p / N           # β:接触あたりの感染率
gamma = 1/d              # γ:回復率（隔離率）
# 微分方程式
def SIR_Model(v, t, beta, gamma):  # 関数名:SIR_Model, v=v(t)（従属変数:v,独立変数:t）, パラメータ:betaとgamma
# （4）数値積分変数S,I,R ベクトル場リスト v = [v[0], v[1], v[2]] = [S, I, R]
  S = v[0]
  I = v[1]
  R = v[2]
  dSdt = - beta*S*I              # dsdt ≡ ds/dt =  - beta*S*I
  dIdt = beta*S*I - gamma*I      # dIdt ≡ dI/dt = beta*S*I - gamma*I
  dRdt = gamma*I                 # dRdt ≡ dR/dt = gamma*I
  return [dSdt, dIdt, dRdt]      # 微分の値を返す
# 初期条件
Init = [N-100, 100, 0]         # t=0における S=N-100（人）, 初期感染者数I=100（人）, R=0（人）

# （5）数値積分scipy.integrate
# 積分器 関数odeint(微分方程式（関数名）, 初期条件, 独立変数（リスト）, パラメータargs（タブル）)
Results = odeint(SIR_Model, Init, t, args = (beta, gamma))
# SIR_Modelの返し値 R,S,Iの微分の値 → 積分器odeintで積分 → Results[0]=S,Results[1]=I,Results[2]=R

# （6）結果表示 データプロットによるグラフ表示
# 点(t,S=Results[0]),点(t,I=Results[1]),点(t,R=Results[2]) のプロット
plt.plot(t[::10],Results[::10,0], color = "green", linewidth = 1.0, label = "S:未感染者")
plt.plot(t[::10],Results[::10,1], color = "red",   linewidth = 1.0, label = "I:感染者")
plt.plot(t[::10],Results[::10,2], color = "blue",  linewidth = 1.0, label = "R:免疫獲得者")
# グラフの見た目設定
plt.title('SIRモデル odeintによるシミュレーション（m={},T={}）'.format(m,T))  # グラフタイトル パラメータmとTの値表示
plt.yticks(np.arange(0,N+0.1,N/10))    # y軸 目盛りの配分 0からN（=1000万）までを10等分 N/10（=100万）刻み Nを含めるためNをN+0.1としておく
plt.gca().set_yticklabels(['{:.0f}%'.format(x/(N/100)) for x in plt.gca().get_yticks()])   # y軸目盛りを％表示に変更
plt.xlabel('時間（日）')                              # 横軸ラベル
plt.ylabel('人数（総人口1000万人に対する割合）')      # 縦軸ラベル
plt.grid(True)                                        # グリッド表示
plt.legend()                                          # 凡例表示
# 設定反映しプロット描画
plt.show()
