import scipy
from scipy import optimize 
import math
import numpy as np
def h(x):
    return np.log(1-x)+2*x+200/10000000

def BinarySearch(upper,lower,err):
    up=upper
    low=lower
    count=0
    while True:
        count=count+1
        mid=(up+low)/2
        y=h(mid)
        if abs(y)<err: #解が発見された
            break
        elif h(low)*y<0: #解は下限と中間点の間にある
            up=mid
        else:                     #解は上限と中間点の間にある
            low=mid
    print(\
        "数値解は",mid,\
        "\nその時の関数の値は",y,\
        "\n計算回数は",count,"です")

BinarySearch(0.5,0.8,1/100000000000000)