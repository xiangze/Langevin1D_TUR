import numpy as np
import matplotlib.pyplot as plt

# 1次元ランジュバン方程式の微分形
# 力A,分散sq2beta
def dlangevin1D(x,A,sq2beta,N=1):
    return A(x)+sq2beta*np.random.random(N)

def dlangevin1D_sq(x,A,sq2beta,dt,N=1):
    return A(x)*dt+sq2beta*np.random.random(N)*np.sqrt(dt)
    
#時間刻みdt, 時間Tの間の1次元ランジュバン方程式の時間発展
def langevin1D(A,sq2beta,T,dt=0.1):
    x=0
    data=[]
    for i in range(int(T)):
        x=dlangevin1D(x,A,sq2beta)*dt #dt
        data.append(x)
    return np.array(data).reshape(-1)

def langevin1D_sq(A,sq2beta,T,dt=0.1,N=1):
    x=0
    data=[]
    for i in range(T):
        x=dlangevin1D_sq(x,A,sq2beta,dt)
        data.append(x)
    return np.array(data).reshape(-1)

def histogram_p(data,binsize,l=1e-10):
    p, bin_edges=np.histogram(data,binsize)
    p=p/data.shape[0]+l
    return p,bin_edges

def calc_score(p,dx):
    #score=∇_x logp
    lp=np.log(p)
    return (lp[1:]-lp[:-1])/dx

### entoropy production ###
def make_entropy_prod(data,A,D,binsize=200):
#    D=1/2*sq2beta*sq2beta
    p, bin_edges=histogram_p(data,binsize)
    #print(p.sum())
    dx=(data.max()-data.min())/binsize
    #score=∇_x logp
    score=calc_score(p,dx)    
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]    
    s=(Ax-D*score)*(Ax-D*score)/D
    return (s*p[:-1]).sum()

def make_entropy_rate(data,A,D,binsize=200):
    return make_entropy_prod(data,A,D,binsize)/data.shape[0]

#### mean and variance of f #####
def make_RHS_a(f,data,A,D,dt=0.1,binsize=200):
    #<f(AP-D∇P)>=∫dx Pf (A-Dscore) =∫dx Pf(A-D∇P/P)=∫dx PfA-fD∇P=∫dx PfA+DP∇f=<fA+D∇f>
    p, bin_edges=histogram_p(data,binsize)
#   print(p)
    score=calc_score(p,(data.max()-data.min())/binsize)    
    dx=(data.max()-data.min())/binsize
    fx=[f(x+dx/2) for x in bin_edges[:-2]]
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]
    #<f*F>
    aveJ=(fx*(Ax-D*score)*p[:-1]).sum()
    #<fDf>
    varJ=2*np.array([f(x)*f(x)*D for x in data]).mean()
    return 2*aveJ*aveJ/(varJ),aveJ,varJ

def make_RHS_b(f,df,data,A,D,dt=0.1):
    #<fA+D∇f>
    aveJ=np.array([A(x)*f(x)+D*df(x) for x in data]).mean()    
    #<fDf>?
    varJ=2*np.array([f(x)*f(x)*D for x in data]).mean()
    return 2*aveJ*aveJ/(varJ),aveJ,varJ

def make_RHS_v(f,data,dt=0.1,binsize=200):
    p, bin_edges=histogram_p(data,binsize)
    dx=(data.max()-data.min())/binsize
    #f(x)○dx=f(x+x/2)*dx
    fdx=np.array([f(x+dx/2) for x in bin_edges[:-2]])/dt
    aveJ=(fdx*p[:-1]).sum()   
    v2=(fdx*fdx).mean()
#    print(v2)
    var=(v2-aveJ*aveJ)
    return (2*aveJ*aveJ)/(var*dt*dt),aveJ,var

def make_RHS_v2(f,data,dt=0.1):
    #f(x)○dx=f(x+x/2)*dx
    fdx=np.array([f((data[i]+data[i+1])/2) for i in range(len(data)-1)])/dt
    aveJ=fdx.mean()   
    v2=(fdx*fdx).mean()
#    print(v2)
    var=(v2-aveJ*aveJ)
    return (2*aveJ*aveJ)/(var*dt*dt),aveJ,var

def calc_ratio_fromdata(s:str,data,f,df,A,D,dt,binsize):
    rs={}
    if("v1" in s):
        rs["v1"]=make_RHS_v(f,data,dt,binsize)
    if("v2" in s):
        rs["v2"]=make_RHS_v2(f,data,dt)
    if("a" in s):
        rs["a"]=make_RHS_a(f,data,A,D,dt,binsize)
    if("b" in s):
        rs["b"]=make_RHS_b(f,df,data,A,D,dt)
    return rs

def calc_ratio_langevin1D(s:str,f,df,A,sq2beta,T,dt,binsize):
    data=langevin1D(A,sq2beta,T,dt)
    D=1/2*sq2beta*sq2beta
    return calc_ratio_fromdata(s,data,f,df,A,D,dt,binsize)

def calc_ratio_langevin1D_sq(s:str,f,df,A,sq2beta,T,dt,binsize):
    data=langevin1D_sq(A,sq2beta,T,dt)
    D=1/2*sq2beta*sq2beta
    return calc_ratio_fromdata(s,data,f,df,A,D,dt,binsize)

if __name__ == '__main__':
    pass