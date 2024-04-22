import numpy as np
import matplotlib.pyplot as plt

# 1次元ランジュバン方程式の微分形
# 力A,分散sq2beta
def dlangevin1D(x,A,sq2beta,rng,N=1):
    return A(x)+sq2beta*rng.standard_normal(N)

def dlangevin1D_sq(x,A,sq2beta,dt,rng,N=1):
    return A(x)*dt+sq2beta*rng.standard_normal(N)*np.sqrt(dt)

    
#時間刻みdt, 時間Tの間の1次元ランジュバン方程式の時間発展
def langevin1D(A,sq2beta,T,dt=0.1,seed=1):
    rng = np.random.default_rng(seed)
    x=0
    data=[]
    for i in range(int(T)):
        x=x+dlangevin1D(x,A,sq2beta,rng)*dt 
        data.append(x)
    return np.array(data).reshape(-1)

def langevin1D_sq(A,sq2beta,T,dt=0.1,N=1,seed=1):
    rng = np.random.default_rng(seed)
    x=0
    data=[]
    eps=rng.standard_normal(T)
    for i in range(T):
        x=x+A(x)*dt+sq2beta*eps[i]*np.sqrt(dt) 
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
def make_entropy_rate(data,A,D,binsize=200):
#    D=1/2*sq2beta*sq2beta
    p, bin_edges=histogram_p(data,binsize)
    #print(p.sum())
    dx=(data.max()-data.min())/binsize
    #score=∇_x logp
    score=calc_score(p,dx)    
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]    
    s=(Ax-D*score)*(Ax-D*score)/D
    return (s*p[:-1]).sum() #,(s*p[1:]).sum()

def make_entropy_rate_debug(data,A,D,binsize=200):
    p, bin_edges=histogram_p(data,binsize)
    dx=(data.max()-data.min())/binsize
    score=calc_score(p,dx)    
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]    
    s=(Ax-D*score)*(Ax-D*score)/D
    return (s*p[:-1]).sum() ,(s*p[1:]).sum(),np.array(Ax).sum(),score.sum(),(Ax-D*score).sum()

#def make_entropy_rate(data,A,D,dt,binsize=200):
#    return make_entropy_prod(data,A,D,binsize)/(data.shape[0]*dt)

#### mean and variance of f #####
def make_RHS_a(f,data,A,D,binsize=200):
    #<f(AP-D∇P)>=∫dx Pf (A-Dscore) =∫dx Pf(A-D∇P/P)=∫dx PfA-fD∇P=∫dx PfA+DP∇f=<fA+D∇f>
    p, bin_edges=histogram_p(data,binsize)
#   print(p)
    score=calc_score(p,(data.max()-data.min())/binsize)    
    dx=(data.max()-data.min())/binsize
    fx=[f(x+dx/2) for x in bin_edges[:-2]]
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]
    #<f*F>
    fdx=fx*(Ax-D*score)
    aveJ=(fdx*p[:-1]).sum()
    #<fDf>
    #almost same
    #varJJ=2*(D*p[:-1]*fx*fx).sum()
    varJ=2*np.array([f(x)*f(x)*D for x in data]).mean()
    return 2*aveJ*aveJ/(varJ),aveJ,varJ #,varJJ

def make_RHS_eq(data,A,D,binsize=200):
    p, bin_edges=histogram_p(data,binsize)
    score=calc_score(p,(data.max()-data.min())/binsize)    
    dx=(data.max()-data.min())/binsize
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]
    fx=(Ax-D*score) #(A-D∇)p)/p
    #<f*F>
    aveJ=(fx*fx*p[:-1]).sum()
    #<fDf>
    #almost same
    varJ=2*(fx*fx*D*p[:-1]).sum()
    #2*np.array([fx*fx*D for x in data]).mean()
    return 2*aveJ*aveJ/(varJ),aveJ,varJ

def make_RHS_b(f,df,data,A,D):
    #<fA+D∇f>
    aveJ=np.array([A(x)*f(x)+D*df(x) for x in data]).mean()    
    #<fDf>?
    varJ=2*np.array([f(x)*f(x)*D for x in data]).mean()
    return 2*aveJ*aveJ/(varJ),aveJ,varJ

def make_RHS_v(f,data,dt=0.1):
    #f(x)○dx=(f(x)+f(x))/2*dx
    fdx=np.array([ (f(data[i+1])+f(data[i]))/2*(data[i+1]-data[i]) for i in range(len(data)-1)])/dt
    aveJ=fdx.mean()
    v2=(fdx*fdx).mean()
    var=(v2-aveJ*aveJ)
    return (2*aveJ*aveJ)/(var*dt),aveJ,var*dt

def tmp(A,sq2beta,T,dt):
    rng=np.random.default_rng(1)
    data=langevin1D_sq(A,sq2beta,T,dt,seed=2)
    #d1=np.array([dlangevin1D_sq(d,A,sq2beta,dt,rng) for d in data])
    d1=np.array([A(d)*dt+sq2beta*rng.standard_normal(1)*np.sqrt(dt)  for d in data])
    #d2=np.array([A(d)*dt  for d in data])
    d2=np.array([A(d)*dt+sq2beta*rng.standard_normal(1)*np.sqrt(dt)  for d in data])
#    print(d1,d2)
    print(d1.sum(),d2.sum())

def compare_ave(data,f,df,A,sq2beta,dt,binsize=200,print_formulars=False):
    rng=np.random.default_rng(1)
    D=1/2*sq2beta*sq2beta
    p, bin_edges=histogram_p(data,binsize)
    score=calc_score(p,(data.max()-data.min())/binsize)    
    dx=(data.max()-data.min())/binsize
    fx=[f(x+dx/2) for x in bin_edges[:-2]]
    Ax=[A(x+dx/2) for x in bin_edges[:-2]]
    Ja=fx*(Ax-D*score)    #<f*F>
    
    aveJa=(Ja*p[:-1]).sum()
    Jb=np.array([A(x)*f(x)+D*df(x) for x in data])
    di=range(len(data)-1)

    fdx=np.array([ (f(data[i+1])+f(data[i]))/2*(data[i+1]-data[i]) for i in di])/dt
    if(print_formulars):
        print("formulars")
        print("data mean",data.sum(),data.mean())
        dx=np.array([data[i+1]-data[i] for i in di])
        print("dx mean",dx.sum(),dx.mean())
        a=np.array([f(data[i])*dx[i] for i in di])/dt
        print("f*dx ",a.sum(),a.mean())        
        a=np.array([f(data[i])*A(data[i])*dt for i in di])/dt
        print("f*Adt ",a.sum(),a.mean())        
        a=np.array([(f(data[i+1])-f(data[i]))*dx[i] for i in di])/dt
        print("dfdt ",a.sum(),a.mean())        
        
        print(0,fdx.sum(),fdx.mean())
        _fdx=np.array([ (f(data[i+1])+f(data[i]))/2*dx[i] for i in di])/dt
        print(1,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ (f(data[i+1])-f(data[i]))/2*dx[i]+f(data[i])*dx[i] for i in di])/dt
        print(2,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ (f(data[i+1])-f(data[i]))/2*dx[i]+f(data[i])*A(data[i])*dt for i in di])/dt #*
        print(3,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ df(data[i])*(data[i+1]-data[i])/2*dx[i]+f(data[i])*A(data[i])*dt for i in di])/dt
        print(4,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ df(data[i])/2*dx[i]*dx[i]+f(data[i])*A(data[i])*dt for i in di])/dt
        print(5,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ df(data[i])*D*dt+f(data[i])*A(data[i])*dt for i in di])/dt #
        print(6,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ df(data[i])*D+f(data[i])*A(data[i]) for i in di])
        print(7,_fdx.sum(),_fdx.mean())
        _fdx=np.array([ df(x)*D+f(x)*A(x) for x in data])
        print(8,_fdx.sum(),_fdx.mean())
        #A(x)*dt+sq2beta*np.random.random(N)*np.sqrt(dt)

    ave=fdx.mean()
    #almost same?
    aveJb=Jb.mean()
    varJb=2*np.array([f(x)*f(x)*D for x in data]).mean()   
    varJa=2*(D*p[:-1]*fx*fx).sum()
    v2=(fdx*fdx).mean()
    var=(v2-ave*ave)
    rhs_a=(2*aveJa*aveJa)/(varJa)
    rhs_v=(2*ave*ave)/(var*dt)
    rhs_b=(2*aveJb*aveJb)/(varJb)
#    assert(rhs_a>=rhs_v)
    return [aveJa,Jb.mean(),ave], [varJa,varJb,var*dt],[rhs_a,rhs_b,rhs_v,rhs_a/rhs_v,rhs_b/rhs_v]  

def compare_langevin_ave(f,df,A,sq2beta,T,dt,binsize,with_entropy=False,print_formulars=False):
    D=1/2*sq2beta*sq2beta
    data=langevin1D_sq(A,sq2beta,T,dt)
    if(with_entropy):
        return compare_ave(data,f,df,A,sq2beta,dt,binsize,print_formulars),make_entropy_rate(data,A,D,binsize)            
    else:
        return compare_ave(data,f,df,A,sq2beta,dt,binsize,print_formulars)

def calc_ratio_fromdata(s:str,data,f,df,A,D,dt,binsize):
    rs={}
    if("v" in s):
        rs["v"]=make_RHS_v(f,data,dt)
    if("a" in s):
        rs["a"]=make_RHS_a(f,data,A,D,binsize)
    if("eq" in s):
        rs["eq"]=make_RHS_eq(data,A,D,binsize)
    if("b" in s):
        rs["b"]=make_RHS_b(f,df,data,A,D)
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