import argparse
import numpy as np
import matplotlib.pyplot as plt
import Langevin1D_TUR as L

#parameters
binsize=200
sq2beta=0.1
T=100000
dt=0.1

A=lambda x: -x*x*x+x
f=lambda x:x
df=lambda x:1

def show_param():
    print("T:%d"%T)
    print("dt:%g"%dt)
    print("sq2beta:%g"%sq2beta)
    print("binsize:%g"%binsize)

def show_ratio1(d:dict,s:str,a:str,b:str,r:float=1.0):
    if(s=="rhs"):
        print( d[a][0],d[b][0],d[a][0]/d[b][0]*r)
    if(s=="mean"):
        print( d[a][1],d[b][1],d[a][1]/d[b][1]*r)
    if(s=="var"):    
        print( d[a][2],d[b][2],d[a][2]/d[b][2]*r)

def show_ratio(d:list,s:str,a:str,b:str):
    print(s+a+"/"+s+b)
    for i in d:
        show_ratio1(i,s,a,b)

def show_ratio_all(d:list,a:str,b:str):
    show_ratio(d,"mean",a,b)
    show_ratio(d,"var",a,b)
    show_ratio(d,"rhs",a,b)

def show_RHS(a,b,varname,varrange,calc_ratio=L.calc_ratio_langevin1D,binsize=binsize,T=T,dt=dt,sq2beta=sq2beta):
    print("show RHS %s vs %s %s"%(a,b,varname))
    d={"binsize":binsize,"dt":dt,"T":T,"sq2beta":sq2beta}
    res=[]
    for v in varrange:
        d[varname]=v
        res.append(calc_ratio(a+b,f,df,A,d["sq2beta"],int(d["T"]),d["dt"],d["binsize"]))
    show_ratio_all(res,a,b)

def show_av_dt(T=100000,calc_ratio=L.calc_ratio_langevin1D,sq2beta=sq2beta,dt=dt,binsize=binsize):
    show_RHS("v1","a","dt",np.arange(0.01,0.1,0.01),calc_ratio)

def show_bv2_dt(T=100000,calc_ratio=L.calc_ratio_langevin1D,sq2beta=sq2beta,dt=dt,binsize=binsize):
    show_RHS("v2","b","dt",np.arange(0.01,0.1,0.01),calc_ratio)
 
def show_v12_dt(T=100,calc_ratio=L.calc_ratio_langevin1D,sq2beta=sq2beta,dt=dt,binsize=binsize):
    show_RHS("v1","v2","dt",np.arange(0.01,0.1,0.01),calc_ratio)
    
def show_dt(a,b,calc_ratio=L.calc_ratio_langevin1D):
    show_RHS(a,b,"dt",np.arange(0.01,0.1,0.01),calc_ratio)

def show_T(a,b,calc_ratio=L.calc_ratio_langevin1D):
    show_RHS(a,b,"T",np.arange(1e5,1e6,1e5),calc_ratio)

def show_hightemp(a,b,calc_ratio=L.calc_ratio_langevin1D):
    show_RHS(a,b,"sq2beta",np.arange(0.01,0.1,0.01),calc_ratio)
    
def show_entropy_rate(varname,varrange,type="sq",T=1e6,make_entropy_rate=L.make_entropy_rate):
    print("entropy production rate:%s type=%s:"%(varname,type))
    d={"dt":dt,"T":T,"sq2beta":sq2beta,"binsize":binsize,"A":A}
    show_param()
    for v in varrange:
        d[varname]=v
        data=L.langevin1D_sq(d["A"],d["sq2beta"],int(d["T"]),d["dt"])
        d["D"]=1/2*d["sq2beta"]*d["sq2beta"]
        print("%s=%g"%(varname,v),end=":")
        print(make_entropy_rate(data,d["A"],d["D"],d["binsize"]))

def show_entropy_rate_dt(type="sq",T=1e6,make_entropy_rate=L.make_entropy_rate):
    show_entropy_rate("dt",np.arange(0.01,0.1,0.01),make_entropy_rate=make_entropy_rate,type=type)    
    
def show_entropy_rate_T(dt=0.1,make_entropy_rate=L.make_entropy_rate):
    show_entropy_rate("T",np.arange(1e5,2e6,2e5),make_entropy_rate=make_entropy_rate)

def show_entropy_rate_binsize(dt=0.1,make_entropy_rate=L.make_entropy_rate):
    show_entropy_rate("binsize",[50,100,150,200],make_entropy_rate=make_entropy_rate)

def show_entropy_rate_temp(type="sq",T=1e6,dt=0.1,make_entropy_rate=L.make_entropy_rate):
    show_entropy_rate("sq2beta",np.arange(1e-6,1e-5,1e-6),T=T,make_entropy_rate=make_entropy_rate)

def entoropy_vs_eq(varname:str,varrange):
    print("entropy prod rate vs RHS(eq),%s"%varname)
    d={"dt":dt,"T":T,"sq2beta":sq2beta,"binsize":binsize,"A":A}
    for v in varrange:
        D=1/2*d["sq2beta"]*d["sq2beta"]
        print("T=%g"%v,end=":")
        d[varname]=v
        data=L.langevin1D_sq(A,d["sq2beta"],int(d["T"]),d["dt"])
        print(L.make_entropy_rate(data,A,D,d["binsize"]),end=",")
        print(L.make_RHS_eq(data,A,D,d["binsize"]))
    
entoropy_vs_eq_T=lambda :entoropy_vs_eq("T",np.arange(1e5,2e6,2e5))
entoropy_vs_eq_dt=lambda :entoropy_vs_eq("dt",np.arange(0.01,0.1,0.01))

compare_ave=False
show_entropy=False
compare_RHS=True
#compare_RHS=False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='コマンドライン引数の例')
    parser.add_argument('-T', '--Total_step')  
    parser.add_argument('-dt', '--time_step')  
    parser.add_argument('-binsize', '--binsize')  
    parser.add_argument('-sq2beta', '--sq2beta')  
    
    if(compare_RHS):
        show_param()
        show_RHS("a","v","dt",np.arange(0.01,0.1,0.01),calc_ratio=L.calc_ratio_langevin1D_sq)        
        show_RHS("b","v","dt",np.arange(0.01,0.1,0.01),calc_ratio=L.calc_ratio_langevin1D_sq)

        show_T("a","v",calc_ratio=L.calc_ratio_langevin1D_sq)
        show_T("b","v",calc_ratio=L.calc_ratio_langevin1D_sq)
        #show_RHS("v","a","binsize",[100,200,300,400],calc_ratio=L.calc_ratio_langevin1D_sq)
        show_RHS("a","b","binsize",[100,200,300,400],calc_ratio=L.calc_ratio_langevin1D_sq)

#        show_RHS("a","b","sq2beta",[100,200,300,400],calc_ratio=L.calc_ratio_langevin1D_sq)

    if(show_entropy):
    #    show_entropy_rate_dt("sq")
    #    show_entropy_rate_T()
    #    show_entropy_rate_binsize()
        #show_entropy_rate_temp()
        show_entropy_rate("sq2beta",[0.01,0.1,0.01],make_entropy_rate=L.make_entropy_rate_debug)
        #entoropy_vs_eq_T()    
        #entoropy_vs_eq_dt()    

#    for dt in np.arange(0.01,0.1,0.01):
#        print("dt=%g"%dt)
#        print(L.compare_langevin_ave(f,df,A,0.001,100000,dt,binsize))
    if(compare_ave):
        A=lambda x: -x*x*x+x+1
        dA=lambda x: -3*x*x+1
        f=lambda x:x*x
        df=lambda x:2*x
        for sq2beta in np.arange(0.0001,0.001,0.0001):
    #    for sq2beta in np.arange(0.01,0.1,0.01):
            print("sq2beta=%g"%sq2beta)
            print(L.compare_langevin_ave(f,df,A,sq2beta,T,dt,binsize))
        f=lambda x:x
        df=lambda x:1
        for sq2beta in np.arange(0.0001,0.001,0.0001):
            print("sq2beta=%g"%sq2beta)
            print(L.compare_langevin_ave(f,df,A,sq2beta,T,dt,binsize))
        A=lambda x: -x*x*x+x
        dA=lambda x: -3*x*x+1
        for sq2beta in np.arange(0.0001,0.001,0.0001):
            print("sq2beta=%g"%sq2beta)
            print(L.compare_langevin_ave(f,df,A,sq2beta,T,dt,binsize))
        f= A
        df=dA
        for sq2beta in np.arange(0.0001,0.001,0.0001):
            print("sq2beta=%g"%sq2beta)
            print(L.compare_langevin_ave(f,df,A,sq2beta,T,dt,binsize))

        #L.tmp(A,sq2beta,100000,dt)   
        #print(L.compare_langevin_ave(f,df,A,0.,400,dt))
        #dt=0.01
        #print(L.compare_langevin_ave(f,df,A,0.01,10000,dt))
        #f=lambda x:x
        #df=lambda x:1

 #   for dt in np.arange(0.01,0.1,0.01):
 #       data=L.langevin1D_sq(A,sq2beta,T,dt)
 #       print(L.make_RHS_a(f,data,A,D,dt,binsize))

 #   for T in np.arange(1e5,2e6,2e5):
 #       data=L.langevin1D_sq(A,sq2beta,int(T),dt)
 #       print(L.make_RHS_a(f,data,A,D,dt,binsize))
