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

#def show_param():

def show_ratio1(d:dict,s:str,a:str,b:str):
    if(s=="rhs"):
        print( d[a][0],d[b][0],d[a][0]/d[b][0])
    if(s=="mean"):
        print( d[a][1],d[b][1],d[a][1]/d[b][1])
    if(s=="var"):    
        print( d[a][2],d[b][2],d[a][2]/d[b][2])

def show_ratio(d:list,s:str,a:str,b:str):
    print(s+a+"/"+s+b)
    for i in d:
        show_ratio1(i,s,a,b)

def show_ratio_all(d:list,a:str,b:str):
    show_ratio(d,"mean",a,b)
    show_ratio(d,"var",a,b)
    show_ratio(d,"rhs",a,b)

def show_av_dt(T=100000,sq2beta=sq2beta,dt=dt,binsize=binsize):
    dts=np.arange(0.01,0.1,0.01) 
    a=[L.calc_ratio_langevin1D("av1",f,df,A,sq2beta,T,dt,binsize) for dt in dts]
    show_ratio(a,"mean","v1","a")
    print("varD/var")
    for i,dt in zip(a,dts):
        print( i["v1"][2],i["a"][2], i["v1"][2]/i["a"][2]*dt*dt) #var
    show_ratio(a,"rhs","v1","a")

def show_bv2_dt(T=100000,sq2beta=sq2beta,dt=dt,binsize=binsize):
    dts=np.arange(0.01,0.1,0.01) 
    a=[L.calc_ratio_langevin1D("bv2",f,df,A,sq2beta,T,dt,binsize) for dt in dts]
    show_ratio(a,"mean","v2","b")
    print("varD/var")
    for i,dt in zip(a,dts):
        print( i["v2"][2],i["b"][2], i["v2"][2]/i["b"][2]*dt*dt)
    show_ratio(a,"rhs","v2","b")

def show_v12_dt(T=100,sq2beta=sq2beta,dt=dt,binsize=binsize):
    dts=np.arange(0.01,0.1,0.01) 
    a=[L.calc_ratio_langevin1D("v1v2",f,df,A,sq2beta,T,dt,binsize) for dt in dts] 
    show_ratio(a,"mean","v1","v2")
    show_ratio(a,"var","v1","v2")

def show_hightemp(a,b):
    l=[L.calc_ratio_langevin1D(a+b,f,df,A,sq2beta,T,dt,binsize) for sq2beta in np.arange(0.01,0.1,0.01) ]
    show_ratio_all(l,a,b)
    
def show_v12_temp():
    a=[L.calc_ratio_langevin1D("v1v2",f,df,A,sq2beta,T,dt,binsize) for sq2beta in np.arange(0.01,0.1,0.01) ]
    show_ratio(a,"mean","v1","v2")
    show_ratio(a,"var","v1","v2")

def show_entropy_rate_temp(type="sq"):
    D=1/2*sq2beta*sq2beta
    for dt in np.arange(0.01,0.1,0.01):
        if(type=="sq"):
            data=L.langevin1D_sq(A,sq2beta,T,dt)
        else:
            data=L.langevin1D(A,sq2beta,T,dt)
        print(L.make_entropy_rate(data,A,D,binsize))

#show_v12(100000)
show_av_dt(100000)
#show_bv2_dt(100000)
#show_v12_temp()
