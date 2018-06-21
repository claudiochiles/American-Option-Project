#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:13:12 2018

@author: Claudio
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.sparse
np.seterr(divide='ignore', invalid='ignore')
import math
from numpy.linalg import inv
from scipy.linalg import solve
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import scipy.sparse
import math
from scipy.stats.mstats import gmean
    

class EuropeanOption(object):
    def __init__(self,stock_price,strike_price):
        self.stock_price = stock_price
        self.strike_price= strike_price
        self.strike_price_1 = strike_price[0]
        self.strike_price_2 = strike_price[1]
        self.strike_price_3 = strike_price[2]
    
    def European_Call_Payoff(self,strike_price):
        european_call_payoff = np.maximum(self.stock_price-strike_price,0)
        return european_call_payoff
        
    def European_Put_Payoff(self,strike_price):
        european_put_payoff = np.maximum(strike_price-self.stock_price,0)
        return european_put_payoff
      
    def Black_Scholes_Explicit_FD_AO(self,t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price,stock_max,stock_min ):           
        
        X = np.log(self.stock_price/strike_price)
        k = interest_rate/(0.5*volatility**2)
        kd = (interest_rate - dividend_yield)/(0.5*volatility**2)
        N = Nplus - Nminus
        dt = (0.5*(volatility**2)*maturity_date)/M
        Xzero = np.log(stock_min/strike_price)
        Xmax = np.log(stock_max/strike_price)
        dX = (Xmax-Xzero)/N
        alpha = dt/(dX*dX)
        solution_mesh = np.zeros((N+1,M+1))
        g = np.zeros((N+1,M+1))
        Xmesh = np.arange(Xzero,Xmax,dX)
        Xmesh2 = np.append(Xmesh,Xmax)
        
        #payoff matrix
        for j in range(0,N+1):
            for i in range(1,M+1):
                g[j,i] = np.exp((0.25*(kd-1)**2 + k)*((i)*dt))*np.maximum(np.exp(0.5*(kd+1)*(j+Nminus)*dX)-
                           np.exp(0.5*(kd-1)*(j+Nminus)*dX),0)
            g[j,0] = np.maximum(np.exp(0.5*(kd+1)*(j+Nminus)*dX)-np.exp(0.5*(kd-1)*(j+Nminus)*dX),0)
        g[0,:]=0
        
        #Boundary conditions of the u matrix
        solution_mesh[:,0] = g[:,0]
        solution_mesh[0,:] = g[0,:]
        solution_mesh[N,:] = g[N,:]
        
        a = alpha
        b = 1 - 2*alpha
        c = alpha
        
        y = np.zeros((N+1,M+1))
        
        for p in range(0,M):
            y[1:N,p+1] = b*solution_mesh[1:N,p]+a*solution_mesh[2:N+1,p]+c*solution_mesh[0:N-1,p]
            solution_mesh[1:N,p+1] = np.maximum(y[1:N,p+1],g[1:N,p+1])
        
        uresult = np.interp(X,Xmesh2,solution_mesh[:,M])
        
        explicit_value = strike_price*strike_price**(0.5*(kd-1))*self.stock_price**(-0.5*(kd-1))*np.exp((-0.25*(kd-1)**2 - k)*0.5*volatility**2*(maturity_date-t))*uresult         
        return explicit_value

    def Black_Scholes_Implicit_FD_AO(self,t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price,stock_max,stock_min,omega,tol ):
        X = np.log(self.stock_price/strike_price)
        tau = (0.5*volatility**2)*(maturity_date-t)
        k = interest_rate/(0.5*volatility**2)
        kd = (interest_rate - dividend_yield)/(0.5*volatility**2)
        N = Nplus - Nminus
        u = np.zeros((N+1,M+1))
        dt = (0.5*(volatility**2)*maturity_date)/M
        Xzero = np.log(stock_min/strike_price)
        Xmax = np.log(stock_max/strike_price)
        dX = (Xmax-Xzero)/N
        alpha = dt/(dX*dX)
       
        g = np.zeros((N+1,M+1))
        Xmesh = np.arange(Xzero,Xmax,dX)
        Xmesh2=np.linspace(Xzero,Xmax,N+1)
        Tmesh = np.arange(0,(0.5*volatility**2*maturity_date),dt)
        Tmesh2 = np.append(Tmesh,(0.5*volatility**2*maturity_date))
        
        #payoff matrix
        for j in range(0,N+1):
            for i in range(1,M+1):
                g[j,i] = np.exp((0.25*(kd-1)**2 + k)*((i-1)*dt))*np.maximum(np.exp(0.5*(kd+1)*(j+Nminus-1)*dX)-
                np.exp(0.5*(kd-1)*(j+Nminus-1)*dX),0)
            g[j,0] = np.maximum(np.exp(0.5*(kd+1)*(j+Nminus-1)*dX)-np.exp(0.5*(kd-1)*(j+Nminus-1)*dX),0)
        g[0,:]=0
        
        
        #Boundary conditions of the u matrix
        u[:,0] = g[:,0]
        u[0,:] = g[0,:]
        u[N,:] = g[N,:]
        
        a = -alpha
        b = 1 + 2*alpha
        c = -alpha
        z = np.linspace(0,0,N-1)
        for p in range(0,M):
            x = np.maximum(u[1:N,p],g[1:N,p+1])
            temp = np.zeros((N-1,))
            temp[0] = a*g[0,p+1]
            temp[-1] = c*g[N,p+1]
            bt = u[1:N,p]-temp
            #b = b_one[0,:]
            xold = 1000*x
            n = len(x)
            while np.linalg.norm(xold-x)>tol:
                xold = x
                for i in range(0,n):
                    if i==0:
                        z = (bt[i]+alpha*x[i+1])/(1+2*alpha)
                        x[i] = np.maximum(omega*z + (1-omega)*xold[i],g[i,p])
                    elif i==n-1:
                        z = (bt[i]+alpha*x[i-1])/(1 + 2*alpha)
                        x[i] = np.maximum(omega*z + (1-omega)*xold[i],g[i,p])
                    else:
                        z = (bt[i]+alpha*(x[i-1]+x[i+1]))/(1 + 2*alpha)
                        x[i] = np.maximum(omega*z + (1-omega)*xold[i],g[i,p])
                u[1:N,p+1] = x
              
        uresult = np.interp(X,Xmesh2,u[:,M])
        implicit_value = strike_price*strike_price**(0.5*(kd-1))*self.stock_price**(-0.5*(kd-1))*np.exp((-0.25*(kd-1)**2 - k)*0.5*volatility**2*(maturity_date-t))*uresult        
        return implicit_value
        
    def Black_Scholes_Crank_Nicholson_FD_AO(self,t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price,stock_max,stock_min,omega,tol ):
        
        X = np.log(self.stock_price/strike_price)
        k = interest_rate/(0.5*volatility**2)
        kd = (interest_rate - dividend_yield)/(0.5*volatility**2)
        N = Nplus - Nminus
        u = np.zeros((N+1,M+1))
        dt = (0.5*(volatility**2)*maturity_date)/M
        Xzero = np.log(stock_min/strike_price)
        Xmax = np.log(stock_max/strike_price)
        dX = (Xmax-Xzero)/N
        alpha = dt/(dX*dX)
        g = np.zeros((N+1,M+1))
        Xmesh2=np.linspace(Xzero,Xmax,N+1)
        Tmesh = np.arange(0,(0.5*volatility**2*maturity_date),dt)
        Tmesh2 = np.append(Tmesh,(0.5*volatility**2*maturity_date))
        
        #payoff matrix
        for j in range(0,N+1):
            for i in range(1,M+1):
                g[j,i] = np.exp((0.25*(kd-1)**2 + k)*((i-1)*dt))*np.maximum(np.exp(0.5*(kd+1)*(j+Nminus-1)*dX)-
                np.exp(0.5*(kd-1)*(j+Nminus-1)*dX),0)
            g[j,0] = np.maximum(np.exp(0.5*(kd+1)*(j+Nminus-1)*dX)-np.exp(0.5*(kd-1)*(j+Nminus-1)*dX),0)
        g[0,:]=0
        
        #Boundary conditions of the u matrix
        u[:,0] = g[:,0]
        u[0,:] = g[0,:]
        u[N,:] = g[N,:]
        
        a= -alpha
        b = 1 + 2*alpha
        c = -alpha
        zmatrix = np.zeros((N+1,M+1))
        for p in range(0,M):
            temp = np.zeros((N-1,))
            temp[0] = a*g[0,p+1]
            temp[-1] = c*g[N,p+1]
            zmatrix[1:N,p] = (1-alpha)*u[1:N,p]+0.5*alpha*(u[2:N+1,p]+u[0:N-1,p])
            bt = zmatrix[1:N,p]-temp
            x = np.maximum(u[1:N,p],g[1:N,p+1])
            #b = b_one[0,:]
            xold = 1000*x
            n = len(x)
            while np.linalg.norm(xold-x)>tol:
                xold = x
                for i in range(0,n):
                    if i==0:
                        z = (bt[i]+0.5*alpha*x[i+1])/(1+ alpha)
                        x[i] = np.maximum(omega*z + (1-omega)*xold[i],g[i,p])
                    elif i==n-1:
                        z = (bt[i]+0.5*alpha*x[i-1])/(1 + alpha)
                        x[i] = np.maximum(omega*z + (1-omega)*xold[i],g[i,p])
                    else:
                        z = (bt[i]+0.5*alpha*(x[i-1]+x[i+1]))/(1 + alpha)
                        x[i] = np.maximum(omega*z + (1-omega)*xold[i],g[i,p])
                u[1:N,p+1] = x
              
        uresult = np.interp(X,Xmesh2,u[:,M])
        crank_value = strike_price*strike_price**(0.5*(kd-1))*self.stock_price**(-0.5*(kd-1))*np.exp((-0.25*(kd-1)**2 - k)*0.5*volatility**2*(maturity_date-t))*uresult        
        return crank_value
        
strike_price = [8,8,8]
maturity_date = 1
stock_price4 = 4
stock_price6 = 6
stock_price8 = 8
stock_price11 = 11
stock_price12 = 12
stock_price15 = 15
stock_max = 150
stock_min = 0.4
t = 0
interest_rate = 0.1
dividend_yield = 0.08 
dividend_yield3 = 0.03 
dividend_yield5 = 0.05 
dividend_yield6 = 0.06 
dividend_yield8 = 0.08 
dividend_yield11 = 0.11   
volatility = 0.4
omega = 1.2
tol = 0.001
M =200
Nplus = 100
Nminus = -100
##Table 4.1
payoff_S4 = EuropeanOption(stock_price4,strike_price)
payoff_S6 = EuropeanOption(stock_price6,strike_price)
payoff_S8 = EuropeanOption(stock_price8,strike_price)
payoff_S11 = EuropeanOption(stock_price11,strike_price)
payoff_S12 = EuropeanOption(stock_price12,strike_price)
payoff_S15 = EuropeanOption(stock_price15,strike_price)

explicit_S4 = payoff_S4.Black_Scholes_Explicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min )
implicit_S4 = payoff_S4.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )
crank_S4 = payoff_S4.Black_Scholes_Crank_Nicholson_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )

explicit_S6 = payoff_S6.Black_Scholes_Explicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min )
implicit_S6 = payoff_S6.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )
crank_S6 = payoff_S6.Black_Scholes_Crank_Nicholson_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )

explicit_S8 = payoff_S8.Black_Scholes_Explicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min )
implicit_S8 = payoff_S8.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )
crank_S8 = payoff_S8.Black_Scholes_Crank_Nicholson_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )

explicit_S11 = payoff_S11.Black_Scholes_Explicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min )
implicit_S11 = payoff_S11.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )
crank_S11 = payoff_S11.Black_Scholes_Crank_Nicholson_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )

explicit_S12 = payoff_S12.Black_Scholes_Explicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min )
implicit_S12 = payoff_S12.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )
crank_S12 = payoff_S12.Black_Scholes_Crank_Nicholson_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )

explicit_S15 = payoff_S15.Black_Scholes_Explicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min )
implicit_S15 = payoff_S15.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )
crank_S15 = payoff_S15.Black_Scholes_Crank_Nicholson_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield,volatility,strike_price[0],stock_max,stock_min,omega,tol )



"Make the dataframe for the Error values"
df_table = pd.DataFrame(columns=['Explicit','Implicit','Crank Nicholson'], index=['4','6','8','11','12','15'])
df_table.loc['15'] = pd.Series({'Explicit':explicit_S15, 'Implicit':implicit_S15, 'Crank Nicholson':crank_S15})
df_table.loc['12'] = pd.Series({'Explicit':explicit_S12, 'Implicit':implicit_S12, 'Crank Nicholson':crank_S12})
df_table.loc['11'] = pd.Series({'Explicit':explicit_S11, 'Implicit':implicit_S11, 'Crank Nicholson':crank_S11})
df_table.loc['8'] = pd.Series({'Explicit':explicit_S8, 'Implicit':implicit_S8, 'Crank Nicholson':crank_S8})
df_table.loc['6'] = pd.Series({'Explicit':explicit_S6, 'Implicit':implicit_S6, 'Crank Nicholson':crank_S6})
df_table.loc['4'] = pd.Series({'Explicit':explicit_S4, 'Implicit':implicit_S4, 'Crank Nicholson':crank_S4})
df_table.index.name = 'S'

implicit_S4D3 = payoff_S4.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield3,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S6D3 = payoff_S6.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield3,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S8D3 = payoff_S8.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield3,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S11D3 = payoff_S11.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield3,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S12D3 = payoff_S12.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield3,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S15D3 = payoff_S15.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield3,volatility,strike_price[0],stock_max,stock_min,omega,tol )

implicit_S4D5 = payoff_S4.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield5,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S6D5 = payoff_S6.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield5,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S8D5 = payoff_S8.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield5,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S11D5 = payoff_S11.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield5,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S12D5 = payoff_S12.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield5,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S15D5 = payoff_S15.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield5,volatility,strike_price[0],stock_max,stock_min,omega,tol )

implicit_S4D6 = payoff_S4.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield6,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S6D6 = payoff_S6.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield6,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S8D6 = payoff_S8.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield6,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S11D6 = payoff_S11.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield6,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S12D6 = payoff_S12.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield6,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S15D6 = payoff_S15.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield6,volatility,strike_price[0],stock_max,stock_min,omega,tol )

implicit_S4D8 = payoff_S4.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield8,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S6D8 = payoff_S6.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield8,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S8D8 = payoff_S8.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield8,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S11D8 = payoff_S11.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield8,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S12D8 = payoff_S12.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield8,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S15D8 = payoff_S15.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield8,volatility,strike_price[0],stock_max,stock_min,omega,tol )

implicit_S4D11 = payoff_S4.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield11,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S6D11 = payoff_S6.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield11,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S8D11 = payoff_S8.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield11,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S11D11 = payoff_S11.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield11,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S12D11 = payoff_S12.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield11,volatility,strike_price[0],stock_max,stock_min,omega,tol )
implicit_S15D11 = payoff_S15.Black_Scholes_Implicit_FD_AO(t,maturity_date,M,Nminus,Nplus,interest_rate,dividend_yield11,volatility,strike_price[0],stock_max,stock_min,omega,tol )


"Make the dataframe for the Error values"
df_table2 = pd.DataFrame(columns=['d=0.03','d=0.05','d=0.06','d=0.08','d=0.11'], index=['4','6','8','11','12'])
#df_table.loc['15'] = pd.Series({'d=0.03':implicit_S15D3, 'd=0.05':implicit_S15D5, 'd=0.06':implicit_S15D6, 'd=0.08':implicit_S15D8, 'd=0.11':implicit_S15D11})
df_table2.loc['12'] = pd.Series({'d=0.03':implicit_S12D3, 'd=0.05':implicit_S12D5, 'd=0.06':implicit_S12D6, 'd=0.08':implicit_S12D8, 'd=0.11':implicit_S12D11})
df_table2.loc['11'] = pd.Series({'d=0.03':implicit_S11D3, 'd=0.05':implicit_S11D5, 'd=0.06':implicit_S11D6, 'd=0.08':implicit_S11D8, 'd=0.11':implicit_S11D11})
df_table2.loc['8'] = pd.Series({'d=0.03':implicit_S8D3, 'd=0.05':implicit_S8D5, 'd=0.06':implicit_S8D6, 'd=0.08':implicit_S8D8, 'd=0.11':implicit_S8D11})
df_table2.loc['6'] = pd.Series({'d=0.03':implicit_S6D3, 'd=0.05':implicit_S6D5, 'd=0.06':implicit_S6D6, 'd=0.08':implicit_S6D8, 'd=0.11':implicit_S6D11})
df_table2.loc['4'] = pd.Series({'d=0.03':implicit_S4D3, 'd=0.05':implicit_S4D5, 'd=0.06':implicit_S4D6, 'd=0.08':implicit_S4D8, 'd=0.11':implicit_S4D11})
df_table2.index.name = 'S'

"Print the Error table 4.1"
print(df_table)
print(' ')
"Print the Error table 4.2"
print(df_table2)
