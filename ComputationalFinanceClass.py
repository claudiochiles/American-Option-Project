    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:29:46 2018

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
        
    def Bull_Call_Spread(self):
        if self.strike_price_1 < self.strike_price_2:
            bull_call_spread = self.European_Call_Payoff(self.strike_price_1) - self.European_Call_Payoff(self.strike_price_2)
        else:
            print('Warning: inputs are incorrect')
            print('strike_price_1 < strike_price_2 does not hold')
            quit()
        return bull_call_spread
    
    def Bull_Put_Spread(self):
        if self.strike_price_1 > self.strike_price_2:
            bull_put_spread = self.European_Put_Payoff(self.strike_price_1) + self.European_Put_Payoff(self.strike_price_2)
        else:
            print('Warning: inputs are incorrect')
            print('strike_price_1 < strike_price_2 does not hold')
            quit()
        return bull_put_spread
    
    def Bear_Call_Spread(self):
        if self.strike_price_1 > self.strike_price_2:
            bear_call_spread = self.European_Call_Payoff(self.strike_price_1) - self.European_Call_Payoff(self.strike_price_2)
        else:
            print('Warning: inputs are incorrect')
            print('strike_price_1 < strike_price_2 does not hold')
            quit()
        return bear_call_spread
    
    def Collar(self):
        if self.strike_price_1 < self.strike_price_2:
            collar = self.European_Put_Payoff(self.strike_price_1) - self.European_Call_Payoff(self.strike_price_2)
        else:
            print('Warning: inputs are incorrect')
            print('strike_price_1 < strike_price_2 does not hold')
            quit()
        return collar
    
    def Straddle_Strategy(self,strike_price):
        straddle_strategy = self.European_Call_Payoff(strike_price) + self.European_Put_Payoff(strike_price)
        return straddle_strategy
    
    def Strangle_Strategy(self):
        if self.strike_price_1 != self.strike_price_2:
            strangle_strategy = self.European_Call_Payoff(self.strike_price_1) + self.European_Put_Payoff(self.strike_price_2)
        else:
            print('Warning: inputs are incorrect')
            print('strike_price_1 < strike_price_2 does not hold')
            quit()
        return strangle_strategy
    
    def Butterfly_Spread(self,p2):
        if self.strike_price_1 < self.strike_price_2 < self.strike_price_3:
            lambdaa = (self.strike_price_3-self.strike_price_2)/(self.strike_price_3-self.strike_price_1)
            butterfly_spread = p2*lambdaa*self.European_Call_Payoff(self.strike_price_1)+p2*(1-lambdaa)*self.European_Call_Payoff(self.strike_price_3)-p2*self.European_Call_Payoff(self.strike_price_2)
        else:
            print('Warning: inputs are incorrect')
            print('strike_price_1 < strike_price_2 < strike_price_3 does not hold')
            quit()
         
        return butterfly_spread
    
    def Black_Scholes_European_Call(self,t,maturity_date,interest_rate,dividend_yield,volatility,strike_price): 
        d1 = (np.log(self.stock_price/strike_price)+(interest_rate-dividend_yield + 0.5*volatility**2)*(maturity_date-t))/(volatility*np.sqrt(maturity_date - t))
        d2 = (np.log(self.stock_price/strike_price)+(interest_rate-dividend_yield - 0.5*volatility**2)*(maturity_date-t))/(volatility*np.sqrt(maturity_date - t))
        bs_european_call_price = np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.cdf(d1)-np.exp(-interest_rate*(maturity_date - t))*strike_price*norm.cdf(d2)
        bs_european_call_delta = np.exp(-dividend_yield*(maturity_date-t))*norm.cdf(d1)
        bs_european_call_theta = dividend_yield*np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.cdf(d1)-((volatility*np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.pdf(d1))/(2*np.sqrt(maturity_date-t)))-interest_rate*np.exp(-interest_rate*(maturity_date-t))*strike_price*norm.cdf(d2)
        bs_european_call_rho = (maturity_date - t)*np.exp(-interest_rate*(maturity_date - t))*strike_price*norm.cdf(d2)
        bs_european_call_gamma =(np.exp(-dividend_yield*(maturity_date - t))*norm.pdf(d2))/(self.stock_price*volatility*np.sqrt(maturity_date-t))
        bs_european_call_vega = np.sqrt(maturity_date - t)*np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.pdf(d1)
        
        
        return bs_european_call_price, bs_european_call_delta, bs_european_call_theta,bs_european_call_gamma, bs_european_call_rho,bs_european_call_vega
    
    def Black_Scholes_European_Put(self,t,maturity_date,interest_rate,dividend_yield,volatility,strike_price): 
        d1 = (np.log(self.stock_price/strike_price)+(interest_rate-dividend_yield + 0.5*volatility**2)*(maturity_date-t))/(volatility*np.sqrt(maturity_date - t))
        d2 = (np.log(self.stock_price/strike_price)+(interest_rate-dividend_yield - 0.5*volatility**2)*(maturity_date-t))/(volatility*np.sqrt(maturity_date - t))
        bs_european_put_price = -np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.cdf(-d1)+np.exp(-interest_rate*(maturity_date - t))*strike_price*norm.cdf(-d2)
        bs_european_put_delta = -np.exp(-dividend_yield*(maturity_date-t))*norm.cdf(-d1)
        bs_european_put_theta = -dividend_yield*np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.cdf(-d1)-((volatility*np.exp(-dividend_yield*(maturity_date-t))*self.stock_price*norm.pdf(d1))/(2*np.sqrt(maturity_date-t)))+interest_rate*np.exp(-interest_rate*(maturity_date-t))*strike_price*norm.cdf(-d2)
        bs_european_put_rho = -(maturity_date - t)*np.exp(-interest_rate*(maturity_date - t))*strike_price*norm.cdf(-d2)
        bs_european_put_gamma =(np.exp(-interest_rate*(maturity_date - t))*strike_price*norm.pdf(d2))/((self.stock_price**2)*volatility*np.sqrt(maturity_date-t))
        bs_european_put_vega = np.sqrt(maturity_date - t)*np.exp(-interest_rate*(maturity_date-t))*strike_price*norm.pdf(d2)
        
        return bs_european_put_price, bs_european_put_delta, bs_european_put_theta,bs_european_put_gamma,bs_european_put_rho, bs_european_put_vega  

    def Black_Scholes_Explicit_FD_EO(self,t,maturity_date,M,N,interest_rate,dividend_yield,volatility,initial_condition, boundary_condition,strike_price,stock_max,stock_min ):
        deltaTau = maturity_date/M
        deltaS = (stock_max-stock_min)/N
        ic_call = 0
        ic_put = 1
        i = np.linspace(1,N+1,N+1)
        Sprice = deltaS * i 
        initial = Sprice[0]
        end = Sprice[N]
        Sprice = Sprice[1:N]
        p = np.linspace(1,M+1,M+1)
        Tau = deltaTau * p
        Tau = Tau[1:M]
        varray = []
        #if np.all(deltaTau <= (deltaS**2)/2): 
        l = deltaTau * ((volatility**2* (Sprice/deltaS)**2)/2 - ((interest_rate - dividend_yield) * (Sprice/deltaS)/2))
        d = deltaTau * ((1/deltaTau) - (volatility**2 * (Sprice/deltaS)**2) - interest_rate)
        u = deltaTau * ((volatility**2* (Sprice/deltaS)**2)/2 + ((interest_rate - dividend_yield) * (Sprice/deltaS)/2))
        A = np.zeros((N-1,N-1))
        v = []
        for j in range(1,N-2):
            A[j,j] = d[j]
            A[j,j-1] = l[j]
            A[j,j+1] = u[j]       
        if boundary_condition == 1:
            A[0,0] = d[0] + 2 * l[0]
            A[0,1] = u[0] - l[0]
            A[N-2,N-2] = d[-1] + 2 * u[-1]
            A[N-2,N-3] = l[-1] - u[-1]
        else:
            A[0,0] = d[0]
            A[0,1] = u[0]
            A[N-2,N-2] = d[-1]
            A[N-2,N-3] = l[-1]
        
        ##put
        if initial_condition == ic_put:
            v = np.maximum(strike_price-Sprice,0)
            for k in range(M):
                v = A @ v
                if boundary_condition == 0:
                    v[0] = v[0]+l[0]*(np.exp(-interest_rate*(k+1)*deltaTau)*strike_price -np.exp(-dividend_yield*(k+1)*deltaTau)*stock_min)
                    v[-1] = 0
                    #v[0] = v[0]+v[0]*initial
            varray.append(v)
            bs_explicit_fd_eo_price=varray
        ##call
        if initial_condition == ic_call:
            v = np.maximum(Sprice-strike_price,0)
            for k in range(M): 
                v = A @ v
                if boundary_condition == 0:
                    v[0] = 0
                    v[-1] = v[-1]+u[-1]*(np.exp(-dividend_yield*(k+1)*deltaTau)*stock_max - np.exp(-interest_rate*(k+1)*deltaTau)*strike_price)
                    #v[-1] = stock_max - strike_price*np.exp(-interest_rate*(k+1)*de
                    #v[-1] = stock_max - strike_price*np.exp(-interest_rate*(k+1)*deltaTau)
            
            varray.append(v)
            bs_explicit_fd_eo_price=varray
       
            
        if 0 < volatility**2 * stock_max**2 * (deltaTau/(deltaS**2)) <= 0.5:
            return bs_explicit_fd_eo_price
        else:
            print('bad inputs')
        
    def Black_Scholes_Implicit_FD_EO(self,t,maturity_date,M,N,interest_rate,dividend_yield,volatility,initial_condition, boundary_condition,strike_price,stock_max,stock_min ):
        deltaTau = maturity_date/M
        deltaS = (stock_max-stock_min)/N
        ic_call = 0
        ic_put = 1
        i = np.linspace(1,N+1,N+1)
        Sprice = deltaS * i 
        initial = Sprice[0]
        end = Sprice[N]
        Sprice = Sprice[1:N]
        varray = []
        #if np.all(deltaTau <= (deltaS**2)/2): 
        l = deltaTau * (-(volatility**2 * (Sprice/deltaS)**2)/2 + ((interest_rate - dividend_yield) * (Sprice/deltaS))/2)
        d = deltaTau * ((1/deltaTau) + (volatility**2 * (Sprice/deltaS)**2) + interest_rate)
        u = deltaTau * (-(volatility**2 * (Sprice/deltaS)**2)/2 - ((interest_rate - dividend_yield) * (Sprice/deltaS))/2)
        A = np.zeros((N-1,N-1))
        v = []
        for j in range(1,N-2):
            A[j,j] = d[j]
            A[j,j-1] = l[j]
            A[j,j+1] = u[j]       
        if boundary_condition == 1:
            A[0,0] = d[0] + 2 * l[0]
            A[0,1] = u[0] - l[0]
            A[N-2,N-2] = d[-1] + 2 * u[-1]
            A[N-2,N-3] = l[-1] - u[-1]
        else:
            A[0,0] = d[0]
            A[0,1] = u[0]
            A[N-2,N-2] = d[-1]
            A[N-2,N-3] = l[-1]
        
        ##put
        if initial_condition == ic_put:
            v = np.maximum(strike_price-Sprice,0)
            for k in range(M):
                v = np.linalg.solve(A,v)
                if boundary_condition == 0:
                    v[0] = v[0]+-l[0]*(np.exp(-interest_rate*(k+1)*deltaTau)*strike_price -np.exp(-dividend_yield*(k+1)*deltaTau)*stock_min)
                    v[-1] = 0
                    #v[-1] = v[-1] + u[-1] * (stock_max - strike_price)
                    #v[0] = v[0]+v[0]*initial
            varray.append(v)
            bs_implicit_fd_eo_price=varray
        ##call
        if initial_condition == ic_call:
            v = np.maximum(Sprice-strike_price,0)
            for k in range(M): 
                v = np.linalg.solve(A,v)
                if boundary_condition == 0:
                    v[0] = 0
                    v[-1] = v[-1]+-u[-1]*(np.exp(-dividend_yield*(k+1)*deltaTau)*stock_max - np.exp(-interest_rate*(k+1)*deltaTau)*strike_price)
                    #v[-1] = stock_max - strike_price*np.exp(-interest_rate*(k+1)*deltaTau)
            
            varray.append(v)
            bs_implicit_fd_eo_price=varray

            
        if deltaTau > 0:
            return bs_implicit_fd_eo_price
        else:
            print('bad inputs')
    def Black_Scholes_CrackNicholson_FD_EO(self,t,maturity_date,M,N,interest_rate,dividend_yield,volatility,initial_condition, boundary_condition,strike_price,stock_max,stock_min):   
        deltaTau = maturity_date/M
        deltaS = (stock_max-stock_min)/N
        ic_call = 0
        ic_put = 1
        i = np.linspace(1,N+1,N+1)
        Sprice = deltaS * i 
        #initial = Sprice[0]
        #end = Sprice[N]
        Sprice = Sprice[1:N]
        varray = []
        #if np.all(deltaTau <= (deltaS**2)/2): 
        lE = deltaTau * ((volatility**2* (Sprice/deltaS)**2)/2 - ((interest_rate - dividend_yield) * (Sprice/deltaS)/2))
        dE = deltaTau * ((1/deltaTau) - (volatility**2 * (Sprice/deltaS)**2) - interest_rate)
        uE = deltaTau * ((volatility**2* (Sprice/deltaS)**2)/2 + ((interest_rate - dividend_yield) * (Sprice/deltaS)/2))
        AE = np.zeros((N-1,N-1))
        lI = deltaTau * (-(volatility**2 * (Sprice/deltaS)**2)/2 + ((interest_rate - dividend_yield) * (Sprice/deltaS))/2)
        dI = deltaTau * ((1/deltaTau) + (volatility**2 * (Sprice/deltaS)**2) + interest_rate)
        uI = deltaTau * (-(volatility**2 * (Sprice/deltaS)**2)/2 - ((interest_rate - dividend_yield) * (Sprice/deltaS))/2)
        AI = np.zeros((N-1,N-1))
        v = []
        for j in range(1,N-2):
            AE[j,j] = dE[j]
            AE[j,j-1] = lE[j]
            AE[j,j+1] = uE[j] 
            AI[j,j] = dI[j]
            AI[j,j-1] = lI[j]
            AI[j,j+1] = uI[j]       
        if boundary_condition == 1:
            AE[0,0] = dE[0] + 2 * lE[0]
            AE[0,1] = uE[0] - lE[0]
            AE[N-2,N-2] = dE[-1] + 2 * uE[-1]
            AE[N-2,N-3] = lE[-1] - uE[-1]
            AI[0,0] = dI[0] + 2 * lI[0]
            AI[0,1] = uI[0] - lI[0]
            AI[N-2,N-2] = dI[-1] + 2 * uI[-1]
            AI[N-2,N-3] = lI[-1] - uI[-1]
        else:
            AE[0,0] = dE[0]
            AE[0,1] = uE[0]
            AE[N-2,N-2] = dE[-1]
            AE[N-2,N-3] = lE[-1]
            AI[0,0] = dI[0]
            AI[0,1] = uI[0]
            AI[N-2,N-2] = dI[-1]
            AI[N-2,N-3] = lI[-1]

        if initial_condition == ic_put:
            v = np.maximum(strike_price-Sprice,0)
            for k in range(M):
                v = np.linalg.solve((np.add(AI,np.identity(len(AI)))),((np.add(AE,np.identity(len(AE))))@v))
                if boundary_condition == 0:
                    v[0] = v[0]+(0.5*lE[0]-0.5*lI[0])*(np.exp(-interest_rate*(k+1)*deltaTau)*strike_price -np.exp(-dividend_yield*(k+1)*deltaTau)*stock_min)
                    v[-1] = 0
                    #v[-1] = v[-1] + u[-1] * (stock_max - strike_price)
                    #v[0] = v[0]+v[0]*initial
            varray.append(v)
            bs_theta_scheme_fd_eo_price=varray
        ##call
        if initial_condition == ic_call:
            v = np.maximum(Sprice-strike_price,0)
            for k in range(M): 
                v = np.linalg.solve((np.add(AI,np.identity(len(AI)))),((np.add(AE,np.identity(len(AE))))@v))
                if boundary_condition == 0:
                    v[0] = 0
                    v[-1] = v[-1]+(0.5*uE[-1]-0.5*uI[-1])*(np.exp(-dividend_yield*(k+1)*deltaTau)*stock_max - np.exp(-interest_rate*(k+1)*deltaTau)*strike_price)#*(np.exp(-dividend_yield*(k+1)*deltaTau)*stock_max - np.exp(-interest_rate*(k+1)*deltaTau)*strike_price)
                    #(np.exp(-dividend_yield*(k+1)*deltaTau)*stock_max - np.exp(-interest_rate*(k+1)*deltaTau)*strike_price))
                    #v[-1] = v[-1]+-uI[-1]*(np.exp(-dividend_yield*(k+1)*deltaTau)*stock_max - np.exp(-interest_rate*(k+1)*deltaTau)*strike_price)
                    #v[-1] = stock_max - strike_price*np.exp(-interest_rate*(k+1)*deltaTau)
            varray.append(v)
            bs_theta_scheme_fd_eo_price=varray
        #if theta != 0 & theta != 1 & theta!= 0.5:
            #print('theta has to be either 0,1/2 or 1')
        if deltaTau > 0:
            return bs_theta_scheme_fd_eo_price
        else:
            print('bad inputs')
        
    
    def Black_Scholes_Theta_Scheme_FD_EO(self,t,maturity_date,M,N,interest_rate,dividend_yield,volatility,initial_condition, boundary_condition,strike_price,stock_max,stock_min,theta):
        Explicit = np.asarray(self.Black_Scholes_Explicit_FD_EO(t,maturity_date,M,N,interest_rate,dividend_yield,volatility,initial_condition, boundary_condition,strike_price,stock_max,stock_min))
        Implicit = np.asarray(self.Black_Scholes_Implicit_FD_EO(t,maturity_date,M,N,interest_rate,dividend_yield,volatility,initial_condition, boundary_condition,strike_price,stock_max,stock_min))
        #zzz = 5*Explicit
        bs_theta_scheme_fd_eo_price = (1-theta)*Explicit+theta*Implicit
        return bs_theta_scheme_fd_eo_price
    
    def Gaussian_RBF(self,x,y,e):
        r = (abs(x-y))
        phi_ga_rbf = np.exp(-e*r)**2
        phi_x_ga_rbf = (2*e**2)*(x-y)*(np.exp(-e*r)**2)
        phi_xx_ga_rbf = (2*e**2)*(np.exp(-e*r)**2)*((2*e**2)*((x-y)**2) + 1)
        return phi_ga_rbf,phi_x_ga_rbf,phi_xx_ga_rbf

    def Multiquadric_RBF(self,x,y,e):
        r = (abs(x-y))**2
        phi_multiq_rbf = np.sqrt((e**2)+r)
        phi_multiq_x_rbf = (x-y)/np.sqrt((e**2)+r)
        phi_multiq_xx_rbf = (1/np.sqrt((e**2)+r))-(((x-y)**2)/(np.sqrt((e**2)+r)**3))
        return phi_multiq_rbf,phi_multiq_x_rbf,phi_multiq_xx_rbf
    
    def Inverse_Multiquadric_RBF(self,x,y,e):
        r = (abs(x-y))**2
        phi_invmultiq_rbf = 1/np.sqrt((e**2)+r)
        phi_invmultiq_x_rbf = -1*((x-y)/((np.sqrt((e**2)+r))**3))
        phi_invmultiq_xx_rbf = (2*((x-y)**2)-(e**2))/(np.sqrt((e**2)+r)**5)
        return phi_invmultiq_rbf,phi_invmultiq_x_rbf,phi_invmultiq_xx_rbf 
    
    def Inverse_Quadratic_RBF(self,x,y,e):
        r = (abs(x-y))**2
        phi_invq_rbf = 1/((e**2)+r)
        phi_invq_x_rbf = -2*(x-y)/((x-y)**2 + e**2)**2
        phi_invq_xx_rbf = ((8*((x-y)**2))/((x-y)**2 + e**2)**3)- (2/(((x-y)**2) + (e**2))**2)
        return phi_invq_rbf,phi_invq_x_rbf,phi_invq_xx_rbf
    
    def Black_Scholes_Global_RBF_EO(self,t,maturity_date,M,N,interest_rate,volatility,initial_condition,rbf_function, boundary_condition,strike_price,stock_max,stock_min,e): 
        x = np.log(self.stock_price)
        y = np.log(self.stock_price)
        x,y = np.meshgrid(x,y)
        Tau = maturity_date - t
        deltaTau = maturity_date/M
        ic_call = 0
        ic_put = 1
        lambdaa = np.zeros((N-1,N-1))
        L = rbf_function[0]
        Lx = rbf_function[1]
        Lxx = rbf_function[2]
        
        #P = -inv(L)*((interest_rate-0.5*volatility**2)*Lx + 0.5*volatility**2*Lxx-interest_rate*L)
        P = solve(L,(np.subtract(np.add(np.multiply((0.5*volatility**2),Lxx),(np.multiply(interest_rate,L))),np.multiply((interest_rate- 0.5*volatility**2),Lx))))
        ##put
        identity = np.identity(len(P))
        tau2 = np.multiply(0.5,deltaTau)
        H = identity+(0.5*deltaTau*P)
        G = identity - (0.5*deltaTau*P)
        v = np.zeros((N-1,N-1))
        varray = []
        bs_global_rbf_eo_price = []
        ##put
        if ((initial_condition == ic_put) & (boundary_condition == 0)):
            for i in range(0,N-1):
                v[:,0] = np.maximum(strike_price-np.exp(x[i]),0)
            v[0,0] = np.exp(-interest_rate*Tau)*strike_price
            v[-1,0] = 0
            lambdaa[:,0] = solve(L,v[:,0])
            for k in range(1,N-2):
                lambdaa[:,k] = solve(G,(H @ lambdaa[:,k-1]))
                v[:,k] = L @ lambdaa[:,k]
                v[0,k] = np.exp(-interest_rate*Tau)*strike_price
                v[-1,k] = 0
                lambdaa[:,k] = solve(L,v[:,k])
            varray = L @ lambdaa[:,-2]
            bs_global_rbf_eo_price = varray
        
        ##call
        if ((initial_condition == ic_call) & (boundary_condition == 0)):
            for i in range(0,N-1):
                v[:,0] = np.maximum(np.exp(x[i])-strike_price,0)
            v[0] = 0
            v[-1,0] = stock_max - np.exp(-interest_rate*Tau)*strike_price
            lambdaa[:,0] = solve(L,v[:,0])
            for k in range(1,N-2):
                lambdaa[:,k] = solve(G,(H @ lambdaa[:,k-1]))
                v[:,k] = L @ lambdaa[:,k]
                v[0,k] = 0
                v[-1,k] = stock_max - np.exp(-interest_rate*Tau)*strike_price
                lambdaa[:,k] = solve(L,v[:,k])
            varray = L @ lambdaa[:,-2]
            bs_global_rbf_eo_price = varray
       
        return bs_global_rbf_eo_price

    def Black_Scholes_RBF_FD_DO(self, t,maturity_date, interest_rate, volatility, stock_min, stock_max, N, M,
                                rbf_function, initial_condition, boundary_condition, e,strike_price):
        x = np.log(self.stock_price)
        y = np.log(self.stock_price)
        x,y = np.meshgrid(x,y)
        Tau = maturity_date - t
        deltaTau = maturity_date/M
        ic_call = 0
        ic_put = 1
        L = rbf_function[0]
        Lx = rbf_function[1]
        Lxx = rbf_function[2]

        W = np.zeros(shape=(N-1, M-1))
        v = np.zeros((N-1,1))
        index = np.arange(N)
        V = np.zeros((N-1,N-1))
        for i in index:
            if i == 0:
                zphi = np.zeros(shape=(1, 2))
                w = np.zeros(shape=(2, 2))
                w[0, 0] = L[0, 0]
                w[0, 1] = L[1, 0]
                w[1, 0] = L[0, 1]
                w[1, 1] = L[1, 1]
                zphi[0, 0] = (interest_rate - volatility ** 2 / 2) * Lx[0, 0] - interest_rate * L[0, 0]
                zphi[0, 1] = (interest_rate - volatility ** 2 / 2) * Lx[0, 1]- interest_rate * L[0, 1]
                W[0, [0, 1]] = solve(w, zphi.T).T
            elif i == N - 1:
                zphi = np.zeros(shape=(1, 2))
                w = np.zeros(shape=(2, 2))
                w[0, 0] = L[N - 3, N - 3]
                w[0, 1] = L[N - 2, N - 3]
                w[1, 0] = L[N - 3, N - 2]
                w[1, 1] = L[N - 2, N - 2]
                zphi[0, 0] = (interest_rate - volatility ** 2 / 2) * Lx[N - 2, N - 3] - \
                             interest_rate * L[N - 2, N - 3]
                zphi[0, 1] = (interest_rate - volatility ** 2 / 2) * Lx[N - 2, N - 2] - \
                             interest_rate * L[N - 2, N - 2]
                W[N - 2, [N - 3, N - 2]] = solve(w, zphi.T).T
            else:
                zphi = np.zeros(shape=(1, 3))
                w = np.zeros(shape=(3, 3))
                w[0, 0] = L[i - 2, i - 2]
                w[0, 1] = L[i-1, i - 2]
                w[0, 2] = L[i , i - 2]
                w[1, 0] = L[i - 2, i - 1]
                w[1, 1] = L[i - 1, i - 1]
                w[1, 2] = L[i , i - 1]
                w[2, 0] = L[i - 2, i ]
                w[2, 1] = L[i-1, i ]
                w[2, 2] = L[i , i ]
                zphi[0, 0] = (interest_rate - volatility ** 2 / 2) * Lx[i-1, i - 2] + \
                             0.5 * volatility ** 2 * Lxx[i-1, i - 2] - interest_rate * L[i-1, i - 2]
                zphi[0, 1] = (interest_rate - volatility ** 2 / 2) * Lx[i-1, i-1] + \
                             0.5 * volatility ** 2 * Lxx[i-1, i-1] - interest_rate * L[i-1, i-1]
                zphi[0, 2] = (interest_rate - volatility ** 2 / 2) * Lx[i-1, i] + \
                             0.5 * volatility ** 2 * Lxx[i-1, i] - interest_rate * L[i-1, i]
                W[i-1, [i - 2, i-1, i ]] = solve(w, zphi.T).T
                
        if ((initial_condition == ic_put) & (boundary_condition == 0)):
            for i in range(0,N-1):
                v[:,0] = np.maximum(strike_price-np.exp(x[i]),0)
            v[0,0] = np.exp(-interest_rate*Tau)*strike_price
            v[-1,0] = 0
            V[:,0] = v[:,0]
            identity = np.identity(len(W))
            one = (identity - 0.5 * deltaTau * W)
            two = (identity + 0.5 * deltaTau * W)
            for k in range(1,M-1):
               V[:,k] = solve(one,(two@V[:,k-1]))
               V[0, k] = np.exp(-interest_rate*(maturity_date-t))*strike_price 
               V[-1, k] = 0
        varray = V[:,-1]
        bs_global_rbf_do_price = varray
                
                ##call
        if ((initial_condition == ic_call) & (boundary_condition == 0)):
            for i in range(0,N-1):
                v[:,0] = np.maximum(np.exp(x[i])-strike_price,0)
            v[0] = 0
            v[-1,0] = stock_max - np.exp(-interest_rate*Tau)*strike_price
            V[:,0] = v[:,0]
            identity = np.identity(len(W))
            one = (identity - 0.5 * deltaTau * W)
            two = (identity + 0.5 * deltaTau * W)
            for k in range(1,M-1):
               V[:,k] = solve(one,(two@V[:,k-1]))
               V[0, k] = 0 
               V[-1, k] = stock_max - np.exp(-interest_rate*(maturity_date-t))*strike_price
            varray = V[:,-1]
            bs_global_rbf_do_price = varray
        
        return bs_global_rbf_do_price
        
    def Geometric_Brownian_Motion_Trajectory(self, trajec, T, mu, sigma, S0, dt):
        H=[]
        L=[]
        for k in range (0,trajec):
           length = round(T/dt)
           t = np.linspace(0, T, round(T/dt))
           W = np.random.standard_normal(size = round(T/dt)) 
           W = np.cumsum(W)*np.sqrt(dt)
           for j in range(1,length):
               X = (mu-0.5*sigma*2)*(t[j]-t[j-1]) + sigma*np.sqrt((t[j]-t[j-1]))*W 
               S = S0*np.exp(X)
           H.append(S[-1])
           L.append(S)
        return H,L
    
    def Geometric_Brownian_Motion_Jump(self, path_number, step_number, T, r, div, sigma, lam_jump, mu_jump, sigma_jump, S0):
        Z1 = np.random.normal(0,1,size = (path_number,step_number))
        Z2 = np.random.normal(0,1,size = (path_number,step_number))
        dt = T / step_number
        lam = lam_jump * dt
        N = np.random.poisson(lam = lam, size = (path_number,step_number))
        S = np.zeros((path_number,step_number),dtype = float)
        M = np.zeros((path_number,step_number),dtype = float)
        X = np.zeros((path_number,step_number),dtype = float)
        for i in range(0,path_number):
            for j in range(0,step_number):
                if N[i,j] > 0:
                    M[i,j] = mu_jump * N[i,j] + sigma_jump * np.sqrt(N[i,j]) * Z2[i,j]
        S[:,0] = S0
        X[:,0] = np.log(S0)
        for j in range(1,step_number):
            X[:,j] = X[:,j-1] + ((r - div) - 0.5 * (sigma ** 2)) * dt + sigma * np.sqrt(dt) * Z1[:,j] + M[:,j]
            S[:,j] = np.exp(X[:,j])
        return S
    
    def Arithmetic_Average_Price_Asian_Call(self, path_number, step_number, T, r, div, sigma, lam_jump, mu_jump, sigma_jump, S0):
        arithmean = np.zeros((path_number,1))
        geometric_brownian_motion_jump = self.Geometric_Brownian_Motion_Jump(path_number,step_number,T,r,div,sigma,lam_jump,mu_jump,sigma_jump,S0)
        for i in range(0,path_number):
            arithmean[i] = np.average(geometric_brownian_motion_jump[i,:])
        asian_call_payoff = np.exp(-r * T) * np.maximum(arithmean - self.strike_price,0)
        asian_call = np.average(asian_call_payoff)
        return asian_call, asian_call_payoff, geometric_brownian_motion_jump
    
    def Geometric_Average_Price_Asian_Call(self, T, r, geometric_brownian_motion_jump):
        path_number = np.size(geometric_brownian_motion_jump,0)
        geomean = np.zeros((path_number,1))
        for i in range(0,path_number):
            geomean[i] = gmean(geometric_brownian_motion_jump[i,:])
        geo_asian_call_payoff = np.exp(-r * T) * np.maximum(geomean - self.strike_price,0)
        geo_asian_call = np.average(geo_asian_call_payoff)
        return geo_asian_call, geo_asian_call_payoff
    
    def BS_Geometric_Average_Price_Asian_Call(self, T, r, div, sigma, S0):
        b = 0.5 * (r + div + (1/6) * (sigma ** 2))
        sigma_tilda = sigma / np.sqrt(3)
        d1 = (np.log(S0//self.strike_price_1) + (r - b + 0.5 * (sigma_tilda ** 2)) * T) / (sigma_tilda * np.sqrt(T))
        d2 = d1 - sigma_tilda * np.sqrt(T)
        bs_geo_avg_price = S0 * np.exp(-b * T) * norm.cdf(d1) - self.strike_price_1 * np.exp(-r * T) * norm.cdf(d2)
        return bs_geo_avg_price
        
    def Control_Variables_Arithmetic_Average_Asian_Call(self, asian_call_payoff, geo_asian_call_payoff, bs_geo_avg_price):
        data = np.append(asian_call_payoff,geo_asian_call_payoff,axis = 1)
        data = pd.DataFrame(data = data)
        optimal_b = (data.cov().as_matrix())[0,1] / (data.cov().as_matrix())[1,1]
        reduction_factor = 1 / (1 - ((data.corr().as_matrix())[0,1]) ** 2)
        control_variable = asian_call_payoff - optimal_b * (geo_asian_call_payoff - bs_geo_avg_price)
        price_control_variable = np.average(control_variable)
        return reduction_factor, control_variable, price_control_variable
      
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
        
