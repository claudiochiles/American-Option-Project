#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:27:44 2018

@author: Claudio
"""

from ComputationalFinanceClass import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.interpolate import interpolate

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


"Print the Error table"
print(df_table)

