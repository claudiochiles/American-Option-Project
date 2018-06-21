#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:57:44 2018

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
##Table 4.2
payoff_S4 = EuropeanOption(stock_price4,strike_price)
payoff_S6 = EuropeanOption(stock_price6,strike_price)
payoff_S8 = EuropeanOption(stock_price8,strike_price)
payoff_S11 = EuropeanOption(stock_price11,strike_price)
payoff_S12 = EuropeanOption(stock_price12,strike_price)
payoff_S15 = EuropeanOption(stock_price15,strike_price)

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


"Print the Error table"
print(df_table2)

