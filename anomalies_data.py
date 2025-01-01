""" 
Data loader for anomalies and all other data used in the paper. 


Please cite the following paper when using this code:
    Stefano Giglio, Bryan Kelly, Serhiy Kozak "Equity Term Structures without Dividend Strips Data"
    Journal of Finance, 2024. Forthcoming

====================
Author: Serhiy Kozak
Date: November 2023
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import *

n_var = 'n'
ret_var = 'ret'
dp_var = 'dp_div'

def load_portf_vars(path, n_portf, extra_variables=None):

    def get_portf_list_from_folder(path, filename_fmt):
        import glob, re
        folders = glob.glob(path + filename_fmt.format('*'))
        portf_names = [re.search(filename_fmt.format('(.*)'), p)[1] for p in folders]
        return portf_names

    variables = [n_var, ret_var, dp_var]
    if extra_variables:
        variables += extra_variables
        
        
    filename_fmt = '{}{}_{}.csv' # var, num of portf, portf (example: ./ret3_size.csv)
    sp1, spN = 'p1', 'p{}'.format(n_portf)

    # list portfolio names based on the files present
    portf_names = get_portf_list_from_folder(path, filename_fmt.format(ret_var, n_portf, '{}'))
    
    # merge all files
    dfs = {}
    for v in variables: # query variables of interest (ret, dp, etc.)
        dfv = None
        
        # load and merge individual portfolio sorts (size, value, mom, etc.)
        for p in portf_names:
                
                # read from file
                data = pd.read_csv(path + filename_fmt.format(v, n_portf, p), 
                                                     parse_dates=True, index_col=[0])[[sp1, spN]].rename({sp1: p+'_'+sp1, spN: p+'_'+spN}, axis=1)
    
                # merge
                dfv = pd.concat((dfv, data), axis=1) if dfv is not None else data

        # save a table for a given variable  
        dfs[v] = dfv
        
    return dfs, portf_names



# TODO: check and drop any undiversified periods in the middle of sample

def find_undiversified_portfolios(df_num_firms, min_firms_in_portf=100, max_undiversified_time_periods=120):
    import re
    dfn = df_num_firms
    num_undiversified_periods = (~(dfn > min_firms_in_portf)).sum()
    portf_to_drop = list(set([re.match(r'(.+)_p\d+', c)[1] for c in dfn.columns[num_undiversified_periods > max_undiversified_time_periods]]))
    return portf_to_drop


     
def load_data(datapath, anompath, n_portf, bonds_max_maturity=15, market_is_avg_portfolio=False, L=12, 
                            retx_implied_div=False, d0=1900, dT=2100, min_firms_in_portf=100, portf_prefixes=None, max_undiversified_time_periods=120):
        
        sd = '_'
        sp1, spN = 'p1', 'p{}'.format(n_portf)

        # load FF factors
        import pandas_datareader.data as web
        FACT = web.DataReader('F-F_Research_Data_Factors', "famafrench",start=1900)[0]  
        FACT.index.names = ['date']
        FACT.index = FACT.index.to_timestamp()
        # FACT.index += MonthEnd(0)

        
        # load S&P500
        SP500 = pd.read_csv(datapath + 'sp500.csv', parse_dates=True, index_col=[0])
        SP500['dpm'] = ((SP500.vwretd - SP500.vwretx) * SP500.spindx.shift(1)).rolling(L).sum() / SP500.spindx
        SP500['rm'] = SP500.vwretd
        SP500.index = [d.replace(day=1) for d in SP500.index.tolist()]
        SP500 = SP500[['rm', 'dpm']]
        
        # load bond yields
        BONDS = pd.read_csv(datapath + 'bond_yields.csv', parse_dates=True, index_col=[0]) # continuously compounded (logs)
        BONDS.drop(['SVENY16', 'SVENY17', 'SVENY18', 'SVENY19', 'SVENY20'], axis=1, inplace=True)
        by = BONDS[['FBY{:02}'.format(m+1) for m in range(5)] + ['SVENY{:02}'.format(m+1) for m in range(5,bonds_max_maturity)]] / 100
        rf1y = by.values[:, :1] 
        p = (-by * np.arange(1,bonds_max_maturity+1)).values
        # 3-mo MA for smoothness
        fs = pd.DataFrame((p[:, :-1] - p[:, 1:] - rf1y), index=by.index, columns=['fs'+str(i) for i in range(2, bonds_max_maturity+1)]).rolling(3).mean()
        brx = pd.DataFrame((p[L:, :-1] - p[:-L, 1:] - rf1y[:-L,:])/np.arange(2,bonds_max_maturity+1), index=by.index[L:], 
                                             columns=['brx'+str(i) for i in range(2, bonds_max_maturity+1)]) # annual returns, up until then end o
        # Timing t: R_{t-1 -> t}, DP_t, BY_t
        
        # load aggregates
        import collections.abc
        anompath_is_list = isinstance(anompath, collections.abc.Sequence) and not isinstance(anompath, str)
        agg_path = (anompath[0] if anompath_is_list else anompath) + '../../aggregates.csv'
        AGG = pd.read_csv(agg_path, parse_dates=True, index_col=[0])[['vwret', 'vwretx', dp_var]]

        # load portfolios
        if anompath_is_list:
            Ps = []
            portf_names = []
            for i, path in enumerate(anompath):
                prefix = portf_prefixes[i]+':' if portf_prefixes and i < len(portf_prefixes) else ''

                # read data
                P, names = load_portf_vars(path, n_portf)
                
                # append a prefix to all column names
                for p in P.values():
                    p.columns = [prefix+c for c in p.columns]

                # append a prefix to all var names
                portf_names += [prefix+n for n in names]

                Ps += [P]

            # merge all tables across datasets
            P = {}
            for k in Ps[0].keys():
                P[k] = pd.concat([p[k] for p in Ps], axis=1)
                
        else:
            P, portf_names = load_portf_vars(anompath, n_portf)
            
            
        undiversified_portfolios = find_undiversified_portfolios(P[n_var], max_undiversified_time_periods=max_undiversified_time_periods)   
        print('Dropping these variables: ', undiversified_portfolios)
        cols_to_drop = [p+sd+sp1 for p in undiversified_portfolios] + [p+sd+spN for p in undiversified_portfolios]
        R = P[ret_var].drop(cols_to_drop, axis=1)
        R.columns = [c+sd+ret_var for c in R.columns]
        DP = P[dp_var].drop(cols_to_drop, axis=1)
        DP.columns = [c+sd+dp_var for c in DP.columns]
        avarnames = portf_names 
        for p in undiversified_portfolios:
            avarnames.remove(p)
        

        # merge all
        DATA = pd.concat((FACT, SP500, BONDS, AGG, R, DP, fs, brx), join='inner', axis=1)
        


        # drop undiversified periods
        DATA.dropna(inplace=True)

        print('===============> Final sample: {} -- {}'.format(DATA.index[0], DATA.index[-1]))
        
        
        # bond yields
        fb_yields = DATA[['FBY{:02d}'.format(y+1) for y in range(5)]] / 100
        gsw_yields = DATA[['SVENY{:02d}'.format(y+1) for y in range(bonds_max_maturity)]] / 100 
        
        # scale
        scale = 1.#/22.
        
        # aggregates
        rf = np.log(1.0 + DATA['RF']/100*scale)
        # rf = np.log(1.0 + DATA['FBY01']/100/12)
        
        # long ends of each anomaly
        reL = np.log(1.0 + DATA[[v+sd+spN+sd+ret_var for v in avarnames]]*scale).sub(rf,axis=0) # need to subtract rf in logs in order for it to cancel with -rf from SDF in the Euler equation
        reL.columns = avarnames # rename columns
        dpL = np.log(1.0 + DATA[[v+sd+spN+sd+dp_var for v in avarnames]]*scale)
        dpL.columns = avarnames # rename columns
        
        # short ends of each anomaly
        reS = np.log(1.0 + DATA[[v+sd+sp1+sd+ret_var for v in avarnames]]*scale).sub(rf,axis=0)
        reS.columns = avarnames # rename columns
        dpS = np.log(1.0 + DATA[[v+sd+sp1+sd+dp_var for v in avarnames]]*scale)
        dpS.columns = avarnames # rename columns
                
        # long-short
        reLS = reL - reS
        dpLS = dpL - dpS
                
        # all assets
        re_all = reL.join(reS, lsuffix='L', rsuffix='S')
        dp_all = dpL.join(dpS, lsuffix='L', rsuffix='S')
        
        # aggregate market    
        rme = np.log(1.0 + DATA[['rm']]*scale).sub(rf, axis=0) 

        # aggregate D/P
        dpm = np.log(1.0 + DATA[['dpm']]*scale) # div growth is much more volatile for ret implied d/p due to dividend reinvestment in the market
        
        # market = EW of all L&S portfolios
        if market_is_avg_portfolio:
                rme = re_all.mean(axis=1) 
                dpm = dp_all.mean(axis=1)
        
        # bond yields, continuously compounded (logs)
        by = fb_yields.iloc[:, :5].join(gsw_yields.iloc[:, 5:])
        fs = DATA[['fs'+str(y) for y in range(2,bonds_max_maturity+1)]]
        brx = DATA[['brx'+str(y) for y in range(2,bonds_max_maturity+1)]]    

        return reLS, dpLS, re_all, dp_all, rme, dpm, by, fs, brx, re_all.columns, avarnames
    
    