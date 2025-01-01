import numpy as np 

# paths
codepath = './'
datapath = './data/'

anompath = datapath + 'Anomalies/17NOV21/NYSE/monthly/'

respath_root = './results/'


# constants
npc = 4         # number of PCs of D/P used to predict returns
L = 12          # horizon at which to predict returns
n_portf = 3     # number of portfolios in a sort
diag_rho_r = False # restrict returns to load only on own D/Ps
rm_loads_only_on_dpm = False # restrict the Rm equation in the state space to be predicted only with own D/P
debug = True    # set to True to redirect output to temp folder
compute_se = False # automatically compute standard errors for most plots
include_idiosync_shocks = True # include idiosyncratic shocks in \Delta p and PD equations for test assets?
market_is_avg_portfolio = False # re-define market as an average of long & short ends of all portfolios
retx_implied_div = False # use dividends implied by ret & retx for anomalies
load_BBK_yields = False # load and compare to BBK yields (in addition to BMSY)
newey_west = False # Use Newey-West SE
dtype = np.float64 # use double-precision arithmetic throughout
mat2plot = [1,2,5,7] # list of maturities to plot on the same figure for time-series figures
cummat2plot = [1,3,5,20,30]#50,100] # list of maturities to plot for prices of cumulative dividends
oos_start_date = '2025-01-01'   # beginning of OOS
mS = 2     # maturity of the short leg (for yield/return slopes use N minus mS, e.g., 7-1, or 7-2)
output_intermediate_data_results = True # output intermediate data and results

robustness_run = False # if True, terminate after Table 6 (robustness) entry is computed for a given model

# max number of periods to compute div strip prices for
p_max_t = 15 #30 # increase this if longer maturities are needed; currently only matuirties up to bonds_max_maturity are plotted
p_reporting_freq = 25 # report div strip prices in 25y increments


# maturities to plot
bonds_max_maturity = 15
maturities = [1, 2, 3, 5, 7, 10, bonds_max_maturity]
max_maturity = bonds_max_maturity
#mat2idx = maturities(2:end)-1

# start date (to compare to dividend strips papers which start in 2003)
#start_date = 0

# titles for plots
var_titles = {
  'EY': 'Equity yields',
  'FEY': 'Forward equity yields',
  'r': 'Strip returns',
  'rfwd': 'Strip forward returns',
  'EY_slopes': 'Equity yield slopes',
  'FEY_slopes': 'Forward equity yield slopes',
  'r_slopes': 'Strip return slopes',
  'rfwd_slopes': 'Strip forward return slopes',
}

factor_titles = {
    'mkt': 'Market',
    'pc1': 'PC1',
    'pc2': 'PC2',
    'pc3': 'PC3',
    'pc4': 'PC4',
    'pc5': 'PC5',
    'pc6': 'PC6',
    'pc7': 'PC7',
}
