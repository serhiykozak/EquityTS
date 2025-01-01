""" 
Main file to replicate the paper. 


Please cite the following paper when using this code:
    Stefano Giglio, Bryan Kelly, Serhiy Kozak "Equity Term Structures without Dividend Strips Data"
    Journal of Finance, 2024. Forthcoming

====================
Author: Serhiy Kozak
Date: November 2023
"""

#| ### Import dependencies
import datetime, os, time, sys
import matplotlib
import statsmodels.api as sm
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import plot_functions as pf
import savedata as sv
from utils import * 
from constants import *
from model import *
from model import optimizer as optimizer

#%% Initialize 
#| ### Create output folders
respath = respath_root + 'test/'
figpath = respath + 'Figures/'
tblpath = respath + 'Tables/'
os.makedirs(figpath, exist_ok=True)
os.makedirs(tblpath, exist_ok=True)
print('Output folder: ' + respath)


#| ### System timer helper function
timeit_t0 = time.time()
def timeit(reset=False):
    global timeit_t0
    if reset:
        timeit_t0 = time.time()
    else:
        print('{:7.2f}s elapsed.'.format(time.time() - timeit_t0))


##############################################################################
#| # Main program
##############################################################################

#%% Load all data and prepare variables

# load all data
import anomalies_data as anom
input_data =  anom.load_data(
        datapath, anompath, n_portf, bonds_max_maturity=bonds_max_maturity, market_is_avg_portfolio=market_is_avg_portfolio, L=L, retx_implied_div=retx_implied_div,
        d0='1900', dT='2050', portf_prefixes=['a', 'ghz', 'r', 'wf'])#, max_undiversified_time_periods=60) # for _all_ and GHZrps and ratios
rx, dp, test_asset_ret, test_asset_dps, rme, dpm, by, fs, brx, labels, AnomalyNames = input_data 
sv.save_data(input_data, 'input_data', path=respath)  
    

#%%
# check timing
dp0 = (np.exp(test_asset_dps.iloc[:, 0].shift(1)) - 1) / (np.exp(test_asset_dps.iloc[:, 0]) - 1) - 1
r0 = np.exp(test_asset_ret.iloc[:-1, 0]) - 1
assert np.corrcoef(r0, dp0[1:])[0,1] > 0.5 # validate timing: change in dp ~ return

# generate labels for all assets and save them
factor_names = ['mkt'] + ['pc'+str(i) for i in range(1, npc)]
labels = factor_names + labels.to_list()
# sv.save_data(labels, 'labels', path=respath)  
with open(respath + 'labels.csv', 'w') as f:
    f.writelines(','.join(labels))


# weight for test assets in GMM moment conditions
_, N = test_asset_ret.shape
test_assets_gmm_weight = tf.constant(1.0 / np.sqrt(N)) # make GMM invariant to the total number of test assets

# model parameters
estimate_state_space_model = True



#%% Set up model variables and data

def init_vars_and_data(rx, dp, test_asset_ret, test_asset_dps, rme, dpm, by, *args, **kwargs):
    #| Define all model parameters to solve for. Note that certain matrices are restricted, so I define only submatrices which need to be estimated.
    K = 2*npc
    _, N = test_asset_ret.shape

    # state vector: intercept
    c = tf.Variable(tf.zeros([K, 1], dtype=dtype), name='c')


    # state vector: transition matrix
    rho_r = tf.Variable(tf.zeros([npc*(npc-1)+1,1], dtype=dtype), name='rho_r') if rm_loads_only_on_dpm \
        else tf.Variable(tf.zeros([npc] if diag_rho_r else [npc, npc], dtype=dtype), name='rho_r')
    rho_dp = tf.Variable(tf.zeros([npc, npc], dtype=dtype), name='rho_dp')

    # returns loadings (test_beta0, test_beta1 are computed endogenously using risk prices)
    # initial guess: market values of parameters
    test_beta2_ur = tf.Variable(tf.concat([tf.ones([N, 1], dtype=dtype),
                                                                                tf.zeros([N, npc-1], dtype=dtype)], axis=1), name='beta2_ur')

    # D/P equations
    # initial guess: market values of parameters
    test_b0 = tf.Variable(tf.zeros([N, 1], dtype=dtype), name='b0')
    test_b1_dp = tf.Variable(tf.concat([tf.ones([N, 1], dtype=dtype),
                                                                            tf.zeros([N, npc-1], dtype=dtype)], axis=1), name='b1_dp')


    # combine into a single list
    variables_F_only = [c, rho_r, rho_dp] # for state space only estimation
    variables_test_assets_only = [test_beta2_ur, test_b0, test_b1_dp]
    #variables = variables_F_only + variables_test_assets_only # for joint estimation



    # construct PCs. Note: PCs are extracted in the training sample (see construct_pcs() -- it respects oos_start_date)
    # Timing: Month t -> returns in month t, D/P as of beginning of the month
    R, DP, Q, D = construct_pcs(rx, dp, rme, dpm)

    # construct MA and build a state vector
    F, test_asset_ret, test_asset_dps, by, dates = prepare_state_space_vars(
                    R, DP, test_asset_ret, test_asset_dps, by, L)
        # Timing: t -> F_{t}=(r_{t}, dp_{t}): returns over L months ending at the end of month t; end-of-month DP

    ## define variables
    # constants
    T, K = F.shape # number of time periods -- reduce due to MA construction above
                                                    # number of factors (npc returns + npc D/Ps)
    assert(K == 2*npc)

    # weights of test assets in PCs
    W = np.concatenate((np.ones([1, N])/N,
                                            np.kron(np.array([[1], [-1]]), Q[:, :npc-1]).T), axis=0)


    # Consistency check: reconstruct PCs
    pcs_ret = test_asset_ret @ W.T
    pcs_dp = test_asset_dps @ W.T

    assert not market_is_avg_portfolio or (F.iloc[:, 0] - pcs_ret.iloc[:,0]).abs().mean() < 1e-14  # only true when mkt is defined as mean of all L&S portfolios
    assert npc < 2 or (F.iloc[:, 1] - pcs_ret.iloc[:,1]).abs().mean() < 1e-14 
    assert npc < 3 or (F.iloc[:, 2] - pcs_ret.iloc[:,2]).abs().mean() < 1e-14 

    assert not market_is_avg_portfolio or np.mean(np.abs(F.iloc[:, npc+0].values - pcs_dp.iloc[:,0].values)) < 1e-15 
    assert npc < 2 or np.mean(np.abs(F.iloc[:, npc+1].values - pcs_dp.iloc[:,1].values)) < 1e-15 
    assert npc < 3 or np.mean(np.abs(F.iloc[:, npc+2].values - pcs_dp.iloc[:,2].values)) < 1e-15 


        
    #| Load recession indicators for conditional plots
    import pandas_datareader.data as web  # module for reading datasets directly from the web
    rec = web.DataReader('USREC', 'fred', start=1972)
    # rec_6m = rec.rolling(7, center=True).sum() > 0 # mark 3 months before/after a recession as a recession period
    dd_rec_y = rec.loc[dates].reset_index()
    dd_rec_r = rec.loc[dates[L:]].reset_index()
    trough_idx_y = dd_rec_y.query('USREC == 1').index
    trough_idx_r = dd_rec_r.query('USREC == 1').index
    peak_idx_y = dd_rec_y.query('USREC == 0').index
    peak_idx_r = dd_rec_r.query('USREC == 0').index


    #| ## Define model variables and data

    #| Combine all data in a list of TensorFlow constant arrays

    ## Merged data
    data_F = tf.constant(F.values, dtype=dtype)
    data = [tf.constant(F.values, dtype=dtype),
                    tf.constant(test_asset_ret.values, dtype=dtype),
                    tf.constant(test_asset_dps.values, dtype=dtype),
                    tf.constant(by.values, dtype=dtype),
                    tf.constant(rec.loc[dates].values.astype(dtype), dtype=dtype)]

    if output_intermediate_data_results:
        sv.save_data(data, 'data', path=respath)  
        sv.save_data(dates, 'dates', path=respath)  

    return F, test_asset_ret, test_asset_dps, by, dates, data, data_F, variables_test_assets_only, variables_F_only, W, D, trough_idx_y, trough_idx_r, peak_idx_y, peak_idx_r


F, test_asset_ret, test_asset_dps, by, dates, data, data_F, variables_test_assets_only, variables_F_only, W, D, trough_idx_y, trough_idx_r, peak_idx_y, peak_idx_r = init_vars_and_data(*input_data)  

oos_start_idx = F[:oos_start_date].shape[0] # time index where OOS data starts
T = data_F.shape[0]


# Summary tables
if output_intermediate_data_results:
    #summary_table(dp, transpose=True)
    summary_table(F.iloc[:, :npc])
    summary_table(F.iloc[:, npc:])
    
    F.to_csv(tblpath + 'F.csv')
   
    
#%% FIGURE 1: Time-series of factor yields.
pf.timeseries_plot(F.index.values, F.iloc[:, npc:], se=None, filename=figpath+'Figure1',
                                        title='Factor yields', legends=[factor_titles[n] for n in factor_names])


#%% FIGURE 2: Time-series of factor returns.
pf.timeseries_plot(F.index.values, F.iloc[:, :npc], se=None, filename=figpath+'Figure2',
                                        title='Factor returns', legends=[factor_titles[n] for n in factor_names])


#%% TABLE A.6: Duration spanning test
if 'durL' in labels:
  F_dur = F.join(pd.DataFrame(test_asset_ret['durL'] - test_asset_ret['durS'], columns=['dur']))
  dur_reg = sm.OLS(F_dur['dur'], sm.add_constant(F_dur[F.columns])).fit()
  print(dur_reg.summary())
  
  with open(tblpath + 'TableA6.tex', 'w') as f:
      from stargazer.stargazer import Stargazer
      f.writelines(Stargazer([dur_reg]).render_latex())
       
    

#%%
#| ## Estimation
# wrapper for scipy optimizer to work with Tensorflow
import external_optimizer_sk as o

fmt_npc = '{:^' + str(3*npc + 2) + '}'
fmt_mae_p = '{:^' + str(3*int(p_max_t/p_reporting_freq) + 2) + '}'


def estimate_state_space_model(variables_F_only, data_F, npc, oos_start_idx):
    #| ### Estimate the state space model (ignore all test assets)

    # state space optimization callback for logging
    def callback_state_space(j, outputs, variables):
        obj, m, u, stats = outputs
        R2u = stats['R2u']*100.0
    
        # print header
        if j % 30 == 0:
            print(('{:6}:  {:^12} '+fmt_npc+fmt_npc).format(
                    'step', 'obj', 'R2r_pcs', 'R2dp_pcs'))
    
        # print progress
        print(('{:6d}: {:12.9f} '+fmt_npc+fmt_npc).format(
                j, obj, vec2str(R2u[:npc]), vec2str(R2u[npc:])))
    
    # initialize the Scipy optimizer
    optimizer = o.ScipyOptimizerInterface(estimate_state_space, variables_F_only,
                                                                                method='BFGS', #'L-BFGS-B',
                                                                                options={'maxiter': 100000, 'disp': True,
                                                                                                'ftol': 1e-16, 'gtol': 1e-9})
    # estimate the state space model
    res = optimizer.minimize(extra_params=[data_F, npc, oos_start_idx],
                                                    loss_callback=callback_state_space)
    
    # store results
    variables_F_only = optimizer._vars
        
    return variables_F_only


if estimate_state_space_model:
    variables_F_only = estimate_state_space_model(variables_F_only, data_F, npc, oos_start_idx)
    timeit()

# use the state space estimates as initial guess in the full estimation
variables = variables_F_only + variables_test_assets_only

estimate_state_space_model = False
    
    
#%%
def estimate_main_model(dates, variables, data, W, npc, oos_start_idx):

    #| ### Estimate the joint model (main estimation loop)

    # joint estimation callback for logging
    def callback_full_model(j, outputs, variables):
        #obj, mc_pen, m, e, stats, EY = outputs
        max_maturity = bonds_max_maturity
        mkt_idx = 0

        if j % 100 == 0:
            obj, m, stats, series, _ = compute_strips_prices(variables, W, data, npc, N, oos_start_idx)

            # print header
            if j % 3000 == 0:
                pf.timeseries_plot(dates.values, tf.stack([series['EY'][i-1, :, mkt_idx] for i in (1, 2, 5, 7)]).numpy().T,
                                                        title='Equity yields (mkt)', filename=figpath+'EY_mkt_ts_{}pcs'.format(npc))
                print(('{:6}:  {:^12}{:^7}{:^7}{:^7}'+fmt_npc+fmt_npc+fmt_mae_p+fmt_mae_p).format(
                        'step', 'obj', 'MCpen', 'R2r', 'R2dp', 'R2r_pcs', 'R2dp_pcs', 'Ph', 'Ph_mkt'))

            # print progress
            R2u = stats['R2u']*100.0
            print(('{:6d}: {:12.9f} {:.0e}{:7.2f}{:7.2f} '+fmt_npc+fmt_npc+fmt_mae_p+fmt_mae_p).format(
                    j, obj, stats['pen'], 100.0*stats['R2r'], 100.0*stats['R2dp'],
                    vec2str(R2u[:npc]), vec2str(R2u[npc:]),
                    vec2str(stats['mae_p']*100.0), vec2str(stats['mae_pm']*100.0)))

    # initialize the Scipy optimizer
    optimizer = o.ScipyOptimizerInterface(estimate_full_model, variables,
                                            method='BFGS', # L-BFGS-B, CG, TNC, BFGS
                                            options={'maxiter': 10000, 'disp': True,
                                                            'ftol': 1e-16, 'gtol': 1e-9})
    # estimate the joint model
    extra_params = [W, data, npc, N, oos_start_idx]#, S1] # S1 = tfp.math.pinv(S)
    res = optimizer.minimize(extra_params=extra_params,
                                                    loss_callback=callback_full_model)

    # store results
    variables = optimizer._vars

    # report and plot final estimates
    callback_full_model(0, [], variables)

    return optimizer, extra_params, variables



optimizer, optimizer_extra_params, variables = estimate_main_model(dates, variables, data, W, npc, oos_start_idx)

# unpack results
c, rho_r, rho_dp, test_beta2_ur, test_b0, test_b1_dp = variables

# save the estimates
if output_intermediate_data_results:
    sv.save_data(variables, 'variables', path=respath)  

# compute all variables of interest for a fully estimated model
obj, m, stats, series, _ = compute_strips_prices(variables, W, data, npc, N, oos_start_idx)
if output_intermediate_data_results:
    sv.save_data(obj, 'obj', path=respath)  
    sv.save_data(stats, 'stats', path=respath)  
    sv.save_data(series, 'series', path=respath)  

timeit()


#%% load BMSY data
import pandas as pd
# load BMSY yields (end-of-month yields)
BMSY = pd.read_csv(datapath + 'bmsy_div_yields.csv', parse_dates=True, index_col=[0])

# sample start date
idx_t0 = 0
asset_idx = 0
dd = dates
idx_t0_BMSY = (dd >= BMSY.index[0]).tolist().index(True)



#%% TABLE 6: Robustness: (a) Main specification: Full sample
# compute gradients
bmsy_mat = [1, 2, 5, 7]
rmse = []
for i, mat_idx in enumerate(bmsy_mat):
    # compute RMSE
    fey = BMSY.iloc[:, [i]].join(pd.DataFrame(series['FEY'][mat_idx-1, idx_t0_BMSY:, asset_idx], index=dd[idx_t0_BMSY:]))
    rmse += [np.sqrt(((fey.iloc[:, 0] - fey.iloc[:, 1])**2).mean())]
    
writefilestr(tblpath + 'Table6_rmse_{}pcs.csv'.format(npc), '&'.join(['{:.3f}'.format(x) for x in rmse]))
writefilestr(tblpath + 'Table6_rmse_ave_{}pcs.csv'.format(npc), '{:.3f}'.format(np.array(rmse).mean()))
print('RMSE: ', rmse)
print('Average EY RMSE: {:.3f}'.format(np.array(rmse).mean()))



#%% FIGURE 4: Dynamics of model-implied yields in the Bansal et al. (2017) sample.
v = 'EY'
ls_offset = 0
mat = 'ts{}-{}'.format(min(mat2plot), max(mat2plot))
legends = [str(m)+'Y' for m in mat2plot]
mat_idx = np.array(mat2plot) - 1    
pf.timeseries_plot(dd[idx_t0_BMSY:].values, extract_series(series[v], max_maturity, idx_t0_BMSY, asset_idx, ls_offset=ls_offset).numpy().T[:, mat_idx], 
                   filename=figpath+'Figure4', title=var_titles[v], legends=legends)


#%% FIGURE 7: Dynamics of model-implied forward equity yields for the aggregate market for different maturities.
v = 'FEY'
ls_offset = 0
mat = 'ts{}-{}'.format(min(mat2plot), max(mat2plot))
legends = [str(m)+'Y' for m in mat2plot]
mat_idx = np.array(mat2plot) - 1    
pf.timeseries_plot(dd.values, extract_series(series[v], max_maturity, 0, asset_idx, ls_offset=ls_offset).numpy().T[:, mat_idx], 
                   filename=figpath+'Figure7', title=var_titles[v], legends=legends)


#%% a function to compute GMM standard errors of model parameters
def compute_parameter_cov_matrix(d, instruments_shocks, m, idx_t0=0):
    """
    Computes GMM standard errors for all model parameters.

    Parameters
    ----------
    d : 2D Tensor
        A Jacobian matrix of first derivatives of all moments w.r.t. model's paramaters
        
    instruments_shocks : tuple ((instruments), (shocks))
        
    m : Tensor
        Evaluated moments (averages). Used for verification. 
        
    Returns
    -------
    TYPE
        Tensor. Cov matrix of all model parameters (K x K)

    """
    
 # compute moment functions at each point in time
    print('Computing moment functions...')
    timeit(reset=True)
    #@tf.function
    def batch_kron(x, y, t, td=0):
        return x[t, :, None] @ y[None, t+td, :]

    Tis = instruments_shocks[0][0].shape[0]

 
    M = tf.stack([tf.concat([tf.reshape(batch_kron(Z, e, t), [-1]) for Z, e in zip(*instruments_shocks)], axis=0) 
                                for t in range(Tis)])
    # verify momement conditions match averages of moments realizations
    assert np.abs((tf.reduce_sum(M, axis=0)/Tis - m).numpy()).sum() < 1e-12
    timeit()
    
    # compute spectral moment variance-covariance matrix
    print('Computing the spectral moment variance-covariance matrix...')
    timeit(reset=True)
    # @tf.function
    def gmmS(M, L=1):
        nm = M.shape[1]
        S = tf.zeros((nm, nm), dtype=dtype)
        for j in range(1-L,L): # HH errors
            for t in range(-min(j,0), Tis-max(j,0)):
                w = (L - abs(j))/L if newey_west else 1.0
                S += batch_kron(M, M, t, j)*w
        return S / Tis
    S = gmmS(M, 2*L if newey_west else L) 
    # S = gmmS(M, int(1.5*L)) 
    # S = gmmS(M, 2*L, newey_west=True) 
    #S1 = tfp.math.pinv(S) # weighting matrix for efficient GMM
    timeit()
    
    
    # compute the variance-covariance matrix of model's parameters and their standard errors
    d_tr = tf.transpose(d)
    dd1 = tf.linalg.inv(d_tr @ d) # (A*d)^{-1} = (d'd)^{-1}, for W = I
    b_cov =  dd1 @ d_tr @ S @ d @ tf.transpose(dd1) / Tis
    
    return b_cov



def compute_mean_moments_cov_matrix(series, variables_mm_reg, extra_params, asset_idx=0, idx_t0=0, ls_offset=0, parallel_iterations=64, experimental_use_pfor=False):
    """
    Initialize a new optimizer instance to compute mean moments. 
    No optimization is done here -- the initial values must be set to correct solutions!
    
    This function creates mean moments for all variables in variables_mm_names,
    that is, it adds new moment conditions and variables (intercepts).
    
    The function returns a new instance of an optimizer, an associated parameter cov matrix,
    as well as new series dictionary which contains parameters of intercept (intercepts).
    These parameters are names as the original variables in variables_mm_names with a prefix 'm'.
    """

    data = extra_params[1]
    T = data[0].shape[0] - L - idx_t0
    
    # create a list of mean moment variables
    variables_mm = []

    # Provide correct solutions to all moments here!
    for y_vname, x_vnames in variables_mm_reg:
        tf_var_name = mm_var_name (y_vname, x_vnames, asset_idx, ls_offset, labels)
        
        if x_vnames and isinstance(x_vnames, (list, tuple)): # regression
            y = series[y_vname][:bonds_max_maturity, series[y_vname].shape[1]-T:, asset_idx] - \
                    series[y_vname][:bonds_max_maturity, series[y_vname].shape[1]-T:, asset_idx+ls_offset]*(ls_offset>0)
            # for X variables use the matching asset_index unless a single time-series is provided
            X = tf.concat([series[v][None, series[v].shape[0]-T:, asset_idx*(asset_idx < series[v].shape[1])] if v else tf.ones((1, T), dtype=dtype) # timing: r_t on pd_t;   ey_t on pd_t
                                         for v in x_vnames], axis=0)
            variables_mm += [tf.Variable(tf.linalg.lstsq(tf.transpose(X), tf.transpose(y)), dtype=dtype, name=tf_var_name)]
        
        else: # means
            variables_mm += [tf.Variable(tf.reduce_mean(series[y_vname][:bonds_max_maturity, series[y_vname].shape[1]-T:, asset_idx] - \
                            series[y_vname][:bonds_max_maturity, series[y_vname].shape[1]-T:, asset_idx+ls_offset]*(ls_offset>0), axis=1)[None, :], dtype=dtype, name=tf_var_name)]
            
        
    
    # extend the set of moment variables
    variables_ext = variables + variables_mm

    # initialize the Scipy optimizer: use to compute moments jacobian only. No reoptimization -- make sure values provided for new moments are correct!
    optimizer_mm = o.ScipyOptimizerInterface(compute_strips_prices_mean_moments, variables_ext)
    
    # compute moment gradients
    print('Computing the gradient of the moment conditions wrt parameters...')
    d, outputs = optimizer_mm.moments_jacobian(extra_params + [variables_mm_reg, asset_idx, idx_t0, ls_offset, labels])

    gmm_obj, m, stats, series, error_moments = outputs
    
    # compute model parameters' SEs
    # extract estimates and data
    u, e_r, e_dp, Z_u, Z_dp, (e_m_list, z_m_list) = error_moments
    u_r = tf.concat([u, e_r*test_assets_gmm_weight], axis=1) 

    instruments_shocks = ((Z_u, Z_dp, u[:, :npc]) + tuple(z_m_list),  # implicit use of an indicator column [0, 0, ..., 0, 1, 1, ..., 1]
                                                (u_r, e_dp*test_assets_gmm_weight, e_r*test_assets_gmm_weight) + tuple(e_m_list))

    
    b_cov_mm = compute_parameter_cov_matrix(d, instruments_shocks, m)


    return optimizer_mm, b_cov_mm, series


#%%
#| ### Compute standard errors of model's parameter estimates
#| GMM stadard errors. First, compute a gradient of all moment conditions wrt parameters. Then stack all moment condition realizations (at every $t$) to compute the spectral variance-covariance matrix of moment conditions with 12 lags (HH approach). Lastly, combine these two estimates using standard GMM formulas to get standard errors of estimated model parameters.

# compute the gradient of the moment conditions wrt parameters
print('Computing the gradient of the moment conditions wrt parameters...')
d, outputs = optimizer.moments_jacobian(optimizer_extra_params)
sv.save_data(outputs, 'outputs', path=respath)

# extract estimates and data
(gmm_obj, m, mkt_clearing_penalty, c, rho, Sigma, lambda0, Lambda, b0, b1, beta0, beta1, beta2,
    F0, F1, R, DP, F0fs, F1fs, Rfs, DPfs, u, e_r, e_dp, Z_u, Z_dp) = outputs

# compute moment functions at each point in time
# u, e_r, e_dp, Z_u, Z_dp, e_m_list = error_moments
u_r = tf.concat([u, e_r*test_assets_gmm_weight], axis=1) 
instruments_shocks = ((Z_u, Z_dp, u[:, :npc]), 
                      (u_r, e_dp*test_assets_gmm_weight, e_r*test_assets_gmm_weight))

b_cov = compute_parameter_cov_matrix(d, instruments_shocks, m)
se = optimizer.unpack( tf.math.sqrt(tf.linalg.diag_part(b_cov)) )

# diagnostic: t-stats of PC ret loadings on D/Ps
b = variables
tstats_r_dp = b[1] / se[1]
print('t-stats of PC ret loadings on D/Ps:')
print(tstats_r_dp.numpy())


#%%
## output all equity yields [END OF MONTH]
# MKT
ey = tf.transpose(series['EY'], perm=(1, 2, 0))[:, :, :]
ey_wide = tf.reshape(ey[:, 0, :], shape=(T, -1)).numpy()
colnames = ['EY{}'.format(m) for m in range(1,1+p_max_t)]
ey_mkt = pd.DataFrame(ey_wide, index=dd, columns=colnames) 
ey_mkt.index.name = 'date'
ey_mkt.to_csv(tblpath + 'EY_mkt.csv', date_format='%Y/%m')


# C-S
ey_wide = tf.reshape(ey[:, npc:, :bonds_max_maturity], shape=(T, -1))
colnames = list(np.array([['EY{}_{}'.format(m, l) for m in range(1,1+bonds_max_maturity)] for l in labels[npc:]]).flatten())
ey_cs = pd.DataFrame(ey_wide.numpy(), index=dd, columns=colnames) 
ey_cs.index.name = 'date'
ey_cs.to_csv(tblpath + 'EY_cs.csv', date_format='%Y/%m')
    

## output all forward equity yields
# MKT
fey = tf.transpose(series['FEY'], perm=(1, 2, 0))[:, :, :]
fey_wide = tf.reshape(fey[:, 0, :], shape=(T, -1)).numpy()
colnames = ['FEY{}'.format(m) for m in range(1,1+bonds_max_maturity)]
fey_mkt = pd.DataFrame(fey_wide, index=dd, columns=colnames) # NEED TO OFFSET DATES???!!!
fey_mkt.index.name = 'date'
fey_mkt.to_csv(tblpath + 'FEY_mkt.csv', date_format='%Y/%m')


# C-S
fey_wide = tf.reshape(fey[:, npc:, :bonds_max_maturity], shape=(T, -1))
colnames = list(np.array([['FEY{}_{}'.format(m, l) for m in range(1,1+bonds_max_maturity)] for l in labels[npc:]]).flatten())
fey_cs = pd.DataFrame(fey_wide.numpy(), index=dd, columns=colnames) 
fey_cs.index.name = 'date'
fey_cs.to_csv(tblpath + 'FEY_cs.csv', date_format='%Y/%m')
    


#%% TABLE A.3: Anomaly portfolios mean excess returns, %, annualized
import json
with open('anom_names.txt') as json_file:
    anom_names = json.load(json_file)     


tL = test_asset_ret.mean().iloc[:int(N/2)]
tL.index = AnomalyNames
tS = test_asset_ret.mean().iloc[int(N/2):]
tS.index = AnomalyNames
anom_table = pd.concat((tS, tL), axis=1)
anom_table.columns = ['Short', 'Long']
anom_table['LS'] = anom_table.Long - anom_table.Short
anom_table.index = [anom_names.get(a) for a in anom_table.index]
anom_table.sort_index(inplace=True)

if len(anom_table) % 2:
  anom_table = anom_table.append(pd.DataFrame([[np.nan]*3], index=[''], columns=anom_table.columns))
mid_split = int(anom_table.shape[0]/2)
latextable(tblpath + 'TableA3_part1.tex', 100*anom_table.iloc[:mid_split])
latextable(tblpath + 'TableA3_part2.tex', 100*anom_table.iloc[mid_split:], na_rep='')

#%%  TABLE 1: Percentage of variance explained by anomaly PCs
Ds = 100*(D / D.sum())[:10].reshape(1, 10)
tD = pd.DataFrame(np.concatenate((Ds, np.cumsum(Ds).reshape(1, 10)), axis=0), index=['\% var. explained', 'Cumulative'])
latextable(tblpath + 'Table1_part1.tex', tD)

R_, DP_, Q_, D_ = construct_pcs(test_asset_ret, test_asset_dps, rme, dpm)
Ds_ = 100*(D_ / D_.sum())[:10].reshape(1, 10)
tD_ = pd.DataFrame(np.concatenate((Ds_, np.cumsum(Ds_).reshape(1, 10)), axis=0), index=['\% var. explained', 'Cumulative'])
latextable(tblpath + 'Table1_part2.tex', tD_)


#%% output R2 to latex
latextable(tblpath + 'F_R2.tex', 
                     pd.DataFrame([stats['R2u'][:npc].numpy()*100, stats['R2u'][npc:].numpy()*100], 
                                                index=['R', 'D/P'], columns=labels[:npc]).applymap(lambda x: ' {:.2f} '.format(x)))

# PAGE 22: Average R2 for the regression of portfolio returns in Equation (10):
writefilestr(tblpath + 'R2r.txt', '{:.1f}'.format(stats['R2r'].numpy()*100))
# PAGE 22: Average R2 for the regression of dividend yields onto the factor dividend yields in Equation (9):
writefilestr(tblpath + 'R2dp.txt', '{:.1f}'.format(stats['R2dp'].numpy()*100))
    
#%% TABLE 2: Estimates of the dynamics of the factors Fₜ
K = 2*npc    
#rho_est = pd.DataFrame(np.concatenate((b[0].numpy(), np.concatenate((b[1].numpy(), b[2].numpy()), axis=0)), axis=1)) 
rho_est = pd.DataFrame(np.concatenate((b[0].numpy(), build_rho_from_parts(b[1], b[2], K, npc, diag_rho_r)[:, npc:].numpy()), axis=1)) 
rho_est_R2 = pd.concat((rho_est, pd.DataFrame(stats['R2u'].numpy()*100, columns=['$R^2$'])), axis=1)
rho_std = pd.DataFrame(np.concatenate((se[0].numpy(), build_rho_from_parts(se[1], se[2], K, npc, diag_rho_r)[:, npc:].numpy()), axis=1)) 
rho_std = rho_std.applymap(lambda x: ' \\small{{({:.2g})}} '.format(x))
#rho_tstat = (rho_est / rho_std).applymap(lambda x: ' \\small{{({:.2f})}} '.format(x))
tbl = pd.concat([rho_est_R2.applymap(lambda x: ' {:.2f} '.format(x)), 
                                 rho_std]).sort_index().reset_index(drop=True)
if npc == 4:
    tbl.index = ['$r_{mkt}$', '', '$r_{pc1}$', '', '$r_{pc2}$', '', '$r_{pc3}$', '', '$y_{mkt}$', '', '$y_{pc1}$', '', '$y_{pc2}$', '', '$y_{pc3}$', '']
latextable(tblpath + 'Table2.tex', tbl)
rho_est.to_csv(tblpath + 'rho_est.csv')


#%% TABLE 3: Risk-neutral estimates of the dynamics of the factors Fₜ
Sl = Sigma @ lambda0
SL = Sigma @ Lambda
c_rn = c - Sl
rho_rn = rho - SL
#tbl = pd.DataFrame(np.concatenate((c_rn.numpy(), rho_rn[:, npc:]), axis=1)).applymap(lambda x: ' {:.2g} '.format(np.round(x+1e-9, 2)))
tbl = pd.DataFrame(np.concatenate((c_rn[:npc].numpy(), c_rn[npc:].numpy(), rho_rn[npc:, npc:]), axis=1)).applymap(lambda x: ' {:.2f} '.format(np.round(x+1e-9, 2)))
if npc == 4:
    tbl.index = ['MKT', 'PC1', 'PC2', 'PC3']
latextable(tblpath + 'Table3.tex', tbl)


#%% Cross-section of risk premia for the 51 anomalies: cross-sectional R2
X = sm.add_constant(F.iloc[:, :npc])
betas = scipy.linalg.solve(X.T @ X, X.T @ test_asset_ret).T[:, 1:]
R2cs = sm.OLS(np.mean(test_asset_ret, axis=0), sm.add_constant(betas)).fit().rsquared
writefilestr(tblpath + 'R2cs.txt', '{:.1f}'.format(R2cs*100))

#%% Output sample start/end dates
writefilestr(tblpath + 'd0.txt', rx.index[0].strftime('%B %Y'))
writefilestr(tblpath + 'dT.txt', rx.index[-1].strftime('%B %Y'))


idx_t0 = 0
asset_idx = 0


#%%
#| # Results

#%% PLOT functions
#| ## Functions
#| ### Helper functions

#%%
#| #### Wrappers for plot functions to compute and standard errors when requested, or plot estimates only
#| Plot functions
def genfn(asset_idx, variable, mat='mat', t0=0, ls_offset=0): # generate a file name for a plot
    if isinstance(mat, (int, float)):
        mat = 'ts' + str(mat)
    fname = figpath + '{}_{}_{}'.format(variable, mat, labels[asset_idx])
    if ls_offset:
        fname += '_ls'
    if isinstance(t0, int):
        if t0 > 0:
            fname += '_' + dd[t0].strftime('%Y')
    else:
        fname += '_cond'
    return fname

def maturity_plot(series, variable, asset_idx, max_maturity=bonds_max_maturity, t0=0, ls_offset=0, # t0 is either a scalar (t0:end) or an array of indices to extract
                                    title='', compute_se=False, extra_params=[], show=True, newfig=True, filename='', **plot_options): # standard errors are not adjusted for sample variation! Use with caution!
    if compute_se:
        se, b = compute_by_maturity_se(b_cov, optimizer, extra_params, variable, asset_idx,
                                                                     max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
    else:
        v = extract_series(series[variable], max_maturity, t0, asset_idx, ls_offset=ls_offset)
        se, b = None, tf.reduce_mean(v, axis=1)

    if filename == '':
        filename = genfn(asset_idx, variable, t0=t0, ls_offset=ls_offset) if show else ''
    pf.maturity_plot(b, se, filename=filename, title=title, show=show, newfig=newfig, **plot_options)
    
    return b, se


def maturity_plot_se(series, b_cov, optimizer, variable, variable_m, asset_idx, max_maturity=bonds_max_maturity, idx_t0=0, ls_offset=0, # t0 is either a scalar (t0:end) or an array of indices to extract
                                    title='', compute_se=False, extra_params=[], show=True, newfig=True, plot_se1=True, plot_se2=True, filename='', **plot_options):
    if compute_se:
        # model-based standard errors
        se, b = compute_mean_statistic_se(b_cov, optimizer, extra_params, variable_m, asset_idx, idx_t0=idx_t0, ls_offset=ls_offset, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)

        if plot_se2:    
            # HAC standard errors for means
            # se2 = np.concatenate([sm.OLS(series[variable][m, idx_t0:, 0].numpy().T, np.ones((series[variable].shape[1]-idx_t0,1))).fit(cov_type='HAC',cov_kwds={'maxlags':int(1.5*L)}).bse for m in range(bonds_max_maturity)])#tf.math.reduce_std(series[variable][:bonds_max_maturity, 0:12:, 0], axis=1)/np.sqrt(series[variable].shape[1]/12)
            se2 = np.concatenate([sm.OLS(series[variable][m, idx_t0:, asset_idx].numpy().T, np.ones((series[variable].shape[1]-idx_t0,1))).fit(cov_type='HAC',cov_kwds={'kernel':'bartlett', 'maxlags':int(1.5*L)}).bse for m in range(bonds_max_maturity)])#tf.math.reduce_std(series[variable][:bonds_max_maturity, 0:12:, 0], axis=1)/np.sqrt(series[variable].shape[1]/12)
        else:
            se2 = None
            
    else:
        v = series[variable_m]
        se, se2, b = None, None, v

    if filename == '':
        filename = genfn(asset_idx, variable_m, t0=idx_t0, ls_offset=ls_offset) if show else ''
    pf.maturity_plot2(b, se if plot_se1 else None, se2, filename=filename, title=title, show=show, newfig=newfig, **plot_options)
    
    return b, se, se2


def cond_maturity_plot(series, variable, asset_idx, peak_idx_y, trough_idx_y, ls_offset=0, title='', compute_se=False, extra_params=[], **plot_options): # standard errors are not adjusted for sample variation! Use with caution!
    maturity_plot(series, variable, asset_idx, t0=peak_idx_y, ls_offset=ls_offset, title=title, compute_se=compute_se, extra_params=extra_params, show=False)
    maturity_plot(series, variable, asset_idx, t0=trough_idx_y, ls_offset=ls_offset, title=title, compute_se=compute_se, extra_params=extra_params, newfig=False,
                                line_color='#d62728', legends=['Normal times', 'Recessions'])


def timeseries_plot(dates, series, variable, asset_idx, maturity_idx=0, t0=0, ls_offset=0, filename='',
                                        title='', compute_se=False, extra_params=[], show=True, newfig=True, **plot_options):
    if compute_se:
        se, b, cov = compute_timeseries_se(b_cov, optimizer, extra_params, variable, asset_idx,
                                                                     maturity_idx=maturity_idx, t0=t0, ls_offset=ls_offset)
    else:
        b = extract_series_mat(series[variable], maturity_idx, t0, asset_idx, ls_offset=ls_offset)
        se, cov = None, None

    if filename == '':
        filename = genfn(asset_idx, variable, mat=maturity_idx+1, t0=t0, ls_offset=ls_offset) if show else ''
    pf.timeseries_plot(dates, b, se, filename=filename, title=title, show=show, newfig=newfig, **plot_options)
    
    return b, se, cov

#| ### Yields, dividend growth, and returns by-maturity plots

def plot_yields(asset_idx, idx_t0=0, ls_offset=0):
    maturity_plot(series, 'P', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Equity strip rel. prices', compute_se=compute_se, extra_params=optimizer_extra_params)
    maturity_plot(series, 'BY', 0, t0=idx_t0, ls_offset=0, title='Bond yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    maturity_plot(series, 'EY', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Equity yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    maturity_plot(series, 'FEY', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Forward equity yields', compute_se=compute_se, extra_params=optimizer_extra_params)

def plot_dg(asset_idx, idx_t0=0, ls_offset=0):
    maturity_plot(series, 'logEg', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Exp. div. growth', compute_se=compute_se, extra_params=optimizer_extra_params)

def plot_returns(asset_idx, idx_t0=0, ls_offset=0):
    maturity_plot(series, 'ER', asset_idx, t0=idx_t0,ls_offset=ls_offset,  title='Strips risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    maturity_plot(series, 'r', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Realized risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    maturity_plot(series, 'rfwd', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Realized fwd risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    maturity_plot(series, 'logER_htm', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='HTM risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    if idx_t0 == 0: # the following two plots are unconditional SR ONLY when idx_t0=0 (std is always based on full sample)
        maturity_plot(series, 'rstd', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='SR (mean realized ret)', compute_se=compute_se, extra_params=optimizer_extra_params)
        maturity_plot(series, 'ERstd', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='SR', compute_se=compute_se, extra_params=optimizer_extra_params)
        maturity_plot(series, 'std', asset_idx, t0=idx_t0, ls_offset=ls_offset, title='Strip returns std. dev.', compute_se=compute_se, extra_params=optimizer_extra_params)
        maturity_plot(series, 'beta', asset_idx, t0=idx_t0, ls_offset=ls_offset, title=r'Strip market $\beta$', compute_se=compute_se, extra_params=optimizer_extra_params)
        maturity_plot(series, 'alpha', asset_idx, t0=idx_t0, ls_offset=ls_offset, title=r'Strip CAPM $\alpha$', compute_se=compute_se, extra_params=optimizer_extra_params)
        maturity_plot(series, 'alphar', asset_idx, t0=idx_t0, ls_offset=ls_offset, title=r'Strip realized CAPM $\alpha$', compute_se=compute_se, extra_params=optimizer_extra_params)

def recession_plots(asset_idx, compute_se=compute_se, ls_offset=0):
    if asset_idx == 0:
        cond_maturity_plot(series, 'BY', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Bond yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'EY_slopes', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Equity yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'EY', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Equity yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'FEY_slopes', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Forward equity yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'FEY', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Forward equity yields', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'ERstd', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='SR', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'ER', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Strips risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'logER_htm', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='HTM risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'logEg', asset_idx, peak_idx_y, trough_idx_y, ls_offset=ls_offset, title='Exp. div. growth', compute_se=compute_se, extra_params=optimizer_extra_params)

    cond_maturity_plot(series, 'rstd', asset_idx, peak_idx_r, trough_idx_r, ls_offset=ls_offset, title='SR (mean realized ret)', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'r', asset_idx, peak_idx_r, trough_idx_r, ls_offset=ls_offset, title='Realized risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)
    cond_maturity_plot(series, 'rfwd', asset_idx, peak_idx_r, trough_idx_r, ls_offset=ls_offset, title='Realized fwd risk premia', compute_se=compute_se, extra_params=optimizer_extra_params)

#| ### Time-series plots

def plot_EY_decomposition(asset_idx, mat_idx, idx_t0=0, ls_offset=0, filename=''):
    pf.timeseries_plot(dd[idx_t0:].values, extract_series_mat(series['logER_htm'], mat_idx-1, idx_t0, asset_idx, ls_offset=ls_offset), show=False)
    pf.timeseries_plot(dd[idx_t0:].values, extract_series_mat(series['logEg'], mat_idx-1, idx_t0, asset_idx, ls_offset=ls_offset), newfig=False, show=False, recession_bars=False)
    pf.timeseries_plot(dd[idx_t0:].values, extract_series_mat(series['EY'], mat_idx-1, idx_t0, asset_idx, ls_offset=ls_offset), newfig=False, legends=['HTM ER', 'Eg', 'EY'], recession_bars=False,
                                    title='Equity yield decomosition ({}Y)'.format(mat_idx), filename=filename)

def plot_dg0_ts(asset_idx, idx_t0=0, ls_offset=0, filename=''):
    pf.timeseries_plot(dd[idx_t0:].values, extract_series_mat(series['logEg'], 0, idx_t0, asset_idx, ls_offset=ls_offset), show=False)
    pf.timeseries_plot(dd[idx_t0:-L].values, extract_series_mat(series['dgrf'], [], idx_t0, asset_idx, ls_offset=ls_offset), newfig=False, legends=['Eg', 'Realized div. growth'], recession_bars=False,
                                    title='Expected vs. realized div. growth', filename=filename)

def plot_cum_ts(asset_idx, mat_idx, idx_t0=0, ls_offset=0):
    mat = 'tscum'
    legends = [str(m)+'Y' for m in mat_idx]
    mat_idx = np.array(mat_idx) - 1
    pf.timeseries_plot(dd[idx_t0:].values, np.cumsum(extract_series(series['P'], p_max_t, idx_t0, asset_idx, ls_offset=ls_offset).numpy().T, axis=1)[:, mat_idx], filename=genfn(asset_idx, 'EY', mat=mat, t0=idx_t0, ls_offset=ls_offset),
                                        title='Price of cumulative div', legends=legends)

def plot_yields_ts(asset_idx, mat_idx, idx_t0=0, ls_offset=0):
    if not isinstance(mat_idx, int):
        mat = 'ts{}-{}'.format(min(mat_idx), max(mat_idx))
        legends = [str(m)+'Y' for m in mat_idx]
        mat_idx = np.array(mat_idx) - 1    
        pf.timeseries_plot(dd[idx_t0:].values, extract_series(series['EY'], max_maturity, idx_t0, asset_idx, ls_offset=ls_offset).numpy().T[:, mat_idx], filename=genfn(asset_idx, 'EY', mat=mat, t0=idx_t0, ls_offset=ls_offset),
                                            title='Equity strip yields', legends=legends)
        pf.timeseries_plot(dd[idx_t0:].values, extract_series(series['FEY'], max_maturity, idx_t0, asset_idx, ls_offset=ls_offset).numpy().T[:, mat_idx], filename=genfn(asset_idx, 'FEY', mat=mat, t0=idx_t0, ls_offset=ls_offset),
                                            title='Forward strip yields', legends=legends)
        pf.timeseries_plot(dd[idx_t0:].values, extract_series(series['ER'], max_maturity, idx_t0, asset_idx, ls_offset=ls_offset).numpy().T[:, mat_idx], filename=genfn(asset_idx, 'ER', mat=mat, t0=idx_t0, ls_offset=ls_offset),
                                            title='Equity strip risk premia', legends=legends)
        pf.timeseries_plot(dd[idx_t0:].values, extract_series(series['logER_htm'], max_maturity, idx_t0, asset_idx, ls_offset=ls_offset).numpy().T[:, mat_idx], filename=genfn(asset_idx, 'logERhtm', mat=mat, t0=idx_t0, ls_offset=ls_offset),
                                            title='Equity strip HTM risk premia', legends=legends)
        pf.timeseries_plot(dd[idx_t0:].values, extract_series(series['logEg'], max_maturity, idx_t0, asset_idx, ls_offset=ls_offset).numpy().T[:, [0,1,4,6]], filename=genfn(asset_idx, 'logEg', mat='ts1-7', t0=idx_t0, ls_offset=ls_offset),
                                            title='Exp growth rate', legends=['1Y', '2Y', '5Y', '7Y'])
    else:
        timeseries_plot(dd[idx_t0:].values, series, 'EY', asset_idx, mat_idx-1, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Equity yield ({}Y)'.format(mat_idx))
        timeseries_plot(dd[idx_t0:].values, series, 'FEY', asset_idx, mat_idx-1, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Forward equity yield ({}Y)'.format(mat_idx))
        timeseries_plot(dd[idx_t0:].values, series, 'ER', asset_idx, mat_idx-1, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Strip expected excess returns ({}Y)'.format(mat_idx))
        timeseries_plot(dd[idx_t0:].values, series, 'logER_htm', asset_idx, mat_idx-1, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Strip HTM excess returns ({}Y)'.format(mat_idx))
        timeseries_plot(dd[idx_t0+L:].values, series, 'r', asset_idx, mat_idx-1, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Strip realized returns ({}Y)'.format(mat_idx))
        timeseries_plot(dd[idx_t0+L:].values, series, 'r_slope7_1', asset_idx, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Strip slope ret 7-1')
        timeseries_plot(dd[idx_t0:].values, series, 'dur', asset_idx, t0=idx_t0, ls_offset=ls_offset, compute_se=compute_se, extra_params=optimizer_extra_params, title='Macaulay duration')



#%% FIGURE 3: Model-implied forward equity yields vs. forward equity yield data.
compute_se = True

import pandas as pd

def pinv(cov, var_unexplained=1e-3): # invert a cov matrix by keeping only the number of eigenvalues which explain 1-var_unexplained fraction of total variance
    Q, d, _ = np.linalg.svd(cov)
    ix = 1-d.cumsum()/d.sum() >= var_unexplained
    ix[ix.sum()] = True # include the last EV that allows us to reach the threshold
    d[ix] = 1.0 / d[ix]
    d[~ix] = 0.0
    cov_inv = Q @ np.diag(d) @ Q.T
    return cov_inv
    
    
def test_bmsy(b, cov, bmsy):
    # merge
    m = pd.DataFrame(dd[idx_t0_BMSY:].values).reset_index().merge(bmsy.reset_index().reset_index(), left_on=0, right_on='date', how='inner').dropna()
    idx_model = m.index_x
    idx_bmsy = m.index_y
    
    b_ = b.numpy()[idx_model]
    cov_ = cov.numpy()[np.ix_(idx_model, idx_model)]
    bmsy_ = bmsy.values[idx_bmsy]
    
    # Wald test
    chi2_stat = (b_ - bmsy_) @ pinv(cov_) @ (b_ - bmsy_)
    pvalue = scipy.stats.chi2.sf(chi2_stat, b_.shape[0]) # dof=N?
    chi2_crit = scipy.stats.chi2.isf(.05, b_.shape[0]) 
    rmse = np.sqrt(((b_ - bmsy_)**2).mean())
    
    return chi2_stat, pvalue, rmse#, chi2_crit 

    

# compute gradients
bmsy_mat = [1, 2, 5, 7]
rmse = []
bmsy_stats = []
mean_se_list = []
i = 0
for mat_idx in bmsy_mat:
    # plot
    b, se, cov = timeseries_plot(dd[idx_t0_BMSY:].values, series, 'FEY', asset_idx, mat_idx-1, t0=idx_t0_BMSY, compute_se=True, extra_params=optimizer_extra_params,
                                    title='Forward yields ({}Y)'.format(mat_idx), show=False)
    plt.plot(BMSY.iloc[:, i], lw=1)
    
    bmsy_stats += [test_bmsy(b, cov, BMSY.iloc[:, i])]
    print(bmsy_stats)
    
    mean_se = se.numpy().mean()
    mean_se_list += [se.numpy().mean()]  
    print(f'{mat_idx}Y: mean(se)={mean_se:.03f}')
    
    
    l = ['Model-implied', 'Strips-implied']
    if load_BBK_yields and mat_idx <= 2:
        plt.plot(BBK.iloc[:, i], lw=1)
        l += ['BBK fwd yields']
    plt.legend(l, frameon=False, loc='upper center', ncol=len(l))
    plt.savefig(figpath + f'Figure3_mat{mat_idx}.pdf', bbox_inches="tight")
    # plt.show()
    
    # compute RMSE
    fey = BMSY.iloc[:, [i]].join(pd.DataFrame(series['FEY'][mat_idx-1, idx_t0_BMSY:, asset_idx], index=dd[idx_t0_BMSY:]))
    rmse += [np.sqrt(((fey.iloc[:, 0] - fey.iloc[:, 1])**2).mean())]
 
    # increase counter
    i += 1


#%% FIGURE 5: Estimated term structure of forward risk premia.
asset_name = 'mkt'
asset_idx = labels.index(asset_name)
ls_offset = 0


def vtitle(v, def_value=None):
    return var_titles[v] if v in var_titles.keys() else (def_value if def_value is not None else v)
    
y_vname = 'r'


# standard errors for the models with additional mean/regression-based moments to be evaluated and plotted (e.g., for standard errors in factor means plots)
optimizer_mm, b_cov_mm, series_mm = compute_mean_moments_cov_matrix(series, variables_mm_reg, optimizer_extra_params, asset_idx=asset_idx, idx_t0=idx_t0_BMSY, ls_offset=ls_offset) # OOS

optimizer_mm0, b_cov_mm0, series_mm0 = compute_mean_moments_cov_matrix(series, variables_mm_reg, optimizer_extra_params, asset_idx=asset_idx, idx_t0=0, ls_offset=ls_offset) # full sample


# plot
for y_vname in ['rfwd']: # , 'r'
    vname = mm_var_name(y_vname, None, asset_idx, ls_offset, labels)

    # plot full-sample SE vs BMSY se (Hansen Hodrick)
    maturity_plot_se(series_mm, b_cov_mm, optimizer_mm, y_vname, vname, asset_idx, idx_t0=idx_t0_BMSY, ls_offset=ls_offset, title=vtitle(y_vname, vname),
                                     compute_se=True, extra_params=optimizer_extra_params, legends=[asset_name], plot_se1=False, plot_se2=True, show=False)
    maturity_plot_se(series_mm0, b_cov_mm0, optimizer_mm0, y_vname, vname, asset_idx, idx_t0=0, ls_offset=ls_offset, title=vtitle(y_vname, vname), 
                                     compute_se=True, extra_params=optimizer_extra_params, newfig=False, plot_se2=False, line_color='#d62728', line_style='--',
                                     legends=['Post-2004 sample', 'Full sample'], filename=figpath + 'Figure5')



#%% FIGURE 6: Decomposition of the 2-year equity yield in the Bansal et al. (2017) sample.
asset_idx = 0
plot_EY_decomposition(asset_idx, 2, idx_t0=idx_t0_BMSY, filename=figpath+'Figure6')


#%% FIGURE 8: Term structures of equity yields conditional on NBER recessions.
asset_idx = 0
idx_t0 = 0

for v in ['EY', 'EY_slopes']:#, 'FEY', 'FEY_slopes']:
    
    if v == 'EY':
      se_rec, b_rec = compute_mean_statistic_se(b_cov_mm0, optimizer_mm0, optimizer_extra_params, f'reg_mkt_{v}_usrec', asset_idx, idx_t0=idx_t0, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
      se_exp, b_exp = compute_mean_statistic_se(b_cov_mm0, optimizer_mm0, optimizer_extra_params, f'reg_mkt_{v}_usexp', asset_idx, idx_t0=idx_t0, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
      
      pf.maturity_plot(b_exp, se_exp, show=False, newfig=True)
      pf.maturity_plot(b_rec, se_rec, show=True, newfig=False,
                                       line_color='#d62728', legends=['Normal times', 'Recessions'], title=var_titles[v],
                                       filename=figpath+'Figure8')

    # test of higher means in rec
    se_rec_test, b_rec_test = compute_mean_statistic_se(b_cov_mm0, optimizer_mm0, optimizer_extra_params, f'reg_mkt_{v}_c_usrec', asset_idx, idx_t0=idx_t0, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
    tstats = b_rec_test[1:,:] / se_rec_test[1:,:]

    tbl_stats = pd.DataFrame([b_rec_test[1,:].numpy().tolist(), tstats.numpy().tolist()[0]], 
                             index=['Diff.', ''], columns=range(1, bonds_max_maturity+1))[[2, 3, 4, 5, 7, 10, 15]]
    tbl_stats.to_csv(tblpath+f'Figure8_{v}.tex', float_format='%.2f', sep='&', header=False, lineterminator='\\\\\n')
    print(tbl_stats)


#%% FIGURE 9: Decomposition of the 5-year equity yield.
asset_idx = 0
plot_EY_decomposition(asset_idx, 5, filename=figpath+'Figure9')


#%%  FIGURES 10, 11: Slope 7-1 of forward equity yields for small and large stocks, value and growth stocks.
# combined slope plots
for i, variable in enumerate(['size', 'value']):
    timeseries_plot(dd[idx_t0:].values, series, 'FEY_slope7_1', labels.index(variable + 'L'), t0=idx_t0, 
                                    compute_se=True, extra_params=optimizer_extra_params, show=False, line_color="#3F5D7D")
    timeseries_plot(dd[idx_t0:].values, series, 'FEY_slope7_1', labels.index(variable + 'S'), t0=idx_t0, title='Fwd equity yield slope 7-1',
                                    compute_se=True, extra_params=optimizer_extra_params, newfig=False, line_color='#d62728', 
                                    legends=['Long', 'Short'], recession_bars=False,
                                    filename=figpath+f'Figure{10+i}')

    
#%% FIGURE 12 (a): Term structure of risk premia for long-short portfolios: SMB, HML, and MOM risk premia with standard error bounds
# NOTE: Adding cross-sectional moments makes the number of moments too large, leading to precision loss
#       To reproduce standard errors in the paper, redefine variables_mm_reg below and re-initialize compute_mean_statistic_se() and maturity_plot_se()
#       functions. The former uses a global variable variables_mm_reg

anom_to_plot = ['sizeL', 'valueL', 'mom12L', 'growthL', 'profL', 'strevL', 'ivolL', 'ageL', 'durL']
ls_offset = int(N/2)
idx_t0=0

# # 
# variables_mm_reg = ( ('r', None), )   # means

# optimizer_mm1, b_cov_mm1, series_mm1 = compute_mean_moments_cov_matrix(series, variables_mm_reg, optimizer_extra_params, asset_idx=labels.index('sizeL'), idx_t0=idx_t0, ls_offset=ls_offset)
# optimizer_mm2, b_cov_mm2, series_mm2 = compute_mean_moments_cov_matrix(series, variables_mm_reg, optimizer_extra_params, asset_idx=labels.index('valueL'), idx_t0=idx_t0, ls_offset=ls_offset)
# optimizer_mm3, b_cov_mm3, series_mm3 = compute_mean_moments_cov_matrix(series, variables_mm_reg, optimizer_extra_params, asset_idx=labels.index('mom12L'), idx_t0=idx_t0, ls_offset=ls_offset)
    
    
# #%
# y_vname = 'r'
# anom_to_plot = ['sizeL', 'valueL', 'mom12L', 'growthL', 'profL', 'strevL', 'ivolL', 'ageL', 'durL']
# vname = mm_var_name(y_vname, None, labels.index('sizeL'), ls_offset, labels) 
# maturity_plot_se(series_mm1, b_cov_mm1, optimizer_mm1, y_vname, vname, labels.index('sizeL'), idx_t0=idx_t0, ls_offset=ls_offset, 
#                                  compute_se=True, extra_params=optimizer_extra_params, newfig=True, show=False, plot_se2=False, se_band=1.)#, legends=anom_to_plot)

# vname = mm_var_name(y_vname, None, labels.index('valueL'), ls_offset, labels) 
# maturity_plot_se(series_mm2, b_cov_mm2, optimizer_mm2, y_vname, vname, labels.index('valueL'), idx_t0=idx_t0, ls_offset=ls_offset, 
#                                  compute_se=True, extra_params=optimizer_extra_params, newfig=False, show=False, plot_se2=False, se_band=1., legends=anom_to_plot, line_color='#d62728', line_style='--')

# vname = mm_var_name(y_vname, None, labels.index('mom12L'), ls_offset, labels) 
# maturity_plot_se(series_mm3, b_cov_mm3, optimizer_mm3, y_vname, vname, labels.index('mom12L'), idx_t0=idx_t0, ls_offset=ls_offset, title='Realized risk premia',
#                                  compute_se=True, extra_params=optimizer_extra_params, newfig=False, show=True, plot_se2=False, se_band=1., legends=['size', 'value', 'momentum'], line_color="#7D5F3F", line_style=':',
#                                  filename=figpath+'Figure12a')
    


#%% FIGURE 12 (b): Term structure of risk premia for long-short portfolios: Select long-short portfolios
anom_to_plot = ['sizeL', 'valueL', 'mom12L', 'growthL', 'profL', 'strevL', 'ivolL', 'ageL']#, 'durL']

for y_vname in ['r']:#, 'rfwd', 'EY', 'FEY']:
    
    for i, asset_name in enumerate(anom_to_plot):
        asset_idx = labels.index(asset_name)
        
        maturity_plot(series, y_vname, asset_idx, t0=0, ls_offset=ls_offset, title=vtitle(y_vname, vname), compute_se=False, newfig=(i==0), 
                                    show=(i==len(anom_to_plot)-1), legends=[a[:-1] for a in anom_to_plot],
                                    filename=figpath+'Figure12b')


#%% FIGURE 13: Expected and realized dividend growth for the market index and momentum stocks.
# (a) Market
plot_dg0_ts(0, filename=figpath+'Figure13a')

# (b) Momentum
idx_mom = labels.index('mom12L') # Long leg of momentum portfolio
plot_dg0_ts(idx_mom, filename=figpath+'Figure13b')


#%% FIGURE A.18: Comparison of standard errors for forward risk premia by maturity.
asset_name = 'mkt'
asset_idx = labels.index(asset_name)
ls_offset = 0

for y_vname in ['rfwd']:#, 'r', 'FEY', 'EY']: 
    vname = mm_var_name(y_vname, None, asset_idx, ls_offset, labels) 
    maturity_plot_se(series_mm0, b_cov_mm0, optimizer_mm0, y_vname, vname, asset_idx, idx_t0=0, ls_offset=ls_offset, title=vtitle(y_vname, vname), 
                                        compute_se=True, extra_params=optimizer_extra_params, filename=figpath+'FigureA18a', line_color2="#d62728", hatch='/')

    maturity_plot_se(series_mm, b_cov_mm, optimizer_mm, y_vname, vname, asset_idx, idx_t0=idx_t0_BMSY, ls_offset=ls_offset, title=vtitle(y_vname, vname), 
                                        compute_se=True, extra_params=optimizer_extra_params, filename=figpath+'FigureA18b', line_color2="#d62728", hatch='/')



#%% NEW TABLE 4: Actual vs. fitted yields
def autocorr(x, lag=12):
  xm = x - x.mean(axis=0)
  xv = x.var(axis=0)
  return (xm[lag:, :]*xm[:-lag, :]).mean(axis=0) / xv
  
BMSY_FEY = 100*BMSY.join(pd.DataFrame(series['FEY'].numpy()[np.array(bmsy_mat)-1, idx_t0_BMSY:, 0].T, index=dd[idx_t0_BMSY:], columns=[f'fey{m}' for m in bmsy_mat]))
bmsy, fey = BMSY_FEY.iloc[:, :len(bmsy_mat)].values, BMSY_FEY.iloc[:, len(bmsy_mat):].values
bmsy_fey_diff = bmsy - fey
df_bmsy = pd.DataFrame([bmsy.mean(axis=0).tolist() + fey.mean(axis=0).tolist() + bmsy_fey_diff.mean(axis=0).tolist(),
                        bmsy.std(axis=0).tolist() + fey.std(axis=0).tolist() + bmsy_fey_diff.std(axis=0).tolist(),
                        autocorr(bmsy).tolist() + autocorr(fey).tolist() + autocorr(bmsy_fey_diff).tolist()], 
                       index=['Mean', 'S.D.', 'AC'], columns=[f'{m}Y' for m in bmsy_mat]*3)
latextable(tblpath + 'Table4_part1.tex', df_bmsy, float_format='%.2f')
print(df_bmsy, '\n')

# slopes
bmsy_slopes = bmsy[:, 1:] - bmsy[:, :1]
fey_slopes = fey[:, 1:] - fey[:, :1]
bmsy_fey_diff_slopes = bmsy_slopes - fey_slopes
df_bmsy_slopes = pd.DataFrame([[np.nan] + bmsy_slopes.mean(axis=0).tolist() + [np.nan] + fey_slopes.mean(axis=0).tolist() + [np.nan] + bmsy_fey_diff_slopes.mean(axis=0).tolist(),
                               [np.nan] + bmsy_slopes.std(axis=0).tolist() + [np.nan] + fey_slopes.std(axis=0).tolist() + [np.nan] + bmsy_fey_diff_slopes.std(axis=0).tolist(),
                               [np.nan] + autocorr(bmsy_slopes).tolist() + [np.nan] + autocorr(fey_slopes).tolist() + [np.nan] + autocorr(bmsy_fey_diff_slopes).tolist()], 
                              index=['Mean', 'S.D.', 'AC'], columns=[f'{m}Y' for m in bmsy_mat]*3)
latextable(tblpath + 'Table4_part2.tex', df_bmsy_slopes, float_format='%.2f')
print(df_bmsy_slopes)

#%% TABLE 5: Dividend growth predictability: BMSY sample
# sample start date
idx_t0 = idx_t0_BMSY

bmsy_mat = [1, 2, 5, 7]
BY = by.copy()
BY.columns = [f'FBY{m+1:02}' for m in range(BY.shape[1])]
B = BMSY.join(BY[[f'FBY{m:02}' for m in bmsy_mat]])
for m in [1, 2, 5, 7]:
    B[f'bmsy{m}'] = B[f'dy{m}'] + B[f'FBY{m:02}'] # end-of-month



r2_mats = list(range(7))
r2_list = []
r2_h_list, b_h_list = [], []
portfolios = [labels[0]] + labels[npc:]
DPs = np.concatenate([F.iloc[:, npc:2*npc], test_asset_dps], axis=1)
for pname in portfolios:
    i = labels.index(pname) 
    print(pname)

    # across horizons  
    r2_h_dp, r2_h_ey, r2_h_eg, r2_h_bmsy = [], [], [], []
    for h in r2_mats: # predictive horizon (-1)
        # cumulative dg: add annual measurements
        dg_cum = pd.Series( series['dgrf'][idx_t0:, i].numpy()+by['FBY01'][idx_t0:-L].values, index=dd[idx_t0:-L] ).rolling(1+h*L).apply(lambda x: x[::L].sum()).shift(-h*L) # bonds are continuously compounded (logs)
        dg_cum.dropna(inplace=True)
        dp = pd.DataFrame(np.exp(DPs[idx_t0:-L*(h+1),i])-1, columns=['dp'], index=dd[idx_t0:-L*(h+1)])
        ey = pd.DataFrame(np.exp((1+h)*series['EY'][h,idx_t0:-L*(h+1),i].numpy())-1, columns=['ey'], index=dd[idx_t0:-L*(h+1)])
        eg = pd.DataFrame(np.exp((1+h)*series['logEg'][h,idx_t0:-L*(h+1),i].numpy())-1, columns=['eg'], index=dd[idx_t0:-L*(h+1)])
        D = pd.merge(pd.DataFrame(np.exp(dg_cum)-1, columns=['dg']), np.exp((1+h)*B[[f'bmsy{m}' for m in bmsy_mat]])-1, left_index=True, right_index=True) \
                .join(dp).join(ey).join(eg)
        

        # predicting dg using DP
        r2_h_dp += [sm.OLS(D['dg'], sm.add_constant(D['dp'])).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}).rsquared]

        # predicting dg using EY
        r2_h_ey += [sm.OLS(D['dg'], sm.add_constant(D['ey'])).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}).rsquared]

        # predicting dg using EY
        r2_h_eg += [sm.OLS(D['dg'], sm.add_constant(D['eg'])).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}).rsquared]
        
        c = f'bmsy{h+1}'
        if c in B.columns:
            reg_h_ey = sm.OLS(D['dg'], sm.add_constant(D[c])).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}) # add rf?
            r2_h_bmsy += [reg_h_ey.rsquared]
        else:
            r2_h_bmsy += [None]
        
    r2_h_list += [r2_h_dp + r2_h_ey + r2_h_bmsy + r2_h_eg]
    


r2_h_list_df = pd.DataFrame(r2_h_list, columns=['{}{}'.format(v, m+1) for v in ['dp', 'ey', 'bmsy', 'eg'] for m in r2_mats]) #, index=portfolios
print(r2_h_list_df.mean())


x = r2_h_list_df.mean().values.reshape(4, 7)*100
r2_df = pd.DataFrame(x, columns=['{}y'.format(m+1) for m in  r2_mats], index=['D/P', 'EY (model)', 'EY (BMSY)', '$\E[\Delta d - r_f]$'])
latextable(tblpath + 'Table5_bmsy.tex'.format(npc), r2_df, header=False)


pvalues = []
for h in r2_mats:
    r2_est = r2_h_list_df[['{}{}'.format(v, h+1) for v in ['dp', 'ey']]] # r2_list_df

    # t-test for mean equality: Calculate the T-test for the means of two independent samples of scores
    m=r2_est.mean()
    z = (m[1]-m[0])/np.sqrt((r2_est.var() / r2_est.shape[0]).sum())
    pvalue = scipy.stats.t.sf(z, r2_est.shape[0])# one-sided
    pvalues += [pvalue]
    print(h+1, ': ', pvalue)

# latextable(tblpath + 'pvalues_mean_equality_{}pcs.csv'.format(npc), pd.Series(pvalues), header=True, float_format='%.2f')
writefilestr(tblpath + 'Table5_bmsy_p.tex'.format(npc), '&'.join(['{:.2f}'.format(x) for x in pvalues]))



#%% TABLE 5: Dividend growth predictability: full sample
idx_t0 = 0


r2_mats = list(range(7))# + [6, 9]
r2_list = []
r2_h_list, b_h_list = [], []
portfolios = [labels[0]] + labels[npc:]
DPs = np.concatenate([F.iloc[:, npc:2*npc], test_asset_dps], axis=1)
for pname in portfolios:
    i = labels.index(pname) 
    reg_dp, reg_ey = None, None
    # try:
        
    reg_dp = sm.OLS(np.exp(series['dgrf'][idx_t0:, i].numpy() + by['FBY01'][idx_t0:-L])-1, sm.add_constant(np.exp(DPs[idx_t0:-L,i])-1)).fit(cov_type='HAC',cov_kwds={'maxlags':12}) # add rf?
    # reg_dp = sm.OLS(np.exp(series['dgrf'][idx_t0+L:, i].numpy())-1, 
    #                 sm.add_constant(np.concatenate([np.exp(series['dgrf'][idx_t0:-L, i].numpy()[:,None])-1, np.exp(DPs[idx_t0+L:-L,i][:,None])-1], axis=1))).fit(cov_type='HAC',cov_kwds={'maxlags':12}) # add rf?   

    # reg_ey = sm.OLS(np.exp(series['dgrf'][idx_t0:, i].numpy() + 0*by['FBY01'][idx_t0:-L])-1, sm.add_constant(np.exp(series['logEg'][0,idx_t0:-L,i].numpy()-1))).fit(cov_type='HAC',cov_kwds={'maxlags':12}) # add rf?
    reg_ey = sm.OLS(np.exp(series['dgrf'][idx_t0:, i].numpy() + by['FBY01'][idx_t0:-L])-1, sm.add_constant(np.exp(series['EY'][0,idx_t0:-L,i].numpy())-1)).fit(cov_type='HAC',cov_kwds={'maxlags':12}) # add rf?
    # reg_ey = sm.OLS(np.exp(series['dgrf'][idx_t0+L:, i].numpy())-1, 
    #                  sm.add_constant(np.concatenate([np.exp(series['dgrf'][idx_t0:-L, i].numpy()[:,None])-1, np.exp(series['EY'][0,idx_t0+L:-L,i][:,None])-1], axis=1))).fit(cov_type='HAC',cov_kwds={'maxlags':12}) # add rf?
    #print(reg.summary())
    
    # restricted R2
    dg = np.exp(extract_series_mat(series['dgrf'], [], idx_t0, i, ls_offset=0)) - 1
    logEg = np.exp(extract_series_mat(series['logEg'], 0, idx_t0, i, ls_offset=0)[:-L]) - 1
    # R2_restr = 1 - (dg - logEg).var()/dg.var()
    R2_restr = 1 - ((dg - logEg)**2).mean()/(dg**2).mean()
    
    print('R2: {:5.2f}%  --> {:5.2f}% ---> {:5.2f} ({})'.format(reg_dp.rsquared*100, reg_ey.rsquared*100, R2_restr*100, pname))
    r2_list += [[reg_dp.rsquared, reg_ey.rsquared, R2_restr]]

    # across horizons  
    r2_h_dp, r2_h_ey, r2_h_eg, b_h_dp, b_h_ey, b_h_eg = [], [], [], [], [], []
    for h in r2_mats: # predictive horizon (-1)
        # cumulative dg: add annual measurements
        dg_cum = pd.Series( series['dgrf'][idx_t0:, i].numpy()+by['FBY01'][idx_t0:-L].values, index=dd[idx_t0:-L] ).rolling(1+h*L).apply(lambda x: x[::L].sum()).shift(-h*L) # bonds are continuously compounded (logs)
        dg_cum.dropna(inplace=True)
        # if h:
        #   dg_cum = dg_cum[:-h*L]

        # predicting dg using DP
        reg_h_dp = sm.OLS(np.exp(dg_cum)-1, sm.add_constant(np.exp(DPs[idx_t0:-L*(h+1),i])-1)).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}) # add rf?
        # reg_h_dp = sm.OLS(np.exp(dg_cum[L:])-1, sm.add_constant(np.concatenate([
        #   np.exp(series['dgrf'][idx_t0:-L*(h+1), i].numpy()[:,None])-1,
        #   np.exp(DPs[idx_t0+L:-L*(h+1),i])[:,None]-1], axis=1))).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}) # add rf?
        
        r2_h_dp += [reg_h_dp.rsquared]  
        b_h_dp += [reg_h_dp.params.tolist()]   #[reg_h_dp.tvalues[1]]  
        
        # predicting dg using EY
        # reg_h_ey = sm.OLS(np.exp(dg_cum)-1, sm.add_constant(np.exp((1+h)*series['logEg'][h,idx_t0:-L*(h+1),i].numpy())-1)).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}) # add rf?
        
        reg_h_ey = sm.OLS(np.exp(dg_cum)-1, sm.add_constant(np.exp((1+h)*series['EY'][h,idx_t0:-L*(h+1),i].numpy())-1)).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}) # add rf?

        r2_h_ey += [reg_h_ey.rsquared]  
        b_h_ey += [reg_h_ey.params.tolist()]# + [reg_h_ey.bse.tolist()]  #[reg_h_ey.tvalues[1]]  
        
        # predicting dg using Eg
        reg_h_eg = sm.OLS(np.exp(dg_cum)-1, sm.add_constant(np.exp((1+h)*series['logEg'][h,idx_t0:-L*(h+1),i].numpy())-1)).fit(cov_type='HAC',cov_kwds={'maxlags':(h+1)*L}) # add rf?
        r2_h_eg += [reg_h_eg.rsquared]  
        
            
        
    r2_h_list += [r2_h_dp + r2_h_ey + r2_h_eg]
    b_h_list += [b_h_ey] #[b_h_dp + b_h_ey]
    
    
r2_h_list_df = pd.DataFrame(r2_h_list, columns=['{}{}'.format(v, m+1) for v in ['dp', 'ey', 'eg'] for m in r2_mats]) #, index=portfolios
print(r2_h_list_df.mean())


# output the table of R2s by horizon
x = r2_h_list_df.mean().values.reshape(3, 7)*100
r2_df = pd.DataFrame(x, columns=['{}y'.format(m+1) for m in  r2_mats], index=['D/P', 'EY (model)', '$\E[\Delta d - r_f]$'])
latextable(tblpath + 'Table5_fs.tex'.format(npc), r2_df, header=False)


np.array(b_h_list).mean(axis=0)


#% FIGURE A.22: Distributions of dividend growth predictability R2
import seaborn as sns
# sns.displot(data=r2_list, kind='hist', fill=True, height=5, aspect=1.5, bins=10, kde=True)
pvalues = []
for h in r2_mats:
    r2_est = r2_h_list_df[['{}{}'.format(v, h+1) for v in ['dp', 'ey']]] # r2_list_df
    sns.displot(data=r2_est, kind='kde', fill=True, height=3, aspect=1.5, bw_adjust=1.0, legend=False)
    plt.gca().legend((f'{h+1}-year equity yield','D/P'), frameon=False, loc='best')
    plt.xlim([0, 0.55])
    for i in range(2):
        plt.axvline(x=r2_est.mean()[i], color=sns.color_palette()[i], ls='--', lw=1.5)
    plt.savefig(figpath + f'FigureA22_mat{h+1}.pdf', bbox_inches="tight")
    
    # t-test for mean equality: Calculate the T-test for the means of two independent samples of scores
    m=r2_est.mean()
    z = (m[1]-m[0])/np.sqrt((r2_est.var() / r2_est.shape[0]).sum())
    pvalue = scipy.stats.t.sf(z, r2_est.shape[0])# one-sided
    # pvalue = scipy.stats.ttest_ind(r2_est.iloc[:, 1], r2_est.iloc[:, 0]).pvalue/2#, alternative='less')
    pvalues += [pvalue]
    print(h+1, ': ', pvalue)

# latextable(tblpath + 'pvalues_mean_equality_{}pcs.csv'.format(npc), pd.Series(pvalues), header=True, float_format='%.2f')
writefilestr(tblpath + 'Table5_fs_p.tex'.format(npc), '&'.join(['{:.2f}'.format(x) for x in pvalues]))
writefilestr(tblpath + 'FigureA22_pvalues.tex'.format(npc), '&'.join(['{:.2f}'.format(x) for x in pvalues]))



#%% TABLE A.5: Bootstrap vs. model-impled standard errors
latextable(tblpath + 'model_fey_mean_se.tex', 100*pd.DataFrame(np.array(mean_se_list)[None, :], index=['model']), float_format='%.2f')
# repeat the above piece of code with idx_t0=0  to produce model_fey_mean_se_full_sample.tex


#%
mean_se_list = []
for mat_idx in bmsy_mat:
    se, b, cov = compute_timeseries_se(b_cov, optimizer, optimizer_extra_params, 'FEY', asset_idx, maturity_idx=mat_idx-1, t0=0, ls_offset=0)
    
    mean_se = se.numpy().mean()
    mean_se_list += [se.numpy().mean()]  
                                           
latextable(tblpath + 'model_fey_mean_se_full_sample.tex', 100*pd.DataFrame(np.array(mean_se_list)[None, :], index=['model']), float_format='%.2f')
print(100*pd.DataFrame(np.array(mean_se_list)[None, :], index=['model']))



#%%
#| #### Compare realized returns to BMSY

#pd1 = -tf.math.log(tf.math.exp(DP) - 1.0) #dg_rf = R - DP - pd1 + pd #r1 = (tf.math.exp(DP0) - 1.0) / P[0, :, :] * tf.math.exp(dg_rf) - 1.0
if asset_idx > npc:
    mkt_DP = test_asset_dps[labels[asset_idx]]
    r = test_asset_ret[labels[asset_idx]]
else:
    r = F.iloc[:, asset_idx]
    mkt_DP = F.iloc[:, npc+asset_idx]

pd = -np.log(np.exp(mkt_DP) - 1.0)
dg_rf = r - mkt_DP - pd + pd.shift(L)
dg_rf.name = 'dgrf'
#pf.timeseries_plot(dg_rf.index[idx_t0:], dg_rf[idx_t0:], se=None, filename='', title='')
#pf.timeseries_plot(dd[idx_t0+L:], series['dgrf'][idx_t0:-L,asset_idx], newfig=True)

dg_rf.dropna().to_csv(tblpath + 'dgrf.csv')

#-------------------------------

B = BMSY.join(by[['FBY01', 'FBY02']])
B['ey1'] = B.dy1 + B.FBY01 # end-of-month
B['ey2'] = B.dy2 + B.FBY02

#-------------------------------

B2 = B.join(dg_rf)
B2['r1'] = B2.ey1.shift(L) + B2.dgrf # end-of-month EY & end-of-month annual growth in div
B2['r2'] = 2*B2.ey2.shift(L) - B2.ey1 + B2.dgrf
#B2.head()

# # standard errors of strip returns in the data
# [((np.exp(B2[r])-1).mean(), (np.exp(B2[r])-1).std()/np.sqrt(len(B)/12)) for r in ['r1', 'r2']]

#%% FIGURE A.15: One-year returns of the 1- and 2-year dividend strips in the data and in our model.
b, se, cov = timeseries_plot(dd[idx_t0+L:].values, series, 'r', asset_idx, 0, t0=idx_t0, compute_se=True, extra_params=optimizer_extra_params, title='Strip realized returns (1Y)', show=False)
pf.timeseries_plot(B2.index, np.exp(B2.r1)-1, se=None, filename=figpath + 'FigureA15a', title='Div. strips returns', newfig=False, recession_bars=False)
import pandas as pd
pd.DataFrame(series['r'][0, idx_t0:, asset_idx], index=dd[idx_t0+L:].values, columns=['r1_model']).join(np.exp(B2.r1)-1).dropna().corr()
bmsy_stats += [test_bmsy(b, cov, np.exp(B2.r1)-1)]
print(np.array(bmsy_stats))

b, se, cov = timeseries_plot(dd[idx_t0+L:].values, series, 'r', asset_idx, 1, t0=idx_t0, compute_se=True, extra_params=optimizer_extra_params, title='Strip realized returns (2Y)', show=False)
pf.timeseries_plot(B2.index, np.exp(B2.r2)-1, se=None, filename=figpath + 'FigureA15b', title='Div. strips returns', newfig=False, recession_bars=False)
pd.DataFrame(series['r'][1, idx_t0:, asset_idx], index=dd[idx_t0+L:].values, columns=['r2_model']).join(np.exp(B2.r2)-1).dropna().corr()
bmsy_stats += [test_bmsy(b, cov, np.exp(B2.r2)-1)]
print(np.array(bmsy_stats))

#%% TABLE A.4: Model accuracy in fitting yields on observed dividend strips.
bmsy_stats_df_EY = pd.DataFrame(bmsy_stats[:4], index=['{}-year'.format(m) for m in bmsy_mat], columns=['Wald statistic', '$p$-value', 'RMSE'])
latextable(tblpath + 'TableA4.tex'.format(npc), bmsy_stats_df_EY, header=False, float_format='%.2f')



#%% APPENDIX

#%% FIGURE A.14: Average market beta of dividend strips by maturity.
asset_idx = 0
maturity_plot(series, 'beta', asset_idx, t0=0, ls_offset=0, title=r'Strip market $\beta$', compute_se=True, extra_params=optimizer_extra_params,
              filename=figpath+'FigureA14')




#%%
#%% APPENDIX C: Calibration moments

full_sample_stats = False

#| ## BMSY (Bansal's sample)
#| ### The market portfolio (a proxy for S&P 500)
asset_idx = 0#idx_big

# sample start date
idx_t0 = idx_t0_BMSY


# Stats for Table IA.I:
def ci(b, se):
    return b - 1.96*se, b + 1.96*se

# slopes
mat = [7, 15] # maturities to use for slopes
var = ['EY', 'FEY', 'r', 'rfwd'] # variables

if full_sample_stats:
    idx_t0 = 0
    sample = 'fullsample'
    cov = b_cov_mm0
    opt = optimizer_mm0

else:
    idx_t0 = idx_t0_BMSY
    sample = 'origsample'
    cov = b_cov_mm
    opt = optimizer_mm
  

est = {}
for v in var:
    # se, b = compute_by_maturity_se(b_cov, extra_params, f'{v}_slopes', 0, max_maturity=max(mat), t0=0)
    se, b = compute_mean_statistic_se(cov, opt, optimizer_extra_params, f'mean_mkt_{v}_slopes', asset_idx, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
    
    print(f'\n% {sample}: {v}_mat_mkt')
    for m in mat:
        print(f'{sample}_{v}_{m}_{mS} = {b[m-1]:.3f};')
        print(f'{sample}_{v}_{m}_{mS}_se = {se[m-1]:.3f};')
        print('{}_{}_{}_{}_ci = [{:.3f}, {:.3f}];'.format(sample, v, m, mS, *ci(b[m-1], se[m-1])))
    
    # print(f'{sample}_nmonths = {T};')  

    print('\n\n')


# levels

mat = [1, 2, 7, 15] # maturities to use for slopes
var = ['r', 'rfwd', 'EY', 'FEY'] # variables

est = {}
for v in var:
    # se, b = compute_by_maturity_se(b_cov, extra_params, f'{v}', 0, max_maturity=max(mat), t0=idx_t0) 
    
    # print(f'% Sample 2004-onwards: {v}_mat_mkt_2004')
    # for m in mat:
    #   print(f'origsample_{v}_{m} = {b[m-1]:.3f};')
    #   print(f'origsample_{v}_{m}_se = {se[m-1]:.3f};')
    #   print('origsample_{}_{}_ci = [{:.3f}, {:.3f}];'.format(v, m, *ci(b[m-1], se[m-1])))
    
    # print('origsample_nmonths = {};'.format(T-idx_t0))  
     

    # se, b = compute_by_maturity_se(b_cov, extra_params, f'{v}', 0, max_maturity=max(mat), t0=0)
    se, b = compute_mean_statistic_se(cov, opt, optimizer_extra_params, f'mean_mkt_{v}', asset_idx, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
    
    print(f'\n% {sample}: {v}_mat_mkt')
    for m in mat:
        print(f'{sample}_{v}_{m} = {b[m-1]:.3f};')
        print(f'{sample}_{v}_{m}_se = {se[m-1]:.3f};')
        print('{}_{}_{}_ci = [{:.3f}, {:.3f}];'.format(sample, v, m, *ci(b[m-1], se[m-1])))
    
    # print(f'{sample}_nmonths = {T};')  

    print('\n\n')

#% FEY slopes -- model implied SE
np.set_printoptions(precision=4)
for y_var in ['FEY_slopes', 'EY_slopes']:#, 'r_slopes', 'rfwd_slopes']:
        # se, b = compute_mean_statistic_se(b_cov_mm0, optimizer_mm0, extra_params, f'reg_mkt_{y_var}_c_pd', asset_idx)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
        se, b = compute_mean_statistic_se(cov, opt, optimizer_extra_params, f'reg_mkt_{y_var}_c_pd', asset_idx, labels=labels)#, asset_idx, max_maturity=max_maturity, t0=t0, ls_offset=ls_offset)
        se, b = se[1, :], b[1, :] # slopes
        # b, se = compute_tsreg_se_hac(series, asset_idx, y_var, 'pd', t0=0)

        print(f'\n\nRegression of {y_var} (N-{mS}) on pd ({"BMSY" if idx_t0 else "full"} sample):')
        print('\nb:\n', b)
        print('\nse:\n', se)
        print('\nt-stat:\n', b / se)
        


#% regression of slopes on pd: estimates and SE of the slope coefficient --> Sample HAC errors 

def compute_tsreg_se_hac(series, asset_idx, y_var, x_var, t0=0):
    b, se = [], []
    X = sm.add_constant(series[x_var].numpy()[t0+L:, asset_idx].T)
    for m in range(bonds_max_maturity):
        Y = series[y_var].numpy()[m, t0+L:, asset_idx].T
        reg = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'kernel':'uniform', 'maxlags':L})
        b += [reg.params[1]]
        se += [reg.bse[1]]
    
    return np.array(b), np.array(se)
                                

np.set_printoptions(precision=4)
for y_var in ['FEY_slopes', 'EY_slopes']:
    for t0 in [0, idx_t0]:
        b, se = compute_tsreg_se_hac(series, asset_idx, y_var, 'pd', t0=t0)


        print(f'\n\nRegression of {y_var} (N-{mS}) on pd ({"BMSY" if t0 else "full"} sample):')
        print('\nb:\n', b)
        print('\nse:\n', se)
        print('\nt-stat:\n', b / se)

