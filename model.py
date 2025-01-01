""" 
Model estimation functions.


Please cite the following paper when using this code:
    Stefano Giglio, Bryan Kelly, Serhiy Kozak "Equity Term Structures without Dividend Strips Data"
    Journal of Finance, 2024. Forthcoming

====================
Author: Serhiy Kozak
Date: November 2023
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from constants import *


# variables_mm_reg supplies a tuple of pairs ( (y_var, [x_var1, x_var2, ...]), ... ) which implements a regression of y on xs
# Examples: 
#     - ('EY', None) or ('EY', [None]): compute the mean of EY
#     - ('EY', [None, 'pd']): compute intercepts and slopes of a regression of EY on a const and pd

# additional GMM moments to produce standard errors for means and regression estimates for these variables (None for intercept = mean)
variables_mm_reg = ( ('EY', None), ('FEY', None), ('r', None), ('rfwd', None),   # means
                     ('EY_slopes', None), ('FEY_slopes', None), ('r_slopes', None), ('rfwd_slopes', None),   # means
                     ('EY_slopes', [None, 'pd']), ('FEY_slopes', [None, 'pd']),  # regressions (use None for intercept)
#                     ('r', ['usrec']), ('rfwd', ['usrec']), ('r', ['usexp']), ('rfwd', ['usexp']), ('r', [None, 'usrec']), ('rfwd', [None, 'usrec']),
#                     ('r_slopes', ['usrec']), ('rfwd_slopes', ['usrec']), ('r_slopes', ['usexp']), ('rfwd_slopes', ['usexp']), ('r_slopes', [None, 'usrec']), ('rfwd_slopes', [None, 'usrec']) ) # conditional means recession/expansion
                     ('EY', ['usrec']), ('FEY', ['usrec']), ('EY', ['usexp']), ('FEY', ['usexp']), ('EY', [None, 'usrec']), ('FEY', [None, 'usrec']),
                     ('EY_slopes', ['usrec']), ('FEY_slopes', ['usrec']), ('EY_slopes', ['usexp']), ('FEY_slopes', ['usexp']), ('EY_slopes', [None, 'usrec']), ('FEY_slopes', [None, 'usrec']) ) # conditional means recession/expansion


optimizer = None

#| # Model estimation functions
#| ## Data loading and preparation functions
#| ### PCA: Extract factors and predictors

#| Extract PCs of excess returns using the *correlation* matrix (equivalent to extracting PCs of standardized data). Adjust sign in a way that in-sample risk premium of every PC is positive. Use returns' eigenvectors to compute D/P ratio for every PC.

def construct_pcs(rx, dp, rme, dpm):
  # construct PCs of L-S portfolios
  Q, D, _ = np.linalg.svd(np.corrcoef(rx[:oos_start_date].dropna().T)) # use the corr matrix to construct eigenvectors!
                                                              # use only IS returns to construct eigenvectors!
  Q = Q * np.sign(np.mean(rx @ Q, axis=0)).values # adjust signs so that all ER are positive (in sample)

  # construct returns & D/Ps based on the eigenvectors above
  R = pd.concat([rme, rx @ Q[:,:npc-1]], axis=1) # market + all portfolio sorts
  R.columns = ['pc'+str(i) for i in range(npc)]
  DP = pd.concat([dpm, dp @ Q[:,:npc-1]], axis=1)
  DP.columns = ['pc'+str(i) for i in range(npc)]

  return R, DP, Q, D

#| ### Construct state space variables: R, DP [Timing $\forall t: R_{t-1\rightarrow t}, DP_t, \text{BY}_{t}$]

def prepare_state_space_vars(R, DP, test_asset_ret, test_asset_dps, by, L, shift_dp=True):
  ## MA of log portfolio returns
  # 1-year MA of returns
  R = R.rolling(window=L).sum()#.shift(-L)
  DP1 = DP.shift(-1) if shift_dp else DP # get end-of-month DP to coincide with return

  # 1-year MA of test asset returns
  test_asset_ret = test_asset_ret.rolling(window=L).sum()#.shift(-L)

  ## VAR state variables (factors)
  F = R.join(DP1, rsuffix='_dp')

  # drop all NAs
  F = F.dropna()
  dd = F.index

  test_asset_ret = test_asset_ret.loc[dd]
  if shift_dp:
    test_asset_dps = test_asset_dps.shift(-1) # get end-of-month DP to coincide with return
  test_asset_dps = test_asset_dps.loc[dd] 
  by = by.loc[dd] # bond yields are reported as end-of-month

  assert(not F.isna().any(axis=None))
  assert(not test_asset_ret.isna().any(axis=None))
  assert(not test_asset_dps.isna().any(axis=None))
  #assert(not by.isna().any(axis=None))

  return F, test_asset_ret, test_asset_dps, by, dd



#| ## TensorFlow functions for model estimation

#| ### Helper function: compute the $R^2$ using dependent variable $Y_t$ and residuals $\varepsilon_t$:
#| $$R^2=1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_{t=1}^T{\varepsilon_t^2}}{\sum_{t=1}^T{Y_t}}$$
@tf.function
def R2(Y, u, axis=0):
  return tf.constant(1.0, dtype=dtype) - tf.reduce_sum(u**2,axis=axis)/tf.reduce_sum(Y**2,axis=axis)

#| ### Helper function: reconstruct transition matrix $\rho$ from parts $\rho_{r,y}, \rho_{y,y}$ in
#| $$\rho=\left[\begin{array}{cc}
#| 0_{p\times p} & \rho_{r,y}\\
#| 0_{p\times p} & \rho_{y,y}
#| \end{array}\right],$$
#| where $\rho_{r,y}$ could potentially be further restricted to be a $p\times p$ diagonal matrix (we don not currently impose this), based on the evidence in HKS that own DPs are the strongest predictors of PC returns.
@tf.function
def build_rho_from_parts(rho_r, rho_dp, K, npc, diag_rho_r):
  if rm_loads_only_on_dpm:
    rho = tf.concat([tf.zeros([K, npc], dtype=dtype), # R_M loads only on DP_M 
                    tf.concat([tf.concat([tf.concat([rho_r[0:1], tf.zeros((1, npc-1), dtype=dtype)], axis=1), tf.reshape(rho_r[1:], (npc-1,npc))], axis=0),
                                rho_dp], axis=0)], axis=1)
  else:
    rho = tf.concat([tf.zeros([K, npc], dtype=dtype),
                      tf.concat([tf.linalg.diag(rho_r) if diag_rho_r else rho_r,
                                rho_dp], axis=0)], axis=1)

  return rho

#| ### State-space standalone estimation function
#| GMM moment conditions for the state space VAR separately ignoring all test assets (used only as an initial guess and for verification purposes).
#|
#| Take a factor model with k factors (returns + D/P ratios), $F_{t}\equiv\left[r_{t};\,y_{t}\right]$, and $p=\frac{1}{2}k$ assets, which are a part of the state vector $F_{t}$:
#| $$\underbrace{F_{t+1}}_{k\times1}=\underbrace{c}_{k\times1}+\underbrace{\rho}_{k\times k}F_{t}+u_{t+1},$$
#| where $\text{var}_{t}\left(u_{t+1}\right)=\Sigma,$ assuming homoskedasticity.
#|
#| Moment conditions:
#| $$g_T(\theta) = \frac{1}{T}\sum_{t=1}^{T}Z_t^{\prime}u_{t+1},$$
#| where instruments $Z_t$ include only lagged valuation ratios:
#| $$Z_t = \left[1, y_t \right]^\prime,$$
#|
#| and a vector of parameters to estimate contains an intercept and non-zero elements of the transition matrix:
#| $$\theta=[c, \rho_{r,y}, \rho_{y,y}]^{\prime}.$$
#|
#| We use an identity weighting matrix for the GMM objective:
#| $$\hat{\theta} = \arg\min_\theta g_T(\theta)^\prime g_T(\theta).$$
@tf.function
def estimate_state_space(variables, F, npc, oos_start_idx):
  # unpack inputs
  c, rho_r, rho_dp = variables

  # pre-process data
  F0 = F[:oos_start_idx,:][:-L,:] # F_{t}
  F1 = F[:oos_start_idx,:][L:,:]  # F_{t+1}

  T, K = F0.shape

  # transition matrix rho is restricted: zero loading on ret; diagonal for returns on dp
  rho = build_rho_from_parts(rho_r, rho_dp, K, npc, diag_rho_r)

  # state dynamics errors
  u = F1 - tf.transpose(c) - F0 @ tf.transpose(rho) # state dynamics error u_{t+1}

  # instruments for moment conditions in e: use state space D/P ratios
  Z = tf.concat([tf.ones([T,1],dtype=dtype), F0[:, npc:]], axis=1)

  # moment conditions
  m = tf.reshape(tf.transpose(Z) @ u, [-1]) / T # orthogonality of everything with D/Ps of PCs

  # GMM objective: m'm
  gmm_obj = tf.reduce_sum(m**2, axis=0)*T*T

  # other statistics
  R2u = R2(F1, u)
  stats = {'R2u': R2u}

  return gmm_obj, m, u, stats

#| ### GMM moment conditions for the state space VAR and all test assets (main estimation function)
#|
#| This is the main estimation function. It estimates the state space dynamics and all returns jointly, that is, using the following equations:
#|
#|
#| $$\underbrace{F_{t+1}}_{k\times1}=\underbrace{c}_{k\times1}+\underbrace{\rho}_{k\times k}F_{t}+u_{t+1},\tag{1}$$
#|
#| $$\underbrace{r_{t+1}}_{p\times1}-r_{f,t}=\underbrace{\beta_{0}}_{p\times1}+\underbrace{\beta_{1}}_{p\times k}F_{t}+\underbrace{\beta_{2}}_{p\times k}u_{t+1}+v_{t+1}+\epsilon_{t+1},\tag{2}$$
#|
#| $$ y_{t}=b_{0}+b_{1}F_{t}+\epsilon_{t}.\tag{3} $$
#|
#|
#| The following vector of parameters is to be estimated:
#| $$\theta=[c, \rho_{r,y}, \rho_{y,y}, \beta_{2, u_r}, b_0, b_{1, y}]^{\prime},$$
#|
#| where we restrict $\beta_2$ to load only on shock to returns, that is, $\beta_2 = [\beta_{2, u_r}, \mathbf{0}_{p\times p}]$, and $b_1 = [\mathbf{0}_{p\times p}, b_{1,y}]$ is restricted to load only on valuation ratios.
#|
#|
#| The function proceed in several steps:
#|
#| 1.   Given current values of $c, \rho_{r,y}, \rho_{y,y}$, the transition matrix $\rho$ is reconstructed, shocks $u_{t+1}$ in (1) are computed, and $\Sigma$ is estimated.
#| 2.   Given current values of $\beta_{2, u_r}$ and $b_{1, y}$, variables $\beta_2$ and $b_1$ are constructed for test assets, respectively, by imposing restrictions that $\beta_2$ loads only on shock to returns and and $b_1$ loads only on valuation ratios.
#| 3.   For state variables (market and PCs), $\beta$s are given by $c, \rho$. Additionally, because D/P are specifically included in the state space, it must be that $b_{0}=0$ and $b_{1} = [\mathbb{0}_{p\times p}, I_{p\times p}], \beta_2 = [I_{p\times p}, \mathbb{0}_{p\times p}]$ for assets in the state vector.
#| 4.   For given parameter values at any step, we compute prices of risk
#| parameters $\lambda$ and $\Lambda$ in
#| \begin{equation}
#| \underbrace{\lambda_{t}}_{k\times1}=\underbrace{\lambda}_{k\times1}+\underbrace{\Lambda}_{k\times k}\underbrace{F_{t}}_{k\times1}
#| \end{equation}
#| via a simple solution to the
#| system of $p=\frac{1}{2}k$ equations for variables in the state space:
#| \begin{align}
#| \underbrace{\beta_{2}\Sigma}_{p\times k}\underbrace{\vphantom{\beta_{2}\Sigma}\lambda}_{k\times1} & =\underbrace{\beta_{0}}_{p\times1}+\frac{1}{2}\text{diag}\left[\beta_{2}\Sigma\beta_{2}^{'}\right],\tag{4}\\
#| \underbrace{\beta_{2}\Sigma}_{p\times k}\underbrace{\vphantom{\beta_{2}}\Lambda}_{k\times k} & =\underbrace{\beta_{1}}_{p\times k}.\tag{5}
#| \end{align}
#| Note that parameters $\lambda$ and $\Lambda$ inherit restrictions
#| on physical dynamics. Specifically, only the first $p$ elements of
#| $\lambda$ are non-zero, since only shocks to PC returns are priced;
#| similarly, $\Lambda$ takes the following form:
#| $$
#| \Lambda=\left[\begin{array}{cc}
#| 0_{p\times p} & \tilde{\Lambda}\\
#| 0_{p\times p} & 0_{p\times p}
#| \end{array}\right],
#| $$
#| where $\tilde{\Lambda}$ is an $p\times p$ matrix of risk price loadings
#| on D/Ps, which are all fully pinned down by the physical dynamics
#| as well.
#| 5.    Given the estimates of parameters $\lambda$ and $\Lambda$ we can now compute $\beta_0, \beta_1$ parameters for all test assets using equations (4) and (5) applied to test assets.
#| 6.    Lastly, we stack state and test asset return and D/P equations, as well as all parameters. We then use them to estimate errors in equations (1)-(3), instruments, and moment conditions. The moment conditions are as follows:
#| * Using the state vector dynamics in (1), we construct
#| shocks $u_{t+1}$ and interact them with instruments which contain
#| a vector of ones and D/P ratios in $F_{t}$. This corresponds to standard
#| OLS moments with additional restrictions that elements of $\rho$
#| corresponding to loadings on lagged returns are all zero.
#| * Using return equations in (2), we construct shocks to
#| returns, $v_{t+1}+\epsilon_{t+1}$, and interact them with instruments
#| which contain a vector of ones, D/P ratios in $F_{t}$, as well as
#| contemporaneous returns shocks in $u_{t+1}$ -- all standard OLS
#| moment conditions as well.
#| * Using the dividend price equation (3) we construct residuals
#| $\epsilon_{t}$ and interact them with instruments which contain a
#| vector of ones and contemporaneous D/P ratios in the state vector.
#|
#| Individual assets' moments are all normalized by a square root of
#| the number of test assets, $\sqrt{n}$, to keep their contribution
#| to GMM objective invariant of $n$. We use an identity GMM weighting
#| matrix in the GMM objective:
#| $$\hat{\theta} = \arg\min_\theta g_T(\theta)^\prime g_T(\theta).$$
#|
#| The following parameters are estimated by GMM:
#|
#| 1.   State space vector variables:
#| * Intercept $c$: $k$ parameters
#| * Loadings $\rho$: $k$ loadings onto D/P ratios, $k\times p$ parameters
#| 2.   Asset-specific parameters:
#| * Intecepts of D/P equations $b_{0}$: $n$ parameters
#| * Loadings of assets' D/Ps onto state-space D/Ps: $n\times p$ (loadings
#| on returns are restricted to zeros)
#| * Loadings of assets' returns onto state-space shocks to returns: $n\times p$
#| (loadings on D/Ps are restricted to zeros). All other loadings of
#| test assets' returns are pinned down by no arbitrage and SDF risk
#| prices.
#|
#|
#| The spectral density covariance matrix of
#| moments uses $12$ lags and follows the approach in Hansen ().
#|
#| '''
#| Note that in this joint system risk prices are effectively identified via a weighted (dynamic) cross-sectional regression. If one uses GLS instead, the solution should coincide with a two-step approach: first estimate the state space, then price all assets. Potential motivation is bias in $R_m$ loadings on market's D/P due to high persistence of the latter. Test assets mix market's D/P with other D/Ps, reduce the bias and potentially lead to better estimates. Note that when anomaly portfolio price changes have no shocks, all equations have to be used jointly.
#| '''
#|
#| Given the risk prices and loadings on shocks, slopes of test asset returns on the market's D/P ratio are restricted by no-arbitrage conditions. Risk prices, therefore, need to be estimated in a way that is consistent with time-series predictability of each time serie for every test asset. This leads to more stringent restrictions than the loading of market returns on the aggregate D/P ratio (not only has it to predict returns on the market, but also predict returns on all anomalies reasonably well).
#|
#| Imposing no-arbitrage restrictions on $\beta_0$ and $beta_1$ together with fitting D/P to the data allows us to satisfy the price consistency restriction (all strip orices add up to the total price of an asset). When $\beta_0$ and $beta_1$ are allowed to be freely estimated, the consistency restriction is violated, which has implications for dividends.
#|
#| Asset specific shocks $v_{t+1}$ and $\epsilon_{t+1}$ are assumed to be measurement errors. Within the model all assets should be priced perfectly without these error terms (all assets are diversified portfolios). In fact, pricing of all strips and their adding up to a total price relies on absence of these terms? (we still perform variance correction... test if this changes the results!)
@tf.function
def estimate_full_model(variables, W, data, npc, N, oos_start_idx, bootstrap_sample=False, gmm_weighting_matrix=None):
  # unpack all variables
  if not bootstrap_sample:
    F, test_asset_ret, test_asset_dps, _, _ = data # Timing: t -> F_{t+1}, r_{t+1}, dp_{t+1}
    
    F0 = F[:oos_start_idx,:][:-L,:] # F_{t}
    F1 = F[:oos_start_idx,:][L:,:]  # F_{t+1}
    R1 = test_asset_ret[:oos_start_idx,:][L:,:]  # R_{t+1}
    DP1 = test_asset_dps[:oos_start_idx,:][L:,:]  # DP_{t+1}
    
    F0fs = F[:-L,:] # F_{t}
    F1fs = F[L:,:]  # F_{t+1}
    R1fs = test_asset_ret[L:,:]  # R_{t+1}
    DP1fs = test_asset_dps[L:,:]  # DP_{t+1}

    assert N == test_asset_ret.shape[1]
    
  else:
    # OOS analysis is not supported in this mode
    F0, F1, R1, DP1 = data 
    F0fs, F1fs, R1fs, DP1fs = F0, F1, R1, DP1


  [c, rho_r, rho_dp, test_beta2_ur, test_b0, test_b1_dp] = variables

  T, K = F0.shape
  test_assets_gmm_weight = tf.constant(1.0 / np.sqrt(N)) # make GMM invariant to the total number of test assets

  # slices to extract ret or dps
  ret_idx = slice(0, npc)
  dps_idx = slice(npc, 2*npc)

  # transition matrix rho is restricted: zero loading on ret; diagonal for returns on dp
  rho = build_rho_from_parts(rho_r, rho_dp, K, npc, diag_rho_r)

  ## Step 1:
  # state dynamics errors
  u = F1 - tf.transpose(c) - F0 @ tf.transpose(rho) # state dynamics error u_{t+1}

  # covariance of residuals
  uc = u - tf.reduce_mean(u,axis=0) # Remove mean
  Sigma = tf.transpose(uc) @ uc / T
  #Sigma = tfp.stats.covariance(u)

  ## Step 2:
  # beta2: restrict to load only on ret shocks (u_r)
  test_beta2 = tf.concat([test_beta2_ur, tf.zeros([N, npc], dtype=dtype)], axis=1)

  # b1: restrict to load only on D/Ps
  test_b1 = tf.concat([tf.zeros([N, npc], dtype=dtype), test_b1_dp], axis=1)

  ## Step 3:
  # Betas of assets in the state space, implied loadings
  state_beta0 = c[ret_idx]
  state_beta1 = rho[ret_idx,:]
  state_beta2 = tf.concat([tf.eye(npc, dtype=dtype),
                            tf.zeros([npc, npc], dtype=dtype)], axis=1)

  # implied D/P equation parameters for assets in the state space
  state_b0 = tf.zeros([npc, 1], dtype=dtype);
  state_b1 = tf.concat([tf.zeros([npc, npc], dtype=dtype), tf.eye(npc, dtype=dtype)], axis=1)

  # state_var = 0.5*tf.reshape(tf.linalg.diag_part(state_beta2 @ Sigma @ tf.transpose(state_beta2) + .0), [npc,1]) # variance correction terms for state variables in the Euler equation
  
  ## Step 4:
  # Solve for risk prices
  b2S = state_beta2 @ Sigma
  Lambda = tf.concat([tf.linalg.solve(b2S[:, :npc], state_beta1),
                      tf.zeros([npc, K], dtype=dtype)], axis=0)

  ## Step 5:
  # compute test_beta0, test_beta1
  tb2S = test_beta2 @ Sigma
  test_beta1 = tb2S @ Lambda
  test_e_r_uncentered = R1 - F0 @ tf.transpose(test_beta1) - u @ tf.transpose(test_beta2) # return errors, v_{t+1} + e_{t+1}
  test_sigma2_r = tf.transpose(tf.reduce_mean((test_e_r_uncentered -
    tf.reduce_mean(test_e_r_uncentered,axis=0))**2, axis=0, keepdims=True))*include_idiosync_shocks
  
  # compute the remaining piece of risk prices (lambda0)
  test_var = 0.5*tf.reshape(tf.linalg.diag_part(test_beta2 @ Sigma @ tf.transpose(test_beta2) + test_sigma2_r), [N,1])

  if market_is_avg_portfolio:
    state_var = W @ test_var
  else: # handle the market separately
    state_var = tf.concat((0.5 * state_beta2[:1,:] @ Sigma @ tf.transpose(state_beta2[:1,:]) + .0,
                            (W @ test_var)[1:]), axis=0)
  lambda0 = tf.concat([tf.linalg.solve(b2S[:, :npc], state_beta0 + state_var), # zero idios adjustment for L-S portfolios (ave ~ 0); market is actual portfolio so no adj is needed
    tf.zeros([npc, 1], dtype=dtype)], axis=0) 
  
  test_beta0 = tb2S @ lambda0 - test_var # drop test_sigma2_r if no idiosyncratic shocks

  
  ## Step 6:
  # stack market and test asset parameters, returns and DPs
  beta0 = tf.concat([state_beta0, test_beta0], axis=0)
  beta1 = tf.concat([state_beta1, test_beta1], axis=0)
  beta2 = tf.concat([state_beta2, test_beta2], axis=0)
  b0 = tf.concat([state_b0, test_b0], axis=0)
  b1 = tf.concat([state_b1, test_b1], axis=0)
  R = tf.concat([F1[:, ret_idx], R1], axis=1) # t+1
  DP = tf.concat([F1[:, dps_idx], DP1], axis=1) # t+1

  # errors
  e_r = R - tf.transpose(beta0) - F0 @ tf.transpose(beta1) - u @ tf.transpose(beta2) # return errors, v_{t+1} + e_{t+1}
  e_dp = DP - tf.transpose(b0) - F1 @ tf.transpose(b1) # DP pricing errors, e_{t+1}
  #v = e_r - e_dp # idiosyncratic shocks to assets' prices, v_{t+1}

  # return errors and moments. Errors: (1) F dynamics, (2) return errors (automatically 0 for mkt & PCs)
  u_r = tf.concat([u, e_r*test_assets_gmm_weight], axis=1)


  # instruments for moment conditions in e: use state space D/P ratios
  Z_u = tf.concat([tf.ones([T,1],dtype=dtype), F0[:, dps_idx]], axis=1) # t

  # instruments for moment conditions in e_dp: contemporaneous D/P ratios
  Z_dp = tf.concat([tf.ones([T,1],dtype=dtype), F1[:, dps_idx]], axis=1) # t+1


  # moment conditions
  m_u = tf.transpose(Z_u) @ u_r # orthogonality with lagged D/Ps of PCs
  m_dp = tf.transpose(Z_dp) @ (e_dp*test_assets_gmm_weight) # orthogonality with contemporaneous D/Ps of PCs
  m_r = tf.transpose(u[:, ret_idx]) @ (e_r*test_assets_gmm_weight) # extra orthogonality conditions wrt u_{t+1} for returns

  # concatenate all moment conditions into one long vector
  m = tf.concat([tf.reshape(m_u, [-1]),
                 tf.reshape(m_dp, [-1]),
                 tf.reshape(m_r, [-1])#, tf.reshape(m_arb, [-1])
                ], axis=0) / T

  # GMM objective: m'm
  if gmm_weighting_matrix is not None:
    gmm_obj = tf.reshape(m, (1, m.shape[0])) @ gmm_weighting_matrix @ tf.reshape(m, (m.shape[0], 1)) *T*T
  else:
    gmm_obj = tf.reduce_sum(m**2, axis=0)*T*T #+ mkt_clearing_penalty # automatically satisfied for D/Ps!

  # mkt clearing penalty for all weights [only bs; ignore returns errors (beta0, beta1 are imposed by no arbitrage; b2 can be precisely estimated?)]
  test_parameters = tf.concat([test_b0, test_b1], axis=1) # test_beta2,
  state_parameters = tf.concat([state_b0, state_b1], axis=1) # state_beta2,
  mkt_clearing_penalty = tf.reduce_sum((W @ test_parameters - state_parameters)[1:,:]**2)
  # mkt_clearing_penalty is no longer zero since mkt is not an EW average of L&S portfolios

  # full-sample 
  Rfs = tf.concat([F1fs[:, ret_idx], R1fs], axis=1) # t+1
  DPfs = tf.concat([F1fs[:, dps_idx], DP1fs], axis=1) # t+1

  return (gmm_obj, m, mkt_clearing_penalty,
          c, rho, Sigma, lambda0, Lambda,
          b0, b1, beta0, beta1, beta2,
          F0, F1, R, DP,
          F0fs, F1fs, Rfs, DPfs,
          u, e_r, e_dp,
          Z_u, Z_dp)

#| ###  Compute dividend strips and all other derived quantities of interest
#| Estimate a model and compute all derived quantities (this function should be called once after estimation is complete; it performs much more work than the one above and should not be used to search for a solution). This function is differentiated to construct plots with standard error bounds; it is not, however, used to find a solution.
#|
#| The function performs the following steps:
#|
#| 1. Estimate main (joint) model's parameters by calling estimate_full_model() function above
#| 2. Compute risk-neutral state space parameters for pricing
#| 3. Back out the implied $\Delta p_t$ process and its risk-neutral parameters
#| 4. Construct shocks and compute their variances
#| 5. For each maturity, compute strip prices, expected next period's strip prices, as well as expected dividends (PV) using pricing recursions
#| 6. Compute summary statistics
#| 7. Compute all series
#| 8. Output all results so that derivateives of any serie can be computed
@tf.function
def compute_strips_prices(variables, W, data, npc, N, oos_start_idx, data_ext=None): # make predictions for data_ext = (F, test_asset_ret, test_asset_dps, by) if provided
 
  # use a full vector of factors F for predictions unless another vector is passed
  if data_ext is None:
    # unpack all variables
    F, test_asset_ret, test_asset_dps, by, usrec = data
  else:    
    F, test_asset_ret, test_asset_dps, by = data_ext

  variables_model, variables_mm = variables[:6], variables[6:]
  
  ## Step 1: estimate main model parameters
  (gmm_obj, m, mkt_clearing_penalty,
    c, rho, Sigma, lambda0, Lambda,
    b0, b1, beta0, beta1, beta2,
    F0is, F1is, Ris, DPis,
    F0, F1, _, _,
    u, e_r, e_dp,
    Z_u, Z_dp) = estimate_full_model(variables_model, W, data, npc, N, oos_start_idx, False)

  
  T, K = F0.shape

  # slices to extract ret or dps
  ret_idx = slice(0, npc)
  dps_idx = slice(npc, 2*npc)

  ## Step 2: risk-neutral dynamics parameters
  Sl = Sigma @ lambda0
  SL = Sigma @ Lambda
  c_rn = c - Sl
  rho_rn = rho - SL

  ## Step 3: Back out the implied delta_p process & risk-neutral gamma parameters
  # delta_p process
  g0 = beta0 - b0 - b1 @ c
  g1 = beta1 - b1 @ rho
  g2 = beta2 - b1

  # Risk-neutral gamma parameters
  g0_rn = g0 - g2 @ Sl
  g1_rn = g1 - g2 @ SL

  ## Step 4: Construct shocks and compute their variances
  v = e_r - e_dp # idiosyncratic shocks to assets' prices, v_{t+1}

  # variances
  sigma2_r = tf.transpose(tf.reduce_mean((e_r - tf.reduce_mean(e_r,axis=0))**2, axis=0, keepdims=True))*include_idiosync_shocks
  sigma2_v = tf.transpose(tf.reduce_mean((v - tf.reduce_mean(v,axis=0))**2, axis=0, keepdims=True))*include_idiosync_shocks

  ## Step 5: For each maturity, compute strip prices, expected next period's strip prices, as well as expected dividends using pricing recursions
  # iteratively price all dividend strips of this test asset
  # initial parameter values
  a1 = b0 + 0.5*(sigma2_r - sigma2_v); d1 = b1 # drop sigma2_r, sigma2_v is no idiosyncratic shocks!!!
  a2 = a1*0; d2 = d1*0
  a1_rn = a1; a2_rn = a2; d1_rn = d1; d2_rn = d2

  # allocate all containers
  sh = tf.TensorShape((F.shape[0], N+npc))
  P_cum = tf.zeros(sh, dtype=dtype) # cumulative prices
  P = tf.TensorArray(dtype, size=p_max_t, element_shape=sh) # all div stirips prices
  PV = tf.TensorArray(dtype, size=p_max_t, element_shape=sh) # all expected PV of dividends
  EP = tf.TensorArray(dtype, size=p_max_t, element_shape=sh) # all expected strips prices next period
  Ph = tf.TensorArray(dtype, size=int(p_max_t/p_reporting_freq), element_shape=sh) # history of P at checkpoints given by P_checkpoints
  #e_arb = tf.zeros(sh, dtype=dtype)

  # add up all strips prices t = 1-100
  for i in range(p_max_t):
    # precompute variance adjustments
    dgSigma1rn = 0.5*(tf.reshape(tf.linalg.diag_part((d1_rn+g2) @ Sigma @ tf.transpose(d1_rn+g2)),b0.shape) + sigma2_v) # drop sigma2_r, sigma2_v is no idiosyncratic shocks!!!
    dgSigma2rn = 0.5*(tf.reshape(tf.linalg.diag_part((d2_rn+g2) @ Sigma @ tf.transpose(d2_rn+g2)),b0.shape) + sigma2_v)

    ## Step 5.1: Recursions for expected strips prices next period
    a1e = a1_rn + g0 + d1_rn @ c + dgSigma1rn
    a2e = a2_rn + g0 + d2_rn @ c + dgSigma2rn
    d1e = g1 + d1_rn @ rho
    d2e = g1 + d2_rn @ rho

    EPt_1 = (tf.exp(tf.transpose(a1e) + F @ tf.transpose(d1e)) -
        tf.exp(tf.transpose(a2e) + F @ tf.transpose(d2e))) # this is really E[P]/e^rf
    EP = EP.write(i, EPt_1) # NOTE: EP is shifted by one period; EP[0,:,:] is meaningless!

    ## Step 5.2: Recursions for dividend strip prices
    a1_rn = a1_rn + g0_rn + d1_rn @ c_rn + dgSigma1rn
    a2_rn = a2_rn + g0_rn + d2_rn @ c_rn + dgSigma2rn
    d1_rn = g1_rn + d1_rn @ rho_rn
    d2_rn = g1_rn + d2_rn @ rho_rn

    Pt = (tf.exp(tf.transpose(a1_rn) + F @ tf.transpose(d1_rn)) -
          tf.exp(tf.transpose(a2_rn) + F @ tf.transpose(d2_rn))) # price at a given maturity

    # save a history of dividend strip prices
    P = P.write(i, Pt) # store all prices
    P_cum += Pt # tf.math.maximum(Pt, tf.constant(0.0, dtype=dtype)) # cumulative price

    # save cumulative price at a checkpoint
    if (i+1) % p_reporting_freq == 0:
      Ph = Ph.write(int((i+1)//p_reporting_freq - 1), P_cum)

    ## Step 5.3: Recursions for expected dividends (PV)
    a1 = a1 + g0 + d1 @ c + 0.5*(tf.reshape(tf.linalg.diag_part(
        (d1+g2) @ Sigma @ tf.transpose(d1+g2)),b0.shape) + sigma2_v)
    a2 = a2 + g0 + d2 @ c + 0.5*(tf.reshape(tf.linalg.diag_part(
        (d2+g2) @ Sigma @ tf.transpose(d2+g2)),b0.shape) + sigma2_v)
    d1 = g1 + d1 @ rho
    d2 = g1 + d2 @ rho

    PVt = (tf.exp(tf.transpose(a1) + F @ tf.transpose(d1)) -
            tf.exp(tf.transpose(a2) + F @ tf.transpose(d2)))
    PV = PV.write(i, PVt) # store

  # stack all TensorArrays
  P = P.stack()
  PV = PV.stack()
  Ph = Ph.stack()
  EP = EP.stack() # NOTE: EP is shifted by one period; EP[0,:,:] is meaningless!


  ## Step 6: Compute summary statistics
  R2u = R2(F1is, u)
  R2r = tf.reduce_mean(R2(Ris, e_r))
  R2dp = tf.reduce_mean(R2(DPis, e_dp))

  mae_p = tf.reduce_mean(tf.abs(Ph[:,:,npc:] - 0.0), axis=[1, 2])
  mae_pm = tf.reduce_mean(tf.abs(Ph[:,:,0] - 0.0), axis=[1]) # market
  mae_r = tf.reduce_mean(tf.abs(e_r))
  mae_dp = tf.reduce_mean(tf.abs(e_dp))
  stats = {'pen': mkt_clearing_penalty, 'R2u': R2u, 'R2r': R2r, 'R2dp': R2dp,
            'mae_p': mae_p, 'mae_pm': mae_pm, 'mae_r': mae_r, 'mae_dp': mae_dp}

  ## Step 7: Compute all series
  # WARNING: TF can't differentiate exp(log(ER_htm)) which leads to missing gradients
  # Be careful to avoid these redundant operations below

  # a vector of maturities
  mat = np.array(range(1, p_max_t+1)).reshape([p_max_t,1,1])

  # bond yields
  BY = tf.expand_dims(tf.transpose(by), axis=2)
  #BY1 = tf.expand_dims(tf.transpose(by[L:, :]), axis=2)

  # equity yields
  DP = tf.concat([F[:, dps_idx], test_asset_dps], axis=1) 
  pd = -tf.math.log(tf.math.exp(DP) - 1.0)
  EY = (-tf.math.log(P) - pd) / mat
  EY_slope7_1 = EY[6:7, :, :] - EY[0:1, :, :] # 7-1 slope needs to be computed here if standard errors are needed

  # forward equity yields
  FEY = EY[:bonds_max_maturity, :, :] - BY
  FEY_slope7_1 = FEY[6:7, :, :] - FEY[0:1, :, :] # 7-1 slope needs to be computed here if standard errors are needed

  # log expected hold-to-maturity return in excess of risk-free rate, log E_t[ R_{t,t+n} / R_{f,t,t+n} ]
  ER_htm = (PV / P) - 1.0 # [not annualized] # ** (1.0 / mat) # see BMSY
  logER_htm = tf.math.log(ER_htm + 1.0) / mat

  # expected returns on strips (theoretical)
  ER = tf.concat([ER_htm[0:1, :, :],
                  EP[1:, :, :]/P[1:, :, :] - 1.0], axis=0)

  # excess realized returns on strips
  R = tf.concat([F[:, ret_idx], test_asset_ret], axis=1) 
  dg_rf = R[L:,:] - DP[L:,:] - pd[L:,:] + pd[:-L,:] 
  r1 = tf.math.exp(R - DP)[L:,:]*(tf.math.exp(DP[L:,:]) - 1.0) / P[0, :-L, :] - 1.0 # CHECK! There might be a mistake here (need to use future R & DP ?)
  r =  tf.concat([tf.reshape(r1, [1, sh[0]-L, N+npc]), # HTM return for 1 year
                  P[:-1, L:, :] / P[1:, :-L, :] * tf.math.exp(R - DP)[L:, :] - 1.0], axis=0) # returns with maturity above 1 year
  r_slope7_1 = r[6:7, :, :] - r[0:1, :, :] # 7-1 slope needs to be computed here if standard errors are needed
  # Note: dgrf can also be baked out from (A12): dgrf = r_1 - e_1

  # standard deviation and SR
  std = tf.math.sqrt(tf.reduce_mean(tf.math.square(r - tf.reduce_mean(r, axis=1, keepdims=True)), axis=1, keepdims=True))
  rstd = r / std # strip returns scaled by full-sample std (to compute SRs)
  ERstd = ER / std # strip conditional expected returns scaled by full-sample std (to compute SRs)

  # beta
  mkt = tf.expand_dims(F[L:,:1], axis=0)
  mkt_mean = tf.reduce_mean(mkt, axis=1, keepdims=True)
  covRM = tf.reduce_mean((mkt - mkt_mean)*(r - tf.reduce_mean(r, axis=1, keepdims=True)), axis=1, keepdims=True)
  varM = tf.reduce_mean(tf.math.square(mkt - mkt_mean), axis=1, keepdims=True)
  beta = covRM / varM
  
  # alpha (using unconditional betas)
  alpha = ER - beta*(tf.transpose(beta0) + F @ tf.transpose(beta1))[:, :1]
  alphar = r - beta*F[L:, :1] # realized alpha

  # returns on forward strips
  rb = tf.concat([tf.constant(0.0, shape=(1,sh[0]-L,1), dtype=dtype),
                  tf.math.exp(BY[1:,:-L,:]*mat[1:bonds_max_maturity]-BY[:-1,L:,:]*mat[:bonds_max_maturity-1,:,:]) - tf.math.exp(BY[0:1,:-L,:])], axis=0)
  rfwd = (r[:bonds_max_maturity,:,:] + 1.0) / (rb + 1.0) - 1.0

  # expected real div growth, E_t[ G_{t,t+n} / R_{f,t,t+n} ] # TODO NOTE: Use forward yields to compute E_t[ G_{t,t+n} ]
  # Eg = ((ER_htm + 1.0) * P / (tf.math.exp(DP0) - 1.0)) - 1.0 # **(1 / mat) - 1.0
  #logEg = tf.math.log(((ER_htm + 1.0) * P / (tf.math.exp(DP) - 1.0))) / mat # **(1 / mat) - 1.0
  logEg = logER_htm - EY # a simpler formula (see A12) -- matches the formula above perfectly (verified)

  # Macaulay duration
  dur = tf.reduce_sum(P * mat, axis=0)

  
  # EY & FEY slopes
  EY_slopes = EY - EY[mS-1, :, :]
  FEY_slopes = FEY - FEY[mS-1, :, :]
  r_slopes = r - r[mS-1, :, :]
  rfwd_slopes = rfwd - rfwd[mS-1, :, :]

    
  ## Step 8: compile a map of calculated series and output all results
  series = {'P': P, 'PV': PV, 'EP': EP,
            'BY': BY, 'EY': EY, 'FEY': FEY, 'EY_slope7_1': EY_slope7_1, 'FEY_slope7_1': FEY_slope7_1, 'r_slope7_1': r_slope7_1, 
            'EY_slopes': EY_slopes, 'FEY_slopes': FEY_slopes, 'r_slopes': r_slopes, 'rfwd_slopes': rfwd_slopes,
            'logER_htm': logER_htm, 'ER': ER, 'r': r, 'logEg': logEg, 'dgrf': dg_rf,
            'dur': dur, 'R': R, 'DP': DP, 'pd': pd, 'pd0': pd[:-L], 'rfwd': rfwd,
            'rstd': rstd, 'ERstd': ERstd, 'std': std, 'beta': beta, 'alpha': alpha, 'alphar': alphar,
            'usrec': usrec, 'usexp': 1.-usrec}

  
  error_moments = (u, e_r, e_dp, Z_u, Z_dp)
  
  return gmm_obj, m, stats, series, error_moments




#@tf.function
def extract_series(v, max_maturity, t0, asset_idx, ls_offset=0):
    v_slice = v[:max_maturity, t0:, asset_idx] if isinstance(t0, int) else tf.gather(v[:max_maturity, :, asset_idx], t0, axis=1)
    if ls_offset:
        v_slice -= v[:max_maturity, t0:, asset_idx+ls_offset] if isinstance(t0, int) else tf.gather(v[:max_maturity, :, asset_idx+ls_offset], t0, axis=1)
    return v_slice

#@tf.function
def extract_series_mat(v, maturity_idx, t0, asset_idx, ls_offset=0):
    v_slice = v[maturity_idx, t0:, asset_idx] if v.ndim > 2 else v[t0:, asset_idx]
    if ls_offset:
        v_slice -= v[maturity_idx, t0:, asset_idx+ls_offset] if v.ndim > 2 else v[t0:, asset_idx+ls_offset]
    return v_slice

#| #### Helper functions to compute standard errors for mean returns plots and time-series plots
    
# function to compute standard errors for a by-maturity plot (average across time) -- standard errors are not adjusted for sample variation! Use with caution!
def compute_by_maturity_se(b_cov, optimizer, extra_params, variable, asset_idx, max_maturity=bonds_max_maturity, t0=0, ls_offset=0):
    # callback which extracts a requested output
    #@tf.function
    def by_maturity_gradient_callback(*params):
        v = compute_strips_prices(*params)[3][variable]
        return tf.reduce_mean(extract_series(v, max_maturity, t0, asset_idx, ls_offset=ls_offset), axis=1)

    # compute a jacobian wrt to requested asset
    jac, b = optimizer.func_jacobian(by_maturity_gradient_callback, extra_params, parallel_iterations=64, experimental_use_pfor=False) # EYm_mkt

    # compute standard errors
    se = tf.math.sqrt( tf.linalg.diag_part(jac @ b_cov @ tf.transpose(jac)) )
    return se, b

# function to compute standard errors for a single time series
def compute_timeseries_se(b_cov, optimizer, extra_params, variable, asset_idx, maturity_idx, t0=0, ls_offset=0):
    # callback which extracts a requested output
    #@tf.function
    def time_series_gradient_callback(*params):
        v = compute_strips_prices(*params)[3][variable]
        return extract_series_mat(v, maturity_idx, t0, asset_idx, ls_offset=ls_offset)

    # compute a jacobian wrt to requested asset
    jac, ts = optimizer.func_jacobian(time_series_gradient_callback, extra_params, parallel_iterations=64, experimental_use_pfor=False) # EYm_mkt

    # compure standard errors
    cov = jac @ b_cov @ tf.transpose(jac)
    se = tf.math.sqrt( tf.linalg.diag_part(cov) )
    return se, ts, cov

# function to compute standard errors for a summary statistic (or a matrix of summary stats)
def compute_statistic_se(b_cov, optimizer, extra_params, variable):
    # callback which extracts a requested output
    #@tf.function
    def time_series_gradient_callback(*params):
        return compute_strips_prices(*params)[3][variable]

    # compute a jacobian wrt to requested asset
    jac, b = optimizer.func_jacobian(time_series_gradient_callback, extra_params, parallel_iterations=64, experimental_use_pfor=False) # EYm_mkt

    # compure standard errors
    cov = jac @ b_cov @ tf.transpose(jac)
    se = tf.math.sqrt( tf.linalg.diag_part(cov) )
    return se, b



def compute_mean_statistic_se(b_cov, optimizer, extra_params, variable, asset_idx, idx_t0=0, ls_offset=0, labels=None):
    # callback which extracts a requested output
    #@tf.function
    def time_series_gradient_callback(*params):
        return compute_strips_prices_mean_moments(*params)[3][variable]

    # compute a jacobian wrt to requested asset
    jac, b = optimizer.func_jacobian(time_series_gradient_callback, extra_params + [variables_mm_reg, asset_idx, idx_t0, ls_offset, labels], parallel_iterations=64, experimental_use_pfor=False) # EYm_mkt

    # compure standard errors
    cov = jac @ b_cov @ tf.linalg.matrix_transpose(jac)
    se = tf.math.sqrt( tf.linalg.diag_part(cov) )
    return tf.squeeze(se), tf.squeeze(b)


#| ### Estimate the extended model with supplemetary mean and regression momements; compute standard errors of all parameters and moments

def mm_var_name(y_vname, x_vnames, asset_idx, ls_offset, labels): # mean moment var name
    a = labels[asset_idx]
    s = 'S' if ls_offset > 0 else ''
    if x_vnames and isinstance(x_vnames, (list, tuple)):
        return f"reg_{a}{s}_{y_vname}_{'_'.join([v if v else 'c' for v in x_vnames])}"  
    else:
        return f"mean_{a}{s}_{y_vname}"

def compute_strips_prices_mean_moments(variables_ext, W, data, npc, N, oos_start_idx, variables_mm_reg, asset_idx, idx_t0, ls_offset, labels): 
    """
    Wraps cumpute_strip_prices() to add the necessary variables and moment conditions 
    in order compute mean moments or regression parameters jointly with the rest of the variables.
    Returns a modified objective and an extended list of moment conditions.
    
    variables_mm_reg supplies a tuple of pairs ( (y_var, [x_var1, x_var2, ...]), ... ) which implements a regression of y on xs
    Examples: 
        - ('EY', None) or ('EY', [None]): compute the mean of EY
        - ('EY', [None, [pd]]): compute intercepts and slopes of a regression of EY on a const and pd
    Timing: Contemparaneous regression for yields, prices, and returns. 
                    IMPORTANT: If a predictability reg of r_{t+1} on pd_{t} is needed, simply create a new variable pd0 = pd[:-L] in compute_strips_prices()  
    """
    
    variables, variables_mm = variables_ext[:6], variables_ext[6:]
    
    # evaluate strip prices
    gmm_obj, m, stats, series, error_moments = compute_strips_prices(variables_ext, W, data, npc, N, oos_start_idx)
    T = error_moments[0].shape[0]# - idx_t0
    
    m_list, e_m_list, z_m_list = [], [], []
    for mm, (y_vname, x_vnames) in zip(variables_mm, variables_mm_reg): # WARNING: Global instance of variables_mm_names is being used! Assume same order!
        
        # compute new moments and errors
        y = tf.transpose(series[y_vname][:bonds_max_maturity, (series[y_vname].shape[1]-T):, asset_idx] - \
                                         series[y_vname][:bonds_max_maturity, (series[y_vname].shape[1]-T):, asset_idx+ls_offset]*(ls_offset>0)) # underlying variable

        if x_vnames and isinstance(x_vnames, (list, tuple)):
            # for X variables use the matching asset_index unless a single time-series is provided
            X = tf.concat([series[v][series[v].shape[0]-T:, asset_idx*(asset_idx < series[v].shape[1]), None] if v else tf.ones((T, 1), dtype=dtype) # timing: r_t on pd_t;   ey_t on pd_t
                                         for v in x_vnames], axis=1)
        else:
            X = tf.ones((T, 1), dtype=dtype)
            
        vname = mm_var_name(y_vname, x_vnames, asset_idx, ls_offset, labels) 
            
        subsample_indicators = tf.concat((tf.zeros((idx_t0, 1), dtype=dtype), tf.ones((T-idx_t0,1), dtype=dtype)), axis=0)
        e_m = y - (X @ mm)*subsample_indicators
        z_m = X*subsample_indicators

        m_list += [tf.reshape(tf.transpose(z_m) @ e_m, [-1]) / T] #[tf.reduce_sum(e_m[idx_t0:, :], axis=0) / T] # e*time_indicator
        
        e_m_list += [e_m]
        z_m_list += [z_m]
        
        series[vname] = mm

 
    # extend the list of moments
    error_moments_mm = error_moments + ((e_m_list, z_m_list),)
    
    # if m_list:    
    # append new moment conditions
    m_extended = tf.concat([m] + m_list, axis=0)
    
    # compute the combine objective
    gmm_obj = tf.reduce_sum(m_extended**2, axis=0)*T*T
        

    return gmm_obj, m_extended, stats, series, error_moments_mm


