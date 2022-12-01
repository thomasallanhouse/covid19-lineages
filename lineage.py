import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from tqdm.notebook import tqdm
import scipy.optimize as op

def LineageCounts(df_data, dict_lineages, t_ranges):
    '''
    Function to generate a timeline dataframe of 
    counts by lineage aliases.
    
    df_data : pandas.DataFrame
       2- columns Dataframe with the raw occurences, 
       1st column (Day) are integers counting from 
       the 26/04/2020, 2nd column (paper_lineage)
       are the labels of lineages
    dict_lineages : dict
       Dictionary with the lineage alias as keys 
       and a list of the conforming sublineages as 
       values
    t_ranges : dict
       Dictionary with the lineage alias as keys 
       and time ranges to consider as values
       
    Returns : pandas.DataFrame
       Count of the occurences of the given lineage aliases, 
       with the day number as indices and the lineages
       aliases as columns
    '''
    
    df_counts = (df_data
                 .groupby('Day')
                 .count()
                )
    columns_name = {'paper_lineage':'All'}
    df_counts.rename(columns=columns_name, 
                     inplace=True)

    for i, (key, value) in enumerate(dict_lineages.items()):
        mask = (df_data
                .paper_lineage
                .isin(value)
               )
        df_grouped = (df_data[mask]
                      .groupby('Day')
                     )

        Z = df_grouped.count()
        t0, t1 = t_ranges[key]
        Z = Z.loc[t0:t1, :]
        columns_name = {'paper_lineage':key}
        Z.rename(columns=columns_name,
                inplace=True)
        query = f'`{key}`.notna()'
        indices = (Z
                   .query(query, engine='python')
                   .index
                  )
        x_min, x_max = indices[[0, -1]]
        Z = Z.loc[x_min:x_max+1, :]
        Z = Z.reindex(np.arange(x_min, x_max+1))
        Z.fillna(0, inplace=True)        

        df_counts = df_counts.join(Z, how='outer')
        
    return df_counts

def LineageProbabilities(df_data, dict_lineages, t_ranges):
    '''
    Function to generate a timeline dataframe of 
    probabilities by lineages aliases.
    
    df_data : pandas.DataFrame
       2- columns Dataframe with the raw occurences, 
       1st column (Day) are integers counting from 
       the 26/04/2020, 2nd column (paper_lineage)
       are the labels of lineages
    dict_lineages : dict
       Dictionary with the lineage alias as keys 
       and a list of the conforming sublineages as 
       values
    t_ranges : dict
       Dictionary with the lineage alias as keys 
       and time ranges to consider as values
       
    Returns : pandas.DataFrame
       Probability of occurences of the given lineage 
       aliases, with the day number as indices and the 
       lineages aliases as columns
    '''
    
    df_probabilities = pd.DataFrame()
    df = LineageCounts(df_data, dict_lineages, t_ranges)
    
    for lineage in df.columns[1:]:
        eval_ = f'`{lineage}` / All'
        p_lineage = df.eval(eval_)
        p_lineage.name = f'P_{lineage}'
        df_probabilities = (df_probabilities
                            .join(p_lineage, how='outer')
                           )
        
    return df_probabilities

def LineageOccurences(df_data, dict_lineages):
    '''
    Function to generate a dictionary of the separated 
    occurrences of the lineage aliases.
    
    df_data : pandas.DataFrame
       2- columns Dataframe with the raw occurences, 
       1st column (Day) are integers counting from 
       the 26/04/2020, 2nd column (paper_lineage)
       are the labels of lineages
    dict_lineages : dict
       Dictionary with the lineage alias as keys 
       and a list of the conforming sublineages as 
       values
       
    Returns : dict
       Occurences with the lineage alias as key and a 
       2-columns pandas.DataFrame, 1st column (Day) 
       the day number, 2nd column (lineage alias) vector 
       of ones or zeros for if is the strain in question 
       or not respectively
    '''
    
    occurences = {}
    
    for lineage in dict_lineages.keys():
        
        lineage_list = dict_lineages[lineage]
        query = 'paper_lineage in @lineage_list'
        x_min, x_max = (df_data
                        .query(query)
                        .Day
                        .agg([min, max])
                        .T
                       )
        query = '@x_min <= Day <= @x_max'
        df_lineage = df_data.query(query)
        eval_ = f'paper_lineage = {query}'
        df_lineage.eval(eval_, inplace=True)
        df_lineage.reset_index(drop=True, inplace=True)
        occurences[lineage] = df_lineage
    
    return occurences

def myop(obj_func, initial_theta, bounds):
    '''
    Optimisation function
    '''
    
    opt_res = op.minimize(obj_func, 
                          initial_theta,
                          method="L-BFGS-B", 
                          jac=True,
                          bounds=bounds, 
                          options={'maxfun':100000, 
                                   'maxiter':100000,
                                  }
                         )
    theta_opt, func_min = opt_res.x, opt_res.fun
    
    return theta_opt, func_min

def myboot(y):
    '''
    Bootstraping function
    '''
    
    n = int(np.sum(y))
    p = y / np.sum(y)
    return np.random.multinomial(n, p)

def GPRPreprocess(y):
    '''
    Function to preprocess data to be input to 
    sk-learn Gaussian Processes
    '''
    
    return np.atleast_2d(np.log(y+1)).T

def GPFitting(occurences, df_counts):
    '''
    Function to produce fitting by either ocurrences 
    or counts of lineage alias.
    
    occurences : dict
       Segregation of occurences with the lineage 
       alias as key and a 2-columns pandas.DataFrame
       of similar characteristics as df_data for 
       a single lineage alias
    df_counts : pandas.DataFrame
       Probability of occurences of the given lineage 
       aliases, with the day number as indices and the 
       lineages aliases as columns
       
    Return : dict
       Collection of fitting data with the following keys and values:
       'Pi_store': The probability of the occurence of the lineage alias,
       'Pi_boot': The bootstrapping of the probability, 
       'r_store': The estimated growing rate,
       'r_boot': The bootstrapping of the growing rate.
    '''
    
    nboot = 200 # Number of bootstrap samples
    ncmax = 15000 # Point at which we switch between GPC and GPR
    
    Pi_store = pd.DataFrame()
    Pi_boot = {}
    r_store = pd.DataFrame()
    r_boot = {}

    for i, lineage in enumerate(occurences.keys()):
        X, y = (occurences[lineage]
                .T
                .values
               )
        X = X.reshape(-1, 1)
        m = len(y)
        print(f'Loaded data for {lineage}')

        columns = ['All', lineage]
        df = df_counts[columns].dropna()
        X0 = (df
              .index
              .to_numpy()
             )
        X0min, X0max = X0[[0,-1]]
        X1 = np.atleast_2d(np.arange(X0min, X0max+1)).T

        if (m < ncmax):
            print('Running Gaussian process classification.')
            kernel = 1.0 * RBF(1.0)
            gpc = GaussianProcessClassifier(kernel=kernel, 
                                            copy_X_train=False)
            gpc.fit(X, y)
            Pi = gpc.predict_proba(X1.reshape(-1, 1))[:, 1]
            dr = np.diff(np.log(Pi/(1.-Pi)))

            print('Main fit done. Bootstrap progress:')

            pb = np.zeros((nboot,len(X1)))
            rb = np.zeros((nboot,len(X1)-1))
            for j in tqdm(range(nboot)):
                i_boot = np.random.randint(0, m, m)
                y_boot = y[i_boot]
                X_boot = X[i_boot].reshape(-1, 1)    
                kernel = gpc.kernel_
                gpc_boot = GaussianProcessClassifier(kernel=kernel, 
                                                     optimizer=None, 
                                                     copy_X_train=False
                                                    )
                try:
                    gpc_boot.fit(X_boot, y_boot)
                    pb[j,:] = (gpc_boot
                               .predict_proba(X1
                                              .reshape(-1, 1)
                                             )[:, 1]
                              )
                    rb[j,:] = np.diff(np.log(pb[j,:] /
                                             (1.-pb[j,:]))
                                     )
                except:
                    print(f'Failed on bootstrap {j:.0f}')
                    j -= 1
        else:
            print('Too large for GPC (' + str(m) + ' samples). Running Gaussian Process Regression.')
            yy1 = df.eval(f'All - `{lineage}`')
            yy2 = df[lineage]
            X0 = np.atleast_2d(X0).T
            y1 = GPRPreprocess(yy1)
            y2 = GPRPreprocess(yy2)

            kernel1 = (1.0 * 
                       RBF(length_scale=10.) + 
                       WhiteKernel(noise_level=1)
                      )
            gpr1 = GaussianProcessRegressor(kernel=kernel1, 
                                            alpha=0.0, 
                                            n_restarts_optimizer=10, 
                                            optimizer=myop
                                           )
            gpr1.fit(X0,y1)
            y_mean1 = gpr1.predict(X1, return_std=False)

            kernel2 = (1.0 * 
                       RBF(length_scale=10.) + 
                       WhiteKernel(noise_level=1.0)
                      )
            gpr2 = GaussianProcessRegressor(kernel=kernel2, 
                                            alpha=0.0, 
                                            n_restarts_optimizer=10, 
                                            optimizer=myop
                                           )
            gpr2.fit(X0,y2)
            y_mean2 = gpr2.predict(X1, return_std=False)

            k1store = gpr1.kernel_
            k2store = gpr2.kernel_

            mu1 = y_mean1.reshape(-1)
            mu2 = y_mean2.reshape(-1)
            Pi = ((np.exp(mu2)-1) /
                  (np.exp(mu1)+np.exp(mu2) -
                   2
                  )
                 )

            dmu1 = np.diff(mu1)
            dmu2 = np.diff(mu2)
            dr = (dmu2) - (dmu1)

            print('Main fit done. Bootstrap progress:')

            pb = np.zeros((nboot,len(X1)))
            rb = np.zeros((nboot,len(X1)-1))
            for j in tqdm(range(nboot)):
                y1 = GPRPreprocess(myboot(yy1))
                y2 = GPRPreprocess(myboot(yy2))

                gpr1 = GaussianProcessRegressor(kernel=k1store, 
                                                alpha=0.0, 
                                                n_restarts_optimizer=1
                                               )
                gpr1.fit(X0, y1)
                y_mean1 = gpr1.predict(X1, return_std=False)

                gpr2 = GaussianProcessRegressor(kernel=k2store, 
                                                alpha=0.0, 
                                                n_restarts_optimizer=1
                                               )
                gpr2.fit(X0, y2)
                y_mean2 = gpr2.predict(X1, return_std=False)

                mu1 = y_mean1.reshape(-1)
                mu2 = y_mean2.reshape(-1)
                pb[j,:] = ((np.exp(mu2)-1) /
                           (np.exp(mu1) +
                            np.exp(mu2) - 2
                           )
                          )

                dmu1 = np.diff(mu1)
                dmu2 = np.diff(mu2)
                r = (dmu2) - (dmu1)
                rb[j,:] = r

        X1 = X1.reshape(-1)
        df_Pi = pd.DataFrame(index=X1, 
                             data=Pi, 
                             columns=[lineage]
                            )
        Pi_store = Pi_store.join(df_Pi, how='outer')
        dr = pd.DataFrame(index=X1[1:], 
                          data=dr, 
                          columns=[lineage]
                         )
        r_store = r_store.join(dr, how='outer')
        pb = pd.DataFrame(data=pb.T, index=X1)
        rb = pd.DataFrame(data=rb.T, index=X1[1:])
        Pi_boot[lineage] = pb
        r_boot[lineage] = rb
            
    return {'Pi_store':Pi_store, 
            'Pi_boot':Pi_boot, 
            'r_store':r_store, 
            'r_boot':r_boot
           }

