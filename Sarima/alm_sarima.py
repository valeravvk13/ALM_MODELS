
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import plotly.graph_objects as go

import multiprocessing
from multiprocessing import Process, Manager, cpu_count

from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import itertools

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX



class SARIMA(SARIMAX):
    """
        SARIMA class inherited from statsmodels.tsa.statespace.sarimax.SARIMAX
    """
    def __init__(self, endog, exog=None, scoring='MAE', *args, **kwargs):
        
        self.endog = endog
        self.exog = exog

        self.scoring = scoring
        
        #overloading __init__
        super(SARIMA, self).__init__(endog=self.endog, exog=self.exog, *args, **kwargs)
        
   
    def _diff(self, x, diff_lag):
        if diff_lag:
            return pd.Series(x[:, 0]).diff(diff_lag)[diff_lag:]
        else:
            return x
        
        
    def acf(self, diff_lag=None, **acf_args):
        
        x = self._diff(self.endog, diff_lag)
        return statsmodels.tsa.stattools.acf(x=x, **acf_args)
        
    
    def pacf(self, diff_lag=None, **pacf_args):
        
        x = self._diff(self.endog, diff_lag)
        return statsmodels.tsa.stattools.pacf(x=x, **acf_args)
    
    
    def plot_acf(self, diff_lag=None, **acf_args):
        
        x = self._diff(self.endog, diff_lag)
        sm.graphics.tsa.plot_acf(x=x, **acf_args)
        
    
    def plot_pacf(self, diff_lag=None, **acf_args):
        
        x = self._diff(self.endog, diff_lag)
        sm.graphics.tsa.plot_pacf(x=x, **acf_args)
   
        
        
        
def sarima_proc(sarima_params, thr, manager_):
    """
        function out of class for multiprocessing
    """
    print('-------- start  {}  with params {} '.format(thr, sarima_params['seasonal_order']))
    fitted_sarima = {'model': SARIMA(**sarima_params).fit(disp=0),
                     'params': sarima_params,
                     }
    print('-------- finish  {}  with params {} '.format(thr, sarima_params['seasonal_order']))
    manager_[thr] = fitted_sarima
    return fitted_sarima




class GridSearch(SARIMA):
    """
        model selection Sarima models GridSearch
    """
    
    def __init__(self, estimator, param_grid, 
                 scoring='MAE', #(MAE, MSE, AIC, BIC)
                 n_jobs=1, cv=1, split_size=0.2):
        
        self. __defaults(estimator, param_grid, 
                   scoring, n_jobs, cv, split_size)
      
        
    def __defaults(self, 
                   estimator, 
                   param_grid, 
                   scoring, 
                   n_jobs, 
                   cv, 
                   split_size):
        """
            default params
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = self._n_jobs_selector(n_jobs)
        self._cv_folds = cv
        self.split_size = split_size
        
    
    def _n_jobs_selector(self, n_jobs):
        """
            select number of cpu using
        """
        self.n_jobs = n_jobs if ((n_jobs >= 1) & (n_jobs <= int(cpu_count() / 2))) else int(cpu_count() / 2)
        return self.n_jobs
        
        
    def _prepare_sarima_parameters(self):
        """
            preparing parameters for statsmodels.tsa.statespace.sarimax.SARIMAX 
            Args:
                GridSearchParams - dictionary with sarimax ranges or lists of parameters
            Return:
                list of dictionaries with sarimax parameters
        """
        sarima_params_lst = [] 
        
        all_params = [dict(zip(self.param_grid.keys(), v)) for v in itertools.product(*self.param_grid.values())]
        for params_ in all_params:    #all_params - combinations of params from GridSearchParams (product)
            #create dict (kwarg)
            sarima_params_dict = {}
  
            #adding order
            order = (params_['p'], params_['d'], params_['q'])
            sarima_params_dict['order'] = order
            
            #adding seasonal_order
            seasonal_order = (params_['P'], params_['D'], params_['Q'], params_['s'])  
            sarima_params_dict['seasonal_order'] = seasonal_order
            
            #adding other sarimax parameters
            sarima_params_ = {k: v for k, v in params_.items() if k not in ['p', 'd', 'q', 'P', 'D', 'Q', 's']}
            sarima_params_dict.update(sarima_params_)
            
            #append params (kwargs) to list
            sarima_params_lst.append(sarima_params_dict)
            
        return sarima_params_lst
        
        
    def _train_test_split(self):
        """
            train_test_split 
        """
    
        if self._cv_folds != 1:
            engod_train, endog_test, exog_train, exog_test = None, None, None, None

            if self.estimator.exog is not None:
                splitted_data = train_test_split(self.estimator.endog, self.estimator.exog, 
                                                 test_size=self.split_size, random_state=13, shuffle=False)
                engod_train, endog_test, exog_train, exog_test = splitted_data

            if self.estimator.exog is None:
                splitted_data = train_test_split(self.estimator.endog,
                                                 test_size=self.split_size, random_state=13, shuffle=False)
                engod_train, endog_test = splitted_data
                
        if self._cv_folds == 1:
            if self.estimator.exog is not None:
                engod_train = endog_test = self.estimator.endog
                exog_train = exog_test = self.estimator.exog

            if self.estimator.exog is None:
                engod_train = endog_test = self.estimator.endog
                exog_train = exog_test = None
                
        return engod_train, endog_test, exog_train, exog_test
        
        
    def _scorer(self, model, endog_test, exog_test):
        """
            Scoring of model
            Args:
                model - estimator
                endog_test, exog_test - test_data    
        """
        endog_train = model.data.endog.__len__() if self._cv_folds != 1 else 0
        if exog_test is not None:
            pred = model.predict(endog_train, endog_train + endog_test.shape[0] - 1,
                                 exog=exog_test)
        else:
            pred = model.predict(endog_train, endog_train + endog_test.shape[0] - 1)
            
        scores = {'AIC': model.aic, 
                  'BIC': model.bic,
                  'MAE': mean_absolute_error(pred, endog_test),
                  'MSE': mean_squared_error(pred, endog_test)
                 }
        return scores
        
        
    def cv_data(self, lst_splitted_data):
        self.cv_data = lst_splitted_data
        
        
    def _GridSearch(self):   
        """
            GridSearch with Pool processes
        """
        
        engod_train, endog_test, exog_train, exog_test = self._train_test_split()
        self.cv_data([engod_train, endog_test, exog_train, exog_test])
    
        if self.n_jobs == 1:
            
            res = []
            for sarima_params in self._prepare_sarima_parameters():
                sarima_params_s = sarima_params.copy()
                
                sarima_params['endog'] = engod_train
                sarima_params['exog'] = exog_train

                model_fit = SARIMA(**sarima_params).fit(disp=0)

                res.append({'model': model_fit, 
                            'params': sarima_params_s, 
                            'scores':  self._scorer(model_fit, endog_test, exog_test)})
                
        if self.n_jobs != 1:
            
            #multiprocessing with Pool
            manager = multiprocessing.Manager()
            manager_dict = manager.dict()

            Pool_results = []
            s_params = self._prepare_sarima_parameters()
            threads_name = ['process_{}'.format(i) for i in range(s_params.__len__())]

            pool = multiprocessing.Pool(self.n_jobs)
            pool_proc_args = []
            for param_s, tr_n in zip(s_params, threads_name):
                
                param_s['endog'] = engod_train
                param_s['exog'] = exog_train

                args=(param_s, tr_n, manager_dict)
                pool_proc_args.append(args)

            Pool_results = pool.starmap(sarima_proc, pool_proc_args)
            pool.close()
                
            models = [x['model'] for x in Pool_results] 
            Sa_param = [x['params'] for x in Pool_results] 
            
            
            """
            #multiprocessing with process threads initialization and fixing samples
            
            manager = multiprocessing.Manager()
            manager_dict = manager.dict()

            Pool_results = []
            s_params = self._prepare_sarima_parameters()

            for i in range(int(s_params.__len__() / self.n_jobs) + (s_params.__len__() % self.n_jobs > 0)):

                jobs = []
                params_sample = s_params[i * self.n_jobs: (i + 1) * self.n_jobs]
                threads_name = ['{}_Pool_{}_process_{}'.format(i * self.n_jobs + x,
                                                               i, x) for x in range(len(params_sample))]

                for param_s, tr_n in zip(params_sample, threads_name):

                    param_s['endog'] = engod_train
                    param_s['exog'] = exog_train

                    p = multiprocessing.Process(target=sarima_proc, #sarima_proc - target function out of class
                                                args=(param_s, tr_n, manager_dict))
                    jobs.append(p)

                for proc in jobs:
                    proc.start()

                for proc in jobs:
                    proc.join()

                Pool_results.append({'threads': threads_name,
                                     'manager_dict': manager_dict
                                     })
                                     
                                     
            models = [Pool_results[0]['manager_dict'][proc]['model'] \
                      for proc in Pool_results[0]['manager_dict'].keys()]
            Sa_params = [Pool_results[0]['manager_dict'][proc]['params'] \
                         for proc in Pool_results[0]['manager_dict'].keys()]
        
            def flatten_lst(lst):
                return [item for sublist in lst for item in sublist]
            proc_names = flatten_lst([proc['threads'] for proc in Pool_results])
            """
        
            res = [{'model': model_fit, 
                      'params': {k: v for k, v in sarima_params.items() if k not in ['endog', 'exog']}, 
                      'scores':  self._scorer(model_fit, endog_test, exog_test)} \
                     for model_fit, sarima_params in zip(models, Sa_param)] 

        self.GridSearchResults = res
    
    
    def fit(self):
        """
            GridSearch fit
        """
        self._GridSearch()
        return self.best_estimator_()
        
        
    #return functions 
    #score, model, params, cv_results
    def cv_results_(self):
        try: 
            res = self.GridSearchResults
            return res
        except:
            print('Exception with cv_results')
    
        
    def _best_model_idx(self):
        try:
            idx_ = np.argmin(np.array([x['scores'][self.scoring] \
                                    for x in self.GridSearchResults]))
            return idx_
        except Exception:
            print('Exception with best_model_idx')
        
    
    def best_estimator_(self):
        try:
            self.GridSearchResults[self._best_model_idx()]['model']
            return self.GridSearchResults[self._best_model_idx()]['model']
        except Exception:
            print('Exception with best_estimator')
    
    
    def best_score_(self):
        try:
            self.GridSearchResults[self._best_model_idx()]['scores'][self.scoring]
            return self.GridSearchResults[self._best_model_idx()]['scores'][self.scoring]
        except Exception:
            print('Exception with best_score')
    
    
    def best_params_(self):
        try:
            self.GridSearchResults[self._best_model_idx()]['params']
            return self.GridSearchResults[self._best_model_idx()]['params']
        except Exception:
            print('Exception with best_params')
     
    
  
