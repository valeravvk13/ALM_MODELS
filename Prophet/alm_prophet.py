import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import itertools

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation as cross_validation_prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import json
from fbprophet.serialize import model_to_json, model_from_json
import logging
logging.getLogger("fbprophet").setLevel(logging.ERROR)

import plotly.graph_objects as go




class PROPHET(Prophet):
    """
        PROPHET class inherited from fbprophet.Prophet
    """
    
    def __init__(self, add_seasonality_arg=None, exog_regressors=None,  *args, **kwargs):
        """
            Args:
                add_seasonality_arg - dictionary with args from Prophet.add_seasonality
                exog_regressors - list names of exog features (for Prophet.add_regressor)
                *args, **kwargs fbprophet.Prophet
        """
        self.add_seasonality_arg = add_seasonality_arg
        self.exog_regressors = exog_regressors
        self.__prophet_fit_flg = 0  #__prophet_fit_flg need to return error if fitted twice
        
        super(PROPHET, self).__init__(*args, **kwargs)
        
        
    def fit(self, train_data):
        """
            overloaded fit method
        """
        self._train_data = train_data
        
        if self.__prophet_fit_flg == 0:
            if self.add_seasonality_arg:
                for arg_ in self.add_seasonality_arg:
                    self.add_seasonality(**arg_)

            if self.exog_regressors:
                for name in self.exog_regressors:
                    Prophet.add_regressor(self, name)
        self.__prophet_fit_flg = 1
        
        fitted_prophet = super().fit(self._train_data)
        
        return fitted_prophet
    

    def __default_cv_params(self):
        """
            default_cv_params (using in cross_validation)
            split with test = 0.2 * (data)
        """
        first_date = self._train_data.ds.iloc[0]
        last_date = self._train_data.ds.iloc[-1]
        days_in = (last_date - first_date).days

        default_params = {'initial':'{} days'.format(int(days_in * 0.8)),
                          'period':'{} days'.format(int(days_in * 0.2)),
                          'horizon':'{} days'.format(int(days_in * 0.2)),
                         }
        return default_params
    
    
    def cv_results(self, df_cv):
        """
            Args:
                df_cv - result of fbprophet.diagnostics.cross_validation_prophet
            Return:
                dictionary, keys: (cutoff, metrics(MAE, MSE, RMSE))
                
        """
        lambda_metrics_df_cv = lambda df_cv: {
            'MAE': mean_absolute_error(df_cv.yhat, df_cv.y), 
            'MSE': mean_squared_error(df_cv.yhat, df_cv.y), 
            'RMSE': np.sqrt(mean_squared_error(df_cv.yhat, df_cv.y))         
        }

        metrics_ = df_cv.groupby('cutoff').apply(lambda_metrics_df_cv)
        
        return {'cutoff': metrics_.index.to_list(),
                'metrics': list(metrics_.values)
               }

    
    def cross_validation(self, cv_params=None):
        """
            validate PROPHET
            Args:
                cv_params - params of fbprophet.diagnostics.cross_validation_prophet
        """
        if cv_params is None:
            cv_params = self.__default_cv_params()
            
        df_cv = cross_validation_prophet(self, **cv_params)
        self._cv_score = self.cv_results(df_cv)
        return df_cv
  
            
    def cv_score(self):
        """
            Return:
                score on cross_validation
        """
        try:
            self._cv_score
            return self._cv_score
        except Exception:
            print('validate model first')
            
            
    def predict(self, test_data):
        """
            overloaded predict method
        """
        self._test_data = test_data
        self._prediction = super().predict(self._test_data)
        return self._prediction
    
    
    def saving_model(self, json_file_name):
        """
            saving model to json_file
            Args:
                json_file_name - filename
        """
        try:
            self._train_data
            with open(json_file_name, 'w') as fout:
                json.dump(model_to_json(self), fout)  # Save model
        except Exception:
            print('model not fitted yet')
        
        
    def plot(self, title='forecast'):
        """
            simple plot
        """
        
        full_data = self._train_data.append(self._test_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full_data.ds, y=full_data.y,
                                 mode='lines', name='true_ts'))

        fig.add_trace(go.Scatter(x=self._test_data.ds, y=self._prediction.yhat,
                                 mode='lines', name='prediction'))
        
        fig.update_layout(title=title, autosize=False,
                         width=1000, height=600)

        fig.show()
        
            
            



class GridSearchProphet(PROPHET):
    """
        model selection with GridSearch (PROPHET models)
    """

    def __init__(self, params_grid, cv_params, scoring='MAE'):
        """
            Args:
                params_grid - dictionary. 
                    keys - PROPHET args
                    values - lists of PROPHET params
                cv_params - parameters of PROPHET cross_validation
                scoring - evaluate predictions
        """
        self.params_grid = ParameterGrid(params_grid)
        self.cv_params = cv_params
        self.scoring = scoring
       
    
    def fit(self, train_data):
        """
            every 
        """
        self._train_data = train_data
        
        GridSearchResults = []
        for param in self.params_grid:

            #validate PROPHET and append results
            m = PROPHET(**param)
            m.fit(self._train_data)
            _ = m.cross_validation(self.cv_params)
            GridSearchResults.append({'model': m,
                                      'params': param, 
                                      'scores': m.cv_score()
                                     })
            #print('lap finish')
        self._GridSearchResults = GridSearchResults
        self._sort_GridSearchResults()
        
        return self.best_model_
                                        
     
    def _sort_GridSearchResults(self):
        """
            method sorting GridSearchResults
        """
        #here we can change algorithm
        #idea search from min(max(scores)) (best from worst variants)
        #mean not from all, only several models
        sort_key = lambda cv_lap: np.mean([sc[self.scoring] for sc in cv_lap['scores']['metrics']])
        self._GridSearchResults = sorted(self._GridSearchResults, key=sort_key)

        self.best_model_ = self._GridSearchResults[0]['model']
        self.best_params_ = self._GridSearchResults[0]['params']
        self.best_cv_scores_ = self._GridSearchResults[0]['scores']
        

        
    def GridSearchCVresults(self):
        """
            Return:
                GridSearchResults
        """
        return self._GridSearchResults
    
    
    def best_model(self):
        """
            Return:
                best model
        """
        return self.best_model_
    
    
    def best_params(self):
        """
            Return:
                best params
        """
        return self.best_params_
    
    
    def best_cv_scores(self):
        """
            Return:
                best score 
        """
        return self.best_cv_scores_
    
    
    def __getitem__(self, key):
        """
            Return:
                key result from top from GridSearchResults
        """
        return self._GridSearchResults[key]
    
    def __len__(self):
        """
            Return:
                number of param_grid combinatins
        """
        return self._GridSearchResults.__len__()
    
    
    def models_predictions(self, test_data, number_of_models=5):
        """
            Return:
                list of Prophet forecasts
                (list len = number_of_models)
        """
        forecasts_arr = []
        for cv_res in self._GridSearchResults[:number_of_models]:
            forecasts_arr.append(cv_res['model'].predict(self._test_data))
        return forecasts_arr
        
        
    #useful functions
    def weighted_average_forecast(self, test_data, number_of_models=5, weights=None):
        """
            weighted_average_forecast of top number_of_models models
            on default weights initialize with uniform distribution
        """
        self._test_data = test_data
        
        if weights is None:
            weights = [1 / number_of_models] * number_of_models
        
        if np.sum(weights) == 1:

            forecasts_arr = self.models_predictions(self._test_data, number_of_models)

            av_forc = np.zeros(forecasts_arr[0].shape[0])
            for i, forc in enumerate(forecasts_arr):
                av_forc += np.array(forc.yhat * weights[i])
            
        return av_forc
    
    
    def predict_with_model_id(self, test_data, models_idx_):
        """
            Prophet forecast of model with models_idx_ index
            GridSearchResults sorted from top model (idx = 0)
        """
        self._test_data = test_data
        return self._GridSearchResults[models_idx_]['model'].predict(self._test_data)
        
   
        
        
            
        
 