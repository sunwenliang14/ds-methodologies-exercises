def plot_residuals(y, yhat):
    return plt.scatter(y, y - yhat)


def regression_errors(y, yhat):
    '''
    Returns a dictionary containing various regression error metrics.
    '''
    n = y.size
    residuals = yhat - y
    ybar = y.mean()

    sse = sum(residuals**2)

    ess = sum((yhat - ybar)**2)

    return {'sse': sse,
        'mse': sse / n,
        'rmse': sqrt(sse / n),
        'ess': ess,
        'tss': ess + sse,}


def baseline_mean_errors(y):
    '''
    Returns a dictionary containing various regression error metrics for a
    baseline model which is a model that uses the mean of y as the prediction.
    '''
    baseline_yhat = y.mean()
    n = y.size
    residuals = y - baseline_yhat

    sse = sum(residuals**2)

    return {'sse': sse,
        'mse': sse / n,
        'rmse': sqrt(sse / n),}


def better_than_baseline(y, yhat):
    sse_baseline = baseline_mean_errors(y)['sse']
    sse_model = regression_errors(y, yhat)['sse']
    return sse_model < sse_baseline


def model_significance(model):
    '''
    Given a fitted OLS model from statsmodels, return the model's explained
    variance, and the p-value indicating whether the relationship is statistically significant.
    '''
    return {'r^2': model.rsquared, 'f p-value': model.f_pvalue}





