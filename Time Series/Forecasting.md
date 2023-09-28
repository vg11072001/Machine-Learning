### Some simple forecasting methods

* Average method - Mean
* Naïve method - we simply set all forecasts to be the value of the last observation
* Seasonal naïve method - we set each forecast to be equal to the last observed value from the same season.
* Drift method- A variation on the naïve method is to allow the forecasts to increase or decrease over time, where the amount of change over time (called the drift) is set to be the average change seen in the historical data.

### Transformations and adjustments
* Calendar adjustments
* Population adjustments
* Inflation adjustments
* Mathematical transformations
  * Box-Cox transformations -  logarithms and power transformation
  * power transformations - square roots and cube roots can be used

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/bf923bcf-67ce-4f6f-b519-efe85d5cad78)


* Bias adjustments
* The difference between the simple back-transformed forecast given by and the mean given by (3.2) is called the bias. When we use the mean, rather than the median, we say the point forecasts have been bias-adjusted.


### Residual diagnostics
* Fitted values
* Residuals
* Portmanteau tests for autocorrelation - A test for a group of autocorrelations is called a portmanteau test
  *  Box-Pierce test
  *  Ljung-Box test

### Evaluating forecast accuracy
 * Forecast errors -A forecast “error” is the difference between an observed value and its forecast.
 * Scale-dependent errors - MAE, RMSE
 * Percentage errors - MAPE
 * Scaled errors -mean absolute scaled error
 * Time series cross-validation
 * Pipe operator

### Prediction intervals
* One-step prediction intervals
* Multi-step prediction intervals
* Benchmark methods -
  *  Mean forecasts
  *  Naïve forecasts
  *  Seasonal Naïve forecasts
  *  Drift forecast
* Prediction intervals from bootstrapped residuals - process allows us to measure future uncertainty by only using the historical data.
* Prediction intervals with transformations - the prediction interval should be computed on the transformed scale, and the end points back-transformed to give a prediction interval on the original scale

### The forecast package in R
* he following list shows all the functions that produce forecast objects.
meanf()
naive(), snaive()
rwf()
croston()
stlf()
ses()
holt(), hw()
splinef()
thetaf()
forecast()
*  forecast() function -  function works with many different types of inputs. It generally takes a time series or time series model as its main argument, and produces forecasts appropriately. It always returns objects of class forecast

### Judgmental forecasts
* [Key principles](https://otexts.com/fpp2/judgmental-principles.html)https://otexts.com/fpp2/judgmental-principles.html
* The Delphi method - is to construct consensus forecasts from a group of experts in a structured iterative manner.
*  New product forecasting- Sales force composite, Executive opinion and Customer intentions

### Time series regression models
* Linear model- Linear regression an dmutiple regression.
* Least squares estimation, Fitted values, Goodness-of-fit, Standard error of the regression
* Evaluating the regression model- ACF plot of residuals

