# Time Series Forecasting
## Theory

### Concepts

* Time Series Characteristics
	* Trend 
	* Seasonality - repeats with respect to timing, direction, and magnitude.
 	* Cyclic - up and down in graph, These are the trends with no set repetition over a particular period of time.
  	* Irregular Variation
  	* ETS Decomposition - for Error, Trend and Seasonality.
 
* Types of Data
	* Time series data - recored with time
 	* Cross sectional data -  one or more variables recorded at the same point in time.
  	* Pooled data - combination of time series data and cross sectional data.
 
* Terminology
	* **Dependence**
 	* ****Stationarity**** - **mean value, variance and autocorrelation** of the series that **remains constant** over the time period. If past effects accumulate and the values increase towards infinity then stationarity is not met.

  	> The values should be independent of time and seasonal effects as well.
	>
   	> ![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/3c0f8c1d-2af4-4552-8415-df1543433351)

   	> Make Time Series stationary using following:
   	> 1. Differencing the Series (once or more)
   	> 2. Take the log of the series
   	> 3. Take the nth root of the series
   	> Test for stationary also there

 
 	* **Specification** - **testing of the linear or non-linear relationships** of dependent variables by using time series models such as ARIMA models.
  
  	* **Differencing** - Differencing is used to **make the series stationary** and to control the auto-correlations
  	  > Differencing the series is nothing but subtracting the next value by the current value.
  	  > 
  	* **Exponential Smoothing** -  time series analysis predicts the one next period value based on the past and current value.
  
  	  > It involves **averaging of data** such that the non-systematic components of each individual case or observation cancel out each other.
  	  > The exponential smoothing method is used to predict the short term prediction.
  	  > 
  	* **Curve fitting** - is used **when data is in a non-linear relationship**.
  	* **ARIMA** -  **Auto Regressive Integrated Moving Average**

* Pattern in Time series
	*  ETS Decomposition components-
 		*  Addictive Time Series: **Base Level + Trend + Seasonality + Error**.
		* Multiplicative Time Series: **Base Level x Trend x Seasonality x Error**

* Test for Stationarity
	1. **Augmented Dickey Fuller test** (ADF Test)
        >
        > ADF test will return 'p-value' and 'Test Statistics' output values.
        >
        > * p-value > 0.05: non-stationary.
        >
        > * p-value <= 0.05: stationary.
        >
        > * **Test statistics**: More negative this value more likely we have stationary series. Also, this value should be smaller than critical values(1%, 5%, 10%).
        > 
        > For e.g. If test statistic is smaller than the 5% critical values, then we can say with 95% confidence that this is a stationary series.
	3. **Kwiatkowski-Phillips-Schmidt-Shin – KPSS test** (trend stationary)

        > The KPSS test, on the other hand, is used to test for trend stationarity.
        >
        > The null hypothesis and the P-Value interpretation is just the opposite of ADH test.
 	>
 
	3. **Philips Perron test (PP Test)**

* Test for Seasonality
	1.  **Autocorrelation Function (ACF)** plot - is simply the correlation of a series with its own lags. If a series is significantly autocorrelated, that means, the previous values of the series (lags) may be helpful in predicting the current value.

  	2.  **Partial Autocorrelation Function (PACF)** plot -  also conveys similar information but it conveys the pure correlation of a series and its lag, excluding the correlation contributions from the intermediate lags.
  	
   	3. **Lag Plots**- is a scatter plot of a time series against a lag of itself. It is normally used to check for autocorrelation. If there is any pattern existing in the series, the series is autocorrelated. If there is no such pattern, the series is likely to be random white noise.  
 
* Some process
	* Deseasonalize a Time series
 		* There are multiple approaches to deseasonalize a time series. These approaches are listed below: (1) Take a moving average with length as the seasonal window. This will smoothen in series in the process. (2) Seasonal difference the series (subtract the value of previous season from the current value). (3) Divide the series by the seasonal index obtained from STL decomposition.
   		* If dividing by the seasonal index does not work well, we will take a log of the series and then do the deseasonalizing 

  	* Detrend a Time Series
		* Detrending a time series means to remove the trend component from the time series. There are multiple approaches of doing this as listed below: (1) Subtract the line of best fit from the time series. (2) The line of best fit may be obtained from a linear regression model with the time steps as the predictor. For more complex trends, we may want to use quadratic terms (x^2) in the model. (3) We subtract the trend component obtained from time series decomposition. (4) Subtract the mean. (5) Apply a filter like Baxter-King filter(statsmodels.tsa.filters.bkfilter) or the Hodrick-Prescott Filter (statsmodels.tsa.filters.hpfilter) to remove the moving average trend lines or the cyclical components.
  	
   	*  Smoothening a Time Series
   		* methods: Take a moving average, Do a LOESS smoothing (Localized Regression) and Do a LOWESS smoothing (Locally Weighted Regression)   	
  
* Granger Causality Test
	* Granger Causality test is a statistical test that is used to determine if a given time series and it’s lags is helpful in explaining the value of another series. 

* White noise
	* Have following features:
 		* Constant mean
   		* Constant Variance
     		* No auto correlation
         * How to identify:
           	* Visual Inspection
           	* Global and local checks
           	* Auto Corelation plots
           	* Statistical tests          
 
* Treat missing values in a time series
	* Backward Fill
 	* Linear Interpolation
	* Quadratic interpolation
	* Mean of nearest neighbors
	* Mean of seasonal couterparts

### Models

* Statistical Models Used For Time Series Forecasting
	* Autoregression (AR)

	* Moving Average (MA)

  	* Naive Approach

	* Holt’s Linear Trend Model

	* Autoregressive Moving Average (ARMA)

	* Autoregressive Integrated Moving Average (ARIMA)

	* Seasonal Autoregressive Integrated Moving-Average (SARIMA)

	* Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)

	* Vector Autoregression (VAR)

	* Vector Autoregression Moving-Average (VARMA)

	* Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)

	* Simple Exponential Smoothing (SES)

  	* Theta
  	
   	* Croston 

	* Holt Winter’s Exponential Smoothing (HWES)

* State-of-the-art machine learning models
	* Classical NN- MLP, RNN
 	* XGBOOST
 	* LightGBM
  	* LSTM 
  	* feature engineering
 
* Novel Proven
	* NHITS
 	* NBEASTS
  	* TFT      

* Probabilistic forecasting

* Global an Local models


### Blogs

#### The Power of Time Series Forecasting: Predicting Ecuador’s Grocery Sales with Precision [Link](https://medium.com/@isaacrambo/revealing-the-power-of-time-series-forecasting-predicting-ecuadors-grocery-sales-with-precision-8c3b0bac97be)
> Predict store sales on data from Corporation Favorita, a large Ecuadorian-based grocery retailer.
>
> Using `pyodbc`: to connect to various database management systems and `dotenv`: managing environment variables
> 
> `statsmodels.tsa.seasonal`: offers tools for decomposing time series data into its constituent components (e.g., trend, seasonality, and noise)
>
> ``scipy.stats`` and ``statsmodels.stats.weightstats`` : hypothesis testing, calculating various statistical measures, and performing t-tests
>
> ``pmdarima.arima`` and ``arch.unitroot``: to build forecasting models and assess stationarity and unit root properties of time series data
>
> HYPOTHESIS - Null Hypothesis (H0) and Alternative Hypothesis (H1)
>
>  XGBoost, ARIMA, SARIMA, and ETS models.

[Code on colab](https://colab.research.google.com/drive/1R_C7422mBQJ9M3ZfdTx-5ASeJwUWGmq5#scrollTo=RvUuWQ3JvW5a)


#### An End-to-End Guide on Time Series Forecasting Using FbProphet

>

#### Google’s Temporal Fusion Transformer (2021) [link](https://medium.com/dataness-ai/understanding-temporal-fusion-transformer-9a7a4fcde74b)

> It's powerful model for **multi-horizon** and **multivariate** time series forecasting use cases. It consist of 3 parts - **Past target, Time-dependent exogensous** and **Statis covariates**.

> (1) to capture temporal dependencies at different time scales by a combination of the **LSTM Sequence-to-Sequence** and the **Transformer’s Self-Attention mechanism**, and

> (2) to enrich **learned temporal** representations with static information about measured entities.

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/0514258c-84c9-45cb-b5b8-e0df22461683)

Blocks is diveded into manyparts-
	
> **Gated residual networks** - improve the generalization capabilities of the model across different application scenarios.
>
> **Static covariate encoders** - learn context vectors from static metadata and inject them at different locations.
>
> **Variable selection** - static covariates, past inputs (time-dependent known and unknown) and known future inputs to learned linear transformations of continuous features and entity embeddings of categorical ones.
>
> **Sequence-to-Sequence** - replaces positional encoding that is found in Transformers [2] by using a Sequence-to-Sequence layer.
>
> **Interpretable Multi-head attention** - 1. importance of values based on the relationships between keys and queries.  The outputs of different heads are then combined via concatenation.
> 2. TFT adjusts this definition to ensure interpretability - allows to easily trace back most relevant values and to have shared weights for values across all attention heads.
> 3. Identify significant changes in temporal patterns. This is done by computing an _average attention pattern per forecast horizon and evaluate the distance_ between it and attention weights at each point.
>
> **Quantile regression** - quantification of uncertainty of predicted values at each time step. It predicts quantiles of the distribution of target ŷ using a special quantile loss function. optimization process is forcing the model to provide reasonable over-estimations for upper quantiles and under-estimations for lower quantiles

Code - 
1) [Demand forecasting with the Temporal Fusion Transformer](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
2) [Temporal Fusion Transformer with Darts library](https://unit8co.github.io/darts/examples/13-TFT-examples.html?highlight=temporal+fusion)


#### Forecasting book sales with Temporal Fusion Transformer [link](https://medium.com/@mouna.labiadh/forecasting-book-sales-with-temporal-fusion-transformer-dd482a7a257c)

> The model implementation that is available in Pytorch Forecasting library along with Kaggle’s “tabular playground series.
>
> Pytorch Forecasting is based on Pytorch Lightning and integrates Optuna for hyperparameters tuning.
>
> 

## Codes

* Decomposition of a Time Series

`````
from statsmodels.tsa.seasonal import seasonal_decompose

multiplicative_decomposition = seasonal_decompose(df['Number of Passengers'], model='multiplicative', period=30)

additive_decomposition = seasonal_decompose(df['Number of Passengers'], model='additive', period=30)
`````

* Autocorrelation and Partial Autocorrelation Functions

````
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df['Number of Passengers'].tolist(), lags=50, ax=axes[0])
plot_pacf(df['Number of Passengers'].tolist(), lags=50, ax=axes[1])
````
* Granger Causality Test

````
from statsmodels.tsa.stattools import grangercausalitytests
data = pd.read_csv('/kaggle/input/dataset/dataset.txt')
data['date'] = pd.to_datetime(data['date'])
data['month'] = data.date.dt.month
grangercausalitytests(data[['value', 'month']], maxlag=2)
````
* KPSS & ADF

````
from statsmodels.tsa.stattools import adfuller, kpss
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])

# ADF Test
result = adfuller(df.value.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.value.values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
````


----------------
* Pandas
	* https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#
 	* https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#time-series-related

* TimeGPT
	* https://nixtla.github.io/nixtla/timegpt.html
	* https://github.com/Nixtla/nixtla
	* https://docs.nixtla.io/docs
	* Video - [TimeGPT Launch by Nixtla ](https://youtube.com/playlist?list=PLq3sJIV6w5BoHJ9gFSedwtb_pqk--4K89&feature=shared)  

* Forecasting
	* R
		* https://otexts.com/fpp2/
	* Python
		* Book - https://www.methsoft.ac.cn/scipaper_files/document_files/Manning.Time.Series.Forecasting.in.Python.pdf
 		* Github https://github.com/marcopeix/TimeSeriesForecastingInPython

* [Facebook Prophet](https://facebook.github.io/prophet/docs/quick_start.html) 

* [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/en/stable/index.html) - PyTorch Forecasting aims to ease state-of-the-art timeseries forecasting with neural networks for both real-world cases and research alike. 

* Guide to use [Darts](https://unit8co.github.io/darts/quickstart/00-quickstart.html) - Darts is a Python library for user-friendly forecasting and anomaly detection on time series.


## Advances on Time Series

* Explore **N-Hits, N-Beats** and **Amazon forecasts** and benchmark them against **simple stat, ml models**
* Also **look for models** which provide accuracy and explainability, both not just one example - **GAM, Prophet** provide both meaning individual contribution to overall forecasts by other drivers
* **Transformer - TFT**, **Darts** - explore darts and it's potential use cases, Explore **linear trees**
* You can continue kaggle for sometime some **EDA** as well
* **Benchmark the observations** And maybe let's summarise findings by **accuracy, interpretability and runtimes**


### Amazon Forecast [Link](https://aws.amazon.com/forecast/)

[Guide](https://docs.aws.amazon.com/pdfs/forecast/latest/dg/forecast.dg.pdf) | [Time Series Forecasting Principles with Amazon Forecast](https://d1.awsstatic.com/whitepapers/time-series-forecasting-principles-amazon-forecast.pdf) | [Github](https://github.com/aws-samples/amazon-forecast-samples)

* It is fully managed service that uses machine learning to produce incredibly accurate forecasts. 
Benefits of Amazon Forecast provide Explainability report. Integration of Amazon forecast with other services: S3, Athena, Glue, Sagemaker, Lambda.
Use Cases of Amazon Forecast:
1. improve product demand planning
2. effectively managing resources:

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/f8d92cef-a9ad-4744-96f4-c4d21fb568b9)

 * **Features** **of Amazon Forecast**: 
 1. information about the local weather is automatically included
 2. creates forecasts that are probabilistic estimates.
 3. Amazon Forecast evaluates time-series data in the context of retail.

* **Limitations** - have little historical data, few choices for customization and irregular or non-seasonal trends, Amazon Forecast is not appropriate. 
Hyperparameter tuning is supported only by CNN-QR and DeepAR+.
For the target, metadata, and related time series datasets, only 13, 10, and 25 features are allowed. If there are more, you might have to choose between features.
The Forecast horizon should be the lesser value of between 500 and ⅓ the size of the target dataset.
DeepAR+ which offers the best estimates will only work when the number of observations is > 300.

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/95ec7308-9f4f-4630-8374-4964349ecee6)

* [**Taco Bell Case Study**](https://aws.amazon.com/forecast/resources/?amazon-forecast-whats-new.sort-by=item.additionalFields.postDateTime&amazon-forecast-whats-new.sort-order=desc) - How it help in improving digital availability with ML Forecast
POS- Point of Sale
![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/ce199d75-6a6d-4c73-b893-b73d49876745)
![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/afc44deb-c2a4-4231-912f-e7f2ebe76986)
![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/f84a3725-6fd4-430b-b388-b300b78a7526)
![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/4db8973f-3c47-4c81-8b7a-a127eac99de6)
![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/12977c72-e976-4c01-8268-d3508ade5f88)


* **More Blogs:**
1. [From forecasting demand to ordering – An automated machine learning approach with Amazon Forecast to decrease stockouts, excess inventory, and costs](https://aws.amazon.com/blogs/machine-learning/from-forecasting-demand-to-ordering-an-automated-machine-learning-approach-with-amazon-forecast-to-decrease-stock-outs-excess-inventory-and-costs/)

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/ea32b57a-8839-4b01-a22b-6b016cc1674f)

2. [Part 2: How Retailers are Leveraging Cloud Technologies to Transform their Supply Chains](https://aws.amazon.com/blogs/industries/how-retailers-leverage-cloud-to-transform-supply-chains/)

3. [A Guide to Predicting Future Outcomes with Amazon Forecast](https://onica.com/blog/ai-machine-learning/a-guide-to-predicting-future-outcomes-with-amazon-forecast/)

4. [Improving Forecast Accuracy with Machine Learning](https://aws.amazon.com/solutions/implementations/improving-forecast-accuracy-with-machine-learning/)

5. [Sales Demand Forecasting with Amazon Forecast](https://samuelabiodun.medium.com/sales-demand-forecasting-with-amazon-forecast-4ff81e6db807)
[Dataset](https://github.com/abiodunjames/Predicting-ecommerce-sales-forecast/tree/master/data) - [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

	1. Pre-processed dataset and upload to an s3
	2. Import training data
	3. Create a predictor or choose algorithm - Dataset group, A featurization configuration, A forecast horizon or prediction length, Evaluation parameters and Algorithm or AutoML
	4. Create a forecast
	5. Retrieve forecast

Extra inform by [Guide to Amazon Forecast for FBA Sellers](https://www.fbamasterclass.io/post/amazon-forecast)



### N-BEATS
Nueral Basis Expansion Analysis for Interpretable Time series forecasting

[N-BEATS_Official_Paper.pdf](https://github.com/vg11072001/Machine-Learning/files/12775592/N-BEATS_Neural_basis_expansion_analysis_for_interp.pdf) | [Short Guide](https://ms.unimelb.edu.au/__data/assets/pdf_file/0009/4524228/Duy_Ngoc_Tran_-_Multivariate_time_series_forecasting_with_N-beats_architecture1.pdf) | [N-BEATS Video](https://www.youtube.com/watch?v=p3Xc_TJU8SI) | [GitHub](https://github.com/ServiceNow/N-BEATS/tree/master)


![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/618384f7-d608-4687-b5fa-95d163ac158b)

It focus on **univariate time series point** forecasting problem. Propose a deep neural architecture based on Backward and forward residual links and a very deep stack of fully-connected layers.
It has properties as **interpretable**, aplicable** without modification** to a wide array of targte domain and **fast** to train.
SUMMARY OF CONTRIBUTIONS :
  1. Deep Neural Architecture - Dmonstrate that pure DL using time series specific components outperforms well-established statistical approaches. (Consist of 1,2)
  2. Interpretable DL for Time Series - Feasible to design an architecture with interpretable outputs that can be used by practitioners in very much the same way as traditional decomposition techniques such as the "seasonality-trend-level" approach. (Consist of 3,4)

* Achitechture

It is pure DL architectures.

  1. Basic Block

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/bb06c972-7d58-489f-8187-107eb078dbc5)


  2. Double Residual Stacking of block

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/b8c0d9c6-dd92-4235-a811-6e7980fbd05e)


  * Each branch has one more fully connected layer without activation, and then a linear basis layer that can be either learned or instead engineered to account for different effects such as trend and seasonality.
  * Since the overall global output is a simple sum of partial outputs of each block, knowing the nature of each basis layer allows the user to estimate the contribution of each component, thus providing interpretability.
  * Essentially, M identical stacks with K blocks in each, as in the suggested model, could be represented by a simple MxK block sequence. However, when separated into stacks, all blocks within each stack can share learnable parameters, resulting in better performance. In addition, each stack can be structured in a given way (e.g. a trend block followed by a seasonality block), both for interpretability and better forecasting.

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/c96a8e38-e810-4d02-b167-200e1e464f57)


  3. Learning Trend

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/51fbfc7c-f23a-47e7-a187-a506a64c6bba)

 
  4. Learning Seasonality

![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/84fac699-5db7-4b8a-9825-60f6ee4ad463)


* Ensembling
  * Much more powerful regularization technique than the popular altematives, e.g. **dropout or L2-norm penalty**.
  * Fit on three different metrics:** sMAPE, MASE and MAPE**
  * For every horizon H, individual models are trained on **input windows of different length: 2H, 3H,....7H**. The overall ensemble exhibits a multi-scale aspect
      - use 180 total models to report results on the test set.
      - use the median as ensemble aggregation function.
      - perform a bagging procedure by induding models trained with different random initializations.

  * N-BEATS-G (generic)
  * N-BEATS-I (interpretable)
  * N-BEATS-l+G(ensemble of all models from N-BEATS-G and N-BEATS-I)
  * ![image](https://github.com/vg11072001/Machine-Learning/assets/67424390/98850aab-083b-44a8-9bfb-ea625912f468)


* Blogs
  * [DL reviws: N-BEATS: Neural Basis Expansion Analysis For Interpretable Time Series Forecasting](https://www.dl.reviews/2020/02/23/n-beats/)
  * [N-BEATS: NEURAL BASIS EXPANSION ANALYSIS FOR INTERPRETABLE TIME SERIES FORECASTING](https://kshavg.medium.com/n-beats-neural-basis-expansion-analysis-for-interpretable-time-series-forecasting-91e94c830393)
  * [Unveiling the Untold: Exploring the Depths of N-BEATS for Stock Price Prediction](https://khofifah.medium.com/unveiling-the-untold-exploring-the-depths-of-n-beats-for-stock-price-prediction-2e3a03c0bf57)
  * [Multiple Time Series Forecasting With N-BEATS In Python](https://forecastegy.com/posts/multiple-time-series-forecasting-nbeats-python/#how-to-prepare-time-series-data-for-n-beats-in-python)
  * [Time Series with TensorFlow: Replicating the N-BEATS Algorithm](https://www.mlq.ai/time-series-with-tensorflow-n-beats-algorithm/)
  * https://unit8co.github.io/darts/examples/07-NBEATS-examples.html
  * https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/ar.html#Interpretable-forecasting-with-N-Beats


-----------------

## References

* https://www.machinelearningplus.com/time-series/time-series-analysis-python/?source=post_page-----8c3b0bac97be--------------------------------
* https://www.kaggle.com/code/satishgunjal/tutorial-time-series-analysis-and-forecasting

### Amaozon Forecast
* https://www.geeksforgeeks.org/what-is-amazon-forecast/
* 
