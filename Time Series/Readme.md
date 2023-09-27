# Time Series Forecasting
## Theory

#### The Power of Time Series Forecasting: Predicting Ecuador’s Grocery Sales with Precision
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

* [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/en/stable/index.html) - PyTorch Forecasting aims to ease state-of-the-art timeseries forecasting with neural networks for both real-world cases and research alike. 

* Guide to use [Darts](https://unit8co.github.io/darts/quickstart/00-quickstart.html) - Darts is a Python library for user-friendly forecasting and anomaly detection on time series.


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
