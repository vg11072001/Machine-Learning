### Parameter estimation
#### Bias
The bias of an estimator $( \hat{\theta} )$  is the expected difference between ! $\widehat{\theta }$ and the true parameter:
$$[ \text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] $$
##### Biased Estimator
A biased estimator is one where the expected value of the estimator is not equal to the true parameter it is estimating. The equation for the bias of an estimator $(\hat{\theta}$ (an estimate of a parameter $(\theta)$ ) is given as: 

$$[ \text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta ] $$

###### Key Components: 
- $(\mathbb{E}[\hat{\theta}])$: 
- The expected value of the estimator. 
- $(\theta)$: The true parameter that we are estimating. 
- 
###### Example: 
If you're estimating the population mean $(\mu)$ using the sample mean $(\hat{\mu})$, and if $(\hat{\mu})$ systematically overestimates or underestimates $(\mu)$, then $(\hat{\mu})$ is biased, and you can calculate the bias using the formula above. 

In practice: 
- If $(\text{Bias}(\hat{\theta}) = 0)$, the estimator is unbiased. 
- If $(\text{Bias}(\hat{\theta}) \neq 0),$ the estimator is biased.

Thus, an estimator is unbiased if its bias is equal to zero, and biased otherwise.

##### Read
[[cheatsheet-statistics_stnfrd_cme_basics.pdf#page=1&selection=16,0,16,20|cheatsheet-statistics_stnfrd_cme_basics, page 1]]
[[cheatsheet-probability_stnfrd_cme_basics.pdf]]

#### Unbiased Estimators

An **unbiased estimator** is a statistical estimator that, on average, produces values equal to the true value of the parameter it is estimating. In other words, an estimator is said to be **unbiased** if the expected value of the estimator equals the parameter being estimated.

##### Key Concepts:
- Let $( \theta )$ be the parameter we want to estimate (e.g., the population mean or variance).
- Let $( \hat{\theta} )$ be the estimator of $\$( \theta )$, calculated from a sample.
- The estimator $( \hat{\theta} )$ is **unbiased** if:
  
  $$ 
  \mathbb{E}[\hat{\theta}] = \theta 
  $$

  where $( \mathbb{E}[\hat{\theta}] )$ represents the expected value (average) of the estimator $( \hat{\theta} )$.

##### Example:
###### 1. Sample Mean:
The sample mean 
$$
( \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i )
$$
is an unbiased estimator of the population mean $( \mu )$. This is because:
$$
\mathbb{E}[\bar{X}] = \mu
$$
meaning that, on average, the sample mean equals the population mean.
###### 2. Sample Variance:
The sample variance 
$$ 
( S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2 ) 
$$
is an unbiased estimator of the population variance $( \sigma^2 )$. The factor $$( \frac{1}{n-1} )$$ instead of $( \frac{1}{n} )$ ensures that the sample variance does not underestimate the true population variance.

##### Importance:
- **Unbiasedness** ensures that, on average, we are not systematically overestimating or underestimating the true parameter.
- While an unbiased estimator may not always be the best choice (e.g., it may have high variance), it guarantees that there is no bias in its predictions over many samples.


##### Read :
* [Unbiased Estimator Glossary ](https://www.statlect.com/glossary/unbiased-estimator)
* [What is an unbiased estimator? Proof sample mean is unbiased and why we divide by n-1 for sample var](https://www.youtube.com/watch?v=xJlwSkyeP0k)

### Resources 
* Playlist #Statistics Fundamentals : 60 videos [Link](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
* CME 106: #Statistics and #Probability  [Introduction to Probability and Statistics for Engineers](https://stanford.edu/~shervine/teaching/cme-106/)
* Online Stanford Course: #Statistics [Learning with Python](https://www.edx.org/learn/python/stanford-university-statistical-learning-with-python)