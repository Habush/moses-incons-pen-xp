# 1. Introduction

This document summarizes the experiment set-up and the results of the current Bio-AI work where we’re using background knowledge to get informative features and models that generalize well. The ultimate goal is to integrate this work into the MOSES machine learning pipeline so that MOSES's evolutionary learning will be guided by prior knowledge.

## 1.1 Model

We use the Bayesian framework to integrate prior knowledge into the task of feature selection. Suppose we have a dataset $X$, with $n$ samples and $p$ predictors, where $X = [X_{1}, ..., X_{p}]$ is an _n_ _x_ _p_ matrix. The response variable $Y$ is _n_ _x_ _1_ vector. Also imagine, that we have a prior knowledge about the $p$ predictors in a form a graph which we will the denote it's adjancency matrix by $J$. We have two objectives with respect to model building - 
 - The model's features should be consistent with the prior knowledge. That is, features that are correlated according to the background knowledge are selected together 
- The model has good predictive performace in terms of predicting/classifying the target $Y$.

We use a latent variable model, where the latent variable are denoted by $\boldsymbol{\gamma}$, which act as feature selectors. $\boldsymbol{\gamma} = (\gamma_{1},...,\gamma_{p})$ is a binary vector where $\gamma_{j} = 0$ indicates the exclusion of feature $j$ from the model, and $\gamma_{j} = 1$ indicates its inclusion. We denote the features by an _p_ _x_ _1_ vector $\boldsymbol{\beta}$. We model the feature selection by $p(\gamma ~| ~\beta, Y)$ where assumes this posterior puts most of it mass on $\boldsymbol{\gamma}$ s that select features that are connected in $J$, and carry signal that can be used to predict the response.  Using Bayes rule:

$$p(\boldsymbol{\gamma} ~| ~\boldsymbol{\beta}, Y) = p(\boldsymbol{\beta} ~| ~Y, \boldsymbol{\gamma})p(Y ~| ~\boldsymbol{\gamma})p(\boldsymbol{\gamma})$$

Here $p(\boldsymbol{\gamma})$ is the prior over the $\boldsymbol{\gamma}$ s where we use an Ising prior to integrate $J$ into the model.

$$p(\boldsymbol{\gamma}) = \frac{1}{Z}e^{\eta\boldsymbol{\gamma}^{T}J\boldsymbol{\gamma} ~+ ~\mu\Sigma_{i}\gamma_{i}}$$

In above equation, $Z$ the normalization constant (also called the partition function), $\eta$ is a parameter that controls the connectivity strength and $\mu$ controls the sparsity. We can't directly sample from $p(\boldsymbol{\gamma})$ because that would entail summing over the $2^p$ possible configurations of $\boldsymbol{\gamma}$ vectors. Hence, we need to resort to using approximate inference methods such as Markov Chain Monte Carlo (MCMC). Currently, we use Mixed-HMC sampler proposed by [Zhou 2020](https://arxiv.org/abs/1909.04852) because of the mixed support of the posterior. 


# 2. Experiment Setup

To test whether using a prior knowledge would help us discover better models, we setup two sets of runs where in one set we used the background graph $J$, and in the other set we used a null graph $J_{control}$ where there were no edges between the features. In later case, $p(\boldsymbol{\gamma})$ can be viewed as _i.i.d_ Bernoulli prior. We used a microarray gene expression data of 642 breast cancer tumor samples with 8,414 genes as features. The gene expression levels were encoded as either "low" (0)  or "high" (1). The response variable is also a binary variable that indicates whether the patients had positive or negative outcome after receiving the drug tamoxifen as treatment. We used the regulatory network, which was obtained from [RegNet](https://regnetworkweb.org/home.jsp), of the genes as the background graph.

Due to large size of the predictors, and the limitation of the current sampler as the number of features increase (more on this on section 4), we first had to filter the features using fisher exact test. We used a $p$ value threshold of $0.01$ for pre-selecting the features using fisher's test on the train set. We sampled from the posterior of the two different models (using $J$ and $J_{control}$), and ran Logistic Regression on the selected features.


We calculated the posterior inclusion probabilities $Pr(\gamma_{j} = 1|Y)$ for each feature by the number of iterations where $\gamma_{j} = 1$ over the total number of iterations. When calculating these marignal probabilities we need a threshold $t$, where features whose $Pr(\gamma_{j} = 1|Y) > t$ are selected, and those below the threshold are excluded. In this experment, we used threshold values in the range $[0.1,~0.9]$. _Features that are selected at higher threshold are those that are deemed to be important as they occur frequently in the posterior samples._ For each threshold value, we collected the statistics about the number of selected features, of those that were selected the ones that are related in the background graph $J$, and the train-test scores. We conducted the 45 runs using different seeds for the train/test split, and took the averages.


# 3. Results

We performed a one-sided paired t-test to check if there is any statistical significance in the difference between the mean of train and test scores for each threshold. In this t-test, we checking if the mean of the scores when using the background graph, $J$, is greater than that when we use the null graph - $J_{control}$. The results are summarized in the following table.

| threshold |p_value (train) |p_value (test)
| ----------|----------------|--------------
|0.1        |0.839           |0.161
|0.2        |0.999           |5.2 x $10^{-4}$
|0.3        |0.999           |0.936
|0.4        |0.999           |0.298
|0.5        |0.999           |0.609
|0.6        |0.876           |0.111
|0.7        |2.56 x $10^{-4}$ | 8.66 x $10^{-6}$
|0.8        |2.63 x $10^{-5}$ |6.38 x $10^{-5}$
|0.9        |7.30 x $10^{-8}$ |8.33 x $10^{-5}$

_Table 1: p_value for one-sided paired t-test comparing mean scores of the two models_


In the above table, we can see than we get significant results ($p < 0.05$) for higher threshold values ($t > 0.6$). As stated above, The important features are the ones that are still selected when the values of threshold $t$ are high because $Pr(\gamma_{j} = 1|Y)$ is high enough to be above $t$.  When the threshold values are higher, only few features that are informative will be selected, as they are the ones whose $\gamma$ values are turned on most frequently. 

![Numb of features selected & Num in J](https://imgur.com/i67aVgc.jpg)

_Fig 1: (a) Number of features selected at each thresold (b) Number of features found in the background graph of those selected_


![Train/test AUC score](https://imgur.com/MqjtAR9.jpg)

_Fig 2: (a) The roc-auc score on train set (b) The roc-auc score on test set_


The above two figures summarize the results of the two different models at each threshold. In _Fig 1_(a), we see that the number of features that are selected is decreasing for both models as the threshold value increases. In addition, as it can be seen in _Fig 1(b)_, of those features that are selected, we find more of them in the graph when using the background knowledge. In fact, the model that doesn't use background knowledge doesn't select features that are found in the graph for $t > 0.4$. _From the above results, we can conclude that, the model that uses background knowledge assigns high probability to features that give better performance and are connected in the background graph compared to the one where we don't use prior knowledge._

# 4. Remarks

In the above experiment, we pre-filter the features using fisher's test to reduce their size. This is done so that the sampler will converge, and finish in a reasonable amount of time. As pointed out by [Grathwohl et.al 2021](https://arxiv.org/abs/2102.04509), Mixed-HMC suffers from performance issues as the dimensionality of the data increases. We are currently looking into ways to improve the performance of the sampler. In addition, we note that in this experiment we didn't vary the connectivity $(\eta)$ and sparsity $(\mu)$ parameters - we set them to 1. in both models. We plan to do a parameter sweep once we build a performant sampler. We also plan to repeat this experiment with a different datasets to should the results still hold.

