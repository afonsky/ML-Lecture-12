---
zoom: 1.1
---

# More Examples

<div class="grid grid-cols-[3fr_3fr] gap-5">
<div>

* Classify **phrases** into toxic or not
	* There are billions of phrases,<br> but fewer labeled observations

* Classify **facial emotion** in an image
	* 20 megapixel camera captures<br> ~20 million pixels, but we might have fewer labeled examples of facial emotions
</div>
<div>
	<br>
  <figure>
    <img src="/faces.png" style="width: 400px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://towardsdatascience.com/training-an-emotion-detector-with-transfer-learning-91dea84adeed">https://towardsdatascience.com/training-an-emotion-detector-with-transfer-learning-91dea84adeed</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.9
---

# Curse of Dimensionality. Example 1

* Consider $k$NN for inputs uniformly distributed in a $p$-dimensional unit hypercube
* Consider hypercubical neighborhood around target point to capture a fraction $r$ of observations
	* Expected edge length will be $e_p (r) = r^{1/p}$
<br>

<figure>
  <img src="/ESL_fig_2.6.svg" style="width: 510px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position:relative; top: -50px; left: 650px"><br>Images source:
  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=674">ESL Fig. 2.6</a>
  </figcaption>
</figure>

* With increasing $p$ **we need to capture increasing fraction of the data to form a local average**
* Reducing $r$ is not an option since it will gain variance of the fit

---

# Curse of Dimensionality. Example 2

* Consider $N$ data points uniformly distributed in a $p$-dimensional unit ball centered at the origin
* Consider $k$NN
* The median distance from the origin to the closest data point $d(p, N) = \bigg( 1 - \frac{1}{2}^{1/N}\bigg)^{\frac{1}{p}}$
* Results in that **most data points are closer to the boundary of the sample space than to any other data point**
	* The prediction is much more difficult near the edges of the training sample
		* Extrapolate is harder than interpolate

---

# Curse of Dimensionality. Example 3

* The sampling density is proportional to $N^{\frac{1}{p}}$
* Suppose $N_1 = 100$ represents a dense sample for a single input problem
	* Then $N_{10} = 100^{10}$ is the sample size required for the **same sampling density** with $10$ inputs
		* Thus in high dimensions **all feasible training samples sparsely populate the input space**

---
zoom: 0.93
---

# Curse of Dimensionality. Example 3

* Suppose we have $1000$ training examples $x_i$ generated uniformly on $[−1, 1]^p$
	* $\mathcal{T}$ is training set
* Assume that $Y = f(X) = e^{-8 \lVert X \rVert^2}$
* Suppose we use $1$NN to predict $y_0$ at the test-point $x_0 = 0$
* We can derive **bias–variance decomposition** as $\mathrm{MSE}(x_0) = \mathrm{Var}_\mathcal{T}(\hat{y}_0) + \mathrm{Bias}^2 (\hat{y}_0)$
* At small $p$, the nearest neighbor is close to $0$, so both bias and variance are small
* As the dimension increases, the nearest neighbor tends to move out from the target point, and both bias and variance are incurred

<figure>
  <img src="/ESL_fig_2.7.svg" style="width: 390px !important; position:relative; top: -30px; left: 450px">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position:relative; top: -80px; left: 250px"><br>Images source:
  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=674">ESL Fig. 2.7</a>
  </figcaption>
</figure>

---

# Curse of Dimensionality. Example 4

#### The same setup but the **function is constant in all but one dimension**:<br> $F(X) = \frac{1}{2} (X_1 + 1)^3$
* The variance dominates
<br>

<figure>
  <img src="/ESL_fig_2.8.svg" style="width: 590px !important; position:relative; top: 0px; left: 250px">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position:relative; top: -80px; left: 50px"><br>Images source:
  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=674">ESL Fig. 2.8</a>
  </figcaption>
</figure>

---

# What to Do When $p >> n$?

1. **Reduce dimensionality** and apply methods developed for $n > p$. Examples:
	* E.g. Linear regression. For $X_{n \times p}$, Gram matrix $X X^{\prime}$ is **singular** (non-invertible)
		* Hence, we can't compute $\hat\beta := (X X^{\prime})^{-1} X^{\prime} Y$
			* since we have **more unknowns (parameters) than linear equations**
			* i.e. rank of $X X^{\prime}$ is at most $n < p$
		* In parametric model, we can **regularize** parameters to decrease **effective degrees of freedom** (edf) to be $\leq p$
			* Assumes: most features or parameters are redundant (collinear)
	* E.g. Extract low dimensional features from images 
	* E.g. Compress images to lower resolution
	* E.g. Compress text documents to continuous embedding vector space

---

# What to Do When $p >> n$?

2. Use entirely **new methods** developed for $n << p$

---
zoom: 0.95
---

# Recall: Linear Discriminant Analysis (LDA)

* Multi-class classifier for $K \geq 2$ classes
* Assumes Gaussian features with the same variance, $\Sigma_{p \times p}$
<br>$\mathbb{P}[Y = k | X] := \frac{\mathcal{N}_k (X | \mu_k, \Sigma) \cdot \pi_k}{\sum_{\ell=1}^K \mathcal{N}_\ell (X | \mu_k, \Sigma) \cdot \pi_\ell} \propto \mathcal{N}_k (X | \mu_k, \Sigma) \cdot \pi_k$
	* $p$ means $\mu_k$ for each class
	* $K - 1$ priors $\pi_k$ (as proportions of observation of each class) since $\Sigma_k \pi_k = 1$
	* $p$ variances of diagonal of $\Sigma$
	* $p(p - 1)/2$ covariances in lower triangular of the symmetric $\Sigma$
		* Biggest source of parameters (i.e. degrees of freedom)
* This simplifies to the **linear discriminant functions**:<br> $\delta_k(X) := X^{\prime} \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^{\prime}\mu_k + \log{\pi_k}$
	* With classification decision: $Y(X) := \argmax_k \delta_k (X)$
* In reality, if we just want to compute $\delta_k - \delta_\ell$ and not $\mathbb{P}[Y = k | X]$
	<!-- * Then we only need to estimate $(K - 1)(p + 1)$ parameters} -->

---

# Example. Gene Expression Arrays

* Gene expression matrix ($\mathrm{patient}\times \mathrm{gene}$) $X_{63 \times 2308}$ 
	* $K = 4$ cancer types, $n := n_{\mathrm{train}} := 43$ patients, $n_{\mathrm{test}} := 20$ patients, $p := 2308$ genes

* Then pooled sample variance is:<br><br>
$\hat{\Sigma}_{2308 \times 2308} = \frac{1}{n - K} \sum\limits_k \sum\limits_{y_i = k} (x_i - \hat\mu_k)(x_i - \hat\mu_k)^{\prime}$
	* It has $\frac{2308 \cdot 2307}{2} = 2 660 278$ covariances

* How can we reduce the number of these estimates?

---

# Diagonal-Covariance LDA

* If we assume **within-class independent features** :
	* then we just need to estimate **diagonal** variances of each $\Sigma_k$ (now having $K$ covariance matrices)
* The discriminant score for class $k$ is<br><br>
$\delta_k (x_{p \times 1}) := -\sum\limits_{j=1}^p \frac{(x_j - \bar{x}_{kj})^2}{s_j^2} + 2 \log \pi_k$
	* $s_{p \times 1}^2$ is pooled variance, $\bar{x}_{k, p \times 1}$ is $k$ class’ mean vector (**centroid** vector of class $k$)
	* We standardize each feature’s variance before computing intra-cluster variability
* Note: coverability is no longer estimated
* We still need to estimate $p = 2308$ means per class and $K = 4$ priors
* What would Hastie and Tibshirani propose?

---
zoom: 0.97
---

# Nearest Shrunken Centroids (NSC)

* We can **shrink** class means (centroids), $\bar{x}_k$, towards global mean, $\bar{x}$ 
	* i.e. shrink $\bar{x}_k - \bar{x}$ towards zero
	* Essentially, we want to zero out any “near-zero” centroid difference
* To do so, we standardize the centroid difference as $d_{kj} := \frac{\bar{x}_{kj} - \bar{x}_j}{m_k (s_j + s_0)}$
	* where $m_k := \frac{1}{n_k} - \frac{1}{n} \in (0, 1)$, and $s_0 := \mathrm{median}_j s_j$  keeps $d_{kj}$ close to zero
	* variance of $\bar{x}_{kj} - \bar{x}$ is $m_k^2 \sigma^2$, where $\sigma^2$ is within-class variance
<br>
<div class="grid grid-cols-[3fr_3fr] gap-1">
<div>

* Now, we **soft-threshold** $d_{kj}$  with unknown parameter $\Delta$:<br>
$d_{kj} := \mathrm{sgn}(d_{jk})(\lvert d_{jk} \rvert - \Delta)_{+}$
	* where $\Delta$ is determined via cross validation (CV)
</div>
<div>
	<br>
  <figure>
    <img src="/nsc.png" style="width: 400px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://www.researchgate.net/figure/Thresholding-schemes-a-hard-thresholding-and-b-soft-thresholding_fig4_245331181">https://www.researchgate.net/figure/Thresholding-schemes-a-hard-thresholding-and-b-soft-thresholding_fig4_245331181</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.97
---

# NSC vs Diagonal Covariance LDA

<div class="grid grid-cols-[3fr_3fr] gap-10">
<div>

* Diagonal Covariance LDA
	* leaves 2308 genes
	* 25% test error
* NSC with :
	* leaves 43 genes
		* Easier to interpret
	* **zero** train, test and CV errors
  <figure>
    <img src="/number_of_genes_shrinkage.png" style="width: 400px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://www.researchgate.net/figure/Thresholding-schemes-a-hard-thresholding-and-b-soft-thresholding_fig4_245331181">https://www.researchgate.net/figure/Thresholding-schemes-a-hard-thresholding-and-b-soft-thresholding_fig4_245331181</a>
    </figcaption>
  </figure>
</div>
<div>
  <figure>
    <img src="/centroids_genes.png" style="width: 410px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Images source:
	  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=674">ESL Fig. 18.4</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.97
---

# Ramaswamy’s Dataset

* Gene expression matrix, $X_{198 \times 16063}$
	* $K = 14$ cancer types, $n_{\mathrm{train}} := 144$ patients, $n_{\mathrm{test}} := 54$ patients, $p := 16063$ genes
<br>[Multiclass cancer diagnosis using tumor gene expression signatures, Ramaswamy et. al.](https://www.pnas.org/content/pnas/98/26/15149.full.pdf)
<div class="grid grid-cols-[4fr_5fr] gap-1">
<div>

* Ridge does not zero out parameters
	* So, $L_2$ will “always” use all features
	* So, $L_2$ vs. $L_1$ is **not a fair comparison** for “Numbers of Genes Used”
	* Unless, we threshold  parameters
	* i.e. zero out params close to zero with hard or soft thresholding
* Next, let’s learn about these models
</div>
<div>
<br>
<br>
  <figure>
    <img src="/ESL_table_18.1.png" style="width: 490px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Source:
	  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=675">ESL Table. 18.1</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Regularized Discriminant Analysis (RDA)

* Assuming independence of all features (genes) is too radical
* Instead, let’s regularize covariances (but not variances)
	* This will shrink a common covariance matrix, $\hat\Sigma$, towards its diagonal
<br> $\hat\Sigma(\gamma) := \gamma\hat\Sigma + (1 - \gamma) I_{p \times p} \cdot \color{grey}\underbrace{\color{#006}{\mathrm{diag}\hat\Sigma}}_{[\hat\Sigma_{11},...,\hat\Sigma_{pp}]^\prime} \color{#006}, \gamma \in [0, 1]$
	* where $\mathrm{diag}[...]$ returns diagonal matrix from a vector or returns a vector of diagonal elements 
	* Then $\hat\Sigma(0) = I \cdot \mathrm{diag}\hat\Sigma = {\begin{bmatrix} \hat\Sigma_{11} & ~ & ~ \\ ~ & \ddots & ~ \\ ~ & ~ & \hat\Sigma_{pp}\end{bmatrix}}$ and $\hat\Sigma(1) =  {\begin{bmatrix} \hat\Sigma_{11} & \ldots & \hat\Sigma_{1p} \\ \vdots & \ddots & \vdots \\ \hat\Sigma_{p1} & \ldots & \hat\Sigma_{pp}\end{bmatrix}}$
* RDA classifier is similar to ridge linear regression with categorical response

---

# Logistic Regression with Quadratic Regularization

* Here we use multinomial logistic regression with regularized parameters
	* It handles multi-class output with a **softmax** function
<br>$\mathbb{P}[Y = k | X] = \frac{\exp[\beta_{k0} + X^\prime \beta_k]}{\sum_\ell \exp[\beta_{k0} + X^\prime \beta_k]}$
	* where $\beta_k \in \mathbb{R}^p$ is a vector of all within-class slopes (excluding intercept)

* We maximize penalized **log-likelihood** as:
<br><br>$\max_{\beta_{0k}, \beta_k} \sum\limits_i \log \mathbb{P}[y_i | X_i] - \frac{\lambda}{2} \sum\limits_k \lVert \beta_k \rVert_2^2$

---

# Support Vector Classifier (SVC)

* Regularization is not needed to find a separating hyperplane
	* When $p > n$, the classes are separable
		* Unless 2 identical observations are in different classes
	* In fact, often we add dimensions (via kernels) to locate the better separating hyperplane

<div class="grid grid-cols-[4fr_3fr] gap-1">
<div>

* Note: SVC is a **binary** classifier
	* So, we need to use one-versus-one (OVO) or one-versus-all (OVA) approach
</div>
<div>
  <figure>
    <img src="/svm.gif" style="width: 350px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image (animated gif) source:
	  <a href="https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93">https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.97
---

# Feature Importance

* Hi-Dim problem: it’s important to select important features
	* Feature “importance” needs to be **well-defined**
		* Important doesn’t mean high-variance, but how well it explains the outcome
			* We could add a large-noise feature, which is not important
* SVC & RDA: 
	* no weights => can’t pick features based on weights
* Logistic Regression:
	* Iteratively remove features with small coefficients and refit
		* Larger parameter **magnitude** may imply larger “importance”
	* Lower **p-value** of a coefficient might imply larger “importance”
	* Popular: important features are sticky in regularization paths
* For any model we can rank features by their effectiveness

---

# Feature Importance

  <figure>
    <img src="/ESL_fig_4.13.png" style="width: 350px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=145">ESL Fig. 4.13</a>
    </figcaption>
  </figure>

* Iteratively remove features with small coefficients and refit
	* Larger parameter **magnitude** may imply larger “importance”
* Lower **p-value** of a coefficient might imply larger “importance”
* Popular: important features are sticky in regularization paths

---

# Computational Shortcut for Ridge Models

* Singular-value decomposition (SVD) yields
<br> $X_{n \times p} = U_{n \times n} D_{n \times n} V_{n \times p}^\prime = R V^\prime$
	* where $D$ is diagonal matrix of ordered eigenvalues,<br> $U$ - left eigenvectors, $V$ - right eigenvectors

* We reparametrize
<br> $\hat\beta_{\mathrm{ridge}} = (X^\prime X + \lambda I)^{-1} X^\prime y = V(R^\prime R + \lambda I)^{-1} R^\prime y = V \hat\theta$

* i.e we reduce data matrix from $X_{n \times p}$ to $R_{n \times n}$ with $n << p$ predictors

* This works for all linear models with ridge penalty!
	* It may not work for non-linear models or those with non-ridge penalty

---

# Protein Mass Spectrometry (PMS) Example

* PMS identifies spectral decomposition of proteins in blood
	* Used to diagnose cancer
* Features ordered by **mass over charge ratio**, $\frac{m}{z}$
* Feature values result in “*smooth*” function shape (plus some noise)
	* Hence, the name “*functional data*”

<div class="grid grid-cols-[3fr_4fr] gap-1">
<div>

* The ordering of the features is highly informative
</div>
<div>
  <figure>
    <img src="/mz.png" style="width: 430px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=684">ESL Fig. 18.7</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Fused Lasso

* Suppose our features have some natural **order**
	* Such as in PMS example, voice signal, stock market prices, digital images
* Fused Lasso enforces smoothness of corresponding coefficients
	* assumes no gaps in order, i.e. no jumps, i.e. the coefficient function sampling is smooth
	* Note: Lasso and Ridge penalties uniformly shrink coefficients without accounting for order
* **Fused Lasso**: $\min\limits_{\beta \in \R^p} \lVert Y - \hat{Y}\rVert_2^2 + \lambda_1 \lVert\beta\rVert + \lambda_2 \sum\limits_{j=1}^{p-1}\lvert \beta_{j+1} - \beta_j \rvert$
			
* Still yields sparse solution
* Forces adjacent coefficients to remain similar in values

---
zoom: 0.85
---

# Classification with Latent Features

* Sometimes features are not directly observable
* How do we even measure similarity among observations?

<div class="grid grid-cols-[4fr_3fr] gap-10">
<div>
<br>

#### Examples:
* A corpus of text documents containing different words, phrases, sentences
	* What if they are written in different languages
* A set of digital images
* A set of DNA molecules’ nucleotide sequences
* A set of stocks’ historical prices
* A set of voice signals
	* Amazon needs to recognize “Alexa” pronounced in any voice, at any length, at any sampling rate
</div>
<div>
  <figure>
    <img src="/alexa.png" style="width: 460px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://read.nxtbook.com/ieee/signal_processing/november_2019/speech_processing_for_digital.html">https://read.nxtbook.com/ieee/signal_processing/november_2019/speech_processing_for_digital.html</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.95
---

# Ex. Protein Classification with String Kernels

* Protein molecules with strings of 20 amino acids: different **length** & **composition**
* We can count matched substrings of different length
	* Define a **feature map** $\Phi_m(x) := \{\phi_a(x)\}_{a \in \mathcal{A}_m}$
		* where $\mathcal{A}_m$ is set of subsequences of length $m$, $\phi_a(x)$ - count of $a$ in $x$, $\mathrm{len}(a) = m$
		* $\Phi_m$ is a vector of size $\lvert\mathcal{A}_m\rvert = 20^m$  (assumes uniform distribution of amino acids)
	* String kernel is a similarity measure $K_m(x_1, x_2) := \langle \Phi_m(x_1), \Phi_m(x_2) \rangle$
		* “$\color{red}{\mathrm{LQE}}\color{#006}$” $\in \mathcal{A}_3, \phi_{\color{red}{\mathrm{LQE}}}\color{#006}(x_1) = 1, \phi_{\color{red}{\mathrm{LQE}}}\color{#006}(x_2) = 2$
	* For $m = 4$, we have a feature space with $p = 160K$ dimensions (most with 0 counts)
		* In practice, we could only use substrings which are present in our string
<div class="grid grid-cols-[4fr_5fr] gap-3">
<div>

Then apply earlier algorithms to the new features
</div>
<div>
  <figure>
    <img src="/lqe.png" style="width: 400px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Source:
	  <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=687">ESL p.668</a>
    </figcaption>
  </figure>
</div>
</div>