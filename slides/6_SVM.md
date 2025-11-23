# Support Vector Machines and Higher Dimensions

* Instead of **primal formulation** of classification $f(x) = \theta^T x + b$<br>
SVMs use **dual formulation** $f(x) = \sum\limits_{i=0}^n \alpha_i y_i (x_i^T x) + b$
* Instead of using non-linear terms  $X_i^d, X_i X_j, ...$, we can use a **kernel trick**
	* Consider a **linear kernel** for $u, v \in \R^p$:
$
K(u, v) = \langle u, v \rangle = \sum\limits_{j=1}^p u_j v_j = u \cdot v = u^{\prime}v
$
	* It turns out that SVC optimization can be rewritten via
$
f(x) = \sum\limits_{i=1}^n \alpha_i K(x, x_i)
$
	* To estimate $\alpha_{1:n}$ and $\beta_0$ we need $\frac{n(n-1)}{2}$ **inner products**... , however...
	* $\alpha \neq 0$ only for the support vectors
		* So, estimation simplifies to
$f(x) = \sum\limits_{i \in S}^n \alpha_i K(x, x_i)$,

where $S$ is a set of support vectors (determined during training)

---
zoom: 0.97
---

# We can also add non-linearity via
<br>

* **Polynomial kernel** of degree $d$:
$$
K(u,v) = (1 + \sum\limits_{j=1}^p u_j v_j)^d
$$

* **Radial (basis) kernel** of degree $d$:
$$
K(u,v) = \mathrm{exp}\Big[-\gamma \sum\limits_{j=1}^p (u_j - v_j)^2 \Big] = \mathrm{exp}\Big[-\gamma ||u_j - v_j||_2^2 \Big] ~~~\mathrm{\textcolor{black}{for}} ~~~\gamma \gt 0
$$

* **Neural network kernel**:
$$
K(u,v) = \mathrm{tanh}\Big[a \langle u, v \rangle + r \Big]
$$

---
zoom: 0.92
---

# Radial Basis Kernel, $\mathrm{exp}\big[-\gamma ||u_j - v_j||_2^2 \big]$

<div class="grid grid-cols-[5fr_3fr] gap-1">
<div>

* It is essentially an unscaled Gaussian PDF
* Large $u - v$ implies near zero exponent
* So, in $f(x) = \beta_0 + \sum\limits_{i \in S}^n \alpha_i K(x, x_i)$
	* $\alpha = 0$ for non-support vectors, or
	* $K(x, x_i) \approx 0$ for distant observations $x_i$

</div>
<div>
<br>
<figure>
	<img src="/Normal_Distribution_PDF.svg" style="width: 250px !important">
	<figcaption style="color:#b3b3b3ff; font-size: 11px">Image source:
	  <a href="https://en.wikipedia.org/wiki/Normal_distribution">https://en.wikipedia.org/wiki/Normal_distribution</a>
	</figcaption>
</figure>
</div>
</div>

<br>
<div class="grid grid-cols-[3fr_3fr] gap-1">
<div>

* Hence, only neighboring support vectors participate in forming a decision boundary

</div>
<div>
<figure>
	<img src="/ISLRv2_figure_9.9.png" style="width: 450px !important">
	<figcaption style="color:#b3b3b3ff; font-size: 11px; position:relative; top: -40px; left: -250px">Image source:
	  <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=390">ISLR Fig. 9.9</a>
	</figcaption>
</figure>
</div>
</div>

---

# SVM and feature map

<figure>
	<img src="/svm_feature_map.png" style="width: 800px !important">
	<figcaption style="color:#b3b3b3ff; font-size: 11px">Image source:
	  <a href="https://xavierbourretsicotte.github.io/Kernel_feature_map.html">https://xavierbourretsicotte.github.io/Kernel_feature_map.html</a>
	</figcaption>
</figure>

---
zoom: 0.95
---

# SVM is a Binary Classifier

* ... but can be extended to multiclass.
* Consider $K > 2$ classes and a test observation $x^{*}$
* **One-versus-one** (OVO) approach
	* Fit $\binom{K}{2}$ pairwise SVMs 
	* Assign $x^{*}$ to the most frequently predicted class
	* Disadvantage: lots of models to fit and inference with
* **One-versus-all** (OVA) approach
	* Fit $K$ SVMs for one of $K$ classes versus all other $K - 1$ classes combined
	* Assign $x^{*}$ to the class with greatest $\beta_{0k} + \beta_{1k}^{\prime} x_1^{*} + ... + \beta_{pk}^{\prime} x_p^{*}$ value
		* Where $\beta_{ik}$ are fitted for each class $k = 1:K$
	* Disadvantage: even if we start with **balanced** $K$ classes, we will face **imbalanced** aggregated classes, which will require probability threshold tuning
