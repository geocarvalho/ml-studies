
x- Logistic regression
	* Classification and representation
	* Logistic regression model
	* Multiclass classification
- Regularization
	* Solving the problem of overfitting
## 1. Regression

### 1.1. Classification and representation

#### 1.1.1 Classification
To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification** **problem** in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, $y\in\{0,1\}$. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the label for the training example.

#### 1.1.2. Hypothesis representation
We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for $h{_\theta}(x)$ to take values larger than 1 or smaller than 0 when we know that $y\in \{0, 1\}$. To fix this, let’s change the form for our hypotheses $h_{\theta} (x)$ to satisfy $0 \leq h_{\theta} (x) \leq 1$. This is accomplished by plugging \theta^TxθTx into the Logistic Function.

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

$h_{\theta} = g(\theta^T x)$

$z = \theta^T x$

$g(z) = \frac{1}{1 + e^{-z}}$

The following image shows us what the sigmoid function looks like:
![[Pasted image 20220717112327.png]]

The function $g(z)$, shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

$h_{\theta(x)}$ will give us the **probability** that our output is 1. For example, $h_{\theta}(x) = 0.7$ gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

$h_{\theta}(x) = P(y=1 | x;0) = 1 - P(y=0 | x;0)$

$P(y=10 | x;0) + P(y=1 | x;0) = 1$

#### 1.1.3. Decision Boundary

- In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
$h_{\theta}(x) \geq 0.5 \to y=1$
$h_{\theta}(x) < 0.5 \to y=0$

- The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:
$g(z) \geq 0.5$
$when \; z \geq 0$

- Remember:
$z=0, e^0=1 \implies g(z)=1/2$
$z \to \infty, e^{-\infty} \to 0 \implies g(z)=1$
$z \to -\infty, e^{\infty} \to \infty \implies g(z)=0$

- So if our input to g is $\theta^T X$, then that means:
$h_{\theta}(x) = g(\theta^T X) \geq 0.5$
$when \; \theta^T X \geq 0$

- From these statements we can now say:
$\theta^T X \geq 0 \implies y=1$
$\theta^T X < 0 \implies y=0$

- The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function. Example:

$$\begin{bmatrix} 5 \\ -1 \\ 0 \end{bmatrix}$$
$$y=1 \; if \; 5+(-1)x_1 + 0x_2 \geq 0$$
$$5-x_1 \geq 0$$
$$-x_1 \geq -5$$
$$x_1 \leq 5$$

In this case, our decision boundary is a straight vertical line placed on the graph where $x_1 = 5$, and everything to the left of that denotes y = 1, while everything to the right denotes y = 0.

Again, the input to the sigmoid function g(z) (e.g. $\theta^T X$) doesn't need to be linear, and could be a function that describes a circle (e.g. $z = \theta_0 + \theta_1 x_1^2 +\theta_2 x_2^2$) or any shape to fit our data.

---
### 1.2. Logistic regression model

#### 1.2.1. Cost function
We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:
$$J(\theta) = \frac{1}{m} \sum^m_{i=1}Cost(h_{\theta}(x^{(i)}), y^{(i)})$$
$$Cost(h_{\theta}(x^{(i)}), y^{(i)}) = -log(h_{\theta}(x)) \; if \; y=1$$
$$Cost(h_{\theta}(x^{(i)}), y^{(i)}) = -log(1 - h_{\theta}(x)) \; if \; y=1$$

When y = 1, we get the following plot for $J(\theta)$ vs $h_{\theta}$:
![[Pasted image 20220720224925.png]]

Similarly, when y = 0, we get the following plot for $J(\theta)$ vs $h_{\theta}$:
![[Pasted image 20220720224959.png]]

$$Cost(h_{\theta}(x), y = 0 \; if \; h_{\theta}(x) = y$$
$$Cost(h_{\theta}(x), y \to \infty \; if \; y=0 \; and \; h_{\theta}(x) \to 1$$
$$Cost(h_{\theta}(x), y \to \infty \; if \; y=1 \; and \; h_{\theta}(x) \to 0$$

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression.

#### 1.2.2. Simplified cost function and gradient descent
We can compress our cost function's two conditional cases into one case:
$$Cost(h_{theta}(x), y) = -ylog(h_{theta}(x))-(1-y)log(1-h_{theta}(x))$$

Notice that when y is equal to 1, then the second term $(1-y)\log(1-h_\theta(x))$ will be zero and will not affect the result. If y is equal to 0, then the first term $-y \log(h_\theta(x))$ will be zero and will not affect the result.

We can fully write out our entire cost function as follows:
$$J(\theta) = -\frac{1}{m}\sum^m_{i=1}[y^{(i)}log(h_{theta}(x^{(i)}))+(1-y^{(i)})log(1-h_{theta}(x^{(i)}))]$$
A vectorized implementation is:
$$h=g(X\theta)$$
$$J(\theta) = \frac{1}{m} \dot (-y^T log(h) - (1-y)^T log(1-h))$$

- Gradient descent
Remember that the general form of gradient descent is:
$$Repeat\{$$
$$\theta_j:= \theta_j - \frac{\delta}{m} \sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_j$$
$$\}$$

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:
$$\theta:= \theta - \frac{\delta}{m}X^T(g(X\theta)-\overrightarrow{y})$$

> Suppose you are running gradient descent to fit a logistic regression model with parameter $\theta\in\mathbb{R}^{n+1}θ∈Rn+1$. What is a reasonable way to make sure the learning rate \alphaα is set properly and that gradient descent is running correctly?
> - Plot $J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}$ as a function of the number of iterations and make sure $J(\theta)$ is decreasing on every iteration.

#### 1.2.3. Advanced optimization
"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to opti  descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value θ:
$$J(\theta)$$
$$\frac{\partial}{\partial \theta_j}J(\theta)$$

We can write a single function that returns both of these:
```octave
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()". (Note: the value for MaxIter should be an integer, not a character string - errata in the video at 7:30)
```octave
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

### 1.3. Multiclass classification

#### 1.3.1. One-vs-all
Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.
$$y \in {0,1,...,n}$$
$$h^{(0)}_\theta (x) = P(y=0|x;\theta)$$
$$h^{(1)}_\theta (x) = P(y=1|x;\theta)$$
$$...$$
$$$h^{(n)}_\theta (x) = P(y=n|x;\theta)$$$$prediction = max_i(h^{(i)}_\theta(x))$$
We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

The following image shows how one could classify 3 classes:
![[Pasted image 20220726224918.png]]

**To summarize:**
- Train a logistic regression classifier $h_\theta(x)$ for each class to predict the probability that y = i.
- To make a prediction on a new x, pick the class that maximizes $h_\theta  (x)$

### 1.4. Solving the problem of overfitting
#### 1.4.1. The problem of overfitting
Consider the problem of predicting y from x ∈ R. The leftmost figure below shows the result of fitting a $y = \theta_0 + \theta_1x$ to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.
![[Pasted image 20221120155049.png]]
Instead, if we had added an extra feature $x^2$ , and fit $y = \theta_0 + \theta_1x + \theta_2x^2$ , then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a $5^{th}$ order polynomial $y = \sum_{j=0} ^5 \theta_j x^j$. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of **underfitting**—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of **overfitting**.

**Underfitting**, or **high bias**, is when the form of our hypothesis function $h$ maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, **overfitting**, or **high variance**, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of **overfitting**:
1) Reduce the number of features:
	-   Manually select which features to keep.
	-   Use a model selection algorithm (studied later in the course).

2) Regularization
	-   Keep all the features, but reduce the magnitude of parameters $\theta_j$.
	-   *Regularization works well when we have a lot of slightly useful features*.

#### 1.4.2. Cost function
**Note:** 5:18 - There is a typo. It should be $\sum_{j=1}^{n} \theta _j ^2$ instead of $\sum_{i=1}^{n} \theta _j ^2$

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:
$$\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$$

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$ . Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:
$$min_\theta\ \dfrac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\cdot\theta_3^2 + 1000\cdot\theta_4^2$$

We've added two extra terms at the end to inflate the cost of $\theta_3$ and $\theta_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near zero. This will in turn greatly reduce the values of $\theta_3x^3$ and $\theta_4x^4$ in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms $\theta_3x^3$ and $\theta_4x^4$.
![[Pasted image 20221120162702.png]]

We could also regularize all of our theta parameters in a single summation as:
$$min_\theta\ \dfrac{1}{2m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2$$

The λ, or lambda, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if $\lambda = 0$  is too small ?

#### 1.4.3. Regularized linear regression
**Note:** 8:43 - It is said that X is non-invertible if $m \leq n$. The correct statement should be that X is non-invertible if $m < n$, and may be non-invertible if $m = n$.

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

##### 1.4.3.1. Gradient Descent
We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

$$Repeat:$$

$$\theta_0:= \theta_0 - \alpha \frac{1}{m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_0$$
$$\theta_j:= \theta_j - \alpha [(\frac{1}{m} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j)+\frac{\lambda}{m}\theta_j] \ \ \ \ \ j \in \{1,2,...,n\}$$

The term $\frac{\lambda}{m}\theta_j$ performs our regularization. With some manipulation our update rule can also be represented as:

$$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

The first term in the above equation, $1 - \alpha\frac{\lambda}{m}$ will always be less than 1. Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update. Notice that the second term is now exactly the same as it was before.

##### 1.4.1.3.2. Normal equation
Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

$$\theta = (X^TX+\lambda \cdot L)^{-1} X^Ty$$
$$Where \ L = \begin{bmatrix} 
	0 & & & & \\
	 & 1 & & & \\
	 & & 1 & & \\
	 & & & \ddots &  \\
	 & & & & 1
\end{bmatrix}$$

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number λ.

Recall that if m < n, then $X^TX$ is non-invertible. However, when we add the term λ⋅L, then $X^TX$ + λ⋅L becomes invertible.

#### 1.4.4. Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:
![[Pasted image 20221120200007.png]]

##### 1.4.4.1. Cost Function

Recall that our cost function for logistic regression was:

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) \large]$$

We can regularize this equation by adding a term to the end:

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

The second sum, $\sum_{j=1}^n \theta_j^2$ **means to explicitly exclude** the bias term, $\theta_0$. I.e. the θ vector is indexed from 0 to n (holding n+1 values, $\theta_0$ through $\theta_n$), and this sum explicitly skips $\theta_0$, by running from 1 to n, skipping 0. Thus, when computing the equation, we should continuously update the two following equations:
![[Pasted image 20221120200235.png]]

## 2. Referencias
https://github.com/atinesh-s/Coursera-Machine-Learning-Stanford/tree/master/Week%203