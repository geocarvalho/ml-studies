Status:
Tags:
Links:
---
# Week 1

## 1.1 Welcome to ML
## 1.2 Welcome
- Grew out of work in AI
- New capability for computers
### Examples:
- Database mining: large datasets from growth of automation/web
> E.g. Web click data, medical records, computational biology, engineering.
- Applications can't program by hand.
> E.g. Autonomous helicopter, handwriting recognition, most of NLP, computer vision.
- Self-customizing programs
> E.g. Amazon, Netflix product recommendations
- Understanding human learning (brain, real AI)

## 1.3 What is ML
- Arthur Samuel (1959) - Chess player program.
> "Field of study that gives computers the ability to learn without being explicitly programmed."
- Tome Mitchell (1998) - Well-posed learning problem.
> "A computer program is said to *learn* from experience E with respect to some task T and come performance measure P, if its performance on T, as measured by P, improves with experience E".

**Q:** Suppose your email program watches which emails you do or do not mark as spam, and based on that learns how to better filter spams. What is the task T in this setting?
**A:** Classify emails as spam or not spam
- The experience E is "watching you label as spam or not spam", the performance measure P is "the number (or fraction) of emails correctly classified as spam/not spam".

### ML algorithms:
- Supervised learning: teach the computer how to do something.
- Unsupervised learning: let it learn by itself.
- Others: reinforcement learning, recommender systems.
- Also talk about: pratical advice for applying learning algorithms.

## 1.4 Supervised learning
- For every example we have the label, the algorithm create "right answers".
- **Regression**: Predict continuous valued output (Given a picture of a person, we have to predict their age on the basis of the given picture). 
- **Classification**: Predict a class valued as output (Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.).

**Q:** You’re running a company, and you want to develop learning algorithms to address each of two problems. Problem 1: You have a large inventory of identical items. You want to predict how many of these items will sell over the next 3 months.
Problem 2: You’d like software to examine individual customer accounts, and for each account decide if it has been hacked/compromised. Should you treat these as classification or as regression problems?

**A:** Treat problem 1 as a regression problem, problem 2 as a classification problem.

## 1.5 Unsupervised learning
- The dataset has no label, and we have to find some structure (clusters).
E.g. Google's news, genomics microarray, organize computing clusters, social network analysis, market segmentation, astronomical data analysis.

- Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
- Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)).

**Q:** Of the following examples, which would you address using an unsupervised learning algorithm? (Check all that apply.)
**A:**   Given a set of news articles found on the web, group them into sets of articles about the same stories. Given a database of customer data, automatically discover market segments and group customers into different market segments.

## 1.6 [QUIZ] Introduction
1.   A computer program is said to learn from experience E with respect to some task T and some performance measure P if its performance on T, as measured by P, improves with experience E. Suppose we feed a learning algorithm a lot of historical weather data, and have it learn to predict weather. What would be a reasonable choice for P?
> The probability of it correctly predicting a future date's weather.

2. Suppose you are working on weather prediction, and use a learning algorithm to predict tomorrow's temperature (in degrees Centigrade/Fahrenheit). Would you treat this as a classification or a regression problem?
> Regression

3. Suppose you are working on stock market prediction, and you would like to predict the price of a particular stock tomorrow (measured in dollars). You want to use a learning algorithm for this. Would you treat this as a classification or a regression problem?
> Regression

4. Some of the problems below are best addressed using a supervised learning algorithm, and the others with an unsupervised learning algorithm. Which of the following would you apply supervised learning to? (Select all that apply.) In each case, assume some appropriate dataset is available for your algorithm to learn from.
> In farming, given data on crop yields over the last 50 years, learn to predict next year's crop yields. 
> Examine a web page, and classify whether the content on the web page should be considered "child friendly" (e.g., non-pornographic, etc.) or "adult."

5. Which of these is a reasonable definition of machine learning?
> Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.

## 1.7 Model representation
- Training set of housing prices
	m = number of training examples.
	x's = input variable / features.
	y's = output variable / target variable.
	(x,y) = one training example.
	(x^i,y^i) = i^{th} training example.
Our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.
![[Pasted image 20220402152833.png]]
- How to represent h?
$h_{\theta}(x) = \theta_0 + \theta_1 * x$
> Linear regression with one variable (univariate)
- Regression problem: When the target variable that we’re trying to predict is continuous, such as in our housing example.
- Classification problem: When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say).

## 1.8 Cost function
- We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.
- Choose $\theta_0$ and $\theta_1$ so that $h_0(x)$ is close to $y$ for our training examples $(x,y)$.
$J(\theta_0, \theta_1) = \frac{1}{2m}\sum^{m}_{i=1}(h_{\theta}(x^i)-y^i)^2$
> minimize $\theta_0,\theta_1$ for the cost function $J(\theta_0,\theta_1)$ (squared error function).
- To break it apart, it is $\frac{1}{2}\bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta (x_{i}) - y_{i}$ , or the difference between the predicted value and the actual value.
- This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\left(\frac{1}{2}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term. The following image summarizes what the cost function does:
![[Pasted image 20220402160607.png]]

## 1.9 Cost function - intuition 1
- If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make a straight line (defined by $h_\theta(x)$) which passes through these scattered data points.
- Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_0, \theta_1)$ will be 0. The following example shows the ideal situation where we have a cost function of 0.
 ![[Pasted image 20220409213155.png]]
- When $\theta_1 = 1$, we get a slope of 1 which goes through every single data point in our model. Conversely, when $\theta_1 = 0.5$, we see the vertical distance from our fit to the data points increase.
![[Pasted image 20220409213258.png]]
- This increases our cost function to 0.58. Plotting several other points yields to the following graph:
![[Pasted image 20220409213319.png]]
- Thus as a goal, we should try to minimize the cost function. In this case, $\theta_1 = 1$ is our global minimum.

## 1.10 Cost function - intuition 2
- A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.
![[Pasted image 20220409214336.png]]
- Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for $J(\theta_0,\theta_1)$ and as a result, they are found along the same line. The circled x displays the value of the cost function for the graph on the left when $\theta_0= 800$ and $\theta_1 = -0.15$. Taking another h(x) and plotting its contour plot, one gets the following graphs:
![[Pasted image 20220409214510.png]]
- When $\theta_0 = 360$ and $\theta_1 = 0$, the value of $J(\theta_0,\theta_1)$) in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data.
![[Pasted image 20220409214602.png]]
- The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_1$ and $\theta_0$ tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

## 1.11 [[Gradient descent]]
- So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.
- Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.
- We put $\theta_0$ on the x axis and $\theta_1$ on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.
![[Pasted image 20220409222314.png]]
- We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.
- The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate.
- For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

The gradient descent algorithm is, repeat until convergence:
$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J (\theta_0, \theta_1)$

> where j=0,1 represents the feature index number.

- At each iteration j, one should simultaneously update the parameters $\theta_1$, $\theta_2$,...,$\theta_n$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.
![[Pasted image 20220409223249.png]]

## 1.12 Gradient descent intuition
- In this video we explored the scenario where we used one parameter $\theta_1$ and plotted its cost function to implement a gradient descent. Our formula for a single parameter was , repeat until convergence:
$\theta_1:= \theta_1 - \alpha \frac{d}{d\theta_1}J(\theta_1)$
- Regardless of the slope's sign for $\frac{d}{d\theta_1} J(\theta_1)$, $\theta_1$ eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of $\theta_1$ increases and when it is positive, the value of $\theta_1$ decreases.
![[Pasted image 20220410000431.png]]
- On a side note, we should adjust our parameter $\alpha$ to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong
![[Pasted image 20220410000502.png]]
#### How does gradient descent converge with a fixed step size $\alpha$?
-  The intuition behind the convergence is that $\frac{d}{d\theta_1} J(\theta_1)$ approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:
$\theta_1:=\theta_1-\alpha * 0$
![[Pasted image 20220410000646.png]]

## 1.12 Gradient descent for linear regression
**Note:** [At 6:15 "h(x) = -900 - 0.1x" should be "h(x) = 900 - 0.1x"]
- When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to :

repeat until convergence:

$\theta_0:=\theta_0−\alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x_i)−y_i)$

$\theta_1:=\theta_1−\alpha \frac{1}{m} \sum_{i=1}^m ((h_\theta(x_i)−y_i)x_i)$
> where m is the size of the training set, $\theta_0$ a constant that will be changing simultaneously with $\theta_1$ and $x_{i}, y_{i}$ are values of the given training set (data).
- Note that we have separated out the two cases for $\theta_j$ into separate equations for $\theta_0$ and $\theta_1$; and that for $\theta_1$ we are multiplying $x_{i}$ at the end due to the derivative. The following is a derivation of $\frac {\partial}{\partial \theta_j}J(\theta)$ for a single example :
![[Pasted image 20220410013221.png]]
- The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

- So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.
![[Pasted image 20220410013344.png]]
- The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.

## Matrices and Vectors
Matrices are 2-dimensional arrays:
$[abcdefghijkl]$

The above matrix has four rows and three columns, so it is a 4 x 3 matrix. A vector is a matrix with one column and many rows:
$[wxyz]$

So vectors are a subset of matrices. The above vector is a 4 x 1 matrix.
**Notation and terms**:
- $A_{ij}$ refers to the element in the ith row and jth column of matrix A.
- A vector with 'n' rows is referred to as an 'n'-dimensional vector. 
- $v_i$ refers to the element in the ith row of the vector.
- In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed. 
- Matrices are usually denoted by uppercase names while vectors are lowercase. 
- "Scalar" means that an object is a single value, not a vector or matrix. 
- $\mathbb{R}$ refers to the set of scalar real numbers.
- $\mathbb{R^n}$ refers to the set of n-dimensional vectors of real numbers.
Run the cell bellow to get familiar with the commands in Octave/Matlab. Feel free to create matrices and vectors and try out different things.
```Octave
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector
v = [1;2;3]

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```

## Addition and scalar multiplication
Addition and subtraction are **element-wise**, so you simply add or subtract each corresponding element:
$[acbd] + [wyxz] = [a+wc+yb+xd+z]$

Subtracting Matrices:
$[acbd] - [wyxz] = [a−wc−yb−xd−z]$

To add or subtract two matrices, their dimensions must be **the same**. In scalar multiplication, we simply multiply every element by the scalar value:
$[acbd] * x = [a∗xc∗xb∗xd∗x]$

In scalar division, we simply divide every element by the scalar value:
$[acbd]/ x = [a/xc/xb/xd/x]$

Experiment below with the Octave/Matlab commands for matrix addition and scalar multiplication. Feel free to try out different commands. Try to write out your answers for each command before running the cell below.
```Octave
% Initialize matrix A and B
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s
s = 2

% See how element-wise addition works
add_AB = A + B

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s
```

## Matrix vector multiplication
We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.
$[xy] = [abcdef]∗[xy]=[a∗x+b∗yc∗x+d∗ye∗x+f∗y]$

The result is a **vector**. The number of **columns** of the matrix must equal the number of **rows** of the vector. An **m x n matrix** multiplied by an **n x 1 vector** results in an **m x 1 vector**. 

Below is an example of a matrix-vector multiplication. Make sure you understand how the multiplication works. Feel free to try different matrix-vector multiplications.
```Octave
% Initialize matrix A
A = [1, 2, 3; 4, 5, 6;7, 8, 9]

% Initialize vector v
v = [1; 1; 1]

% Multiply A * v
Av = A * v
```

## Matrix matrix multiplication
 We multiply two matrices by breaking it into several vector multiplications and concatenating the result.
$[a​bc​de​f​]∗[w​xy​z​] = [a∗w+b∗y​a∗x+b∗zc∗w+d∗y​c∗x+d∗ze∗w+f∗y​e∗x+f∗z​]$

An **m x n matrix** multiplied by an **n x o matrix** results in an **m x o** matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix. 

> To multiply two matrices, the number of **columns** of the first matrix must equal the number of **rows** of the second matrix.

For example:
```Octave
% Initialize a 3 by 2 matrix
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix
B = [1; 2]

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1)
mult_AB = A*B

% Make sure you understand why we got that result
```

## Matrix multiplication properties
- Matrices are not commutative: $A∗B \neq B∗A$
- Matrices are associative: $(A∗B)∗C = A∗(B∗C)$

The **identity matrix**, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$[1​0​00​1​00​0​1​]$

When multiplying the identity matrix after some matrix $(A∗I)$, the square identity matrix's dimension should match the other matrix's **columns**. When multiplying the identity matrix before some other matrix $(I∗A)$, the square identity matrix's dimension should match the other matrix's **rows**.
```Octave
% Initialize random matrices A and B
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]
% What happens when we multiply I*A ?
IA = I*A

% How about A*I ?
AI = A*I

% Compute A*B
AB = A*B

% Is it equal to B*A?
BA = B*A

% Note that IA = AI but AB != BA
```

## Inverse and transpose
The **inverse** of a matrix A is denoted $A^{-1}$. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the $pinv(A)$ function and in Matlab with the $inv(A)$ function. Matrices that don't have an inverse are *singular* or *degenerate*.

The **transposition** of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in matlab with the transpose(A) function or A':
$A =[abcdef]$

$A^T=[acebdf]$

In other words:
$A_{ij} = A^T_{ji}$

```Octave
% Initialize matrix A
A = [1,2,0;0,5,6;7,0,9]

% Transpose A
A_trans = A'

% Take the inverse of A
A_inv = inv(A)

% What is A^(-1)*A?
A_invA = inv(A)*A
```

---
References:
