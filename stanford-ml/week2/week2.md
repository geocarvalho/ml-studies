# Week 2 
- Multivariate linear regression
- Computing parameters analytically
- Octave tutorial

## 1. Multivariate Linear regression
### 1.1. Multiple features

**Note:** [7:25 - $\theta^T$ is a 1 by (n+1) matrix and not an (n+1) by 1 matrix]

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

- $x^{(i)}_j$=value of feature j in the $i{th}$ training example
- $x^{(i)}$=the input (features) of the ith training example
- $m$=the number of training examples
- $n$=the number of features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$

In order to develop intuition about this function, we can think about $\theta_0$ as the basic price of a house, $\theta_1$ as the price per square meter, $\theta_2$ as the price per floor, etc. $x_1$ will be the number of square meters in the house, $x_2$ the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

$h_{\theta(x)} = [\theta_0 \; \theta_1 \cdots \theta_n] \begin{bmatrix}
x_0 \\ x_1 \\ \vdots \\ x_n
\end{bmatrix} = \theta^T x$

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume $x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$. This allows us to do matrix operations with theta and x. Hence making the two vectors '$\theta$' and $x^{(i)}$ match each other element-wise (that is, have the same number of elements: n+1).]

### 1.2. Gradient descent for multiple variables
The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

$repeat \; until \; convergece:\{$

$\theta_0:= \theta_0 - \alpha \frac{1}{m} \sum^m_{i=1}(h_{\theta}(x^{(i)}) - y{(i)})\cdot x_0^{(i)}$

$\theta_1:= \theta_1 - \alpha \frac{1}{m} \sum^m_{i=1}(h_{\theta}(x^{(i)}) - y{(i)})\cdot x_1^{(i)}$

$\theta_2:= \theta_2 - \alpha \frac{1}{m} \sum^m_{i=1}(h_{\theta}(x^{(i)}) - y{(i)})\cdot x_2^{(i)}$

$\cdots \}$

In other words:

$repeat \; until \; convergece:\{$

$\theta_j:= \theta_j - \alpha \frac{1}{m} \sum^m_{i=1}(h_{\theta}(x^{(i)}) - y{(i)})\cdot x_j^{(i)}$

$for \; j := 0...n$

The following image compares gradient descent with one variable to gradient descent with multiple variables:
![[Pasted image 20220619200115.png]]

### 1.3. Gradient descent in practice 1 - feature scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

$−1 ≤ x_{(i)}$

or

$−0.5 ≤ x_{(i)}$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

$x_i := \dfrac{x_i - \mu_i}{s_i}$

Where $μ_i$ is the **average** of all the values for feature (i) and $s_i$ is the range of values (max - min), or $s_i$ is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if $x_i$ represents housing prices with a range of 100 to 2000 and a mean value of 1000, then, $x_i := \dfrac{price-1000}{1900}$.

### 1.4. Gradient descent in practice 2 - learning rate
**Debugging gradient descent.** Make a plot with _number of iterations_ on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

**Automatic convergence test.** Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as $10^{−3}$. However in practice it's difficult to choose this threshold value.

![[Pasted image 20220621085527.png]]

It has been proven that if learning rate α is sufficiently small, then J(θ) will decrease on every iteration.

![[Pasted image 20220621085541.png]]

To summarize:

If $\alpha$ is too small: slow convergence.

If $\alpha$ is too large: may not decrease on every iteration and thus may not converge.

### 1.5. Features and polynomial regression
We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine $x_1$ and $x_2$ into a new feature $x_3$ by taking $x_1$ $x_2$.

**Polynomial Regression**

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is $h_\theta(x) = \theta_0 + \theta_1 x_1$ then we can create additional features based on $x_1$, to get the quadratic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$ or the cubic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$

In the cubic version, we have created new features $x_2$ and $x_3$ where $x_2 = x_1^2$ and $x_3 = x_1^3$.

To make it a square root function, we could do: $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

eg. if $x_1$ has range 1 - 1000 then range of $x_1^2$ becomes 1 - 1000000 and that of $x_1^3$ becomes 1 - 1000000000

## 2. Computing parameters analytically
### 2.1. Normal equation
**Note:** [8:00 to 8:44 - The design matrix X (in the bottom right side of the slide) given in the example should have elements x with subscript 1 and superscripts varying from 1 to m because for all m training sets there are only 2 features $x_0$ and $x_1$. 12:56 - The X matrix is m by (n+1) and NOT n by n. ]

Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:

$\theta = (X^T X)^{-1}X^T y$

![[Pasted image 20220621212429.png]]

There is **no need** to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

```markdown
| Gradient Descent           | Normal Equation                               |
|----------------------------|-----------------------------------------------|
| Need to choose alpha       | No need to choose alpha                       |
| Needs many iterations      | No need to iterate                            |
| O ($kn^2$)                 | O (n^3$), need to calculate inverse of $X^TX$ |
| Works well when n is large | Slow if n is very large       
```

With the normal equation, computing the inversion has complexity $\mathcal{O}(n^3)$. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

### 2.2. Normal equation noninvertibility
When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of $\theta$ even if $X^TX$ is not invertible.

If $X^TX$ is **noninvertible,** the common causes might be having :

-   Redundant features, where two features are very closely related (i.e. they are linearly dependent)
    
-   Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).
 
Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

### 2.3. Programming tips from mentors
#### Subject: Confused about "h(x) = theta' * x" vs. "h(x) = X * theta?"
The lectures and exercise PDF files are based on Prof. Ng's feeling that novice programmers will adapt to for-loop techniques more readily than vectorized methods. So the videos (and PDF files) are organized toward processing one training example at a time. The course uses column vectors (in most cases), so h (a scalar for one training example) is theta' * x.

Lower-case x typically indicates a single training example.

The more efficient vectorized techniques always use X as a matrix of all training examples, with each example as a row, and the features as columns. That makes X have dimensions of (m x n). where m is the number of training examples. This leaves us with h (a vector of all the hypothesis values for the entire training set) as X * theta, with dimensions of (m x 1).

X (as a matrix of all training examples) is denoted as upper-case X.

Throughout this course, dimensional analysis is your friend.

#### Subject: Tips from the Mentors: submit problems and fixing program errors

#### **Text:**
This post contains some frequently-used tips about the course, and to help get your programs working correctly.

#### **The Most Important Tip:**
Search the forum before posting a new question. If you've got a question, the chances are that someone else has already posted it, and received an answer. Save time for yourself and the Forum users by searching for topics before posting a new one.

#### **Running your scripts:**
At the Octave/Matlab command line, you do not need to include the ".m" portion of the script file name. If you include the ".m", you'll get an error message about an invalid indexing operation. So, run the Exercise 1 script by typing just "ex1" at the command line.

You also do not need to include parenthesis () when using the submit script. Just type "submit".

You cannot execute your functions by simply typing the name. All of the functions you will work on require a set of parameter values, enter between a set of parenthesis. Your three methods of testing your code are:

1 - use an exercise script, such as "ex1"

2 - use a Unit Test (see below) where you type-in the entire command line including the parameters.

3 - use the submit script.

#### **Making the grader happy:**
The submit grader uses a different test case than what is in the PDF file. These test cases use a different size of data set and are more sensitive to small errors than the ex test cases. Your code must work correctly with any size of data set.

Your functions must handle the general case. This means:

- You should avoid using hard-coded array indexes.

- You should avoid having fixed-length arrays and matrices.

It is very common for students to think that getting the same answer as listed in the PDF file means they should get full credit from the grader. This is a false hope. The PDF file is just one test case. The grader uses a different test case.

Also, the grader does not like your code to send any additional outputs to the workspace. So, every line of code should end with a semicolon.

#### **Getting Help:**
When you want help from the Forum community, please use this two-step procedure:

1 - Search the Forum for keywords that relate to your problem. Searching by the function name is a good start.

2 - If you don't find a suitable thread, then do this:

2a - Find the unit tests for that exercise (see below), and run the appropriate test. Attempt to debug your code.

2b - Take a screen capture of your whole console workspace (including the command line), and post it to the forum, along with any other useful information (computer type, Octave/Matlab version, other tests you've tried, etc).

#### Debugging:
If your code runs but gives the wrong answers, you can insert a "keyboard" command in your script, just before the function ends. This will cause the program to exit to the debugger, so you can inspect all your variables from the command line. This often is very helpful in analysing math errors, or trying out what commands to use to implement your function.

There are additional test cases and tutorials listed in pinned threads under "All Course Discussions". The test cases are especially helpful in debugging in situations where you get the expected output in ex but get no points or an error when submitting.

#### Unit Tests:
Each programming assignment has a "Discussions" area in the Forum. In this section you can often find "unit tests". These are additional test cases, which give you a command to type, and provides the expected results. It is always a good idea to test your functions using the unit tests before submitting to the grader.

If you run a unit test and do not get the correct results, you can most easily get help on the forums by posting a screen capture of your workspace - including the command line you entered, and the results.

#### Having trouble submitting your work to the grader?:
- This section will need to be supplemented with info appropriate to the new submission system. If you run the submit script and get a message that your identity can't be verified, be sure that you have logged-in using your Coursera account email and your Programming Assignment submission password.

- If you get the message "submit undefined", first check that you are in the working directory where you extracted the files from the ZIP archive. Use "cd" to get there if necessary.

- If the "submit undefined" error persists, or any other "function undefined" messages appear, try using the "addpath(pwd)" command to add your present working directory (pwd) to the Octave execution path.

-If the submit script crashes with an error message, please see the thread "Mentor tips for submitting your work" under "All Course Discussions".

-The submit script does not ask for what part of the exercise you want to submit. It automatically grades any function you have modified.

#### Found some errata in the course materials?
This course material has been used for many previous sessions. Most likely all of the errata has been discovered, and it's all documented in the 'Errata' section under 'Supplementary Materials'. Please check there before posting errata to the Forum.

Error messages with fmincg()

The "short-circuit" warnings are due to use a change in the syntax for conditional expressions (| and & vs || and &&) in the newer versions of Matlab. You can edit the fmincg.m file and the warnings may be resolved.

#### Warning messages about "automatic broadcasting"?
See [this](https://www.gnu.org/software/octave/doc/interpreter/Broadcasting.html) link for info.

#### Warnings about "divide by zero"
These are normal in some of the exercises, and do not represent a problem in your function. You can ignore them - Octave senses the issue and substitutes a +Inf or -Inf value so your program continues to execute.

## 3. Octave/Matlab tutorial
### 3.1. Basic operations
```Octave
2^6
1 == 2 % false
1 ~= 2
1&&0 % AND
1||0 %OR
xor(1,0)
PS1('>> ') % change prompt
a=3; $ semicolon supressing output
b='hi';
c=(3>=1);
d=pi;
disp(a);
disp(sprintf('2 decimals: %0.2f', d))
disp(sprintf('6 decimals: %0.6', d))
format long
d
format short
d
A = [1 2; 3 4; 5 6]
v = [1 2 3]
v = [1; 2; 3]
v = 1:0.1:2
v = 1:6
ones(2,3)
C = 2*ones(2,3)
w = ones(1,3)
w = zeros(1,3)
w = rand(1,3)
w = rand(3,3) %between 0 and 1
w = randn(1,3) %gaussian distribution
w = -6 + sqrt(10) *(randn(1, 10000))
hist(w)
hist(w,50)
I = eye(4)
I = eye(6)
help eye
help rand
help 
```

### 3.2. Moving data around
```Octave
A = [1 2; 3 4; 5 6]
sz = size(A)
size(A, 1) % rows n
size(A, 2) % column n
length(A)
length([1;2;3;4;5])
pwd
ls
load featuresX.dat
load priceY.dat
load('featuresX.dat')
who %show variables available
size(featuresX)
size(priceY)
whos %plus details for variables
clear featuresX %remove variable
v = priceY(1:10)
save hello.mat v; %save variable v inside hello.mat file
save hello.txt v -ascii %save as text (ASCII)
A(3,2)
A(2,:) %every element along that row/column
A(:,2)
A([1 3], :) %select elements from 1 and 3 row
A(:,2) = [10; 11; 12] %replace 2 column
A = [A, [100; 101; 102]] %append another column
A(:) %put all elements of A into a single vector
A = [1 2; 3 4; 5 6]
B =[11 12; 13 14; 15 16]
C = [A B] % same as C = [A, B]
C = [A; B]
```

### 3.3. Computing on data
```Octave
A=[1 2; 3 4; 5 6]
B=[11 12; 13 14; 15 16]
C=[1 1; 22]
A * B
A .* B %element-wise multiplication
A .^ B %element-wise squaring
v = [1; 2; 3]
1 ./ v %element-wise reciprocal of v
log(v) %element-wise logarithm of v
exp(v) %element-wise exponentiation
abs(v) %element-wise absolute value
-v %-1*v
v + 1
A' %transpose
a=[1 15 2 0.5]
max(a)
val, ind=max(a)
max(A)
a < 3 %element-wise comparison
find(a < 3)
A = magic(3)
[r, c] = find(A >= 7)
sum(a)
prod(a)
floor(a)
ceil(a)
rand(3)
max(rand(3), rand(3))
max(A,[],1)
max(A,[],2)
max(max(A))
A(:)
max(A(:))
A=magic(9)
sum(A,1)
sum(A,2)
A .* eye(9)
sum(sum(A .* eye(9)))
sum(sum(A .* flipud(eye(9))))
A=magic(3)
temp=pinv(A)
temp * A %identity matrix
```

### 3.4. Plotting data
```Octave
t=[0:0.1:0.98];
y1=sin(2*pi*4*t);
plot(t,y1);
y2=cos(2*pi*4*t);
plot(t,y2);

plot(t,y1);
hold on;
plot(t, y1, 'r');
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my plot')
cd 'path/to/dir'
print -dpng 'myPlot.png'
close

figure(1); plot(t,y1);
figure(2); plot(t,y2;
subplot(1,2,1);
plot(t,y1);
subplot(1,2,2);
plot(t,y2);
axis([0.5 1 -1 1])
clf;

A = magix(5)
imagesc(A)
imagesc(A), colorbar, colormap gray;
A(1,2)
A(4,5)
imagesc(magic(15)), colorbar, colormap gray;

a=1, b=2, c=3
a=1; b=2; c=3
```

### 3.4. Control statements: for, while, if statement
```Octave
v=zeros(10,1)
for i=1:10,
	v(i) = 2^i;
end;
v

indices=1:10;
for i=indices,
	disp(i);
end;

i=1;
while 1 <=5;
	v(i) = 100;
	i = i+1;
end;
v

while true,
	v(i) = 999;
	i = i+1;
	if i == 6,
		break
	end;
end;
v

v(1) = 2;
if v(1)==1,
	disp('The value is one');
elseif v(1) == 2,
	disp('The value is two');
else
	disp('The value is not one or two');
end;
```

- Define function using file
```Octave
function y = squareThisNumber(x)
y=x^2
```

- Run function
```Octave
cd 'path/to/dir/'
squareThisNumber(5)
```

- Octave search path (advanced/optional)
```Octave
addpath('path/to/dir')
cd 'another/path'
squareThisNumber(5)
pwd
```

- Function with multiple values
```Octave
function [y1,y2] = squareAndCubeThisNumber(x)

y1=x^2;
y2=x^3;
```

```Octave
[a,b] = squareAndCubeThisNumber(5);
a
b
```

- Define a function to compute the cost function $J(\theta)$
```Octave
function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing our training examples.
%y is the class labels

m = size(X, 1); % number of training examples
predictions = X*theta; % predictions of hypothesis on all m
sqrErrors = (predictions-y).^2; % squared errors
J = 1/(2*m) * sum(sqrErrors);
```

```Octave
X = [1 1; 1 2; 1 3]
y = [1; 2; 3]
theta = [0;1];

j = costFunctionJ(X,y,theta)

theta = [0;0];
j = costFunctionJ(X,y,theta)
(1^2 + 2^2 + 3^2)/ (2*3)
```

### 3.5. Vectorization
- Unvectorized implementation
```Octave
prediction = 0.0;
for j = 1:n+1,
	prediction = prediction + theta(j) * x(j)
end;
```
- Vectorized implementation
```Octave
prediction = theta' * x;
```

### 3.6. 
```Octave
%1
A = [1 2; 3 4; 5 6];
B = [1 2 3; 4 5 6];
C = A * B %true
C = B' + A %true
C = A' * B %error
C = B + A %error

%2
A=[16 2 3 13;
5 11 10 8;
9 7 6 12;
4 14 15 1]

B = A(:, 1:2) %true
B = A(1:4, 1:2) %true
B = A(0:2, 0:4) %error
B = A(1:2, 1:4) %false

%3
A = magic(10)
x = A(:,1)

v = A * x; %true
v = Ax;
v = x' * A;
v = sum (A * x);

%4
B = magic(7)
v = B(:,2)
w = B(:,3)

z = 0;
for i = 1:7
	z = z + v(i) * w(i)
end

z = sum (v .* w) %true
z = w' * v %true
z = v * w %error
z = w * v %error

%5
X = magic(7)
for i = 1:7
	for j = 1:7
		A(i, j) = log(X(i, j));
		B(i, j) = X(i, j) ^ 2;
		C(i, j) = X(i, j) + 1;
		D(i, j) = X(i, j) / 4;
	end
end

C = X + 1 %true
D = X / 4 %true
A = log (X) %true
B = X ^ 2 % B = X .^ 2 
```