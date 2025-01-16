# Deep Learning
Supervised Machine Learning 
     1.Regression
     2.Classification
Unsupervised Machine Learning:
     1.Clustering
     
## Regression (Linear Regration)
      Cost function (Mean Square Error)
      Optimization algorithm (Gradient Decent)
### Feature Scalling Methods:
     When the dataset feature has larg range such as i.e feature1 = 1000 - 2000, we need to use feature scalling methods to speed up the Gradient Decent algorithm.
    1. Mean Normalization:
         formula: x'= x1-mean/max-min
    2. Zero score normalization
         x' = (x1 - mean)/ σ
       where σ is standard deviation
       σ = √∑(x-mean)^2/n-1
       mean = sum/n

     When to use SoftMax Function as cost function?
          It's used as activation function in multi-class classification ptoblem.

     Make sure the Gradient Decent is working:
       You can chexk by plotting the J(w,b) vs iterations (number of iteration gradient decent takes to update w,b), to see if the value of j(w,b) is decreasing by each iteration, other wise if it's increasing then there should be an error.
       Also, when to declare convergence? 
       Declare convergence by looking at j(w,b)<= ε and ε=10^-3 or 0.001 if the value of j(w,b) i.e cost function is decreasing in the orther of ε i.e (gradient decent found parameters w,b to be close to global minimum) by each iterations after a      certain time then you can declare convergece. 

Learning rate:
     if too large may never converge (i.e overshooting) and if too small, then converging very slowly ( takes long time).
     To fix, small very small and check how the j(w,b) is changing.
     i.e j(w,b) should decrease on every iteration.
     if with J(w,b) does not decrease on very small α then there should be a bug in the code.

Feature Engineering:
     To better improve the model learning, using your intuition, desing a new feature by either transforming or combining the exciting feature is called feature engineering.
     Consider the follwoing example:
     the model is to predict the price of the house, and our dataset has features, x1 = length, x2 = width.
     to help model learn and predict better we add a new feature x3 = x1*x2 ( the area).ù
Polynomial regression:
     Example: j(w,b) = w*x^2 + w*x + b or x^3

## Classification 
     Logistic Resgression
          Sigmoid function:
          g(z) = 1/(1 + exp^(-z))
     Logistic Regression:
          note: Dot product i.e vector multiplications.
          considering w and x are vectors or array or many dimension, that is 1D, 2D or 3D etc.
          j(w.x + b) = 1/(1 + exp^(-w.x + b)) or P(y=1|x:w,b) that is the probability of y/output being 1, when x/input is input and w,b are parameters.
     Decision Boundary:
          


