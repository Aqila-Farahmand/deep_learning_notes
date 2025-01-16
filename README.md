# Deep Learning
Supervised Machine Learning 
     1.Regression
     2.Classification
Unsupervised Machine Learning:
     1.Clustering
     
## Regression

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
       Declare convergence by looking at j(w,b)<= ε and ε=10^-3 or 0.001 if the value of j(w,b) i.e cost function is decreasing in the orther of ε i.e (gradient decent found parameters w,b to be close to global minimum) by each iterations after a      
       certain time then you can declare convergece. 

#### Learning rate(α)
     if too large may never converge (i.e overshooting) and if too small, then converging very slowly ( takes long time).
     To fix, small very small and check how the j(w,b) is changing.
     i.e j(w,b) should decrease on every iteration.
     if with J(w,b) does not decrease on very small α then there should be a bug in the code.

### Feature Engineering:
     To better improve the model learning, using your intuition, desing a new feature by either transforming or combining the exciting feature is called feature engineering.
     Consider the follwoing example:
     the model is to predict the price of the house, and our dataset has features, x1 = length, x2 = width.
     to help model learn and predict better we add a new feature x3 = x1*x2 ( the area).ù
### Polynomial regression:
     Having more than one features.i.e x1 and x2 and the learning function is no longer a stright line.
     1. j(w,b) = w1*x1^2 + w2*x2 + b
     2. circle, j(w,b) = w1.x1^2 * w2.x2^2 + b 
     3. cube (x^3) or any surface that fits the model.

## Classification 
     
    Classification with Logistic Resgression:
          Sigmoid function:
          g(z) = 1/(1 + exp^(-z))
     Logistic Regression (Sigmoid)
          note: Dot product i.e vector multiplications.
          considering w and x are vectors or array or many dimension, that is 1D, 2D or 3D etc.
          j(w.x + b) = 1/(1 + exp^(-w.x + b)) or P(y=1|x:w,b) that is the probability of y/output being 1, when x/input is input and w,b are parameters.
     Decision Boundary:
          linear (w.x +b)
          Non-linear (circle, cube and etc) or polynomial
          Or even more complex decision boundry.
     Cost Function for Logistic Regression:
          y = 1 or y = 0
          x = [x1, x2, ... xm]
          thus:
          J(w,b) = ∑(i=1 to m​) -yi*​log(f​w,b(xi​)) - (1−yi​)*log(1−fw,b​(xi​))
          i.e
          if y = 1;
          j(w,b) = -log (f(xi)
          if y = 0;
          j(w,b) = -log(1-f(xi))
     Minimizing the Cost with Gradient Descent:
          minj(w,b) = minCost
          j(w,b) =  1/(1 + exp^(-w.x + b))
          Repeat {
          wj = wj - α * (∇j(w,b))
          b = b - α * (∇j(w,b))
          } for simultinous update
          
  ### The Problem of Overfitting
   
<img width="1324" alt="regresion_overfit" src="https://github.com/user-attachments/assets/f9e57958-fde0-4732-ba6b-f957980100fa" />

       
<img width="1304" alt="classification_overfit" src="https://github.com/user-attachments/assets/84aa7c31-4cbb-4c6e-94d9-89a6046f06c0"/>

     Adressing overfitting:
          1. Collect more data
               if not possible. then go for 2 and 3.
          2. Select features to include/exlude by picking the most relevent features.
          3. Regularization 
               Reduce the size of the Wj to smaller values let's say close to zero.
     Regularization:
           Penalize all the parameters i.e Wj to very small value! usually minimizing the parameter W rather than b.
           Increasing the regularization parameter λ;
           For a model that includes the regularization parameter λ(lambda), increasing λ will tend to decrease the parameter wj.
          The parmeter λ( reduces overfitting by reducing the size of the parameters.  For some parameters that are near zero, this reduces the effect of the associated features.
