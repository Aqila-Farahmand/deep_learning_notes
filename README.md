# Deep Learning
1. Regression model
2. Cost function (Mean Square Error)
3. Optimization algorithm (Gradient Decent)
4. Feature Scalling Methods:
     When the dataset feature has larg range such as i.e feature1 = 1000 - 2000, we need to use feature scalling methods to speed up the Gradient Decent algorithm.
    1. Mean Normalization:
         formula: x'= x1-mean/max-min
    2. Zero score normalization
         x' = (x1 - mean)/ σ
       where σ is standard deviation
       σ = √∑(x-mean)^2/n-1
       mean = sum/n

#### Note

### When to use SoftMax Function as cost function?
1. It's used as activation function in multi-class classification ptoblem.

### Make sure the Gradient Decent is working:
  You can chexk by plotting the J(w,b) vs iterations (number of iteration gradient decent takes to update w,b), to see if the value of j(w,b) is decreasing by each iteration, other wise if it's increasing then there should be an error.
  Also, when to declare convergence? 
  Declare convergence by looking at j(w,b)<= ε and ε=10^-3 or 0.001 if the value of j(w,b) i.e cost function is decreasing in the orther of ε i.e (gradient decent found parameters w,b to be close to global minimum) by each iterations after a certain time then you can declare convergece. 

### Learning rate:
if too large may never converge (i.e overshooting) and if too small, then converging very slowly ( takes long time).
To fix, small very small and check how the j(w,b) is changing.
i.e j(w,b) should decrease on every iteration.
if with J(w,b) does not decrease on very small α then there should be a bug in the code.
