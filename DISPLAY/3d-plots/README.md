# 3D Plots

Look at the pretty pictures!

## Dataset
Input is normally distributed (with mean 0) along x and y with variance 0.1 and 0.2 respectively.
Ouput is just an arbitrary function I cooked up. It is calculated as 

    z = x*y - x**2 + x - 2*y**2 + y + R
  
Where R is a random perturbation, normally distributed with variance 0.1.

### Data
![High Angle](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/high-angle.png)

### Generating Function as a Wireframe
![Generating Wireframe](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/generating-wireframe.png)

## Least-Squares Approximations
### Least squares approximating plane, using the whole dataset 
![Least Squares](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/least-squares-approximated.png)

## Linear Regression
Only 3 training points are used for the linear regression for this example, specifically to get a poorly-trained model for demonstration. 
Even so, with these three particular points, the approximation is actually pretty good.

### Dataset Projections with Training Data Marked
![Training Data](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-data.png)
![Training Data vs X](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-data-vs-x.png)
![Training Data vs Y](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-data-vs-y.png)

### Output of Linear Regression
![Linear Regression](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/linear-regression-approximated.png)

