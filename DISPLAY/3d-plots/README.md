# 3D Plots

Look at the pretty pictures!

## Dataset
Input is normally distributed (with mean 0) along x and y with variance 0.1 and 0.2 respectively.
Ouput is just an arbitrary function I cooked up. It is calculated as 

    z = x*y - x**2 + x - 2*y**2 + y + R
  
Where R is a random perturbation, normally distributed with variance 0.1.

![Generating Wireframe](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/generating-wireframe.png)

## Least-Squares Approximations
![Least Squares](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/least-squares-approximated.png)

## Linear Regression with Good Points
Only 3 training points are used for the linear regression for this example, specifically to get a poorly-trained model for demonstration. 

With the first random selection of three particular points, the approximation is somewhat reasonable.

### Dataset Projections with Training Data Marked
![Training Data](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-data.png)
![Training Data vs X](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-data-vs-x.png)
![Training Data vs Y](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-data-vs-y.png)
![Training In 3D](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/training-in-3d.png)

### Output of Linear Regression
![Linear Regression](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/linear-regression-approximated.png)
![Linear Regression 2](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/linear-regression-low-angle.png )


## Linear Regression with Good Points
With a different random selection, the approximation is terrible

### Dataset Projections with Training Data Marked
![training-points](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/terrible-sample/training.png)
![training-x](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/terrible-sample/training-x.png)
![training-y](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/terrible-sample/training-y.png)
![training-3d](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/terrible-sample/training-3d.png)

### Output of Linear Regression
![approx-plane](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/terrible-sample/approx-plane.png)
![approx-points](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/3d-plots/pictures/terrible-sample/approx-points.png)
