# Stochastic Gradient Descent

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10)

It doesn't matter much that we're using "hinge" here. We could just as easily use "Perceptron".

    Perceptron() is equivalent to SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None).

Total of 2000 points

## Train on 200 points

![XXX](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/sgd/pictures/training_and_test.png )
![XXX](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/sgd/pictures/sgd_10_iter.png)

## Train on 4 points

![XXX](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/sgd/pictures/tiny_training_set.png)
![XXX](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/sgd/pictures/sgd_tiny_training_10_iter.png )
