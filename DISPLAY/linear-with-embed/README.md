# Linear Predictions of Non-Linear Data Using Embeddings

This demonstrates an interesting phenomenon. 

Consider the following data, with some test samples extracted. 

![ring base](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_base.png)
![train and test](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_train_and_test.png)

In this case, the data hase been labelled either red or blue, and we want to learn to classify it. (This is a supervised learning problem)

We could classify it using, e.g., k-nearest-neighbours: 

![circ_k10](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/knn/pictures/circ_k10.png)

But perhaps we would like to learn a more elegant classifying function of some sort.

Of course, it is not possible to perform a linear classification on such data. 
Attempting to use, e.g. a stochastic gradient descent linear classifier results in quite random classification: 

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10)
    
It doesn't matter too much that we're using `hinge` loss and `l2` penalty in this case. We could just as easily use `perceptron`
in which case we may as well use the `perceptron` api. From the scikitlearn manual:
    
    Perceptron() is equivalent to SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None).

![SGD Bad Prediction](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_bad_predict.png)

However, by using the correct embedding for this data, and embedding the data in 3D, 
it is possible to perform a 3D linear classification to find a plane of best fit. 

Consider, for example, the following embedding: 

    z = 0.5*x + x**2 + 0.5*y + y**2

That looks like this: 

![Embed Top](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd-embed-top.png)
![Embed Semiside](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_embed_semiside.png)
![Embed Side](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_embed_side.png)
![Embed Other](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_embed_other.png)

Now, we can use e.g. SGD classifier. Note that we have three pieces of input data, namely x, y, and our derived z co-ordinate, 
and one piece of labelled data to learn, namely, the colour red or blue.

Then the classifier learns the following separating plane, and classifies the data correctly: 

![Predicted Top](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_predicted_top.png)
![Predicted Semiside](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_predicted_semiside.png)
![Predicted Side](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_predicted_side.png)
![Predicted Other](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/sgd_ring_predicted_other.png)

By finding the intersection of the learned plane with the original embedding function, we can get a separating ellipse for the data.

(To visualize this data I have cheated by actually just running the prediction algorithm on a fine mesh grid, but the theory stands)

![De-projected](https://github.com/mark-chimes/ml_stuff/blob/master/DISPLAY/linear-with-embed/pictures/de-projected.png)

This is only slightly useful in practice. We are giving the algorithm quite a lot of information with that embedding function. 

However, from a theoretical standpoint it is very interesting. 
If you learned, say, multiple layers of linear functions separated by non-linear embeddings, you could fit quite arbitrary data. 
In a very rough sense this is what a deep neural network does. 

# With Thanks To 

This bit of experimentation was inspired by https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ so go look there for more information
and some very nice visualizations! 

![Topology 3D](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/topology_3d.png)


