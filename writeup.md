# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/all_signs.png "Visualization"
[image2]: ./examples/all_signs_grayscale.png "Grayscaling"
[image3]: ./examples/test_sign.png 

[image4]: ./examples/predict1.png "Traffic Sign 1"
[image5]: ./examples/predict2.png "Traffic Sign 2"
[image6]: ./examples/predict3.png "Traffic Sign 3"
[image7]: ./examples/predict4.png "Traffic Sign 4"
[image8]: ./examples/predict5.png "Traffic Sign 5"
[image9]: ./examples/visual_img.png 
[image10]: ./examples/visual.png 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/ClarenceXu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set, all the 43 unique classess are presented below. 


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to use following 2 techniques:
1. normalize the data using mentioned way in the project: (pixel - 128)/ 128 
1. convert the images to grayscale because according to the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" stated that using grayscale improves accuracy. 
 
Here is an example of all traffic sign images after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5x1x12 	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32			        |
| Convolution 5x5x12x32 | 1x1 stride, VALID padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   		        |
| Flatten       		| outputs 400  									|
| Fully connected		| outputs 120 									|
| RELU					|												|
| Fully connected		| outputs 84 									|
| RELU					|												|
| Fully connected		| outputs 10 									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters

EPOCHS = 40
BATCH_SIZE = 128
rate = 0.001
keep_prob = 0.7

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.949
* test set accuracy of 0.938


If a well known architecture was chosen:
* I chose the LeNet solution and adjusted the parameters 
I ran though different combination of parameters, e.g. 
* keep_prob between 0.5 and 1.0, Convolution 5x5x1x12
* EPOCHS between 20 and 50
* increase 1st Convolution layer from 5x5x1x6 to 5x5x1x12
* increase 2nd Convolution layer from 10x10x16 to 10x10x32 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 5 German traffic signs that I found on the web:

![alt text][image3] 

1st image might be difficult to classify because the image is dark and inclined 
2nd image might be difficult to classify because the image is dark and also difficult for humen to identify, meanwhile, the resolution is very low
3rd image might be difficult to classify because the image is partially covered by other object
4th image might be difficult to classify because the image contains 2 traffic signs
5th image should be easy to classify because the image is very clear 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
 
| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8,Speed limit (120km/h) | Speed limit (120km/h)   									| 
| 7,Speed limit (100km/h) | Speed limit (120km/h) 										|
| 12,Priority road  	  | Priority road  												|
| 40,Roundabout mandatory | Speed limit (100km/h) 					 				|
| 25,Road work			| Road work   							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

For 1st image, the model is 100% sure that this is a "Speed limit (120km/h)"  (probability of 1.0). The top five soft max probabilities were

![alt text][image4]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (120km/h)  									| 
| .0     				| Speed limit (80km/h)										|
| .0					| Dangerous curve to the right											|
| .0	      			| Traffic signals					 				|
| .0    			    | Speed limit (30km/h)     							|


For 2nd image, the model predicted wrongly, a very low probability (0.062) for the correct "Speed limit (100km/h)". The top five soft max probabilities were

![alt text][image5] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.937         		| Speed limit (120km/h)  									| 
| 0.062  (label)   			| Speed limit (100km/h)										|
| .0					| Speed limit (80km/h)										|
| .0	      			| Speed limit (70km/h)			 				|
| .0    			    | End of all speed and passing limits   							|

For 3rd image, the model is 100% sure that this is a "Priority road"  (probability of 1.0). The top five soft max probabilities were

![alt text][image6] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         		| Priority road								| 
| 0.0     			| Roundabout mandatory										|
| .0					|  No passing									|
| .0	      			| End of no passing		 				|
| .0    			    | Right-of-way at the next intersection   							|

For 4th image, the model predicted completely wrong, correct sign should be "Roundabout mandatory". The top five soft max probabilities were

![alt text][image7] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.492         		| Speed limit (100km/h)							| 
| 0.290     			| Right-of-way at the next intersection									|
| 0.098					| Beware of ice/snow								|
| 0.063	      			| Double curve	 				|
| 0.022    			    | Road work  							|

For 5th image, the model is 100% sure that this is a "Road work"  (probability of 1.0). The top five soft max probabilities were

![alt text][image8] 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         		| Road work							| 
| 0.0     			| Dangerous curve to the right										|
| .0					| Road narrows on the right								|
| .0	      			| Speed limit (20km/h)				|
| .0    			    | Speed limit (30km/h)  							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image9] 

![alt text][image10] 

In the visualization of the 1st convolutional layer, different feature maps look for the triangle, e.g. FeatureMap 1, 2, 4, 5, 6, 7, 8, 9, 19
In addition, it also looks for the animal sign inside the triangle, e.g. FeatureMap 1, 2, 4, 6, 7, 8, 9
