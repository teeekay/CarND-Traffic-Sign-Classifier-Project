### **Traffic Sign Recognition** 

## Writeup by Tony Knight - 2017/04/21 

![image1](examples\TestSamples\keepright.png "Keep Right!")





[//]: # (Image References)



[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---
###Writeup / README

Here is a link to my [project code](https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_ClassifierV2_2.ipynb)

###Data Set Summary & Exploration

####1. 
I computed statistics on the data set using numpy and Panda and standard python math.  This was completed in code cells 3 through 7 on both the original data, the images converted to contrast enhanced grayscale images and, a subset of equalized training images.   The relevant stats are:

| Image Set        | Count |
|:-----------------|------:|
|Training examples | 34799 |
|Validation examples | 4410 |
|Testing examples | 12630 |

The image data shape is (32, 32, 3) which is in an RGB format.

The number of classes is 43.  After loading the signnames file, I was able to use Pandas Dataframe to generate a listed breakdown of the image class in the validation set by signname.


| Sign Identification                       | Count of Images|
|:---------------------------------------------------|------:|
|Ahead only                                          |  1080 |
|Beware of ice/snow                                  |   390 |
|Bicycles crossing                                   |   240 |
|Bumpy road                                          |   330 |
|Children crossing                                   |   480 |
|Dangerous curve to the left                         |   180 |
|Dangerous curve to the right                        |   300 |
|Double curve                                        |   270 |
|End of all speed and passing limits                 |   210 |
|End of no passing                                   |   210 |
|End of no passing by vehicles over 3.5 metric tons  |   210 |
|End of speed limit (80km/h)                         |   360 |
|General caution                                     |  1080 |
|Go straight or left                                 |   180 |
|Go straight or right                                |   330 |
|Keep left                                           |   270 |
|Keep right                                          |  1860 |
|No entry                                            |   990 |
|No passing                                          |  1320 |
|No passing for vehicles over 3.5 metric tons        |  1800 |
|No vehicles                                         |   540 |
|Pedestrians                                         |   210 |
|Priority road                                       |  1890 |
|Right-of-way at the next intersection               |  1170 |
|Road narrows on the right                           |   240 |
|Road work                                           |  1350 |
|Roundabout mandatory                                |   300 |
|Slippery road                                       |   450 |
|Speed limit (100km/h)                               |  1290 |
|Speed limit (120km/h)                               |  1260 |
|Speed limit (20km/h)                                |   180 |
|Speed limit (30km/h)                                |  1980 |
|Speed limit (50km/h)                                |  2010 |
|Speed limit (60km/h)                                |  1260 |
|Speed limit (70km/h)                                |  1770 |
|Speed limit (80km/h)                                |  1650 |
|Stop                                                |   690 |
|Traffic signals                                     |   540 |
|Turn left ahead                                     |   360 |
|Turn right ahead                                    |   599 |
|Vehicles over 3.5 metric tons prohibited            |   360 |
|Wild animals crossing                               |   690 |
|Yield                                               |  1920 |

I also observed how Pandas could output stats about individual columns in the dataframe.  The individual column values had been calculated on individual images in the grayscale dataset.

|stat |      X_max  |    X_mean   |     X_min   |     X_std   |      X_sum |
|:---:|:---:|:---:|:---:|:---:|:---:|
|count| 34799.000000  | 34799.000000 | 34799.000000 | 34799.000000 |  34799.000000 |  
|mean |    254.995862 |   139.554981 |     6.117503 |    63.027796 | 142904.300181 |  
|std  |      0.131900 |    11.417725 |     7.621655 |     7.089961 |  11691.750853 |  
|min  |    244.000000 |    97.950195 |     1.000000 |    35.793909 | 100301.000000 |  
|25%  |    255.000000 |   132.451172 |     2.000000 |    58.799550 | 135630.000000 |  
|50%  |    255.000000 |   136.599609 |     4.000000 |    64.331295 | 139878.000000 |  
|75%  |    255.000000 |   142.957520 |     6.000000 |    68.217517 | 146388.500000 |  
|max  |    255.000000 |   225.502930 |    70.000000 |    79.894665 | 230915.000000 |  



####2. Exploratory Visualization

The code for this step is contained in the fourth through seventh code cells of the Jupyter notebook.  

Here is an exploratory visualization of the data sets. It is a bar chart showing how the data is distributed between the classes in each of the sets.

![3 Bar charts]( )

I also investigated how the pixel intensities were distributed after contrast enhancing the grayscale images.  There appear to be some interesting artefacts in this data, and I am concerned by how many pixel values were clipped at the maximum level of 255, however, the general shape of the data set looks like a somewhat squished normal distribution, which bodes well for analysis.

![CLAHE Pixel Histogram]()


### Model Architecture Design and Tests

I initially used an implementation of the LeNet model.  To feed the model, I began by using the raw RGB data to see how the model would work.  I was able to get about 90% accuracy on the training data set.  I then switched to using HSV image data by converting the images using openCV functions.  However, this did not help improve results.

I switched to using CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhanced grayscaled versions of the image. I found that there was a large difference in the brightness of the images, and I wanted to try and compensate for this.  I had also read articles indicating that CLAHE was useful in traffic sign recognition (although it was usually done in HSV space not on grayscale images)   The code to convert the images is found in code cell 2 of the notebook.  I tried using both Scipy Image (skimage) and opencv (cv2) routines to complete the CLAHE enhancement after grayscaling the image.  I experimented with different tiling and cliplimit values to obtain both the best visual result, and a good pixel distribution (see image x above).

Here is an example of a traffic sign image before and after grayscaling and CLAHE enhancement.

![alt text][image2]

As a last step, I normalized the image data using the individual image mean and standard deviation to produce images with pixel intensities having a standard deviations of 1 centered at 0.   used in some cases

####2. Starting with the training, validation, and test data sets provided, I used two strategies to try and improve validation accuracy.  I tried equalizing the training data set so that there were an even number of images in each class of image, and this way the model would not be biased to select images that were overrepresented in the training dataset.  However, this did not help, possibly because this limited the size of the training set too much.  I also tried augmenting the training dataset by adding distortions of the original images.  In code cells 11 to 19 I added images that had random amounts of motion blur, reduced scaling, displacement, rotation, and fixed perspective distortions (tilts) up left right and down.

My final training set had 313191 images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... +++++++++


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
