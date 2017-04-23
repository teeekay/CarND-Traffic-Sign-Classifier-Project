### **Traffic Sign Recognition** 

## Writeup by Tony Knight - 2017/04/21 

![image1](https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/keepright.png?raw=true "Keep Right!")

Here is a link to my [project code](https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_ClassifierV2_2.ipynb)

### Data Set Summary & Exploration

I computed statistics on the provided data sets using numpy and Panda and standard python math.  This was completed in code cells 3 through 7 on both the original data, the images converted to contrast enhanced grayscale images and, a subset of equalized training images.   The relevant stats are:

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



#### 2. Exploratory Visualization

The code for this step is contained in the fourth through seventh code cells of the Jupyter notebook.  

Here is an exploratory visualization of the data sets. It is a bar chart showing how the data is distributed between the classes in each of the sets.

![3 Bar charts](https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/datasetclassdistribution1.png?raw=true )

I investigated how the pixel intensities were distributed after contrast enhancing the grayscale images.  There appear to be some interesting artefacts in this data, and I am concerned by how many pixel values were clipped at the maximum level of 255, however, the general shape of the data set looks like a "normal" distribution, which bodes well for analysis.

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/grayscalehistogram1.png?raw=true" alt="Histogram of pixel intensity in Grayscale Dataset" width=400>

After normalizing the data to a standard deviation of 1 and a mean of 0, the pixel intensity distribution looks like this:
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/grayscalehistogram1.png?raw=true" alt="Histogram of pixel intensity in Grayscale Dataset" width=400>



### Model Architecture Design and Tests

I initially used the LeNet-5 implementation of a standard 2 layer feed forward convolutional network activated by softmax and ReLU functions, with a depth   

To feed the model, I started by using the raw RGB image data to see how the model would work.  I was able to get about 90% accuracy on the training data set.  I then switched to using HSV image data by converting the images using the openCV functions.  However, this did not improve results.  I read in the paper by [Sermanet and LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that they were able to get highest accuracy when using only grayscale (Y channel) information, so I decided to try using just grayscale information.

I found that there was a large visual difference in the brightness of the images, and I wanted to try and compensate for this.  I switched to using CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhanced grayscaled versions of the image.   I had also read articles including the Nvidia's ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316v1.pdf) document indicating that CLAHE was useful in traffic sign recognition (although it was usually done in HSV space not on grayscale images).  

The code I used to convert the images is found in code cell 2 of the notebook.  I tested using both Scipy Image (skimage) and opencv (cv2) routines to complete the CLAHE enhancement after grayscaling the image.  I experimented with different tiling and cliplimit values to obtain both the best visual result, and a good pixel distribution(see image x above).

Here is an example of a traffic sign image before and after grayscaling and CLAHE enhancement.

![alt text][image2]

As a last step, I normalized the image data using the individual image mean and standard deviation to produce images with pixel intensities having a standard deviations of 1 centered at 0.

#### 2. 

Starting with the training, validation, and test data sets provided, I used two strategies to try and improve validation accuracy.  First, I tried equalizing the training data set so that there were an equal number of images in each class of image, and in this way the model would not be biased to select images that were over-represented in the training dataset.  However, In my tests, this did not improve validation or test accuracy, possibly because this limited the size of the training set too much.

Secondly , I tried augmenting the training dataset by adding distortions of the original images.  In code cells 11 to 19 I generated copies of the validation set images that had random amounts of motion blur, reductions in scale, displacement, rotation, and fixed perspective distortions (affine transformations that could be called tilts) in each of 4 directions (up left right and down).

| Image Sets after Augmenting        | Count |
|:-----------------|------:|
|- Original Training examples | 34799 |
|- blurred copies | 34799|
|- scaled copies | 34799|
|- displaced copies | 34799|
|- rotated copies | 34799|
|- tilted copies (4 directions) | 139196|
|Total Training Set Size | 313191 |
|Validation Set Size | 4410 |
|Testing Set Size | 12630 |

My final training set had 313191 images. My validation set and test set remained unchanged with 4410 and 12630 images.  I felt that the distortions to the image set would be useful both because it would add additional examples to the test set, and because the distortions applied were likely to occur in the field.

Comments in the Sermanet and Lecun paper indicated that care should be taken to ensure that the validation set did not include images that were taken from the same run of images taken of the same sign, as this would not give the best indication of how well the model worked on general examples of the signs.  I had not checked this.

Here is an example of an original image and a set of the augmented images generated from it:

![alt text][image3]


####3. 

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/MultiScaleLenet.png?raw=true" alt="Multi Scale Convolutional Network Diagram" width="600">
#### Image 3qqq

A summary of the final model is displayed in image qqq, and in Table  
The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| # | Layer         		|     Description	        					|Input  | Output |
|:--|:---------------------|:---------------------------------------------|:------:|:------:|
| 0 | Input         		| Grayscale image   					| 	|32x32x1|
|	|	         		|    					| 
| 1 | Convolution 5x5     	| 1x1 stride, valid padding  	|32x32x1 |28x28x16 |
| 1 |ELU					|												|28x28x16 |28x28x16 |
| 1 | Max pooling 2x2	   	| 2x2 stride,valid padding			|28x28x16 |14x14x16|
|	|	         			|    											| | | 
| 2 | Convolution 5x5	    | 1x1 stride, valid padding   |14x14x16 |10x10x16 |
| 2 | ELU					|												|10x10x16 |10x10x16 |
| 2 | Max pooling 2x2      	| 2x2 stride, valid padding				|10x10x16 |5x5x16 |
| 2 | Flatten   	      	|  				|5x5x16 | 400 |
|	|	         		|    					|  | |
| 3 | Fully connected		|         							| 400 |120 |
| 3 | ELU					|         							| 120 |120 |
| 3 | Dropout				| Dropout of 50% in Training 		| 120 | 120 |
|	|						|												| | |
| 1A | Max pooling 4x4	      	| 4x4 stride, valid padding 				|14x14x16 | 3x3x16 |
| 1A | Flatten   	      	|  								|  3x3x16 | 144|
| 1A | Dropout				| Dropout of 50% in Training		| 144| 144|
|	|						|												| | |
| 4 | Fully connected		| Combined input from Layer 3 and 1A | 264 | 84 |
| 4 | ELU					|         							| 84 | 84 |
|	|						|										| | |
| 5 | Fully Connected		|         				| 84 | 43 |
|	|						|										| | |
| 5A  | Softmax             | Softmax applied to output of model |43 | 43|


####4.
The model was trained in cell ???

During training runs, I tried to optimize hyper parameters including the batch size, training rate, and number of epochs.  I used the Adam (Adaptive Moment Estimation) optimizer as opposed to standard stochastic gradient descent optimizer.  I added in explicit exponential decay of the training rate as training progressed.  My final parameters were

| Parameter | Value |
|:-------|:----:|
|Epochs | 200 |
|Batch Size | 64 |
|Initial Learning Rate | 0.0007 |
|Learning Rate Decay  Rate | 0.985 |
| mu | 0 |
| sigma | 0.1 |

Initially I used the ReLU activation function in layers , but I switched to using ELU after reading that it had been shown to learn faster and perform better.

#### 5. 

I started with the simple LeNet model using RGB input, and then moved to using grayscale input and was able to increase peak validation accuracy from 90% to 93%.  I then augmented the input set and this enabled the model to achieve validation accuracy above 95%.  At this point I experimented with adjusting the CLAHE algorithm, and found that I achieved best results when the cliplimit was increased to 32, and I was able to increase validation accuracy to 96 - At this point I found that I had been running dropout in the validation tests.  I restricted dropout to the training stage, and the validation accuracy increased to 97.8 and my first check on the test set gave a result of 95.6%.

At this point I tried equalizing the input set so that there were an equal number of examples per class in the training set (See code cell 6).  This reduced the number of samples in the training set to approximately 20% of its original size.  Validation accuracy dropped to 95%, and test set accuracy dropped to 92%.

I ran tests substituting the skimage (scikit-image) version of CLAHE instead of the CV2 variant and played with the parameters.  I found I could achieve comparable accuracy results, but that the skimage results were not as visually pleasing and were generally darker, so I switched back to cv2.

I had originally been standardizing the input using the mean and standard deviation of the set, but switched to using the values of the individual images.  This had little effect on the results produced by the model, possibly because the CLAHE enhanced images have relatively similar intensity distributions.

I switched my augmentation process to use a random noise background instead of extending the "SAME" on boundaries which could produce patterns the model might pick up in the rotation, scaling, and displacement distortions.  This appeared to improve validation accuracy to 98.3% - (However, I also fixed a bug which had been limiting rotations and displacements to only positive increments at the same time, so this may have also had an effect!).  The model achieved 96.7% on the test set.  I also tested the model on 7 images collected from the internet (primarily from Google street view in Hanover).  The model identified 6 of the 7 images properly.

I switched the activation function in each layer from to ELU, and saw a small increase in accuracy with each test set.

I then tried testing the use of a multiscale model, feeding a pooled set from the first layer into the fourth layer achieving Training: 99.9 Validation: 95.6 Test 92.9 after 150 epochs.

I moved back to the full (unequalized) training dataset and added augmentations for tilt up, down, left and right.  This resulted in a total of 313,191 images in the full (unequalized) training dataset. Achieved results of Training: 97.5% Validation: 98.3% Test 95.7% Internet clippings 7/7.  I realized I was using a softmax activation in layer 1 and switched this to ELU.  This resulted in slightly higher training accuracy, but slightly lower validation and test results after 100 epochs.

I then tried increasing the width of the layers, adjusting the first convolution to produce 16 feature planes instead of 6. Achieved Training 98.37, validation 98.21 Test 96.3 Internet Images 7/7 after 25 Epochs.

I ran the model a few times for a longer set of Epochs and introduced the use of a decaying learning rate. My final results were Training Accuracy 98.10%, Validation Accuracy 98.84%, Loss 0.082, Test set 96.86,  Internet Images 7 for 7.


###Test a Model on New Images

Here are the 7 images I retrieved from the internet.

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/keepright.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Limit70.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/noentry.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/row_nxt_intersection.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Stop.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Straightahead.png?raw=true" width=100>
<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/yield.png?raw=true" width=100>

I thought the sixth image (straight ahead) might be difficult to classify due to the distortion of the circular shape.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

|Sign Class| Sign Name		|Prediction Class |    Sign Name      | Confidence | 
|:--|----------------------:|:-------------------|:------------------:|:--------:| 
| 38| Keep Right      		| 38| Keep Right								| 1.000 |
| 4 | Speed Limit 70		| 4 | Speed Limit 70							| 1.000 |
| 17| No Entry				| 17| No Entry									| 0.996 |
| 11| RoW at next Intersection |11 | RoW at next Intersection Bumpy Road	| 1.000 |
| 14| Stop 				| 14 | Stop		      							| 1.000 |
| 35| Ahead Only			| 35 |Ahead Only      							| 0.942 |
| 13| Yield					| 13 |Yield		      							| 0.999 |
| 44| Random noise 			| 40 |Roundabout								| 0.279 |

The model was able to correctly guess All 7 traffic signs, giving an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.8%.

The softmax probability for 4 of the guesses was effectively 1.  The model was 99.6% sure of the no Entry prediction, with .003 and .001 chance of Yield and Stop signs respectively.  This is interesting as the no entry sign is circular whereas the Yield and Stop signs have straight edges (triangle and octoganal).  The ahead only sign had the lowest probability of 0.942, with secondary guesses of   which I would have thought the model would have used to  triangular. with an apex at the base - all the other triangular signs have an apex at the top of the image.  The  


| Probability      	| Predicted Class| Predicted Sign Name			|  Notes |
|:-----------------:|:---------------:|:-----------------------------:| :-------|
| 0.942 | 35| Ahead Only  		| |
| 0.047 | 18| General Caution |Vertical line down center, but triangular shape, white background|
| 0.004	| 26| Traffic Signals |Vertically aligned dots down central axis, but triangular shape, white background|
| 0.003	| 3 | Speed Limit 60 |circular, possible arrow head shape between 6 and 0 but white background|
| 0.003	| 37| Go Straight or left |circular, blue background, vertical arrow, but additional horizontal arrow|

Based on the precision and recall report and confusion matrix generated on the results from the test set; 1 image each from classes 11, 12, 13, 33, 40, 2 images from 34, and 8 from 37 were classified as 35, but no class 35 images were classified in any other class (Perfect Precision, but lower recall).
 
I added a random noise image and the model's top five guesses for a matching sign were:

| Probability      	| Predicted Class| Predicted Sign Name			|  Notes |
|:-----------------:|:---------------:|:-----------------------------:| :-------|
| 0.279 | 40|Roundabout mandatory|Recall 91.1 |
| 0.216 | 11|Right-of-way at the next intersection|Recall 92.4 |
| 0.189	| 27|Pedestrians|Recall 56.7 |
| 0.104	| 26 | Traffic Signals |Recall 87.2|
| 0.096	| 12| Priority Road |Recall 98.8|

The low but fairly similar probabilities associated with all 5 top guesses indicates that the noise sample does not fit any of the images particularly well.

Precision and Recall and Confusion Matrix


|Class|	Precision|	Recall|	F-score|	Items|	Sign Description|
|----:|---------:|-------:|-------:|--------:|:-----------|
|0	|98.4	|100.0	|99.2	|60 	|Speed limit (20km/h)|
|1|	97.8	|99.0	|98.4	|720	|Speed limit (30km/h)|
|2|	96.1	|99.3	|97.7	|750	|Speed limit (50km/h)|
|3|	97.0	|94.9	|96.0	|450	|Speed limit (60km/h)|
|4|	99.5	|97.9	|98.7	|660	|Speed limit (70km/h)|
|5|	92.1	|94.6	|93.3	|630	|Speed limit (80km/h)|
|6|	97.3	|95.3	|96.3	|150	|End of speed limit (80km/h)|
|7|	96.7	|98.4	|97.6	|450	|Speed limit (100km/h)|
|8|	99.3	|94.7	|96.9	|450	|Speed limit (120km/h)|
|9|	99.6	|99.2	|99.4	|480	|No passing|
|10|	99.8	|98.8	|99.3	|660	|No passing for vehicles over 3.5 metric tons|
|11|	99.0	|92.4	|95.6	|420	|Right-of-way at the next intersection|
|12|	98.8	|98.8	|98.8	|690	|Priority road|
|13|	98.2	|99.9	|99.0	|720	|Yield|
|14|	99.3	|100.0	|99.6	|270	|Stop|
|15|	89.9	|84.8	|87.3	|210	|No vehicles|
|16|	99.3	|100.0	|99.7	|150	|Vehicles over 3.5 metric tons prohibited|
|17|	100.0	|98.9	|99.4	|360	|No entry|
|18|	95.7	|96.7	|96.2	|390	|General caution|
|19|	100.0	|100.0	|100.0	|60	|Dangerous curve to the left|
|20|	97.6	|92.2	|94.9	|90	|Dangerous curve to the right|
|21|	98.5	|72.2	|83.3	|90	|Double curve|
|22|	98.1	|88.3	|93.0	|120	|Bumpy road|
|23|	95.5	|100.0	|97.7	|150	|Slippery road|
|24|	94.0	|87.8	|90.8	|90	|Road narrows on the right|
|25|	94.8	|98.8	|96.7	|480	|Road work|
|26|	94.0	|87.2	|90.5	|180	|Traffic signals|
|27|	66.7	|56.7	|61.3	|60	|Pedestrians|
|28|	82.2	|98.7	|89.7	|150	|Children crossing|
|29|	83.5	|95.6	|89.1	|90	|Bicycles crossing|
|30|	86.0	|86.0	|86.0	|150	|Beware of ice/snow|
|31|	96.8	|100.0	|98.4	|270	|Wild animals crossing|
|32|	84.1	|96.7	|89.9	|60	|End of all speed and passing limits|
|33|	98.6	|98.1	|98.3	|210	|Turn right ahead|
|34|	96.0	|100.0	|98.0	|120	|Turn left ahead|
|35|	100.0	|96.2	|98.0	|390	|Ahead only|
|36|	100.0	|100.0	|100.0	|120	|Go straight or right|
|37|	87.0	|100.0	|93.0	|60	|Go straight or left|
|38|	99.9	|99.6	|99.7	|690	|Keep right|
|39|	94.7	|100.0	|97.3	|90	|Keep left|
|40|	91.1	|91.1	|91.1	|90	|Roundabout mandatory|
|41|	100.0	|100.0	|100.0	|60	|End of no passing|
|42|	97.8	|96.7	|97.2	|90	|End of no passing by vehicles over 3.5 metric tons|

Here is a graphical representation of the Confusion Matrix

<img src="https://github.com/teeekay/CarND-Traffic-Sign-Classifier-Project/blob/master/examples/TestSamples/Confusion.png?raw=true" alt="Confusion Matrix" width=400>

With this model the worst performance is found with class 27 'pedestrians' which has both bad precision and recall and a dismal F1 rating of 61.3 wheras the rest of the scores are generally 90 or higher
