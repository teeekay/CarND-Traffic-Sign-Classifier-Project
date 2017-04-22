### Remarks:

### Initial model

- Hyperparams set sigma = 0.1, Mu =0

Stages:

1) L1: Conv2d, ReLU, MaxPool
2) L2: Conv2d, softMax, MaxPool, flatten
3) L3: FullyConnected, dropout(50%)
4) L4: FullyConnected, ReLU
5) L5: FullyConnected

runs with RGB files and initial model peaked out at about 90%, with overfitting probably after about 87%
changes to batch size and epochs had little effect - attempted moving batches at 256 and 512, and learning rate down from 0.001 to 0.0005

Inserted a ReLU before dropout on L4 to see effect - no big change

Move mu to 0.1 and sigma to 0.01 - result 5%
Move mu to 1.0 and sigma to 0.01 - result 5% - tensorflow is mixed up! Need to restart environment.


### Preprocessing Optimization:
First test: switch to HSV
- peaked at 86
 Second Test switch to grayscale CLAHE - batch 256, 100 epochs, mu 0, sigma 0.1
Training Accuracy 98%, Validation 91.9%
- test with dropout of 0.75
Training Accuracy 98.6%, Validation 91.4% 
- Reduced batch to 128, 150 epochs, mu 0, sigma 0.1
Training Accuracy 98.6%, Validation 91.4% 
- cut learning rate to 0.0008
Training Accuracy 98.6%, Validation 92.8%
- cut learning rate to 0.0006
Training Accuracy 99.5 %, Validation 94.1 % at 132 epochs, but 98.9 92.6 at 150 epochs 
- cut learning rate to 0.0005, moved to 100 epochs
Training Accuracy 99.7 %, Validation 93.6 % at 100
- cut learning rate to 0.0003, moved to 120 epochs
bad start - ends at 98.1% 89.6% after 120 epochs
- try learning rate of 0.0004, moved to 120 epochs
Training Accuracy 99.4%, Validation 92.6%
- reduce batchsize to 64, switch lr to 0.0005
Training Accuracy 99.5%, Validation 93.1% - solidly in 93% range since 60th epoch
Training Accuracy 99.7%, Validation 93.4% - solidly in 93% range since 80th epoch - peaks in 94%
- Now will try to bolster by adding images to training set
Add sets with blur, scaling, displacement, rotation
Validation Accuracy exceeds Training Accuracy - At epoch 11 93. vs 89.9 until ~ epoch 35 when validation falls below training
at  35: 95.3 to 94.4
at 100: 96.9 to 94.9
at 125: 96.5 to 95.0
at 140: 97.2 to 95.1
- Retried with CLAHE set to 8x8 instead of 4x4 - batch 128 100 epochs - validation accuracy looks lower
at 35: 90.9 and 88.4

- Testing with Standardization about 0, std=1, lr 0.005, batch 128, 100 epochs, clahe 8x8

validation stuck at 87% from Epoch 10 on..
reset learing data?

- Testing with Standardization around 0, std = 1, lr = 0.006, batch 128 100 epochs, clahe 4x4, mu 0, sigma 0.1
EPOCH 14 ...
Training Accuracy   = 0.964
Validation Accuracy = 0.949

|E    | T   |  V  |
|----:|----:|----:|
|10   |95.8 |94.9 |
|35   |97.9 |95.2 |
|50   |98.5 |95.9 |
|75   |98.5 |95.5 |
|100  |98.9 |95.6 |

peaked at 96.3

- Trying again after changing cliplimit in CLAHE to 32.0 (from 2.0) to improve histogram distribution.   Still concerned about clipped bright (value 255) pixels in dataset. running with lr = 0.006, batch 128 100 epochs, clahe 4x4, mu 0, sigma 0.1

100 - 98.5, 96.0

-Realized that I was running dropout on validation run.  Reset so that dropout was 50% on training and none during evaluation tests - 
EPOCH 10 ...
Training Accuracy   = 0.958
Validation Accuracy = 0.970
EPOCH 100 ...
Training Accuracy   = 0.993
Validation Accuracy = 0.978
 - Ran test set after this and got a result of 95.6!
 
- Balanced training set (but not validation set) so there were only 180 examples from each set.
also switched to skimage equalize_adapthist to see if it made a difference - params do not match exactly
getting a darker image, with pixels skewed to darker intensity. 
training accuracy >99.7%, but validation at about 95.3 - maybe should balance validation out
- test accuracy reduced to 92.8

- switched from tiles=8 to tiles=16 to see if it made a difference:
- training 99.4, valid at 94, test at 91.1

Switch back to CV2 but with equalized training set:
at 100 Training: 99.3 Validation: 95.2
at 150 Training: 99.8 Validation: 95.4 Test 92.3

Switch to using standardization by subtracting mean and dividing by std dev of individual images not of set
at 100 Training: 99.6 Validation: 95.0
at 150 Training: 99.9 Validation: 95.1 Test 92.7

Switch to using ELU instead of RELU to see if makes a difference
at 100 Training: 99.6 Validation: 95.7
at 150 Training: 99.8 Validation: 96.2 Test 93.2

MultiScale System - using output from layer 1 to go into layer 4
at 100 Training: 99.6 Validation: 95.0
at 150 Training: 99.9 Validation: 95.6 Test 92.9

Now back to unequalized data set using individual standardization, multiscale, CV2 adaptive equalization, dataset augmentation
at 100 Training: 99.3 Validation: 97.6 Test 95.3

- Brought in use of gray noise background instead of edge extension during augmentation transforms.  Fixed bugs in augmentation that resulted in only negative offsets instead of both positive and negative
at 100 Training 98.5 Validation 98.3 Test 96.7 Internet clippings 6/7

- Added perspective distortions in each of primary directions that would probably be how camera would see - up down left and right.  Almost doubled the training set to 313191 images.
at 100 Training 97.5 Validation 98.3 Test 95.7 Internet clippings 7/7!

- Reran after adding names to model, reset training rate to 0.004
at 100 Training 97.6 Validation 98.2 Test 96.1 Internet clippings 7/7!

- Reran after adding names to activations in model, reset training rate to 0.004 switched softmax on L1 to elu
at 100 Training 98.0 Validation 98.1 Test 96.0 Internet clippings 7/7!

Adjusted with code to collect predictions and to track loss, also stepwise decay of learning rate.

Switched to see if wider conv on layer 1 (16 instead of 6) would make a difference. cut to 25 Epochs
at 25 Training 98.37, validation 98.21 Test 96.3 Internet Images 7/7 but image 35 was not so solid a choice 

tweaked learning rate and decay and reduced to 20 epochs
EPOCH 20 ... Training Accuracy   = 97.79% Validation Accuracy = 98.71% Learning Rate = 0.0001162 Loss = 0.156
Test 96.4
- ran again after reset
 EPOCH 1 ... Training Accuracy   = 84.39% Validation Accuracy = 93.95% Learning Rate = 0.0006451 Loss = 0.880
 ...
 EPOCH 20 ... Training Accuracy   = 97.91% Validation Accuracy = 98.48% Learning Rate = 0.0001162 Loss = 0.107
 INFO:tensorflow:Restoring parameters from ./traffic_model/lenet5 - Test Accuracy = 0.964

- tweaked learning rate and decay and boosted to 100 epochs
 EPOCH 1 ... Training Accuracy   = 78.08% Validation Accuracy = 93.83% Learning Rate = 0.0006723 Loss = 0.707
 ....
 EPOCH 100 ... Training Accuracy   = 97.16% Validation Accuracy = 98.73% Learning Rate = 0.0000076 Loss = 0.123

- had to run again - ran to 150 to see if any difference

    EPOCH 150 ... Training Accuracy   = 97.38% Validation Accuracy = 98.53% Learning Rate = 0.0000008 Loss = 0.055
    Test set - 96.33
- ran again but switched to reduced gradient for learning rate, and smaller batch size - 64
 EPOCH 150 ... Training Accuracy   = 98.28% Validation Accuracy = 98.75% Learning Rate = 0.0000203 Loss = 0.194
 Test set 97.13
 - ran agai due to need to fix model to get out details from layers - cut decay rateof learning rate, got relatively similar results.
 EPOCH 150 ... Training Accuracy = 98.42%, Validation Accuracy = 98.78%, Learning Rate = 0.0001293, Loss = 0.058
 Test set 96.46 (lower)
 
- ran test again to check
 1) EPOCHS = 200, BATCH_SIZE = 64, starter_learning_rate = 0.0007, decay_step = batches_per_epoch * 4, decay_rate = 0.985
 2) EPOCH 200 ... Training Accuracy = 98.10%, Validation Accuracy = 98.84%, Learning Rate = 0.0000008, Loss = 0.082
 3) Test set 96.86%