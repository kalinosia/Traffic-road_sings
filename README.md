# Exercise with TensorFlow 
### My journey, Dominika Kalinowska


## Background
First of all, I had a problem with doing model so 
I done some tutorials with tensorflow. 
It took a while to see the first solution and the first conclusions are
 to one more time repeat theory with doing some good tuturial and work your program for first time.
 ## My code 
 ### 1. Basic
 ```
model = tf.keras.Sequential([
        # [[[..],..[..]]..] -> [......]
        keras.layers.Flatten(input_shape=(IMG_WIDTH,IMG_HEIGHT,3)),
        # RGB [0, 255] ->[0, 1]
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        # Hidden layer with 128 neurons, Dense - fully conected
        keras.layers.Dense(128, activation="relu"), 
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    # Train neural network
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
```
`loss='categorical_crossentropy'`  here's where I have issue: not sparse_categorical_crossentropy because this to 1D.
Here what i get:
```
Epoch 1/10
500/500 [==============================] - 5s 5ms/step - loss: 2.7086 - accuracy: 0.3389
Epoch 2/10
500/500 [==============================] - 2s 4ms/step - loss: 1.6135 - accuracy: 0.6083
Epoch 3/10
500/500 [==============================] - 2s 4ms/step - loss: 1.1515 - accuracy: 0.7237
Epoch 4/10
500/500 [==============================] - 2s 4ms/step - loss: 0.9014 - accuracy: 0.7848
Epoch 5/10
500/500 [==============================] - 2s 4ms/step - loss: 0.7476 - accuracy: 0.8244
Epoch 6/10
500/500 [==============================] - 2s 4ms/step - loss: 0.6430 - accuracy: 0.8473
Epoch 7/10
500/500 [==============================] - 2s 4ms/step - loss: 0.5551 - accuracy: 0.8726
Epoch 8/10
500/500 [==============================] - 2s 4ms/step - loss: 0.5022 - accuracy: 0.8804
Epoch 9/10
500/500 [==============================] - 2s 4ms/step - loss: 0.4459 - accuracy: 0.8953
Epoch 10/10
500/500 [==============================] - 2s 4ms/step - loss: 0.4175 - accuracy: 0.9018
333/333 - 1s - loss: 0.5079 - accuracy: 0.8757
```
Not so bad for first time? But here is where fun begin: more testing, more outputs, more conclusions. 

___
___
---

> I reduce date to `NUM_CATEGORIES = 22` !!!


##### Result:

```
Epoch 1/10
346/346 [==============================] - 3s 4ms/step - loss: 2.1448 - accuracy: 0.4077
Epoch 2/10
346/346 [==============================] - 1s 4ms/step - loss: 1.3320 - accuracy: 0.6509
Epoch 3/10
346/346 [==============================] - 1s 4ms/step - loss: 1.0199 - accuracy: 0.7339
Epoch 4/10
346/346 [==============================] - 1s 4ms/step - loss: 0.8193 - accuracy: 0.7915
Epoch 5/10
346/346 [==============================] - 1s 4ms/step - loss: 0.6878 - accuracy: 0.8252
Epoch 6/10
346/346 [==============================] - 1s 4ms/step - loss: 0.5773 - accuracy: 0.8607
Epoch 7/10
346/346 [==============================] - 1s 4ms/step - loss: 0.5183 - accuracy: 0.8735
Epoch 8/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4515 - accuracy: 0.8922
Epoch 9/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4106 - accuracy: 0.9009
Epoch 10/10
346/346 [==============================] - 1s 4ms/step - loss: 0.3772 - accuracy: 0.9081
231/231 - 1s - loss: 0.3897 - accuracy: 0.9042
```

### 2. Grayscale

In loading `image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` and changing in model `...keras.layers.Flatten(input_shape=(IMG_WIDTH,IMG_HEIGHT))...` not `input_shape=(IMG_WIDTH,IMG_HEIGHT, 3)`.

##### Result:
```
Epoch 1/10
346/346 [==============================] - 1s 2ms/step - loss: 2.2212 - accuracy: 0.3858
Epoch 2/10
346/346 [==============================] - 1s 2ms/step - loss: 1.4902 - accuracy: 0.6048
Epoch 3/10
346/346 [==============================] - 1s 2ms/step - loss: 1.1572 - accuracy: 0.6912
Epoch 4/10
346/346 [==============================] - 1s 2ms/step - loss: 0.9370 - accuracy: 0.7619
Epoch 5/10
346/346 [==============================] - 1s 2ms/step - loss: 0.7885 - accuracy: 0.8041
Epoch 6/10
346/346 [==============================] - 1s 2ms/step - loss: 0.6865 - accuracy: 0.8276
Epoch 7/10
346/346 [==============================] - 1s 2ms/step - loss: 0.6100 - accuracy: 0.8522
Epoch 8/10
346/346 [==============================] - 1s 2ms/step - loss: 0.5541 - accuracy: 0.8602
Epoch 9/10
346/346 [==============================] - 1s 2ms/step - loss: 0.4973 - accuracy: 0.8796
Epoch 10/10
346/346 [==============================] - 1s 2ms/step - loss: 0.4656 - accuracy: 0.8870
231/231 - 0s - loss: 0.4885 - accuracy: 0.8832
``` 
It's worse. 

### 3. More neurons in hidden layer
Changing from 128 to 256.
##### Result:
```
Epoch 1/10
346/346 [==============================] - 5s 7ms/step - loss: 2.0267 - accuracy: 0.4438
Epoch 2/10
346/346 [==============================] - 2s 7ms/step - loss: 1.2011 - accuracy: 0.6822
Epoch 3/10
346/346 [==============================] - 2s 7ms/step - loss: 0.8608 - accuracy: 0.7830
Epoch 4/10
346/346 [==============================] - 2s 7ms/step - loss: 0.6904 - accuracy: 0.8294
Epoch 5/10
346/346 [==============================] - 2s 7ms/step - loss: 0.5671 - accuracy: 0.8552
Epoch 6/10
346/346 [==============================] - 2s 7ms/step - loss: 0.4971 - accuracy: 0.8749
Epoch 7/10
346/346 [==============================] - 2s 7ms/step - loss: 0.4301 - accuracy: 0.8877
Epoch 8/10
346/346 [==============================] - 2s 7ms/step - loss: 0.3988 - accuracy: 0.8958
Epoch 9/10
346/346 [==============================] - 2s 7ms/step - loss: 0.3718 - accuracy: 0.9036
Epoch 10/10
346/346 [==============================] - 2s 7ms/step - loss: 0.3432 - accuracy: 0.9090
231/231 - 1s - loss: 0.3502 - accuracy: 0.9115

```

Little bit better.

Now in our program is 18450 pictures. One image is 30 x 30 x 3 so it's 2700. Our inputs are 49 815 000 if all images was in train!!!
I give 512 neurons and see what happen.  
##### Result:
```
Epoch 1/10
346/346 [==============================] - 7s 13ms/step - loss: 2.0210 - accuracy: 0.4545
Epoch 2/10
346/346 [==============================] - 5s 13ms/step - loss: 1.1217 - accuracy: 0.7053
Epoch 3/10
346/346 [==============================] - 5s 13ms/step - loss: 0.7958 - accuracy: 0.7971
Epoch 4/10
346/346 [==============================] - 5s 13ms/step - loss: 0.6227 - accuracy: 0.8400
Epoch 5/10
346/346 [==============================] - 5s 13ms/step - loss: 0.5275 - accuracy: 0.8662
Epoch 6/10
346/346 [==============================] - 5s 13ms/step - loss: 0.4588 - accuracy: 0.8773
Epoch 7/10
346/346 [==============================] - 5s 13ms/step - loss: 0.4240 - accuracy: 0.8890
Epoch 8/10
346/346 [==============================] - 5s 14ms/step - loss: 0.3657 - accuracy: 0.9049
Epoch 9/10
346/346 [==============================] - 5s 14ms/step - loss: 0.3337 - accuracy: 0.9072
Epoch 10/10
346/346 [==============================] - 5s 13ms/step - loss: 0.3219 - accuracy: 0.9162
231/231 - 1s - loss: 0.3036 - accuracy: 0.9220
```
Try one more time with 50000 neurons. 
##### Result:
```
346/346 [==============================] - 438s 1s/step - loss: 2.9920 - accuracy: 0.4423
Epoch 2/10
346/346 [==============================] - 394s 1s/step - loss: 0.9562 - accuracy: 0.7229
Epoch 3/10
346/346 [==============================] - 401s 1s/step - loss: 0.6875 - accuracy: 0.8023
Epoch 4/10
346/346 [==============================] - 423s 1s/step - loss: 0.5619 - accuracy: 0.8350
Epoch 5/10
346/346 [==============================] - 496s 1s/step - loss: 0.4976 - accuracy: 0.8552
Epoch 6/10
346/346 [==============================] - 423s 1s/step - loss: 0.3784 - accuracy: 0.8910
Epoch 7/10
346/346 [==============================] - 433s 1s/step - loss: 0.3537 - accuracy: 0.8995
Epoch 8/10
346/346 [==============================] - 430s 1s/step - loss: 0.3500 - accuracy: 0.9008
Epoch 9/10
346/346 [==============================] - 422s 1s/step - loss: 0.3220 - accuracy: 0.9039
Epoch 10/10
346/346 [==============================] - 421s 1s/step - loss: 0.2609 - accuracy: 0.9270
231/231 - 78s - loss: 0.2887 - accuracy: 0.9207
```
Result isn't awesome, but look at time! It's take so long!! 
Back to 128 to test more.

### 4. Activation 
#### activation = 'sigmoid'
##### Result:
```
Epoch 1/10
346/346 [==============================] - 7s 5ms/step - loss: 2.1604 - accuracy: 0.4062
Epoch 2/10
346/346 [==============================] - 1s 4ms/step - loss: 1.2925 - accuracy: 0.6669
Epoch 3/10
346/346 [==============================] - 2s 4ms/step - loss: 0.9139 - accuracy: 0.7768
Epoch 4/10
346/346 [==============================] - 1s 4ms/step - loss: 0.7013 - accuracy: 0.8358
Epoch 5/10
346/346 [==============================] - 2s 7ms/step - loss: 0.5578 - accuracy: 0.8723
Epoch 6/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4691 - accuracy: 0.8917
Epoch 7/10
346/346 [==============================] - 1s 4ms/step - loss: 0.3926 - accuracy: 0.9132
Epoch 8/10
346/346 [==============================] - 1s 4ms/step - loss: 0.3506 - accuracy: 0.9195
Epoch 9/10
346/346 [==============================] - 1s 4ms/step - loss: 0.3105 - accuracy: 0.9327
Epoch 10/10
346/346 [==============================] - 1s 4ms/step - loss: 0.2730 - accuracy: 0.9376
231/231 - 1s - loss: 0.3326 - accuracy: 0.9091
```
 
#### activation="linear"
##### Result
```
Epoch 1/10
346/346 [==============================] - 4s 4ms/step - loss: 1.9077 - accuracy: 0.4984
Epoch 2/10
346/346 [==============================] - 1s 4ms/step - loss: 1.0494 - accuracy: 0.7132
Epoch 3/10
346/346 [==============================] - 1s 4ms/step - loss: 0.7866 - accuracy: 0.7898
Epoch 4/10
346/346 [==============================] - 1s 4ms/step - loss: 0.6013 - accuracy: 0.8387
Epoch 5/10
346/346 [==============================] - 1s 4ms/step - loss: 0.5735 - accuracy: 0.8383
Epoch 6/10
346/346 [==============================] - 1s 4ms/step - loss: 0.5284 - accuracy: 0.8593
Epoch 7/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4324 - accuracy: 0.8813
Epoch 8/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4481 - accuracy: 0.8754
Epoch 9/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4139 - accuracy: 0.8887
Epoch 10/10
346/346 [==============================] - 1s 4ms/step - loss: 0.3676 - accuracy: 0.8976
231/231 - 1s - loss: 0.5393 - accuracy: 0.8653
```
#### activation = None
##### Result:
```
Epoch 1/10
346/346 [==============================] - 3s 4ms/step - loss: 1.8961 - accuracy: 0.4967
Epoch 2/10
346/346 [==============================] - 1s 4ms/step - loss: 1.0036 - accuracy: 0.7258
Epoch 3/10
346/346 [==============================] - 1s 4ms/step - loss: 0.7263 - accuracy: 0.8081
Epoch 4/10
346/346 [==============================] - 1s 4ms/step - loss: 0.6647 - accuracy: 0.8177
Epoch 5/10
346/346 [==============================] - 1s 4ms/step - loss: 0.5422 - accuracy: 0.8562
Epoch 6/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4405 - accuracy: 0.8811
Epoch 7/10
346/346 [==============================] - 1s 4ms/step - loss: 0.5062 - accuracy: 0.8653
Epoch 8/10
346/346 [==============================] - 1s 4ms/step - loss: 0.3753 - accuracy: 0.8986
Epoch 9/10
346/346 [==============================] - 1s 4ms/step - loss: 0.4764 - accuracy: 0.8768
Epoch 10/10
346/346 [==============================] - 2s 4ms/step - loss: 0.3727 - accuracy: 0.8963
231/231 - 1s - loss: 0.4398 - accuracy: 0.8793
```
## ... Conv2D ...
#### 1. Start, some code...
```
model = tf.keras.Sequential([
        keras.layers.Flatten(input_shape=(IMG_WIDTH,IMG_HEIGHT,3)),
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),

        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES)
    ])
```
##### Result
```
Epoch 1/10
346/346 [==============================] - 13s 25ms/step - loss: 2.1173 - accuracy: 0.3547
Epoch 2/10
346/346 [==============================] - 9s 25ms/step - loss: 0.6221 - accuracy: 0.8077
Epoch 3/10
346/346 [==============================] - 9s 26ms/step - loss: 0.2815 - accuracy: 0.9221
Epoch 4/10
346/346 [==============================] - 9s 26ms/step - loss: 0.1719 - accuracy: 0.9509
Epoch 5/10
346/346 [==============================] - 9s 26ms/step - loss: 0.1125 - accuracy: 0.9694
Epoch 6/10
346/346 [==============================] - 9s 25ms/step - loss: 0.0858 - accuracy: 0.9743
Epoch 7/10
346/346 [==============================] - 8s 24ms/step - loss: 0.0705 - accuracy: 0.9792
Epoch 8/10
346/346 [==============================] - 8s 24ms/step - loss: 0.0582 - accuracy: 0.9836
Epoch 9/10
346/346 [==============================] - 9s 25ms/step - loss: 0.0379 - accuracy: 0.9889
Epoch 10/10
346/346 [==============================] - 8s 24ms/step - loss: 0.0331 - accuracy: 0.9909
231/231 - 2s - loss: 0.0868 - accuracy: 0.9786
```
####  2. Without `tf.keras.layers.MaxPooling2D()` - it's take so long..
```
model = tf.keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),

        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
           
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])
```
##### Result
```
Epoch 1/10
346/346 [==============================] - 137s 383ms/step - loss: 0.7206 - accuracy: 0.7949
Epoch 2/10
346/346 [==============================] - 134s 386ms/step - loss: 0.0821 - accuracy: 0.9760
Epoch 3/10
346/346 [==============================] - 132s 382ms/step - loss: 0.0339 - accuracy: 0.9902
Epoch 4/10
346/346 [==============================] - 132s 383ms/step - loss: 0.0239 - accuracy: 0.9939
Epoch 5/10
346/346 [==============================] - 132s 382ms/step - loss: 0.0336 - accuracy: 0.9900
Epoch 6/10
346/346 [==============================] - 132s 382ms/step - loss: 0.0177 - accuracy: 0.9946
Epoch 7/10
346/346 [==============================] - 133s 384ms/step - loss: 0.0171 - accuracy: 0.9951
Epoch 8/10
346/346 [==============================] - 132s 381ms/step - loss: 0.0294 - accuracy: 0.9926
Epoch 9/10
346/346 [==============================] - 136s 393ms/step - loss: 0.0153 - accuracy: 0.9956
Epoch 10/10
346/346 [==============================] - 139s 403ms/step - loss: 0.0233 - accuracy: 0.9930
231/231 - 25s - loss: 0.1604 - accuracy: 0.9764
```

####  3. default: `padding="valid"`
```
model = tf.keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),

        layers.Conv2D(16, 3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3,  activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3,  activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])
```
##### Result 
```
Epoch 1/10
346/346 [==============================] - 16s 39ms/step - loss: 1.6720 - accuracy: 0.4911
Epoch 2/10
346/346 [==============================] - 13s 38ms/step - loss: 0.4034 - accuracy: 0.8806
Epoch 3/10
346/346 [==============================] - 13s 38ms/step - loss: 0.1857 - accuracy: 0.9459
Epoch 4/10
346/346 [==============================] - 13s 38ms/step - loss: 0.0973 - accuracy: 0.9719
Epoch 5/10
346/346 [==============================] - 13s 38ms/step - loss: 0.0663 - accuracy: 0.9817
Epoch 6/10
346/346 [==============================] - 13s 38ms/step - loss: 0.0425 - accuracy: 0.9883
Epoch 7/10
346/346 [==============================] - 13s 38ms/step - loss: 0.0375 - accuracy: 0.9893
Epoch 8/10
346/346 [==============================] - 13s 39ms/step - loss: 0.0321 - accuracy: 0.9919
Epoch 9/10
346/346 [==============================] - 13s 37ms/step - loss: 0.0309 - accuracy: 0.9913
Epoch 10/10
346/346 [==============================] - 13s 38ms/step - loss: 0.0167 - accuracy: 0.9950
231/231 - 3s - loss: 0.0693 - accuracy: 0.9827
```

#### 4. `padding="same"` 
```
Epoch 1/10
346/346 [==============================] - 24s 61ms/step - loss: 1.7817 - accuracy: 0.4544
Epoch 2/10
346/346 [==============================] - 21s 60ms/step - loss: 0.3933 - accuracy: 0.8752
Epoch 3/10
346/346 [==============================] - 20s 59ms/step - loss: 0.1604 - accuracy: 0.9536
Epoch 4/10
346/346 [==============================] - 20s 59ms/step - loss: 0.0798 - accuracy: 0.9772
Epoch 5/10
346/346 [==============================] - 20s 59ms/step - loss: 0.0502 - accuracy: 0.9865
Epoch 6/10
346/346 [==============================] - 20s 58ms/step - loss: 0.0390 - accuracy: 0.9897
Epoch 7/10
346/346 [==============================] - 20s 58ms/step - loss: 0.0178 - accuracy: 0.9958
Epoch 8/10
346/346 [==============================] - 20s 59ms/step - loss: 0.0293 - accuracy: 0.9915
Epoch 9/10
346/346 [==============================] - 20s 58ms/step - loss: 0.0272 - accuracy: 0.9918
Epoch 10/10
346/346 [==============================] - 20s 58ms/step - loss: 0.0110 - accuracy: 0.9964
231/231 - 4s - loss: 0.0722 - accuracy: 0.9809
```

## Code and result
##### Model:
```
model = tf.keras.Sequential([

        keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT,3)),
        tf.keras.layers.MaxPooling2D(),  # Max pooling is then used to reduce the spatial dimensions
        tf.keras.layers.Conv2D(64, 3, padding="same", activation='relu'), # We then learn 64 filters
        tf.keras.layers.MaxPooling2D(),  
        tf.keras.layers.Conv2D(128, 3, padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        # Hidden layer with 128 neurons, Dense - fully conected
        tf.keras.layers.Dense(128, activation='relu'),
        # output
        keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    # Train neural network
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # not sparse_categorical_crossentropy bc 1D
                  metrics=['accuracy'])
```
##### Result (on all 43 cat.)
```
Epoch 1/10
500/500 [==============================] - 33s 61ms/step - loss: 2.1360 - accuracy: 0.3980
Epoch 2/10
500/500 [==============================] - 30s 60ms/step - loss: 0.5102 - accuracy: 0.8436
Epoch 3/10
500/500 [==============================] - 30s 60ms/step - loss: 0.1850 - accuracy: 0.9465
Epoch 4/10
500/500 [==============================] - 30s 60ms/step - loss: 0.0879 - accuracy: 0.9761
Epoch 5/10
500/500 [==============================] - 33s 67ms/step - loss: 0.0529 - accuracy: 0.9858
Epoch 6/10
500/500 [==============================] - 31s 63ms/step - loss: 0.0549 - accuracy: 0.9839
Epoch 7/10
500/500 [==============================] - 32s 65ms/step - loss: 0.0323 - accuracy: 0.9909
Epoch 8/10
500/500 [==============================] - 32s 64ms/step - loss: 0.0280 - accuracy: 0.9925
Epoch 9/10
500/500 [==============================] - 31s 62ms/step - loss: 0.0284 - accuracy: 0.9914
Epoch 10/10
500/500 [==============================] - 31s 61ms/step - loss: 0.0251 - accuracy: 0.9932
333/333 - 6s - loss: 0.1159 - accuracy: 0.9704
```

## Tips to start

Start with small date: gtsrb-small which have `NUM_CATEGORIES = 3`!!! Remember to change that when you chane data.

When you load date you can see something. You want to see something to understand what's going on.

`print(image.shape) ` to see that your image is: `(30,30,3)`
When you `print(image)` you will see something like that:
```
[[[212 221 222]
  [249 244 249]
  [255 255 255]
  ...
  [ 97  97  97]
  [127 127 124]
  [183 198 197]]

 [[244 246 242]
  [253 250 250]
  [255 255 255]
  ...
  [100  93  93]
  [149 128 137]
  [238 232 241]]

 [[232 252 246]
  [254 240 238]
  [255 255 255]
  ...
  [ 97  93  93]
  [136 121 144]
  [234 230 248]]

 ...

 [[255 255 255]
  [255 255 251]
  [254 250 250]
  ...
  [100  98  99]
  [104 101 101]
  [105 101  99]]

 [[255 255 255]
  [255 255 255]
  [254 255 255]
  ...
  [ 97  95  96]
  [105  99 100]
  [102 100 101]]

 [[255 255 254]
  [255 255 255]
  [255 255 255]
  ...
  [ 95  95  96]
  [ 98  96  97]
  [ 97  96  98]]]
```
Not very funny. But:
```
plt.imshow(image, cmap=plt.cm.binary)
plt.show()
```
And you can see image. Remember to change BGR to RGB:
`image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) `.

 `print("LEN OF IMAGES: ", len(images))` to see how many is images:  gtsrb-small -> 840
 
 You can:
 ```
plt.hist(labels)
plt.show()
```
to see more about images in categories.

If you can see some random images:
```
# Determine the (random) indexes of the images that you want to see
    traffic_signs = [100, 558, 650, 838]   
    # https://www.datacamp.com/community/tutorials/tensorflow-tutorial
    # Fill out the subplots with the random images that you defined
    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(i, labels.count(i)))
        plt.imshow(images[traffic_signs[i]])
        plt.subplots_adjust(wspace=0.5)

    plt.show()
```

