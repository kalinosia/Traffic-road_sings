# Exercise with TensorFlow 
### My journey, Dominika Kalinowska


## Background
As research continues in the development of self-driving cars, one of the key challenges is [computer vision](https://en.wikipedia.org/wiki/Computer_vision), 
allowing these cars to develop an understanding of their environment from digital images. 
In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, you’ll use [TensorFlow](https://www.tensorflow.org/) to build a neural network to classify road signs based on an image of those signs. 
To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) dataset, 
which contains thousands of images of 43 different kinds of road signs.

## Code
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
##### Result 
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

## Tips 

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

