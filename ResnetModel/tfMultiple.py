#使用RESNET50/101模型进行多分类（>=3）模型训练的脚本
#整体结构和二分类训练脚本类似，只是修改一些参数用于适应多分类模型训练

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import Adam, schedules, SGD, Adagrad
from tensorflow.keras import layers


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
EPOCH = 15
CLASSNUM = 7
MODEL_DIR = "D:/users/leo.ren/pythonCode/ResnetModel/outer101.h5"

model = tf.keras.applications.ResNet101V2(
    include_top=False, input_shape=(224,224,3), classes=7
)

#Dataset path
train_path="D:/users/leo.ren/pythonCode/Clothing3/outer"
test_path="D:/users/leo.ren/pythonCode/Clothing3/outer"

train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_path,target_size=(224, 224),batch_size=BATCH_SIZE,shuffle=True,class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_path,target_size=(224,224),batch_size=BATCH_SIZE,shuffle=False,class_mode='categorical')

#Add customized layers
headModel = layers.GlobalAveragePooling2D()(model.output)   
'''headModel = layers.Flatten()(headModel)       
headModel = layers.Dense(units=1024,activation="relu")(headModel)'''
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(units=CLASSNUM, activation="softmax")(headModel)

model = Model(inputs=model.input, outputs=headModel)
model.summary()

#remember to change the category for model saving
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint(MODEL_DIR, monitor='val_accuracy', mode='max', save_freq= 'epoch')


optimizer = Adagrad(learning_rate=0.01)

model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics= ['accuracy'])
H = model.fit(train_generator,validation_data=test_generator,epochs=EPOCH,verbose=1,callbacks=[mc,es])

#remember to change path for model
model.load_weights(MODEL_DIR)
model.evaluate(test_generator)