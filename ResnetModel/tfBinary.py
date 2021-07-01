#使用RESNET50模型进行二分类模型训练的脚本

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import Adam, schedules, SGD, Adagrad, Adadelta
from tensorflow.keras import layers


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
EPOCH = 10
CLASSNUM = 2
MODEL_DIR = 'D:/users/leo.ren/pythonCode/ResnetModel/hat.h5'

model = tf.keras.applications.ResNet50V2(
    include_top=False, input_shape=(224,224,3), classes=CLASSNUM
)

#训练和测试数据集路径
train_path="D:/users/leo.ren/pythonCode/Clothing3/Hat"
test_path="D:/users/leo.ren/pythonCode/Clothing3/Hat"

#设置数据读入器的参数。括号内为数据增强的选项
train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.15,height_shift_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator()

#从路径读入数据
train_generator = train_datagen.flow_from_directory(train_path,target_size=(224, 224),
    batch_size=BATCH_SIZE,shuffle=True,class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_path,target_size=(224,224),
    batch_size=BATCH_SIZE,shuffle=False,class_mode='binary')

#加入一下客制化输出层用来规定输出的向量大小
headModel = layers.GlobalAveragePooling2D()(model.output)
'''headModel = layers.Flatten()(headModel)
headModel = layers.Dense(units=1024,activation="relu")(headModel)'''
#Dropout层用于减少过拟合
headModel = layers.Dropout(0.2)(headModel)
headModel = layers.Dense(units=1, activation="sigmoid")(headModel)

model = Model(inputs=model.input, outputs=headModel)
model.summary()

#es用于当模型loss连续上升时停止训练
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
#mc用于每次训练完一个epoch进行保存，记得更换模型保存路径
mc = ModelCheckpoint(MODEL_DIR, monitor='val_accuracy', mode='max', save_freq= 'epoch')

#优化器用于确定模型优化策略
optimizer = Adagrad(learning_rate=0.01)

#开始训练
model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics= ['accuracy'])
H = model.fit(train_generator,validation_data=test_generator,epochs=EPOCH,verbose=1,callbacks=[mc,es])

#训练结束后的测试
model.load_weights(MODEL_DIR)
model.evaluate(test_generator)