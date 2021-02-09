from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, multiply, Flatten, Concatenate, Dense, BatchNormalization
from keras.models import Model


def multi_scale_model(model_name, input_shape):
    x_input = Input(input_shape, name='input0')
    
    largeScale = Conv2D(10, (16,16), strides=2, padding='same', activation='relu', name='conv0')(x_input)
    largeScale = MaxPooling2D((2,2), name='mp0')(largeScale)
    largeScale = BatchNormalization(name='bn0')(largeScale)
    largeScale = Conv2D(20, (8,8), strides=2, padding='same', activation='relu', name='conv1')(largeScale)
    largeScale = MaxPooling2D((2,2), name='mp1')(largeScale)
    largeScale = BatchNormalization(name='bn1')(largeScale)
    largeScale = Flatten(name='flat0')(largeScale)
    
    midScale = Conv2D(10, (8,8), strides=2, padding='same', activation='relu', name='conv2')(x_input)
    midScale = MaxPooling2D((2,2), name='mp2')(midScale)
    midScale = BatchNormalization(name='bn2')(midScale)
    midScale = Conv2D(20, (4,4), strides=2, padding='same', activation='relu', name='conv3')(midScale)
    midScale = MaxPooling2D((2,2), name='mp3')(midScale)
    midScale = BatchNormalization(name='bn3')(midScale)
    midScale = Flatten(name='flat1')(midScale)
    
    smallScale = Conv2D(10, (4,4), strides=2, padding='same', activation='relu', name='conv4')(x_input)
    smallScale = MaxPooling2D((2,2), name='mp4')(smallScale)
    smallScale = BatchNormalization(name='bn4')(smallScale)
    smallScale = Conv2D(20, (4,4), strides=2, padding='same', activation='relu', name='conv5')(smallScale)
    smallScale = MaxPooling2D((2,2), name='mp5')(smallScale)
    smallScale = BatchNormalization(name='bn5')(smallScale)
    smallScale = Flatten(name='flat2')(smallScale)
    
    finalFeatures = Concatenate(axis=1, name='cat0')([largeScale, midScale, smallScale])
    X = Dense(10, activation='relu', name='fc0')(finalFeatures)
    X = Dense(2, activation='softmax', name='fc1')(X)

    model = Model(inputs=x_input, outputs=X, name=model_name)
    
    return model


def large_scale_model(input_shape):
    x_input = Input(input_shape, name='input0')
    
    largeScale = Conv2D(10, (16,16), strides=2, padding='same', activation='relu', name='conv0')(x_input)
    largeScale = MaxPooling2D((2,2), name='mp0')(largeScale)
    largeScale = BatchNormalization(name='bn0')(largeScale)
    largeScale = Conv2D(20, (8,8), strides=2, padding='same', activation='relu', name='conv1')(largeScale)
    largeScale = MaxPooling2D((2,2), name='mp1')(largeScale)
    largeScale = BatchNormalization(name='bn1')(largeScale)
    largeScale = Flatten(name='flat0')(largeScale)
    
    X = Dense(10, activation='relu', name='fc0')(largeScale)
    X = Dense(2, activation='softmax', name='fc1')(X)
    
    model = Model(inputs=x_input, outputs=X, name='large_scale_model')
    
    return model


def medium_scale_model(input_shape):
    x_input = Input(input_shape, name='input0')
    
    midScale = Conv2D(10, (8,8), strides=2, padding='same', activation='relu', name='conv2')(x_input)
    midScale = MaxPooling2D((2,2), name='mp2')(midScale)
    midScale = BatchNormalization(name='bn2')(midScale)
    midScale = Conv2D(20, (4,4), strides=2, padding='same', activation='relu', name='conv3')(midScale)
    midScale = MaxPooling2D((2,2), name='mp3')(midScale)
    midScale = BatchNormalization(name='bn3')(midScale)
    midScale = Flatten(name='flat1')(midScale)
    
    X = Dense(10, activation='relu', name='fc0')(midScale)
    X = Dense(2, activation='softmax', name='fc1')(X)
    
    model = Model(inputs=x_input, outputs=X, name='medium_scale_segCNN')
    
    return model


def small_scale_model(input_shape):
    x_input = Input(input_shape, name='input0')
    
    smallScale = Conv2D(10, (4,4), strides=2, padding='same', activation='relu', name='conv4')(x_input)
    smallScale = MaxPooling2D((2,2), name='mp4')(smallScale)
    smallScale = BatchNormalization(name='bn4')(smallScale)
    smallScale = Conv2D(20, (4,4), strides=2, padding='same', activation='relu', name='conv5')(smallScale)
    smallScale = MaxPooling2D((2,2), name='mp5')(smallScale)
    smallScale = BatchNormalization(name='bn5')(smallScale)
    smallScale = Flatten(name='flat2')(smallScale)
    
    X = Dense(10, activation='relu', name='fc0')(smallScale)
    X = Dense(2, activation='softmax', name='fc1')(X)
    
    model = Model(inputs=x_input, outputs=X, name='small_scale_segCNN')
    
    return model
