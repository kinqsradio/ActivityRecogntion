# Standard library imports
import os

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Flatten, TimeDistributed, Softmax, Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
from tensorflow.keras.regularizers import l2

# Dense Model
def initialize_dense_model(input_shape, num_classes, reg_strength=0.001):
    model = Sequential()

    # Flatten the input
    model.add(Flatten(input_shape=input_shape))
 
    # First Dense layer
    model.add(Dense(512, kernel_regularizer=l2(reg_strength)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Second Dense layer
    model.add(Dense(256, kernel_regularizer=l2(reg_strength)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Third Dense layer
    model.add(Dense(128, kernel_regularizer=l2(reg_strength)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    # model.add(Dense(num_classes))
    model.add(Dense(num_classes, activation='softmax'))


    return model


# LSTM Model
def initialize_lstm_model(input_shape, num_classes, reg_strength=0.001):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(1024, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))

    # Second LSTM layer
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))

    # Third LSTM layer
    model.add(LSTM(256))
    model.add(Dropout(0.5))

    # First Dense layer
    model.add(Dense(1024, kernel_regularizer=l2(reg_strength)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Second Dense layer
    model.add(Dense(512, kernel_regularizer=l2(reg_strength)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Third Dense layer
    model.add(Dense(256, kernel_regularizer=l2(reg_strength)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    # model.add(Dense(num_classes))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# CNN Model
def initialize_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = input_shape))
    
    model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))
                                      
    model.add(TimeDistributed(Flatten()))
                                      
    model.add(LSTM(32))
                                      
    model.add(Dense(num_classes, activation = 'softmax'))
    
    return model


# Function to train the model
def train_model(model_name ,model, features, labels, batch_size, epochs, early_stopping_patience=5):
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                #   loss = 'categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(f'{model_name}.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, 
                        callbacks=[early_stopping, reduce_lr, checkpoint])

    # Load the best model
    model.load_weights(f'{model_name}.h5')

    # Save and plot the training history
    history_df = pd.DataFrame(history.history)
    history_df.to_excel(f'{model_name}_training_history.xlsx', index=False)

    plot_performance(history)

    return model

def plot_performance(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.jpg')
    plt.show()

