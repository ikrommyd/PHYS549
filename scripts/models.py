import numpy as np 
import keras
from keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


class MLP:

    def __init__(self, num_of_features, num_of_layers = 2, dim_of_layers = (60,60), activation_of_layers = ('relu','relu'), normalize_batch = True):

        if num_of_features != len(dim_of_layers) != len(activation_of_layers):
            print("Number of layers must have the size as the activation function's list")
            pass

        layers = []

        if normalize_batch:
            layers.append(BatchNormalization(input_shape=(num_of_features,)))
            
        input_shapes = [num_of_features, *dim_of_layers] 
        for dim, act, shape in zip(dim_of_layers, activation_of_layers, input_shapes[:-1]):
            layers.append(Dense(units=dim, activation=act, input_dim=shape)) 
        
        layers.append(Dense(units=2,activation='softmax'))
        self.keras_model = keras.Sequential(layers)
        
    def show_model(self):
        print(self.keras_model.summary())
        
    def save_random_weights(self):
        self.keras_model.save_weights('random_weights.h5')
    
    def find_optimal_lr(self, X_train, y_train, rates, batch_size = 2**16, verbosity = 'auto'):
        
        losses = []

        for lr in tqdm(rates):
            self.keras_model.load_weights('random_weights.h5')
            opt = keras.optimizers.SGD(learning_rate=lr)
            self.keras_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            history = self.keras_model.fit(X_train, y_train, batch_size=batch_size, epochs=10,
                                      shuffle=False, verbose=verbosity)
            losses.append(history.history['loss'][-1])
        
        self.optimal_lr = rates[np.nanargmin(losses)]
        
        plt.figure()
        plt.plot(rates, losses)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.semilogx()
        plt.tight_layout()
        plt.savefig('plots/lr.pdf')
        plt.show()
        
        print(f"The optimal learning rate is {self.optimal_lr}")
        
    def train(self, X_train, y_train, X_test, y_test, batch_size = 2**16):
        
        try:
            lr = self.optimal_lr
        except AttributeError:
            lr = 1.0
            
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
        # callbacks = [early_stopping, model_checkpoint]
        callbacks = [model_checkpoint]

        self.keras_model.load_weights('random_weights.h5')
        opt = keras.optimizers.SGD(learning_rate=lr)
        self.keras_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.keras_model.fit(X_train, y_train, batch_size=batch_size, 
                                       epochs=100, shuffle=False, callbacks = callbacks, 
                                       validation_data=(X_test,y_test))

        self.keras_model.load_weights('keras_model_best.h5')

        return history
        
    def evaluate(self, X_train, y_train, X_test, y_test):

        evaluation_train = self.keras_model.evaluate(X_train,y_train)
        evaluation_test = self.keras_model.evaluate(X_test,y_test)

        print(evaluation_train)
        print(evaluation_test)
        
        return evaluation_train, evaluation_test
    
    def predict(self, X_train, X_test):
        
        predict_array_train = self.keras_model.predict(X_train)
        predict_array_test = self.keras_model.predict(X_test)
        
        return predict_array_train, predict_array_test
