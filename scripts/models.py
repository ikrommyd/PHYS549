'''
This python file contains a class called MLP that defines a multilayer perceptron in keras to be used in our modeling and wraps around it.
This provides ease of use in our modeling notebooks as we avoid writing a lot of lines of code.
There are methods defined that print the model, train the model, find the optimal learning rate for the model, evaluate the model and predict with the model. 
'''

import numpy as np 
import keras
from keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


class MLP:
    '''
    This class defined a multilayer peceptron in keras.

    Parameters
    ----------
    num_of_features : int
        The number of features that the datasset has.
    num_of_layers : int, optional
        The number of hidden layers in the multilayer perceptron. Default is 2.
    dim_of_layers : tuple of ints, optional
        The number of neurons in each layer. Must have length equal to the number of hidden layers. Default is (60,60).
    activation_of_layers : tuple of strings or keras.activations
        Activation functions of each layer. The strings must be labels of keras.activations. Must have length equal to the number of hidden layers. Default is ('relu','relu').
    normalize_batch : bool, optional
        Whether to add a batch normalization layer at the beginning. Default is True.

    Returns
    -------
    Instance of the MLP class
    '''

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
        '''
        Prints the model summary of the keras model.
        '''

        print(self.keras_model.summary())
        
    def save_random_weights(self):
        '''
        Saves the random initial weights into a file called 'random_weights.h5'.
        '''

        self.keras_model.save_weights('random_weights.h5')
    
    def find_optimal_lr(self, X_train, y_train, rates, batch_size = 2**16, verbosity = 'auto'):
        '''
        Finds the optimal learning rate by calculating the minimum loss after 10 epochs for variable values of the learning rate.
        Saves the optimal learning rate as a class argument called 'optimal_lr' and also shows a plot of the value of the loss function after 10 epochs vs learning rate.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples, 2)
            Training target values.
        rates : array-like of shape (n_rates,)
            The learning rates to loop over.
        batch_size = int, optinal
            The batch size to use while training to find the optimal learning rate. Default is 2**16.
        verbosity = 'auto', 0, 1, or 2, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        '''
        
        losses = []

        for lr in tqdm(rates):
            self.keras_model.load_weights('random_weights.h5')
            opt = keras.optimizers.SGD(learning_rate=lr)
            self.keras_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            history = self.keras_model.fit(X_train, y_train, batch_size=batch_size, epochs=10,
                                           verbose=verbosity)
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
        
    def train(self, X_train, y_train, X_test, y_test, learning_rate = 0.01,  epochs = 200, batch_size = 2**16):
        '''
        Trains the keras model. 

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples, 2)
            Training target values.
        X_test : array-like of shape (n_samples, n_features)
            Validation data.
        y_test : array-like of shape (n_samples, 2)
            Validation target values.
        learning rate : real, optional
            Learning rate to use for training. Default is 0.01.
        epochs : int, optional
            Number of epochs to train. Default is 200.
        batch_size = int, optional.
            The batch size to use while training. Default is 2**16.

        Returns
        -------
        Instance of keras.callbacks.History for the training.
        '''
            
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
        # callbacks = [early_stopping, model_checkpoint]
        callbacks = [model_checkpoint]

        self.keras_model.load_weights('random_weights.h5')
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
        self.keras_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.keras_model.fit(X_train, y_train, batch_size=batch_size, 
                                       epochs=epochs, callbacks = callbacks, 
                                       validation_data=(X_test,y_test))

        self.keras_model.load_weights('keras_model_best.h5')

        return history
        
    def evaluate(self, X_train, y_train, X_test, y_test, batch_size = 2**16):
        '''
        Evaluates the keras model.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples, 2)
            Training target values.
        X_test : array-like of shape (n_samples, n_features)
            Validation data.
        y_test : array-like of shape (n_samples, 2)
            Validation target values.
        batch_size = int, optional.
            The batch size to use while evaluating. Default is 2**16.

        Returns
        -------
        Tuple of lists of reals of length 2. Loss and accuracy for the X_train and X_test data sets.
        '''

        evaluation_train = self.keras_model.evaluate(X_train,y_train, batch_size=batch_size)
        evaluation_test = self.keras_model.evaluate(X_test,y_test, batch_size=batch_size)

        print(evaluation_train)
        print(evaluation_test)
        
        return evaluation_train, evaluation_test
    
    def predict(self, X_train, X_test, batch_size = 2**16):
        '''
        Predicts target values with the keras model.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        X_test : array-like of shape (n_samples, n_features)
            Validation data.
        batch_size = int, optional.
            The batch size to use while evaluating. Default is 2**16.

        Returns
        -------
        Tuple of ndarrays of shape (n_samples, 2). Target values for the X_train and X_test data sets.
        '''
        
        predict_array_train = self.keras_model.predict(X_train, batch_size=batch_size)
        predict_array_test = self.keras_model.predict(X_test, batch_size=batch_size)
        
        return predict_array_train, predict_array_test
        