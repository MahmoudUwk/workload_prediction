import numpy as np

def reg_all(X_train,y_train,X_test,reg_model):
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.ensemble import GradientBoostingRegressor,HistGradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    reg_models_names = ["linear_reg","svr_reg","GPR_reg","GBT_reg"]
    ind = [c for c,ele in enumerate(reg_models_names) if ele==reg_model][0]
    regs_all = [LinearRegression(), SVR(kernel= 'linear'),GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()), 
                HistGradientBoostingRegressor()]
    reg = regs_all[ind]
    reg.fit(X_train, y_train)
    
    return reg#,reg.predict(X_train),reg.predict(X_test)





def class_all(X_train,y_train,X_test,class_model):
    from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier
    class_models_names = ["KNN","MLP","GNB","RDF","GBT"]
    ind = [c for c,ele in enumerate(class_models_names) if ele==class_model][0]
    class_all = [KNeighborsClassifier(n_neighbors=3), "keras",
                GaussianNB(),RandomForestClassifier(n_estimators=50,max_depth=12),
                HistGradientBoostingClassifier()]
    classifier = class_all[ind]
    if class_model == "MLP":
        classifier,y_test_pred = MLP_classifier(X_train,y_train,X_test)
    else: 
        classifier.fit(X_train, y_train)
        y_test_pred = classifier.predict(X_test)
    
    return classifier,y_test_pred








def MLP_classifier(X_train,y_train,X_test):
    from keras.layers import Dense
    from keras.models import  Sequential #,load_model
    import tensorflow as tf
    import os
    
    class SaveBestModel(tf.keras.callbacks.Callback):
        def __init__(self, save_best_metric='val_loss', this_max=False):
            self.save_best_metric = save_best_metric
            self.max = this_max
            if this_max:
                self.best = float('-inf')
            else:
                self.best = float('inf')
    
        def on_epoch_end(self, epoch, logs=None):
            metric_value = logs[self.save_best_metric]
            if self.max:
                if metric_value > self.best:
                    self.best = metric_value
                    self.best_weights = self.model.get_weights()
    
            else:
                if metric_value < self.best:
                    self.best = metric_value
                    self.best_weights= self.model.get_weights()
    
    #%%
    out_size = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train)
    n = 2**8

    model = Sequential()
    model.add(Dense(n, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(n, activation='relu'))
    # model.add(Dense(n, activation='relu'))

    model.add(Dense(out_size, activation='softmax'))

    checkpoint = SaveBestModel()
    callbacks_list = [checkpoint]
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=400, batch_size=2048, validation_split =0.2,callbacks=callbacks_list)
    model.set_weights(checkpoint.best_weights)
    # best_epoch = np.argmin(history.history['val_loss'])
    
    y_test_pred = np.argmax(model.predict(X_test),axis=1)
    return model,y_test_pred
















