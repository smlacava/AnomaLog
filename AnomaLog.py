import pandas as pd
import os
import numpy as np
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.externals import joblib  # sklearn >= 0.23
except:
    import joblib                         #sklearn  < 0.23


class AnomaLog():
    def __init__(self, multiclass=True, model_type='DNN'):
        """
        The __init__ method is the initializer of the class.

        :param multiclass: it has to be True in order to use (fit, evaluate and use
                           as analyzer) a multiclass classifier, or False to use a
                           2-class classifier (True by default)
        :param model_type: it can be 'DNN' (Deep Neural Network), 'SVM' (Support
                           Vector Machine), 'DT' (Decision Tree) 'RF' (Random
                           Forest), or 'KNN' (K-Nearest Neighbors) in order to use
                           the respective model as classifier ('DNN' by default)
        """
        self.normal_class = None
        self.classes = None
        self.classes_names = None
        self.fields = None
        self.external_units = 10
        self.internal_units = 50
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.norm_bias = 0
        self.model_type = model_type
        self.multiclass = multiclass
        self.dummies = dict()

    def dataset_reader(self, filename, column_names=None):
        """
        The dataset_reader method allows to read a csv file and returns the dataset,
        dropping the samples having missing values, and to set the columns names.

        :param filename: it is the name of the file (with its path) which contains
                              the dataset
        :param column_names: it has to be a list of strings in order to set the
                              columns names, None otherwise (None by default)

        :return: the dataframe representing the dataset
        """
        aux_df = pd.read_csv(filename, header=None)
        aux_df.dropna(inplace=True, axis=1)
        df = aux_df.copy()
        del aux_df
        if not (column_names is None):
            df.columns = column_names
        return df

    def _model_creation(self, data=None, labels=None):
        """
        The _model_creation method is used in order to create the model when the fit
        method is called.

        :param data: it is the training set (None by default)
        :param labels: it is the list of labels (None by default)
        """
        if self.model_type == 'DNN':
            input_dimension = data.shape[1]
            output_dimension = labels.shape[1]

            self.model = Sequential()
            self.model.add(Dense(self.external_units,
                                 input_dim=input_dimension,
                                 activation='relu'))
            self.model.add(Dense(self.internal_units,
                                 input_dim=input_dimension,
                                 activation='relu'))
            self.model.add(Dense(self.external_units,
                                 input_dim=input_dimension,
                                 activation='relu'))
            self.model.add(Dense(1, kernel_initializer='normal'))
            self.model.add(Dense(output_dimension, activation='softmax'))
            self.model.compile(loss=self.loss, optimizer=self.optimizer)
        elif self.model_type == 'SVM':
            self.model = CalibratedClassifierCV(LinearSVC())
        elif self.model_type == 'DT':
            self.model = DecisionTreeClassifier()
        elif self.model_type == 'RF':
            self.model = RandomForestClassifier()
        elif self.model_type == 'KNN':
            self.model = KNeighborsClassifier(int(data.shape[1] ** (1 / 2)))

    def fit(self, x_train, y_train, validation_fraction=0, epochs_number=1000,
            pat=5):
        """
        The fit method is used in order to create and fit the classifier.

        :param x_train: it is the training set
        :param y_train: it is the list of training labels
        :param validation_fraction: is the fraction of samples to use as validation
                                    samples, and in this case the validation loss
                                    will be evaluated in order to decide when to
                                    stop the training of the classifier (0 by
                                    default, used only for the DNN model)
        :param epochs_number: it is the maximum number of training epochs, if the
                                    classifier does not stop automatically its
                                    fitting before (1000 by default, used only for
                                    the DNN model)
        :param pat: it is the patience, so the maximum number of epochs without any
                                    improvement in the validation loss (if
                                    validation_fraction is not equal to zero) or in
                                    the loss (otherwise) after which the classifier
                                    stops its training (5 by default, used only for
                                    the DNN model)
        """
        if self.model_type == 'DNN':
            self._DNN_fit(x_train, y_train, validation_fraction, epochs_number, pat)
        else:
            self._model_creation()
            self.model.fit(x_train, np.argmax(y_train, axis=1))

    def _DNN_fit(self, x_train, y_train, validation_fraction=0,
                 epochs_number=1000, pat=5):
        """
        The _DNN_fit method is used in order to create and fit the DNN classifier,
        by automatically stopping the training phase after a predefined number of
        epochs or after a certain number of epochs without any improvement in the
        considered loss metric.

        :param x_train: it is the training set
        :param y_train: it is the list of training labels
        :param validation_fraction: is the fraction of samples to use as validation
                                    samples, and in this case the validation loss
                                    will be evaluated in order to decide when to
                                    stop the training of the classifier (0 by
                                    default)
        :param epochs_number: it is the maximum number of training epochs, if the
                                    classifier does not stop automatically its
                                    fitting before (1000 by default)
        :param pat: it is the patience, so the maximum number of epochs without any
                                    improvement in the validation loss (if
                                    validation_fraction is not equal to zero) or in
                                    the loss (otherwise) after which the classifier
                                    stops its training (5 by default)
        """
        self._model_creation(x_train, y_train)
        if validation_fraction == 0:
            monitor = EarlyStopping(monitor='loss',
                                    min_delta=1e-3,
                                    patience=pat,
                                    verbose=1,
                                    mode='auto',
                                    restore_best_weights=True)
            self.model.fit(x_train, y_train, verbose=2, epochs=epochs_number,
                           callbacks=[monitor])
        else:
            x_train, x_val, y_train, y_val = self.split_data(x_train,
                                                             y_train,
                                                             validation_fraction)
            monitor = EarlyStopping(monitor='val_loss',
                                    min_delta=1e-3,
                                    patience=pat,
                                    verbose=1,
                                    mode='auto',
                                    restore_best_weights=True)
            self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                           callbacks=[monitor], verbose=2, epochs=epochs_number)

    def _encode_text(self, df, name, analysisFLAG=False):
        """
        The _encode_text method is used by the compute_dataset method in order to
        encode the textual elements of a column as dummy values.

        :param df: it is the dataframe which represents the dataset
        :param name: it is the name of the column on which compute the dummy values
        :param analysisFLAG: it has to be True if the encoding is related to an
                             analysis of a raw dataset by using a previously fitted
                             analyzer, False otherwise (False by default)

        :return: the managed dataframe which represents the dataset
        """
        if analysisFLAG is False:
            dummies = pd.get_dummies(df[name])
            self.dummies[name] = dummies.columns
        else:
            aux = df[name]
            columns = self.dummies[name]
            aux = aux.astype(pd.CategoricalDtype(categories=columns))
            dummies = pd.get_dummies(aux)
        for x in dummies.columns:
            dummy_name = f"{name}-{x}"
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)
        return df

    def compute_dataset(self, df, labels_column=None, normal_class=None, analysisFLAG=False):
        """
        The compute_dataset method computes the dataset by extracting the dummy
        values corresponding to the textual columns and dropping the rows showing
        missing values.

        :param df: it is the dataframe which represents the dataset
        :param labels_column: it is the name of the column in which are contained
                              the labels related to the samples, and if it is None
                              then all the columns will be considered in computing
                              the features and no labels list will be computed (None
                              by default)
        :param normal_class: it is the name of the non-anomalous class which will be
                              managed and associated to an attribute of the object
                              in order to use it in the evaluation step (optional,
                              None by default)
        :param analysisFLAG: it has to be False if the dataset has to be computed
                              during the fitting of the system, True otherwise
                              (False by default)

        :return: the dataset of the features, and the list of labels (if
                             labels_column is not None)
        """
        df_columns = df.columns
        if not (labels_column is None):
            self.fields = []
            for column in df_columns:
                if column != labels_column:
                    self.fields.append(column)

        aux_df = df.copy()
        aux = df.values[0]
        for i in range(len(df_columns) - 1):
            aux = aux_df[df_columns[i]].values
            if type(aux[0]) is str:
                aux_df = self._encode_text(aux_df, df_columns[i], analysisFLAG)
        aux_df.dropna(inplace=True, axis=1)

        if labels_column is None:
            x = aux_df.to_numpy()
            return x
        else:
            if not (normal_class is None):
                idx = self._class_index(aux_df, normal_class, labels_column)
                if self.multiclass is False:
                    for i in range(len(aux_df[labels_column])):
                        if aux_df[labels_column].values[i] != normal_class:
                            aux_df[labels_column].values[i] = 'Anomalous'

            x_columns = aux_df.columns.drop(labels_column)
            x = aux_df[x_columns].values
            dummies = pd.get_dummies(aux_df[labels_column])
            y = dummies.values

            self._find_classes(aux_df, labels_column, y)

            if not (normal_class is None):
                self.normal_class = int(np.argmax(y[idx]))
            return x, y

    def _find_classes(self, df, labels_column, y):
        """
        The _find_classes method is used by the compute_dataset method in order to
        store the indexes associated to each class label and the ordered names in
        classes and classes_names attributes, respectively.

        :param df: it is the dataframe representing the dataset
        :param labels_column: it is the name of the column in which are contained
                              the labels related to the samples, and if it is None
                              then all the columns will be considered in computing
                              the features and no labels list will be computed (None
                              by default)
        :param y: it is the list of labels
        """
        unique_df = df.drop_duplicates(subset=[labels_column])
        classes = unique_df[labels_column].values.tolist()
        self.classes = []
        self.classes_names = []
        for class_name in classes:
            idx = self._class_index(df, class_name, labels_column)
            self.classes.append(np.argmax(y[idx]))
            self.classes_names.append(class_name)

    def _class_index(self, df, class_name, labels_column):
        """
        The _class_index method is used in order to find the index of the first
        sample of the dataset belonging to a specific class.

        :param df: it is the dataframe representing the dataset
        :param class_name: it is the class of which search the index of the first
                              sample in the dataset
        :param labels_column: it is the name of the column in which are contained
                              the labels related to the samples, and if it is None
                              then all the columns will be considered in computing
                              the features and no labels list will be computed (None
                              by default)

        :return: the index of the first sample belonging to the chosen class
        """
        labels = df[labels_column].values.tolist()
        return labels.index(class_name)

    def split_data(self, data, labels, test_fraction, idx_flag=False):
        """
        The split_data method allows to randomly split the dataset and the related
        list of labels, in order to obtain two different dataset (training set and
        test set) and the related list of labels, and eventually indexes related to
        the samples in the original dataset.

        :param data: it is the dataset which has to be splitted
        :param labels: it is the list of labels
        :param test_fraction: it is the fraction of samples of the whole dataset
                              which have to be used as test set (used as fraction of
                              the whole dataset if a value less than 1 is used, as
                              the number of testing samples otherwise)
        :param idx_flag: it has to be True in order to return also the lists of
                         training and testing indexes (False by default)

        :return: the training set, the test set, the training labels and the test
                 labels if idx_flag = False, also the list of training indexes and
                 the list of test indexes otherwise (all in numpy.array format)
        """

        n_samples = len(data)

        if test_fraction < 1:
            test_fraction = int(test_fraction * n_samples)
        idx = list(np.random.permutation(n_samples))
        idx_test = np.asarray(idx[0:test_fraction])
        idx_train = np.asarray(idx[test_fraction:])
        X_train = np.asarray([data[i] for i in idx_train])
        X_test = np.asarray([data[i] for i in idx_test])
        y_train = np.asarray([labels[i] for i in idx_train])
        y_test = np.asarray([labels[i] for i in idx_test])
        if idx_flag:
            return X_train, X_test, y_train, y_test, idx_train, idx_test
        return X_train, X_test, y_train, y_test

    def evaluate_performance(self, x_test, y_test, norm_bias=0, aucFLAG=False):
        """
        The evaluate_performance method is used to test the classifier, and to show
        the related performance.

        :param x_test: is the test set
        :param y_test: is the list of test labels
        :param norm_bias: it is the value to subtract to the normal class score
                          before predicting the classes to which the samples belong,
                          reducing the missed alarms but incrementing the false
                          positive rate if its value is greater than zero (0 by
                          default)
        :param aucFLAG: it has to be True in order to return also the AUC (Area
                          Under the Curve) value in case of a binary (2-class)
                          classification, False otherwise (False by default)

        :return: the mean accuracy, the false negatives (normal) number, the false
                 positives (anomalous) number, the true positives number, the
                 true negatives number, and the AUC value (if aucFLAG is True in a
                 binary classification)
        """
        if self.model_type == 'DNN':
            pred = self.model.predict(x_test)
        else:
            pred = self.model.predict_proba(x_test)
        pred = self._apply_bias(pred, norm_bias)
        y_pred = np.argmax(pred, axis=1)
        y_eval = np.argmax(y_test, axis=1)
        score, falseNormal, falseAnomalous, trueAnomalous, trueNormal = self._classification_metrics(y_eval, y_pred,
                                                                                                     x_test)
        if self.multiclass is False:
            auc = self._roc(y_pred, pred)
            if aucFLAG is True:
                return score, falseNormal, falseAnomalous, trueAnomalous, trueNormal, auc
        return score, falseNormal, falseAnomalous, trueAnomalous, trueNormal

    def _apply_bias(self, pred, norm_bias):
        """
        The _apply_bias method subtract a static bias from the scores related to the
        normal class.

        :param pred: it is the scores matrix
        :param norm_bias: it is the bias value

        :return: the manages scores matrix
        """
        self.norm_bias = norm_bias
        if norm_bias != 0:
            for i in range(len(pred)):
                pred[i][self.normal_class] += norm_bias
        return pred

    def _classification_metrics(self, y_eval, y_pred, x_test):
        """
        The _classification_metrics method is used to compute some performance
        evaluations on the classifier.

        :param y_eval: is the list of evaluated test labels
        :param y_pred: is the list of predicted test labels
        :param x_test: is the test dataset

        :return: the mean accuracy, the false negatives (normal) number, the false
                 positives (anomalous) number, the true positives number and the
                 true negatives number
        """
        N = len(y_pred)
        if self.model_type == 'DNN':
            score = metrics.accuracy_score(y_eval, y_pred)
        else:
            score = self.model.score(x_test, y_eval)
        print("Accuracy: {}".format(score))
        print()
        if not (self.normal_class is None):
            totalNormal = np.sum([x == self.normal_class for x in y_eval])
            totalAnomalous = len(y_eval) - totalNormal
            falseNormal = np.sum([((y_eval[x] != y_pred[x]) and y_pred[x] == self.normal_class) for x in range(N)])
            falseAnomalous = np.sum([((y_eval[x] != y_pred[x]) and y_pred[x] != self.normal_class) for x in range(N)])
            trueAnomalous = totalAnomalous - falseNormal
            trueNormal = totalNormal - falseAnomalous
            self._confusion_matrix(trueNormal, falseNormal, trueAnomalous, falseAnomalous)
            print()
            print(metrics.classification_report(y_eval, y_pred, labels=self.classes, target_names=self.classes_names))
        return score, falseNormal, falseAnomalous, trueAnomalous, trueNormal

    def _roc(self, y_true, y_score):
        """
        The _roc method shows the ROC (Receiver operating characteristic) curve
        related to the positive (anomalous) class, and computes the correspondent
        AUC (Area Under the Curve) value (this method is used in case of 2-class
        classification).

        :param y_true: it is the list of labels
        :param y_score: it is the matrix which represents the scores related to each
                        class for each sample

        :return: the auc value
        """
        idx = 1
        if self.normal_class == 1:
            idx = 0
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score[:, idx], pos_label=idx)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        lbl = 'ROC curve (area = %0.4f)' % roc_auc
        plt.plot(fpr, tpr, color='darkorange', label=lbl)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend(loc="lower right")
        plt.show()
        return roc_auc

    def _confusion_matrix(self, trueNormal, falseNormal, trueAnomalous, falseAnomalous):
        """
        The _confusion_matrix method computes the 2-class confusion matrix between
        the anomalous and the normal samples.

        :param trueNormal: it is the number of normal samples predicted as normal
        :param falseNormal: it is the number of anomalous samples predicted as
                               normal
        :param trueAnomalous: it is the number of anomalous samples predicted as
                               anomalous
        :param falseAnomalous: it is the number of normal samples predicted as
                               anomalous
        """
        norm_space = "         "
        anomal_space = "         "
        for i in range(6 - len(str(trueNormal))):
            norm_space += " "
        for i in range(6 - len(str(falseNormal))):
            anomal_space += " "
        print("                       Normal       Anomalous")
        print("Predicted Normal        " + str(trueNormal) + norm_space + str(falseAnomalous))
        print("Predicted Anomalous     " + str(falseNormal) + anomal_space + str(trueAnomalous))

    def save(self, filename='IDS'):
        """
        The save method allows to save the classifier model and the settings
        (classifier model, multiclass flag, classes names, classes labels, log
        parameters, index of the normal class and the last used bias for the normal
        score) into a .h5 file (in the case of a DNN) or in a .pkl file (otherwise)
        and a txt file, respectively (eventual files having the same name will be
        replaced).

        :param filename: it is the name of the files (without their extension) in
                         which save the model and the settings ('IDS' by default)
        """
        settings_file = filename + '.txt'
        if os.path.exists(settings_file):
            os.remove(settings_file)

        f = open(settings_file, "a")
        f.write("Model=" + str(self.model_type))
        f.write("\nMulticlass=" + str(self.multiclass))
        f.write("\nNormal=" + str(self.normal_class))
        cn = "Names="
        cl = "Labels="
        flds = "Fields="

        f.write("\n%s" % cn)
        for i in range(len(self.classes_names)):
            f.write("%s " % self.classes_names[i])

        f.write("\n%s" % cl)
        for i in range(len(self.classes)):
            f.write("%s " % str(self.classes[i]))

        f.write("\n%s" % flds)
        for field in self.fields:
            aux_field = field.split()
            if len(aux_field) > 1:
                field = ""
                for a in aux_field:
                    field += a
            f.write("%s " % field)

        f.write("\nBias=%s" % str(self.norm_bias))

        f.close()

        dummy_name = filename + '_dummies.pkl'
        with open(dummy_name, 'wb') as f:
            pickle.dump(self.dummies, f, pickle.HIGHEST_PROTOCOL)

        if self.model_type == 'DNN':
            filename = filename + ".h5"
            self.model.save(filename)
        else:
            filename = filename + ".pkl"
            joblib.dump(self.model, filename)

        print("Dummies values saved in " + dummy_name)
        print("Model saved in " + filename)
        print("Settings saved in " + settings_file)

    def load(self, filename):
        """
        The load method allows to load the classifier model and the settings
        (classes names, classes labels, log parameters and index of the normal
        class) from a .h5 file and a txt file, respectively.

        :param filename: it is the name of the files (without their extension) from
                         which load the model end the settings ('IDS' by default)
        """
        settings_file = filename + '.txt'
        settingFLAG = not (os.path.exists(settings_file))
        h5FLAG = not (os.path.exists(filename + '.h5'))
        pklFLAG = not (os.path.exists(filename + '.pkl'))
        if settingFLAG or (h5FLAG and pklFLAG):
            print("IDS file named " + filename + " not found")
            return

        f = open(settings_file, "r")
        settings = f.readlines()
        for s in settings:
            aux = s.split('=')
            aux = aux[1]
            if "Normal=" in s:
                self.normal_class = int(aux)
            if "Names=" in s:
                self.classes_names = aux.split()
            if "Labels=" in s:
                self.classes = []
                aux_labels = aux.split()
                for idx in aux_labels:
                    self.classes.append(int(idx))
            if "Fields=" in s:
                self.fields = aux.split()
            if "Bias=" in s:
                self.norm_bias = float(aux)
            if "Model=" in s:
                self.model_type = aux
            if "Multiclass=" in s:
                self.multiclass = False
                if aux == 'True':
                    self.multiclass = True
        f.close()

        if self.model_type == 'DNN':
            self.model = load_model(filename + ".h5")
        else:
            self.model = joblib.load(filename + ".pkl")

        with open(filename + '_dummies.pkl', 'rb') as f:
            self.dummies = pickle.load(f)

    def analysis(self, filename, anomalousFLAG=False, n=None, norm_bias=0):
        """
        The analysis method allows to analize a log file, classifying the samples
        belonging to the input file as belonging to the predicted class (the samples
        having missing values will be removed).

        :param filename: it can be a string representing the name of the file (with
                              its path) which has to be analyzed, or the dataframe
        :param anomalousFLAG: it has to be True in order to return the
                              classification related to the only anomalous samples
                              (False by default)
        :param n: it is the number of samples which have to be returned, or None in
                              oder to return all the samples
        :param norm_bias: it is the value to subtract to the normal class score
                              before predicting the classes to which the samples
                              belong, reducing the missed alarms but incrementing
                              the false positive rate if its value is positive (0 by
                              default)

        :return: the dataframe including the predicted labels in the outcome column
        """
        if type(filename) is str:
            aux_df = self.dataset_reader(filename)
        else:
            aux_df = filename.copy()
        aux_df.columns = self.fields
        df = aux_df.copy()

        x = self.compute_dataset(aux_df, analysisFLAG=True)
        if self.model_type == 'DNN':
            pred = self.model.predict(x)
        else:
            pred = self.model.predict_proba(x)

        pred = self._apply_bias(pred, norm_bias)
        y_pred = np.argmax(pred, axis=1)

        labels = list()
        names = self.classes_names
        for idx in y_pred:
            labels.append(names[self.classes.index(idx)])

        df['outcome'] = labels

        if anomalousFLAG is True:
            normal_name = self.classes_names[self.classes.index(self.normal_class)]
            df.drop(df[df['outcome'] == normal_name].index, inplace=True)

        if n is None:
            n = len(labels)

        return df.iloc[0:n, 0:]