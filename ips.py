from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from abc import ABCMeta
from sklearn.svm import SVC, SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

build_acc = []
floor_acc = []

long_scaler = MinMaxScaler()
lat_scaler = MinMaxScaler()


def load(train_path, valid_path):

    train_data_frame = pd.read_csv(train_path)
    train_data_frame = train_data_frame[(train_data_frame.LATITUDE != 0.0) & (train_data_frame.LONGITUDE != 0.0) &
                                        (train_data_frame.BUILDINGID == 2)]

    test_data_frame = pd.read_csv(valid_path)
    test_data_frame = test_data_frame[(test_data_frame.LATITUDE != 0.0) & (test_data_frame.LONGITUDE != 0.0) &
                                      (test_data_frame.BUILDINGID == 2)]

    rest_data_frame = train_data_frame
    valid_data_trame = pd.DataFrame(columns=train_data_frame.columns)
    valid_num = int(len(train_data_frame)/10)

    sample_row = rest_data_frame.sample(valid_num)
    rest_data_frame = rest_data_frame.drop(sample_row.index)

    valid_data_trame = valid_data_trame.append(sample_row)
    train_data_frame = rest_data_frame

    # Split data frame and return
    training_x = train_data_frame.get_values().T[:520].T
    training_y = train_data_frame.get_values().T[[520, 521, 522, 523], :].T

    validation_x = valid_data_trame.get_values().T[:520].T
    validation_y = valid_data_trame.get_values().T[[520, 521, 522, 523], :].T

    testing_x = test_data_frame.get_values().T[:520].T
    testing_y = test_data_frame.get_values().T[[520, 521, 522, 523], :].T

    return training_x, training_y, validation_x, validation_y, testing_x, testing_y


def normalize_x(x_array):
    res = np.copy(x_array).astype(np.float)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if res[i][j] == 100:
                res[i][j] = 0
            else:
                res[i][j] = -0.01 * res[i][j]
    return res


def normalize_y(longs, lats):
    global long_scaler
    global lat_scaler
    longs = np.reshape(longs, [-1, 1])
    lats = np.reshape(lats, [-1, 1])
    long_scaler.fit(longs)
    lat_scaler.fit(lats)
    return np.reshape(long_scaler.transform(longs), [-1]), \
            np.reshape(lat_scaler.transform(lats), [-1])


def reverse_normalizeY(longs, lats):
    global long_scaler
    global lat_scaler
    longs = np.reshape(longs, [-1, 1])
    lats = np.reshape(lats, [-1, 1])
    return np.reshape(long_scaler.inverse_transform(longs), [-1]), \
            np.reshape(lat_scaler.inverse_transform(lats), [-1])


class Model( object):
    __metaclass__ = ABCMeta

    # ML model object
    longitude_regression_model = None
    latitude_regression_model = None

    # Training data
    normalize_x = None
    longitude_normalize_y = None
    latitude_normalize_y = None

    def __init__(self):
        pass

    def _preprocess(self, x, y):
        self.normalize_x = normalize_x(x)
        self.longitude_normalize_y, self.latitude_normalize_y = normalize_y(y[:, 0], y[:, 1])
        self.floorID_y = y[:, 2]
        self.buildingID_y = y[:, 3]

    def fit(self, x, y):
        # Data pre-processing
        self._preprocess(x, y)
        self.longitude_regression_model.fit(self.normalize_x, self.longitude_normalize_y)
        self.latitude_regression_model.fit(self.normalize_x, self.latitude_normalize_y)

    def predict(self, x):
        # Testing
        x = normalize_x(x)
        predict_longitude = self.longitude_regression_model.predict(x)
        predict_latitude = self.latitude_regression_model.predict(x)

        # Reverse normalization
        predict_longitude, predict_latitude = reverse_normalizeY(predict_longitude, predict_latitude)

        # Return the result
        res = np.concatenate((np.expand_dims(predict_longitude, axis=-1),
            np.expand_dims(predict_latitude, axis=-1)), axis=-1)

        return res

    def error(self, x, y):
        _y = self.predict(x)
        dist = np.sqrt(np.square(_y[:, 0] - y[:, 0]) + np.square(_y[:, 1] - y[:, 1]))
        #plot_dist_error(dist)
        #map_plot(_y, y)
        print(min(dist), np.mean(dist), max(dist))

        return dist


class SVM(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = SVR(verbose=True)
        self.latitude_regression_model = SVR(verbose=True)
        self.floor_classifier = SVC(verbose=True)
        #self.building_classifier = SVC(verbose=True)


class RandomForest(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = RandomForestRegressor()
        self.latitude_regression_model = RandomForestRegressor()
        self.floor_classifier = RandomForestClassifier()
        #self.building_classifier = RandomForestClassifier()


class GradientBoostingDecisionTree(Model):
    def __init__(self):
        super().__init__()
        self.longitude_regression_model = GradientBoostingRegressor()
        self.latitude_regression_model = GradientBoostingRegressor()
        self.floor_classifier = GradientBoostingClassifier()
        #self.building_classifier = GradientBoostingClassifier()


def plot_dist_error(dist):
    dist = dist.tolist()
    t = np.arange(0.0, len(dist), 1)
    y_mean = [np.mean(dist)]*len(dist)

    fig, ax = plt.subplots()
    ax.plot(t, dist, label='Error value')
    ax.plot(t, y_mean, label='Mean', linestyle='--', color='red')

    ax.set(xlabel='Data points (ID)', ylabel='Error (m)',
           title='Error value of points using baseline Random Forest')
    ax.grid()

    fig.savefig("error.png")
    plt.legend()
    plt.show()


def map_plot(_y, y):
    # take the first two features
    h = .02  # step size in the mesh

    # Calculate min, max and limits
    x_min, x_max = y[:, 0].min() - 1, y[:, 0].max() + 1
    y_min, y_max = y[:, 1].min() - 1, y[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Put the result into a color plot
    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], alpha=0.5, s=7, marker='o', label='actual position')
    plt.scatter(_y[:, 0], _y[:, 1], alpha=0.3, s=4, marker='o', label='predicted position')
    plt.xlim(xx.min()-20, xx.max()+20)
    plt.ylim(yy.min()-20, yy.max()+20)
    plt.legend()
    plt.grid()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title("Data points")
    plt.tight_layout(pad=0)
    plt.savefig('data_points.png')
    plt.show()


def bar_plot(lst):

    method = ['SVM', 'Random Forest', 'Boosting Decision Tree']  # nom = ['Obama', 'Romney', 'Johson', 'Stein']

    data = lst  # [51.258, 47.384, 0.992, 0.365]
    x_pos = [x for x in range(len(data))]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorsBox = ['r', 'b', 'c', 'y', 'purple', 'green', 'k', 'magenta', 'firebrick']

    for x in range(len(data)):
        ax.bar(x_pos[x], data[x], color=colorsBox[x], label=method[x], align='center')

    #plt.xlabel('Nominees')
    plt.title("Floor classification for buildingID=2")
    plt.ylabel('Accuracy(%)')
    plt.ylim(bottom=50, top=100)
    plt.xticks(x_pos, method)
    plt.legend()
    plt.tight_layout()
    plt.savefig('floor.pdf', bbox_inches=None)
    plt.close()


if __name__ == '__main__':

    train_csv_path = 'TrainingData.csv'
    valid_csv_path = 'ValidationData.csv'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load(train_csv_path, valid_csv_path)
    # Training
    SVM = SVM()
    SVM.fit(train_x, train_y)

    RF = RandomForest()
    RF.fit(train_x, train_y)

    GBDT = GradientBoostingDecisionTree()
    GBDT.fit(train_x, train_y)

    RF.error(test_x, test_y)
