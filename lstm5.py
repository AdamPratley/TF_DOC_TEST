# univariate cnn lstm example
from datetime import datetime
from numpy import array, ndarray
import numpy as np
from keras.models import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dense, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from type_enforced import Enforcer
from sklearn.metrics import mean_squared_error

matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
    }
)

@Enforcer
class EpochData(Callback):
    def __init__(self, max_epochs: int, silent: bool = False):
        self.start: datetime = datetime.now()
        self.elapsed: datetime = datetime.now()
        self.history: list = []
        self.max_epochs = max_epochs
        self.silent = silent

    def plotdata(self):
        data: ndarray = np.array(self.history)

        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel = "epoch"
        plt.ylabel = "loss"
        plt.title(f"loss change by epoch, Elapsed:{self.elapsed}")
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        self.elapsed = datetime.now() - self.start

        print(
            f"{int((epoch/self.max_epochs)*100)}% {str(self.elapsed)[:7]} loss: {'{0:.12f}'.format(logs['loss'])}{' '* 10}\r",
            end="",
        )
        self.history.append((epoch, logs["loss"]))

        if epoch == (self.max_epochs - 1):
            print(
                f"100% {str(self.elapsed)[:7]} loss: {'{0:.12f}'.format(logs['loss'])}{' '* 10}",
                end="\n",
            )

            if self.silent:
                return

            self.plotdata()


@Enforcer
class Job:
    def __init__(
        self,
        pred_amount: int = 10,
        data_amount: int = 80,
        h_layer: int = 1,
        units: int = 1,
        max_epochs: int = 1,
        n_steps_split: int = 4,
        n_features: int = 1,
        n_seq: int = 2,
        n_steps: int = 2,
        optimizer=Adam(),
        silent: bool = False,
    ):
        self.pred_amount = pred_amount
        self.data_amount = data_amount
        self.h_layer = h_layer
        self.units = units
        self.n_steps_split = n_steps_split
        self.raw_seq, self.future_seq = self.__load_data()
        self.X, self.y = self.__split_sequence(self.raw_seq)
        self.n_features = n_features
        self.n_seq = n_seq
        self.n_steps = n_steps
        self.max_epochs = max_epochs
        self.model = Sequential()
        self.new_seq = self.raw_seq.copy()
        self.optimizer = optimizer
        self.history = None
        self.silent = silent
        self.Stopper = EarlyStopping(monitor="loss", patience=50)
        self.X = self.X.reshape(
            (self.X.shape[0], self.n_seq, self.n_steps, self.n_features)
        )

        if (len(self.raw_seq) % (self.n_seq + self.n_steps)) != 0:
            raise ValueError(
                f"Input data must be divisable by {(self.n_seq + self.n_steps)}"
            )

        self.__build_model()

    def __split_sequence(self, sequence):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + self.n_steps_split
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def __load_data(self):
        dataframe = read_csv("csv_data.csv", usecols=[5], engine="c")
        dataset = dataframe.values.astype("float32")
        dataset = dataset.flatten(order="C")

        dataset_length = len(dataset)
        a = dataset_length - self.pred_amount - self.data_amount
        b = dataset_length - self.pred_amount
        c = a + self.pred_amount

        raw_seq = dataset[a:b].tolist()
        future_seq = dataset[c:].tolist()

        return (raw_seq, future_seq)

    def __build_model(self):
        self.model.add(
            TimeDistributed(
                Conv1D(filters=64, kernel_size=1, activation="relu"),
                input_shape=(None, self.n_steps, self.n_features),
            )
        )
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))

        for _ in range(self.h_layer):
            self.model.add(CuDNNLSTM(self.units, return_sequences=True))

        self.model.add(CuDNNLSTM(self.units))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss="mse")

        if self.silent:
            return

        self.model.summary()

    def fit_model(self):
        self.history = self.model.fit(
            self.X,
            self.y,
            epochs=self.max_epochs,
            verbose=0,
            callbacks=[EpochData(self.max_epochs, silent=True), self.Stopper],
        )

    def predict(self):
        for _ in range(self.pred_amount):
            x_input = array(self.new_seq)
            x_input = x_input.reshape(
                (
                    len(self.new_seq) // (self.n_seq + self.n_steps),
                    self.n_seq,
                    self.n_steps,
                    self.n_features,
                )
            )

            yhat = self.model.predict(x_input, verbose=0)
            self.new_seq.append(yhat[0][0])
            self.new_seq.pop(0)


y = []
x = []
scores = []
end = 25
data_amounts = list(range(2,end))
for idx,item in enumerate(data_amounts):
    data_amounts[idx] *=4



for idx, item in enumerate(data_amounts):
    
    start = datetime.now()

    job = Job(
        pred_amount=10,
        data_amount=item,
        h_layer=4,
        units=100,
        max_epochs=50_000,
        n_steps_split=4,
        n_features=1,
        n_seq=2,
        n_steps=2,
        optimizer=Adam(learning_rate=0.000001),
        silent=True,
    )

    job.fit_model()
    print(f"", end="\n")

    job.predict()

    y.append(job.history.history["loss"])
    x.append(job.history.epoch)

    RMSE = mean_squared_error(
        job.future_seq[len(job.future_seq) - job.pred_amount :],
        job.new_seq[len(job.new_seq) - job.pred_amount :],
        squared=False,
    )
    scores.append(RMSE)
    print(f"RMSE: {RMSE}")

    elapsed = (datetime.now() - start) * (end - (idx+1))
    print(f"Estimated time left: {str(elapsed)[:7]}")

# x = list(range(len(y[0])))
for idx, item in enumerate(y):
    plt.plot(x[idx], item, label=f"{data_amounts[idx]}")

plt.legend()
plt.title("Loss Value by Epoch, for $n$ Data Points")
plt.ylabel("Loss Value")
plt.xlabel("Epoch")
plt.show()

for idx, item in enumerate(scores):
    plt.bar(data_amounts[idx], item, label=f"{data_amounts[idx]}")

plt.legend()
plt.title(r"RMSE vs $n$ Data Points")
plt.ylabel("RMSE Value")
plt.xlabel(r"$n$ Data Points")
plt.show()

# xplot = list(range(1,len(job.new_seq)+1))
# plt.plot(xplot, job.new_seq, color="red", label="Prediction")
# plt.plot(xplot, job.future_seq, color="blue", label="Actual")
# plt.legend()
# plt.show()
