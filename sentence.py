import cipheycore
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    Reshape,
    Input
)
import numpy
from tensorflow.keras.models import Sequential, load_model
import random
import sys

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizer_v2.adam import Adam

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/tmp/tflog")

width = 512

def make_model():
    m = Sequential()
    m.add(Input((width,)))
    for i in range(1, 4):
        m.add(Dense(512, activation="relu"))
    m.add(Dropout(0.2))
    for i in range(1, 4):
        m.add(Dense(512, activation="relu"))

    for i in range(1, 4):
        m.add(Dense(256, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return m

# model = make_model()
# model.save("/tmp/model")
# exit(0)


def str_conv(*args):
    ret = numpy.zeros((len(args), width), 'int32')
    for arg_index in range(0, len(args)):
        s = args[arg_index]
        for index in range(0, len(s) - 1):
            ret[arg_index][index] = ord(s[index])
    return ret

model = tf.keras.models.load_model("/mnt/bigs/model")
if len(sys.argv) > 1:
    print(model.predict(str_conv(*sys.argv[1:2]))[0][0])
    exit(0)


hansard = []
twists = []
wikis = []
sentences = []

#/usr/share/dict/words
with open("/mnt/bigs/hansard.txt", "r", encoding="cp1252") as f:
    hansard = f.read().splitlines()
with open("/mnt/bigs/twist.2.txt", "r", encoding="utf8") as f:
    twists = f.read().splitlines()
# with open("/mnt/bigs/wiki-links/all.txt", "r", encoding="utf8") as f:
#     twists = f.read().splitlines()
with open("/mnt/bigs/data/benchmark-v1.0/sentences.txt", "r", encoding="utf8") as f:
    sentences = random.sample(f.read().splitlines(), 1000000)
wordlists = [sentences, hansard, twists]
print(f"Loaded {len(wordlists)} datasets")

lens = dict()
data_size = 0
analysis = cipheycore.start_analysis()
for list in wordlists:
    for word in list:
        cipheycore.continue_analysis(analysis, word)
        lens.setdefault(len(word), 0)
        lens[len(word)] += 1
        data_size += 1
cipheycore.finish_analysis(analysis)
print(f"Analysed frequencies")
"""
model = Sequential()
model.add(Input((width,)))
for i in range(1, 4):
    model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
for i in range(1, 4):
    model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

"""


def generate_ds(number: int):
    def rand_word_len():
        pos = random.randint(0, data_size)
        for elem, count in lens.items():
            pos -= count
            if pos <= 0:
                return elem
        raise 0

    fuzz_rate = 0.5

    ds_x = numpy.zeros((number, width), 'int32')
    ds_y = numpy.empty(number)
    for i in range(0, number - 1):
        if random.uniform(0, 1) < fuzz_rate:
            sample_word, = cipheycore.fuzz(analysis, rand_word_len()),
            ds_y[i] = 0
        else:
            wl = random.choice(wordlists)
            sample_word = random.choice(wl)
            ds_y[i] = 1
        if len(sample_word) > width:
            continue
        for j in range(0, len(sample_word) - 1):
            ds_x[i][j] = ord(sample_word[j])
        if (i % (number // 100)) == 0:
            print(f"generating dataset {(i / number) * 100}% complete")
    return ds_x, ds_y

stop_threshold = 0.995

opt = Adam(lr=0.00001) # , decay=.02
model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
while True:
    ds_x,ds_y = generate_ds(1000000)
    es = EarlyStopping(monitor='val_accuracy', baseline=stop_threshold)
    res = model.fit(ds_x, ds_y, callbacks=[], epochs=16, validation_split=0.2, batch_size=1024)  #
    model.save("/mnt/bigs/model")
    if res.history['val_accuracy'][-1] > stop_threshold:
        break
