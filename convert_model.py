from cipheydists import get_model, get_lite_model
import tensorflow as tf
from pathlib import Path
import sys

def convert_model(src: str, dst : str):
    model = tf.keras.models.load_model(src)
    print(path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    res = converter.convert()
    open(dst, "wb").write(res)

if __name__ == '__main__':
    convert_model(sys.argv[1], sys.argv[2])
