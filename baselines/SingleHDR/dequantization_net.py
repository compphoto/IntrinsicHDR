import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

class Dequantization_net(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def inference(self, input_images):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images)

    def loss(self, predictions, targets):
        """Compute the necessary loss for training.
        Args:
        Returns:
        """
        return tf.reduce_mean(tf.square(predictions - targets))

    def down(self, x, outChannels, filterSize):
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(outChannels, filterSize, 1, 'same')(x), 0.1)
        x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(outChannels, filterSize, 1, 'same')(x), 0.1)
        return x

    def up(self, x, outChannels, skpCn):
        x = tf.image.resize_bilinear(x, 2*tf.shape(x)[1:3])
        x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(outChannels, 3, 1, 'same')(x), 0.1)
        x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(outChannels, 3, 1, 'same')(tf.concat([x, skpCn], -1)), 0.1)
        return x

    def _build_model(self, input_images):
        print(input_images.get_shape().as_list())
        x = tf.nn.leaky_relu(tf.keras.layers.Conv2D(16, 7, 1, 'same')(input_images), 0.1)
        s1 = tf.nn.leaky_relu(tf.keras.layers.Conv2D(16, 7, 1, 'same')(x), 0.1)
        s2 = self.down(s1, 32, 5)
        s3 = self.down(s2, 64, 3)
        s4 = self.down(s3, 128, 3)
        x = self.down(s4, 256, 3)
        # x = self.down(s5, 512, 3)
        # x = self.up(x, 512, s5)
        x = self.up(x, 128, s4)
        x = self.up(x, 64, s3)
        x = self.up(x, 32, s2)
        x = self.up(x, 16, s1)
        x = tf.nn.tanh(tf.keras.layers.Conv2D( 3, 3, 1, 'same')(x))
        output = input_images + x
        return output