from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

keras = tf.keras


ACTION_SET = ['LEFT_SLOW', 'LEFT', 'LEFT_FAST',
              'SLOW',     'NOTHING', 'FAST',
              'RIGHT_SLOW', 'RIGHT', 'RIGHT_FAST']


def fix_dim(matrix):
    return np.expand_dims(np.expand_dims(matrix, 0), 3)


class DataSet:
    def __init__(self, length, road_shape):
        self.length = length
        self.state = fix_dim(road_shape)
        self.action = np.array(0)
        self.reward = np.expand_dims(np.zeros(len(ACTION_SET)), 0)
        self.next_state = fix_dim(road_shape)
        self.n_games = 0
        self.road_shape = road_shape

    def clear(self):
        self.state = fix_dim(self.road_shape)
        self.action = np.array(0)
        self.reward = np.expand_dims(np.zeros(len(ACTION_SET)), 0)
        self.next_state = fix_dim(self.road_shape)
        self.n_games = 0

    def append(self, state, action, reward, next_state):
        self.state = np.append(self.state, fix_dim(state), 0)
        self.action = np.append(self.action, action)
        temp = np.zeros(len(ACTION_SET))
        temp[ACTION_SET.index(action)] = reward
        self.reward = np.append(self.reward, np.expand_dims(temp, 0), 0)
        self.next_state = np.append(self.next_state, fix_dim(next_state), 0)

    def cut_first_sample(self):
        self.state = self.state[1:]
        self.action = self.action[1:]
        self.reward = self.reward[1:]
        self.next_state = self.next_state[1:]


def calc_reward(car):
    reward = 0
    if car.status is 'collision':
        reward -= 500
    elif car.status is 'completed':
        reward += 200
    elif car.status is 'illegal_action':
        reward -= 100
    else:
        reward += car.speed
        reward += car.x

    return reward


# Creating a simple CNN model in keras using functional API


def create_model(IMG_SHAPE):

    inputs = keras.Input(shape=IMG_SHAPE, dtype=float)

    conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    maxpool_1 = keras.layers.MaxPooling2D((2, 2))(conv_1)

    conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1)

    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)

    conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)

    flatten = keras.layers.Flatten()(conv_3)

    dense_1 = keras.layers.Dense(64, activation='relu')(flatten)

    output = keras.layers.Dense(len(ACTION_SET), activation='softmax')(dense_1)

    model = keras.Model(inputs=inputs, outputs=output)

    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy()

    return model, optimizer, loss_fn


# Get the model etc.

def train(model, optimizer, loss_fn, data_set, epoches):
    batch_size = 128
    gamma = 0.8

    x_train = tf.cast(data_set.state, 'float32')
    y_train = data_set.reward + gamma*(model(tf.cast(data_set.next_state, 'float32')))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    train_acc_metric = keras.metrics.CategoricalCrossentropy()

    # Iterate over epochs.
    for epoch in range(epoches):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer. The operations that the layer applies
                # to its inputs are going to be recorded on the GradientTape.

                logits = model(x_batch_train)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric(y_batch_train, logits)

            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * batch_size))
            # if step >= int(x_train.shape[0]) / batch_size:
            #     break

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

    model.save('merge_simulator_net.h5')  # creates a HDF5 file 'my_model.h5'
    print('---------Model Saved---------')


def get_action(state, model, ep=0.3):
    # fill here any pre-processing done on the original state image

    action_set = model(tf.cast(state, 'float32'))
    action = ACTION_SET[int(np.argmax(action_set))]

    return epsilon_greedy(action=action, action_set=ACTION_SET, epsilon=ep)


def epsilon_greedy(action, action_set, epsilon=0.5):

    if np.random.rand() > epsilon:
        return action
    else:
        return action_set[int(np.random.randint(0, len(action_set)-1))]
