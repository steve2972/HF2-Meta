import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

import numpy as np
import random
import time


##############################
# TRAINING FUNCTIONS         #
##############################

def update_weights(old_weights, grads, lr):
    new_weights = list()
    i, j = 0, 0
    while i < len(old_weights) and j < len(grads):
        if old_weights[i].shape == grads[j].shape:
            new_weights.append(old_weights[i] - lr * grads[j])
            i += 1
            j += 1
        else:
            new_weights.append(old_weights[i])
            i += 1
    return new_weights

def weight_scaling_factor(n):
    return 1/n

def weight_difference(old_weights, weights):
    new_weights = list()
    for i in range(len(weights)):
        new_weights.append(old_weights[i] - weights[i])
    return new_weights

def scale_model_weights(weights, scalar):
    weight_final = []
    steps = len(weights)
    for i in range(steps):
        weight_final.append(scalar * weights[i])
    return weight_final

def sum_weights(weight_list):
    avg_weight = list()
    for weight in zip(*weight_list):
        layer_mean = tf.math.reduce_sum(weight, axis=0)
        avg_weight.append(layer_mean)

    return avg_weight

def reduce_grads_like(grads, model):
    new_grads = list()
    i, j = 0, 0
    while i < len(grads) and j < len(model.trainable_weights):
        if  grads[i].shape == model.trainable_weights[j].shape:
            new_grads.append(grads[i])
            i += 1
            j += 1
        else:
            i += 1
    return new_grads

@tf.function
def train_on_batch(model, loss, opt, X, y):
    with tf.GradientTape() as tape:
        y_hat = model(X, training=True)
        loss_value = loss(y, y_hat)
    grads = tape.gradient(loss_value, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))

##############################
# TESTING FUNCTIONS          #
##############################

def test_model(model, comm_round, start_time, dataset, test_rounds):
    old_weights = model.get_weights()
    test_data = dataset.get_test_dataset()
    losses, accs = list(), list()
    for finetune_data, validation_data in test_data:
        try:
            model.fit(finetune_data, epochs=test_rounds, verbose=0)
            loss, acc = model.evaluate(validation_data, verbose=0)
            losses.append(loss)
            accs.append(acc)
        except:
            print("Error")
        model.set_weights(old_weights)
    elps = time.time() - start_time
    acc, loss = np.average(accs), np.average(losses)
    print('round: {:03d} | acc= {:.4f} | loss: {:.4f} | time: {:.2f}'
            .format(comm_round, acc, loss, elps))

    return acc, loss

def evaluate_model(model, dataset, inner_batch_size, shots, classes, test_rounds, num_tests):
    old_weights = model.get_weights()
    losses = list()
    accs = list()
    for _ in range(num_tests):
        data, x_test, y_test = dataset.get_mini_dataset(inner_batch_size, 1, shots, classes, split=True)
        model.fit(data, epochs=test_rounds, verbose=0)
        evaluate = model.evaluate(x_test, y_test, verbose=0)
        losses.append(evaluate[0])
        accs.append(evaluate[1])
        model.set_weights(old_weights)
    acc, std, loss = np.average(accs), np.std(accs), np.average(losses)
    return acc, std, loss