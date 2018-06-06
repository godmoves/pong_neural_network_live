import tensorflow as tf
import cv2  # read in pixel data
import pong  # our class
import numpy as np  # math
import random  # random
# queue data structure. fast appends. and pops. replay memory
from collections import deque


# hyper params
ACTIONS = 3  # up,down, stay
# define our learning rate
GAMMA = 0.99
# for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 150000
ADDITIONAL_OB = 50000
# store our experiences, the size of it
REPLAY_MEMORY = 500000
# batch size to train on
BATCH = 100

# create tensorflow graph


def createGraph():

    # first convolutional layer. bias vector
    # creates an empty tensor with all elements set to zero with a shape
    W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.02))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.truncated_normal([3136, 784], stddev=0.02))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.truncated_normal([784, ACTIONS], stddev=0.02))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
    conv1 = tf.nn.relu(tf.nn.conv2d(
        s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(
        conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(
        conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5


# deep q network. feed in pixel data to graph session
def trainGraph(inp, out, sess):

    # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])  # ground truth
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices=1)
    # cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(action - gt))
    # optimization fucntion to reduce our minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize our game
    game = pong.PongGame()

    # create a queue for experience replay to store policies
    D = deque()

    # intial frame
    frame = game.getPresentFrame()
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # saver
    saver = tf.train.Saver(tf.global_variables())
    writer_long = tf.summary.FileWriter("./logs/long", sess.graph)
    writer_short = tf.summary.FileWriter("./logs/short", sess.graph)

    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.latest_checkpoint('./checkpoints')
    if checkpoint is not None:
        print("Restore Checkpoint %s" % (checkpoint))
        saver.restore(sess, checkpoint)
        print("Model restored.")
        steps = global_step.eval()
        OBSERVE = steps + ADDITIONAL_OB
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialized new Graph")
        steps = 0

    expected_epsilon = INITIAL_EPSILON - steps * (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if expected_epsilon > FINAL_EPSILON:
        epsilon = expected_epsilon
    else:
        epsilon = FINAL_EPSILON

    # training time
    while(1):
        # output tensor
        out_t = out.eval(feed_dict={inp: [inp_t]})[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame, hit_rate, hit_rate_100 = game.getNextFrame(argmax_t)
        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        # add our input tensor, argmax tensor, reward and updated input tensor tos tack of experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # training iteration
        if steps > OBSERVE:

            # get values from our replay memory
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # train on that
            train_step.run(feed_dict={
                           gt: gt_batch,
                           argmax: argmax_batch,
                           inp: inp_batch
                           })

        # update our input tensor the the next frame
        inp_t = inp_t1
        steps += 1

        # record the agent's performance every 100 steps
        if steps % 100 == 0:
            summaries_long = tf.Summary(value=[
                tf.Summary.Value(tag="hit_rate", simple_value=hit_rate)])
            writer_long.add_summary(summaries_long, steps)
            summaries_short = tf.Summary(value=[
                tf.Summary.Value(tag="hit_rate", simple_value=hit_rate_100)])
            writer_short.add_summary(summaries_short, steps)

        # print our where we are after saving where we are
        if steps % 10000 == 0:
            sess.run(global_step.assign(steps))
            saver.save(sess, './checkpoints/pong-dqn', global_step=steps)

        print("TIMESTEP", steps, "/ EPSILON %7.5f" % epsilon, "/ ACTION", maxIndex,
              "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))


def main():
    # create session
    sess = tf.InteractiveSession()
    # input layer and output layer by creating graph
    inp, out = createGraph()
    # train our graph on input and output with session variables
    trainGraph(inp, out, sess)


if __name__ == "__main__":
    main()
