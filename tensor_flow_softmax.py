import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np

# Import data from csv
path = "Data Sets\\GBP-USD Hourly.csv"
data_from_file = pd.read_csv(path).values[:, 1]

size = 12

# Normalise input data
# self.data_from_file = self.normalise_data(self.data_from_file)

# Get true_direction value as every 4th entry, starting from 12
ground_truth = data_from_file[size:]

# Initialise empty vector
all_data = np.zeros([len(ground_truth), size])

# Create n 1x12 vectors and store them in a matrix
for idx in range(len(ground_truth)):
    all_data[idx, :] = data_from_file[idx:idx + size]

# Find the correct direction
true_direction = np.zeros([len(ground_truth), 2])

for idx, dp in enumerate(all_data):
    if dp[-1] <= ground_truth[idx]:
        # Final value of input vector is smaller than true_direction value
        # Thus correct prediction is downwards
        true_direction[idx, :] = [0, 1]
    else:
        # Else, true_direction direction is up
        true_direction[idx, :] = [1, 0]

# Have to take into account time series nature but can shuffle training data
split = 0.8
split_idx = round(len(true_direction) * split)
train_data, test_data = all_data[:split_idx], all_data[split_idx:]
train_true, test_true = true_direction[:split_idx], true_direction[split_idx:]
shfl_idx = np.random.permutation(len(train_true))
train_data, train_true = train_data[shfl_idx], train_true[shfl_idx]

### TENSOR FLOW ###

# Python optimisation variables
learning_rate = 0.5
epochs = 20
batch_size = 1
shuffle_buffer_size = 100

# Initialise numpy data sets as tensor flow data sets
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_true))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_true))

# Shuffle the training data set
train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


# declare the training data placeholders
# input x - 12
x = tf.placeholder(tf.float32, [None, size])
# now declare the output data placeholder - 2 possibilities up or down
y = tf.placeholder(tf.float32, [None, 2])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([size, 32], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([32]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([32, 2], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([2]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.tanh(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

iter = train_dataset.make_one_shot_iterator()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('C:\\Users\\grbea\\OneDrive - University of Cambridge\\4th Year Project\\Code\\Code Outputs')

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(train_data) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = train_data[i:i + batch_size]
            batch_y = train_true[i:i + batch_size]
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: test_data, y: test_true}))


    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: test_data, y: test_true}))
