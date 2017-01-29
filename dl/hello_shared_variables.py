# encoding: utf-8
import tensorflow as tf
import numpy as np

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def my_image_filter(input_images):
    with tf.variable_scope("conv1") as scope:
        print "%s:%s" % (scope.name, scope.reuse)
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2") as scope:
        print "%s:%s" % (scope.name, scope.reuse)
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])

explanation = """
get_variable()就像是在map里面按key取value，其行为被它所属的variable_scope的reuse这个属性控制。

reuse有三种取值，False就只能新建而且不能重名，True就只能用已有的不能新建，None则如果已有就用已有的，没有就新建。

而且整个tensorflow/python/ops/variable_scope.py的代码里（包括 tf.variable_scope() 这个函数），reuse的默认值都是None，按理来说就不会报重复。

但有个潜规则A：每层variable_scope都从上一层variable_scope继承reuse的值，具体实现在 _pure_variable_scope()函数里：

reuse = reuse or old.reuse # Re-using is inherited by sub-scopes.

还有个潜规则B：最顶层的variable_scope被显式初始化为reuse=True，具体实现在get_variable_scope()函数里：

scope = VariableScope(False)

所以，Tensorflow的默认行为就变成了不允许重名，必须新建。

上面说的是实现。

而从意图上看，这应该是故意这么做的，为了避免shared by accident。
"""

print explanation

shape = [5,5,32,32]
image1 = image2 = np.zeros(shape, dtype="float32")

print "\nCASE: default value of `reuse` is False and forbids variable sharing.\n"

global_scope = tf.get_variable_scope()
print "%s:%s" % ('global_scope', global_scope.reuse)

try:
    result1 = my_image_filter(image1)
    result2 = my_image_filter(image2)
except Exception as e:
    print "\nThe following exception is expected:\n"
    print e

print "\nCASE: Setting reuse to None allows creation and variable sharing.\n"

global_scope._reuse = None

result1 = my_image_filter(image1)
result2 = my_image_filter(image2)

print "\nNo exception raised, variables reused."

print "\nCASE: Setting reuse to True forbids creation.\n"

with tf.variable_scope("another_root_scope") as scope:
    try:
        scope._reuse = True

        result1 = my_image_filter(image1)
        result2 = my_image_filter(image2)
    except Exception as e:
        print "\nThe following exception is expected:\n"
        print e
