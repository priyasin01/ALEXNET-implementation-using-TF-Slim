def alexnet(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 96, [11, 11], stride=4, padding=0, scope='conv1')
    
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
    net = slim.conv2d(inputs, 256, [5, 5], stride=1, padding=2, scope='conv2')
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
    net = slim.repeat(net, 2, slim.conv2d, 384, [3, 3],stride=1, padding=1, scope='conv3')
    net = slim.conv2d(inputs, 256, [3, 3], stride=1, padding=1, scope='conv4')
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
    
    
    net = slim.fully_connected(net, 4096, scope='fc5')
    ##net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc6')
    ##net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn='softmax', scope='fc7')
  return net
