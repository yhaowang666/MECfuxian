"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
# import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=10**-6,
            output_graph=True,
            epsilon_max=0.9,
            restore_network=True,
            save_network=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy             # 0 if e_greedy_increment is not None else self.epsilon_max
        self.restore_network = restore_network
        self.save_network = save_network

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # tf.assign将后面一个变量的内容赋给前一个变量
        # 这里面是将eval_net中的参数赋给target_net
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 神经网络一个场景中必不可少的一步，初始化所有参数
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):

        # 网络当先所处的状态
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input

        # 整个网络中Q-现实的定义，它的类型与actions的个数相同
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        # 定义下一个状态，类型与s相同，同时作为回合内下一步的输入

        # ------------------ build evaluate_net ------------------
        # 定义第一个神经网络的名字为eval_net
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            # tf.GraphKeys包含所有graph collection中的标准集合名
            # GLOBAL_VARIABLES:该collection默认加入所有的Variable对象，并且在分布式环境中共享。
            # n_l1 = 10，表示该网络中l1层的神经元个数为10；而w（权重）和b（偏置）初始化为一定范围内的随机数
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 360, \
                tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0.01)  # config of layers

            # first layer. collections is used later when assign to target net
            # 定义第一个隐藏层为l1
            with tf.variable_scope('l1'):
                # w1的大小为n_features行（这里面因为用直角坐标表示位置，所以n_features = 2），n_l1列；
                # feature为状态的特征
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # 采用relu激励函数
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            # 定义第二个隐藏层为l2
            with tf.variable_scope('l2'):
                # w2的大小为n_l1行，n_actions列（输出的结果为所有可能actions对应的Q值）；
                # feature为状态的特征
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # eval_net中l2层输出的结果直接为Q-估计
                self.q_eval = tf.matmul(l1, w2) + b2

        # 定义loss层，用于比较Q-估计和Q-显示的差别
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        # 定义train层，选择RMSProp优化器，目的是最小化loss层输出结果；该层输出结果为_train_op
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # 定义第二个神经网络的名字为target_net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            # 第二个神经网络初始参数和第一个神经网络网络相同
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # target_net中的输入为下一个状态s_
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # tar_net中l2层输出的结果直接为Q-next
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # np.hstack(): 在水平方向上平铺
        transition = np.hstack((s, [[a, r]], s_))
        # print(transition)
        # replace the old memory with new memory
        # memory_counter不断累加，这样通过求余可以不断更新记忆库中的内容
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        # 增加observation的维度，变成两列的矩阵，符合神经网络输入端的要求
        # observation = observation[np.newaxis, :]
        # print(observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值
            print('action_value:', actions_value)
            print('epsilon', self.epsilon)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        # print('epsilon：', self.epsilon)
        return action

    def learn(self):
        # check to replace target parameters
        # 一个回合内，每replace_target_iter步，更新一次
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        # 如果记忆库已满，从[0,memory_size)中随机抽取batch_size个数字，组成一个数组
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # 如果记忆库未满，从[0,memory_counter)中随机抽取batch_size个数字，组成一个数组
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # 根据抽中的索引号选择记忆
        batch_memory = self.memory[sample_index, :]

        # 计算Q-现实
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            # 采用placeholder定义的s、s_、q_target,使用时一定要用feed_dict字典的形式
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params,选择最后面两列
                self.s: batch_memory[:, :self.n_features],  # newest params，选择最前面两列
            })

        # change q_target w.r.t q_eval's action
        # 下面的步骤都是未了对应位置，因为q_target和q_eval里面记录的是每个动作下对应的Q值
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 选择batch_memory中第三列的内容作为动作索引
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 选择batch_memory中第四列的内容作为reward
        reward = batch_memory[:, self.n_features + 1]

        # 根据batch_memory中选择的动作的位置，给q_target对应动作的位置赋一个计算出来的新值
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        # self.cost代表的是loss值的大小，即target_net和eva_net的偏差
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # print('loss', self.loss)
        # print('cost', self.cost)
        # append(object) 是将一个对象作为一个整体添加到列表中，添加后的列表比原列表多一个元素
        self.cost_his.append(self.cost/100)


        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        # print(self.cost_his)
        # print(np.arange(len(self.cost_his)))
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        # plt.switch_backend('agg')
        plt.show()
        # print(self.cost_his)

    def save_net(self):
        if self.save_network:
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, "my_net/save_net.ckpt")
            print("Save to path:", save_path)
            # print(self.cost_his)
            return save_path


    def restore_net(self):
        if self.restore_network:
            saver = tf.train.Saver()
            saver.restore(self.sess, "my_net/save_net.ckpt")
            print("restore in path")



