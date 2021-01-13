"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
# import gym
import time
from Sat_IoT_env_optimization import Sat_IoT

# seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同
np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

# 回合数
MAX_EPISODES = 200
# 每个回合的最大步数
MAX_EP_STEPS = 200
# actor学习率
LR_A = 0.001    # learning rate for actor
# critic学习率
LR_C = 0.001    # learning rate for critic
# 回报衰减系数
GAMMA = 0.9     # reward discount
# 设定保存参数的字典
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)   # actor和critic每隔多少步更新target_net
][0]            # you can try different target replacement strategies
# 记忆库大小
MEMORY_CAPACITY = 10000
# 每次抽取记忆的数量（每次网络更新选取数据的维度）
BATCH_SIZE = 32

# RENDER = False
OUTPUT_GRAPH = True
# ENV_NAME = 'Pendulum-v0'

###############################  Actor  ####################################

class Actor(object):
    # 初始化
    def __init__(self, sess, action_dim, learning_rate, replacement):
        self.sess = sess
        # 动作的个数
        self.a_dim = int(action_dim)
        # 动作的取值范围
        # self.action_bound1 = action_bound1
        # self.action_bound2 = action_bound2
        # 学习率
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.restore_network = False,
        self.save_network = True,
        # Actor模块
        with tf.variable_scope('Actor'):
            # input s, output a
            # eval_net，实时更新神经网络参数，输出选择的动作
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            # target_net，不更新神经网络的参数，每隔固定步数，将eval_net赋给target_net，
            # 输出的动作供critic供评价
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)
        # 获取eval_net网络的参数，赋给e_params
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        # 获取target_net网络的参数，赋给t_params
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        # 如果为hard模式，直接将target_net参数替换次数清零，并且将e_params赋给t_params
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            # 如果为soft模式，根据加权系数tau确定t_params和e_params比例，最终赋给t_params
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    # 搭建网络，返回最佳动作
    def _build_net(self, s, scope, trainable):
        # 初始化变量定义
        with tf.variable_scope(scope):
            # 生成一组符合标准正态分布的tensor对象，均值为0，标准差为0.3
            init_w = tf.random_normal_initializer(0., 0.000003)
            # 生成一个初始值为常量0.1的tensor对象
            init_b = tf.constant_initializer(0.0000001)
            # l1层神经网络的定义，神经元个数为360
            net = tf.layers.dense(s, 360, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            # a层神经网络定义，神经元个数为self.a_dim（输出动作的个数），激励函数为tanh，即输出范围为（-1，1）
            # 在乘以self.action_bound，这样输出的值的范围为（-self.action_bound，self.action_bound）
            # 最终返回scaled_a，即输出的真实动作，取值范围为（0，2）
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                # scaled_a = np.append([], actions)
                # scaled_a = tf.placeholder(tf.float32, shape=[None, self.a_dim], name='scaled_a')
                # print(actions[:int(self.a_dim/2)])
                # actions = tf.Variable(actions)
                # print(scaled_a)
                # scaled_a = np.array((1, self.a_dim))
                # scaled_a = tf.multiply(actions, self.action_bound1, name='scaled_a')  # Scale output to -action_bound to action_bound
                # scaled_a = tf.multiply(actions, self.action_bound2, name='scaled_a')
                # scaled_a = tf.multiply(actions, self.action_bound2, name='scaled_a')
        return (actions + 1)/2

    # 学习更新网络
    def learn(self, s):   # batch update
        # 采用梯度下降法更新Actor网络中的eval_net,学习模块的输入只有当前状态（mini-batch中抽取的记忆）
        self.sess.run(self.train_op, feed_dict={S: s})

        # 如果更新模式为soft，每一步都对target_net进行更新
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        # 如果更新模式为hard，每隔rep_iter_a步替换target_net当中的参数
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    # 选择动作
    def choose_action(self, s):
        # 先将输入s变量转换为array数组
        # s = s[np.newaxis, :]    # single state
        # 根据s直接返回由eval_net输出的最佳动作
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    # 梯度下降策略，Actor网络train模块的定义，输入a_grads为由Critic网络来的动作梯度（表示动作的好坏）
    def add_grad_to_graph(self, a_grads):
        # 梯度下降策略
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            # tf.gradients求动作a关于eval_net中的参数e_params的导数，并对结果乘以一个动作梯度a_grads
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        # train模块的定义
        with tf.variable_scope('A_train'):
            # 学习率为负值
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            # 通过求得的policy_grads对e_params参数更新
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

    def save_net(self):
        if self.save_network:
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, "my_net/actor_save_net_20.ckpt")
            print("Save to path:", save_path)
            # print(self.cost_his)
            return save_path

    def restore_net(self):
        if self.restore_network:
            saver = tf.train.Saver()
            saver.restore(self.sess, "my_net/actor_save_net_20.ckpt")
            print("restore in path")


###############################  Critic  ####################################

class Critic(object):
    # Critic对象初始化
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        # 状态维度（3）
        self.s_dim = state_dim
        # 动作维度（1）
        self.a_dim = action_dim
        # 学习率
        self.lr = learning_rate
        # 未来回报衰减率
        self.gamma = gamma
        self.replacement = replacement
        self.restore_network = False,
        self.save_network = True,

        # Critic模块定义，包含eval_net, target_net
        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            # 防止a的变化导致actor中target_net的更新，即tf.stop_gradient对q_target的反传进行截断
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            # Critic中eval_net的定义，返回的是q_evaluate(和DQN非常类似)
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            # Critic中target_net的定义，返回的是q_target,
            # 输入是下一个状态和选择的动作（Actor中的target_net给出）
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        # q_target的定义，当前回报加上下一个状态的由target_net确定的q_target（也就是target_net的输出）
        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        # TD_error的定义，里面包含了计算得到的q_target和eval_net输出的q_eval的差异（loss）
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        # Critic_train模块的定义，功能:最小化loss
        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # a_grad（动作梯度）的定义，输出结果为q_evaluate关于当前选择动作的导数
        # a_grad越大，代表这个动作需要更新的价值也就越大，对应的Actor针对这个动作更新的幅度也就越大
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        # 针对不同的模式target_net更新方式的确定
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.000001)
            init_b = tf.constant_initializer(0.0000001)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

    def save_net(self):
        if self.save_network:
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, "my_net/critic_save_net_20.ckpt")
            print("Save to path:", save_path)
            # print(self.cost_his)
            return save_path

    def restore_net(self):
        if self.restore_network:
            saver = tf.train.Saver()
            saver.restore(self.sess, "my_net/critic_save_net_20.ckpt")
            print("restore in path")


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        # 可以储存的记忆的数量
        self.capacity = capacity
        # 记忆库的初始化，capacity行，dims列
        self.data = np.zeros((capacity, dims))
        # 已存储的记忆个数
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        # 按照（s, a, r, s_）的方式存储记忆，注意要以array的格式存储
        '''
        print(s)
        print(a)
        print(r)
        print(s_)
        '''
        transition = np.hstack((s, [a], [[r]], s_))
        # 记忆库的覆盖
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # 从 0~capacity-1中随机选出n个整数
        indices = np.random.choice(self.capacity, size=n)
        # 抽取n个样本
        return self.data[indices, :]


env = Sat_IoT()

state_dim = env.n_features
print(state_dim)
# 动作的维度
action_dim = env.K * 2  # 针对每个用户需要输出两个动作
# 动作的取值范围
# 每个任务的第一个动作的取值范围（0，Sat_N）
action_bound1 = env.Sat_N
# 每个任务的第二个动作的取值范围（0，Gat_N）
action_bound2 = env.Gat_N
# all placeholder for tf
# 定义神经网络总的placeholder变量，状态S、回报R、下一个状态S_
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
# 实现Actor类，对象actor输出为最佳动作，输入参数为动作的维度等
actor = Actor(sess, action_dim, LR_A, REPLACEMENT)
# critic接收由actor中target_net输出的动作actor.a_
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
# actor接收由critic输出的动作提升梯度critic.a_grads
actor.add_grad_to_graph(critic.a_grads)
# 激活全部变量
sess.run(tf.global_variables_initializer())
# critic.restore_net()
# actor.restore_net()
M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # control exploration

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()/10**6
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        while True:
            # env.render()

            # Add exploration noise
            a = actor.choose_action(s)  # a的取值范围为（0，2）
            # np.random.normal生成均值为a方差为var的高斯分布随机数，
            # 同时，np.clip截取（0，2）的部分，小于0的数全变为0，大于2的数全变为2

            dim = np.int(action_dim/2)
            a[:dim] = np.clip(np.random.normal(a[:dim]*action_bound1, var), 0, action_bound1)    # add randomness to action selection for exploration
            a[dim:] = np.clip(np.random.normal(a[dim:]*action_bound2, var), 0, action_bound2)
            a = a.astype(int)
            print('a', a)
            s_, r, done = env.step(a)
            s_ /= 10**6
            #
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                # 当记忆库已满时，不断减小方差
                var *= .99999999    # decay the action randomness
                # b_M为记忆库中提取的记忆
                b_M = M.sample(BATCH_SIZE)
                # 截取当前状态部分
                b_s = b_M[:, :state_dim]
                # 截取选取动作部分
                b_a = b_M[:, state_dim: state_dim + action_dim]
                # 索引为负数，是从后往前的顺序
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r
            if done:
                break

print('Running time: ', time.time()-t1)
actor.save_net()
critic.save_net()
