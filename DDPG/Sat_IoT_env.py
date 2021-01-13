# -*- coding = utf-8 -*-
# @Time : 2020/12/9 13:34
# @Author : 王浩
# @File : Sat_IoT_env.py
# @Software : PyCharm

import numpy as np
import time
# import sys
import math


eta = 0.5  # 时延和能耗的比重
K = 100   # 总任务个数
Sat_N = 2    # 定义总卫星个数
Gat_N = 2    # 定义总网关个数
Ter_N = 2    # 定义地面IoT终端的个数
N_k = [8*10**4, 1.2*10**5]   # 定义任务大小区间
w_k = 1   # 每个任务所占的比重
Height = 10**6    # LEO卫星的高度
Cover_radius = 10**6   # LEO卫星的覆盖半径
cycles_average_bit = 1000  # CPU运算能力，单位cycles/bit
Energy_cost = 3*10**-10   # CPU每循环一圈的能源消耗
X_s = 10*10**6  # 单个卫星的总通信容量
Y_g = 50*10**6  # 单个网关的总通信容量
Z_s = 10**10   # 单个卫星的总计算容量，单位cycles/s
Q_g = 50*10**10  # 单个网关的总计算容量，单位cycles/s
J = 20   # 最大完成回报
Time_slot = 10**-2  # 每个时隙长度
c_v = 3*10**8  # 光速
Max_time_slot = 100   # 定义一个最大处理时间的时隙个数


class Sat_IoT(object):
    # 实例的初始化
    def __init__(self):
        # python中的super(Net, self).init()是指首先找到Net的父类（比如是类NNet），
        # 然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，
        # 其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()
        self.Sat_N = Sat_N
        self.Gat_N = Gat_N
        self.K = K
        super(Sat_IoT, self).__init__()
        # self.title('Sat_IoT')
        self._build_Sat_IoT()
        self.U = np.random.randint(N_k[0], N_k[1], size=K).reshape(K, 1)  # 初始化总任务集
        self.state = np.concatenate((self.omega, self.psi, self.phi, self.U, self.X, self.Y, self.Q, self.Z), axis=1)
        self.state_ = self.state.copy()
        # 定义每个时隙剩余要传输/计算的数据大小
        self.omega_t = self.omega.copy()
        self.psi_c = self.psi.copy()
        self.phi_t = self.phi.copy()
        self.phi_c = self.phi.copy()
        self.reward = 0
        # 初始时间
        self.time = 0
        self.solved = 0
        self.time_reward = 0
        self.energy_reward = 0
        self.ok = 0

    def _build_Sat_IoT(self):
        # 状态空间的定义和初始化
        self.omega = np.zeros((K, Sat_N))   # 初始化与卫星有关的任务集，列为总卫星个数
        # omega_l = np.append(omega_l, [[1,0]], axis = 0)
        self.psi = np.zeros((K, Sat_N))       # 初始化由卫星处理的任务集，列为总卫星个数
        self.phi = np.zeros((K, Gat_N))      # 初始化与网关有关的任务集，列为总网关个数
        self.X = np.zeros((K, Sat_N))       # 初始化卫星正在进行的任务占用的通信资源，列为总卫星个数
        self.X_remain = np.ones((1, Sat_N))*X_s  # 卫星剩余的通信资源
        self.X_allocation = np.zeros((1, Sat_N))
        self.Y = np.zeros((K, Gat_N))       # 初始化网关正在进行的任务占用的通信资源，列为网关总个数
        self.Y_remain = np.ones((1, Gat_N)) * Y_g  # 网关剩余的通信资源
        self.Q = np.zeros((K, Gat_N))      # 初始化网关正在进行的任务占用的计算资源，列为网关总个数
        self.Q_remain = np.ones((1, Gat_N)) * Q_g  # 网关剩余的计算资源
        self.Gat_allocation = np.zeros((1, Gat_N))
        self.Z = np.zeros((K, Sat_N))       # 初始化卫星正在进行的任务占用的计算资源，列为总卫星个数
        self.Z_remain = np.ones((1, Sat_N)) * Z_s  # 卫星剩余的计算资源
        self.Z_allocation = np.zeros((1, Sat_N))
        # 初始化位置矩阵
        # self.PL = np.array([[1.1, 1.2, 1.2, 1.3], [1.2, 1.3, 1.1, 1.2]])*10**6
        self.PL = np.random.randint(Height, int(math.sqrt(Height**2 + Cover_radius**2)),
                                    Sat_N*(Gat_N+Ter_N)).reshape(Sat_N, Gat_N+Ter_N)

        # 动作空间的定义和初始化
        # self.A1 = np.arange(2)    # 0代表不安排在该时隙，1代表安排在本时隙
        self.A1 = np.arange(Sat_N+1)  # 0代表不安排在该时隙，其他代表与该任务有关的卫星编号
        self.A2 = np.arange(Gat_N+1)      # 0代表由卫星计算，或者由网关计算
        # self.A4 = np.arange(Gat_N)   #
        self.a = np.zeros((2, K))       # 定义动作的类型，用来储存对每个任务的动作
        self.n_actions = ((Sat_N+1)*(Gat_N+1))**K
        self.n_features = (Sat_N*4+Gat_N*3+1)*K

    def reset(self):
        self.__init__()
        return self.state.reshape(1, self.n_features)

    # 将action有十进制数字转变为矩阵的方式表示
    def Action(self, action):
        '''
        for i in range(K):
            self.a[0][i] = action % (Sat_N+1)
            action = action // (Sat_N+1)
        for i in range(K):
            self.a[1][i] = action % (Sat_N+1)
            action = action // (Sat_N+1)
        '''
        action.astype(int)
        self.a = np.array(action).reshape(2, K)
    # 判断需要分配的总资源以及是否满足限制条件
    def Source_allocation(self):
        # 判断需要分配的总的卫星通信资源
        # print(self.a)
        self.X_allocation = np.zeros((1, Sat_N))
        self.Z_allocation = np.zeros((1, Sat_N))
        self.Gat_allocation = np.zeros((1, Gat_N))
        # 返回2代表计算卸载了根本不存在的任务
        for i in range(K):
            if (self.a[0][i] != 0) & (self.U[i][0] == 0):
                return 2
        for i in range(K):
            if self.a[0][i] != 0:
                self.X_allocation[0][int(self.a[0][i]) - 1] += np.sqrt(self.U[i][0] * eta * w_k)
        # 判断需要分配的总的卫星计算资源
        for i in range(K):
            if (int(self.a[0][i]) != 0) & (int(self.a[1][i]) == 0):
                self.Z_allocation[0][int(self.a[0][i]) - 1] += np.sqrt(self.U[i][0] * eta * w_k)
        # 判断需要分配的总的网关通信和计算资源
        # print(int(self.a[0][0]) != 0,  int(self.a[1][0]) != 0, int(self.a[0][0]) != 0 & (int(self.a[1][0]) != 0))
        for i in range(K):
            if (int(self.a[0][i]) != 0) & (int(self.a[1][i]) != 0):
                # print('lala', int(self.a[1][i]) - 1)
                self.Gat_allocation[0][int(self.a[1][i]) - 1] += np.sqrt(self.U[i][0] * eta * w_k)
                # print('haha', self.Gat_allocation[0][int(self.a[1][i]) - 1])
        for i in range(K):
            if (int(self.a[0][i]) == 0) & (int(self.a[1][i]) != 0):
                return 1
        # 判断该动作是否满足要求
        # print("Y_remain, Gat_allocation", self.Y_remain, self.Gat_allocation)
        # print((np.min(Max_time_slot * self.Y_remain - self.Gat_allocation) < 0))
        if (np.min(Max_time_slot*self.X_remain - self.X_allocation) < 0) | \
                (np.min(Max_time_slot*self.Z_remain/cycles_average_bit - self.Z_allocation) < 0) | \
                (np.min(Max_time_slot*self.Y_remain - self.Gat_allocation) < 0) | \
                (np.min(Max_time_slot*self.Q_remain/cycles_average_bit - self.X_allocation) < 0) :
            return 1
        else:
            return 0

    # 更新剩余的资源
    def Source_remain(self):
        # 更新剩余卫星通信资源
        self.X_remain = np.ones((1, Sat_N))*X_s - np.sum(self.X, axis=0)
        for i in range(Sat_N):
            self.X_remain[0][i] = max(self.X_remain[0][i], 0)
        # print('X_remain:', self.X_remain)
        # 更新剩余卫星计算资源
        self.Z_remain = np.ones((1, Sat_N))*Z_s - np.sum(self.Z, axis=0)
        for i in range(Sat_N):
            self.Z_remain[0][i] = max(self.Z_remain[0][i], 0)
        # print('Z_remain:', self.Z_remain)
        # 更新剩余网关通信资源
        self.Y_remain = np.ones((1, Gat_N))*Y_g - np.sum(self.Y, axis=0)
        for i in range(Gat_N):
            self.Y_remain[0][i] = max(self.Y_remain[0][i], 0)
        # print('Y_remain:', self.Y_remain)
        # 更新剩余卫星计算资源
        self.Q_remain = np.ones((1, Gat_N))*Q_g - np.sum(self.Q, axis=0)
        for i in range(Gat_N):
            self.Q_remain[0][i] = max(self.Q_remain[0][i], 0)
        # print('Q_remain:', self.Q_remain)

    #  更新X,Y,Z,Q,U,psi,phi，对即将执行的新任务分配资源,计算总时延和能耗
    def Source_update(self):
        self.time_reward = 0
        self.energy_reward = 0
        for i in range(K):
            # 更新X,Y,Z,Q,U
            if (self.a[0][i] != 0) & (self.U[i][0] != 0) :
                # 如果任务在该时隙处理，则将总任务集当中对应任务的数据大小放在由卫星有关的任务集
                self.omega[i][int(self.a[0][i]) - 1] = self.U[i][0]
                self.omega_t[i][int(self.a[0][i]) - 1] = self.omega[i][int(self.a[0][i]) - 1]
                # 分配卫星上的通信资源

                self.X[i][int(self.a[0][i]) - 1] = 2/3 * self.X_remain[0][int(self.a[0][i]) - 1] \
                    * np.sqrt(self.U[i][0] * eta * w_k) / self.X_allocation[0][int(self.a[0][i]) - 1]
                # 计算等待时延和卫星地面之间的传输时延
                self.time_reward += self.time*Time_slot + self.U[i][0]/self.X[i][int(self.a[0][i]) - 1]
                # 判断由卫星计算还是网关计算
                if self.a[1][i] == 0:
                    # 由卫星计算的任务集
                    self.psi[i][int(self.a[0][i]) - 1] = self.U[i][0]
                    self.psi_c[i][int(self.a[0][i]) - 1] = self.psi[i][int(self.a[0][i]) - 1]
                    # 计算由卫星处理的能量消耗
                    self.energy_reward += self.U[i][0] / cycles_average_bit * Energy_cost
                    # 分配卫星上的计算资源
                    self.Z[i][int(self.a[0][i]) - 1] = 2/3 * self.Z_remain[0][int(self.a[0][i]) - 1] \
                        * np.sqrt(self.U[i][0] * eta * w_k) / self.Z_allocation[0][int(self.a[0][i]) - 1]
                    # 计算由卫星处理的计算时延和传播时延
                    self.time_reward += self.U[i][0] / self.Z[i][int(self.a[0][i]) - 1] + \
                        2 * self.PL[int(self.a[0][i]) - 1][np.random.randint(0, Ter_N)]/c_v
                else:
                    # 与网关有关的任务集
                    self.phi[i][int(self.a[1][i]) - 1] = self.U[i][0]
                    self.phi_t[i][int(self.a[1][i]) - 1] = self.phi[i][int(self.a[1][i]) - 1]
                    self.phi_c[i][int(self.a[1][i]) - 1] = self.phi[i][int(self.a[1][i]) - 1]
                    # 分配网关通信资源
                    # print('*****', self.Gat_allocation[0][int(self.a[1][i]) - 1])
                    if self.Gat_allocation[0][int(self.a[1][i]) - 1] != 0:
                        self.Y[i][int(self.a[1][i]) - 1] = 2/3 * self.Y_remain[0][int(self.a[1][i]) - 1] \
                            * np.sqrt(self.U[i][0] * eta * w_k) / self.Gat_allocation[0][int(self.a[1][i]) - 1]
                    # 计算卫星网关端的传输时延
                        self.time_reward += self.U[i][0] / self.Y[i][int(self.a[1][i]) - 1]
                    # 分配网关计算资源
                    # print('sadfafasfsgsjghj', self.Gat_allocation[0][int(self.a[1][i]) - 1])
                        self.Q[i][int(self.a[1][i]) - 1] = 2/3 * self.Q_remain[0][int(self.a[1][i]) - 1] \
                            * np.sqrt(self.U[i][0] * eta * w_k) / self.Gat_allocation[0][int(self.a[1][i]) - 1]
                    # 计算网关处理的计算时延和传播时延
                    if self.Q[i][int(self.a[1][i]) - 1] != 0:
                        self.time_reward += self.U[i][0] / self.Q[i][int(self.a[1][i]) - 1] + \
                            2 * (self.PL[int(self.a[0][i]) - 1][np.random.randint(0, Ter_N)] +
                                self.PL[int(self.a[0][i]) - 1][int(self.a[1][i])+Ter_N-1])/c_v
                # 并将U中该位置的任务清零
                self.U[i][0] = 0
        # if self.U.sum() != 0:
            # self.time_reward += self.time
        # print(self.energy_reward, self.time_reward)
        return self.energy_reward, self.time_reward

    # 未处理任务集为0时，重新生成任务
    def update_U(self):
        # ok 代表是否填充新任务
        # 当任务完全处理完毕时，释放U中该位置的任务，并引入新任务
        for i in range(K):
            if (np.sum(self.Z[i]) == 0) & (np.sum(self.Q[i]) == 0) & (self.U[i][0] == 0):
                self.U[i][0] = np.random.randint(N_k[0], N_k[1])
                self.solved += 1
                self.ok = 1
        '''
        if self.U.sum() == 0:
            self.U = np.random.randint(N_k[0], N_k[1], size=K).reshape(K, 1)
            self.time = 0
            self.solved += 1
        '''
    # 经过一个时隙后，系统状态的变化
    def update(self):
        # 更新与卫星有关的任务集,卫星分配的通信资源，计算资源
        self.omega_t -= self.X * Time_slot

        for i in range(K):
            for j in range(Sat_N):
                # 如果传输已经完成，接下来开始计算由卫星处理任务或传到网关
                # print(self.Y[i])
                if self.omega_t[i][j] <= 0:
                    if self.omega_t[i][j] == 0:
                        # 更新卫星计算的任务集，Z单位为cycles/s，cycles_average_bit单位是cycles/bit，
                        # 相除单位为bits/s,同时系统在完成终端到卫星之间的传输任务之后，立即执行下一步操作
                        self.psi_c[i][j] -= self.Z[i][j] / cycles_average_bit * Time_slot
                        self.phi_t[i][j] -= self.Y[i].sum() * Time_slot
                    else:
                        # print(self.psi_c[i][j])
                        self.psi_c[i][j] -= self.Z[i][j] / cycles_average_bit * \
                            (-self.omega_t[i][j]/self.X[i][j])
                        self.phi_t[i][j] -= self.Y[i].sum() * (-self.omega_t[i][j] / self.X[i][j])
                    self.omega_t[i][j] = 0
                    self.omega[i][j] = 0
                    self.X[i][j] = 0
                # 本时隙卫星完成计算任务后，自动释放任务和计算资源
                if self.psi_c[i][j] <= 0:
                    self.psi_c[i][j] = 0
                    self.psi[i][j] = 0
                    self.Z[i][j] = 0
                    self.U[i][0] = 0

        # 更新与网关有关的任务集,网关分配的通信资源，计算资源
        # 更新网关有关的任务集，需要注意，该任务只有在卫星传输已经完成时才会进行处理
        for i in range(K):
            for j in range(Gat_N):
                if self.phi_t[i][j] <= 0:
                    if self.phi_t[i][j] == 0:
                        self.phi_c[i][j] -= self.Q[i][j] / cycles_average_bit * Time_slot
                    else:
                        # print(self.phi_t[i][j])
                        if self.Y[i][j] != 0:
                            # print('ooo', self.Y[i][j])
                            self.phi_c[i][j] -= self.Q[i][j] / cycles_average_bit * \
                                (-self.phi_t[i][j] / self.Y[i][j])
                    # 释放网关通信资源
                    self.phi_t[i][j] = 0
                    self.Y[i][j] = 0
                    # 判断网关计算是否完成
                    # np.maximum(self.phi_c, 0)
                    if self.phi_c[i][j] <= 0:
                        self.phi_c[i][j] = 0
                        self.phi[i][j] = 0
                        self.Q[i][j] = 0
                        self.U[i][0] = 0

        self.update_U()

        self.time += Time_slot

    def step(self, action):
        # time.sleep(0.3)
        self.Action(action)
        # 更新剩余资源
        self.Source_remain()

        # 更新X_allocation，Z_allocation，Gat_allocation
        if_error = self.Source_allocation()
        if if_error == 1:
            print("选择无效动作，初始状态不更新")
            # self.update()
            self.show_system()
            return self.state_.reshape(1, self.n_features), -100, False
        if if_error == 2:
            print("计算卸载不存在的任务，初始状态不更新")
            # self.update()
            self.show_system()
            return self.state_.reshape(1, self.n_features), -10, False

        # print(self.Gat_allocation)
        # 更新X,Y,Z,Q,U,psi,phi
        energy_reward, time_reward = self.Source_update()
        print('energy_reward, time_reward', energy_reward, time_reward)
        if action.sum() == 0 :
            self.reward = -1
        else:
            self.reward = J - (eta*time_reward + (1-eta)*energy_reward) - self.time
        # self.state_ = np.concatenate((self.omega, self.psi, self.phi, self.U, self.X, self.Y, self.Q, self.Z), axis=1)
        self.ok = 0
        while self.ok == 0:
            self.update()
            self.reward -= Time_slot
        print('已完成任务集:', self.solved)
        if self.solved >= 1000:
            done = True
        else:
            done = False
        # print(self.state_.reshape(1, self.n_features), self.reward, done)
        self.state_ = np.concatenate((self.omega, self.psi, self.phi, self.U, self.X, self.Y, self.Q, self.Z), axis=1)
        self.state = self.state_.copy()
        self.show_system()
        return self.state_.reshape(1, self.n_features), self.reward, done

    def show_system(self):
        np.set_printoptions(linewidth=400)
        print('*****************************************************')
        print(self.state_, '\n', self.a, '\n', self.reward)
        print("X", self.X)
        print("X_remain", self.X_remain)
        print("X_allocation", self.X_allocation)
        print("Y", self.Y)
        print("Y_remain", self.Y_remain)
        print('Q', self.Q)
        print('Q_remain', self.Q_remain)
        print('Gat_allocation', self.Gat_allocation)
        print('Z', self.Z)
        print('Z_remain', self.Z_remain)
        print('Z_allocation', self.Z_allocation)
        print('omega_t', self.omega_t)
        print('psi_c', self.psi_c)
        print('phi_t', self.phi_t)
        print('phi_c', self.phi_c)
        print('time', self.time)
        print(self.ok)

