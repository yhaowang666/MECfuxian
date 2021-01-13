from Sat_IoT_env import Sat_IoT
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(1000000):
        # initial observation
        observation = env.reset()
        observation /= 10 ** 6
        # print(observation)
        while True:
            # fresh envenv.reset()
            # env.render()
            # print('****************************************************')
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # print('episode:', episode)
            # print(action)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            observation_ /= 10**6
            # 将当前状态，选择的动作，回报，下一个状态保存到存储库当中去
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                # 过两百步后，每5步学习一次
                RL.learn()

            # swap observation
            # 变为下一个状态（观察值）
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
            # print('Episode:', episode, ', step:', step)

    # end of game
    print('game over')
    # env.destroy()


if __name__ == "__main__":
    # maze game
    env = Sat_IoT()
    # RL方法选择DQN
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.0001,    # 学习效率设为0.0001（）
                      reward_decay=0.9,      # 预计回报衰减
                      e_greedy=0,          # 选择最大Q值对应的动作的概率
                      replace_target_iter=200,   # 每隔200步替换一次target_net的参数
                      memory_size=2000,     # 记忆上限
                      output_graph=True,   # 输出神经网络训练模型
                      restore_network=False,
                      save_network=False
                      )
    RL.restore_net()
    run_maze()
    # env.after(100, run_maze)  # after语句可以实现定时器循环
    # env.mainloop()  # mainloop就进入到事件（消息）循环
    save_path = RL.save_net()
    RL.plot_cost()  # 观看神经网络的误差曲线
