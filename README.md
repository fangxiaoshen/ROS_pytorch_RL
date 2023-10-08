# ROS_pytorch_RL
在turtlebot3，pytorch上使用DQN,DDPG,PPO,SAC算法，在gazebo上实现仿真。
# 仿真配置教程
https://blog.csdn.net/qq2650326396/article/details/124801005
# 实车配置教程
https://blog.csdn.net/qq2650326396/article/details/132076450

# 我设置了新的reward和控制，让小车减少碰撞。在PPO下训练了140回合后的训练效果：
可以看到小车一定不会碰撞，因为我加了一个膨胀的距离给小车。 
[INFO] [1639030326.511617, 504.920000]: Ep: 20 score: 9586.48 memory: 52521 episode_step: 2500.00 time: 2:57:02
[INFO] [1639030394.774092, 68.211000]: Goal!!
[INFO] [1639030395.362659, 68.758000]: Goal position : 0.4, 0.4
[INFO] [1639030496.484066, 169.810000]: Goal!!
[INFO] [1639030497.016797, 170.313000]: Goal position : 6.5, 4.5
[INFO] [1639030570.563280, 243.808000]: Goal!!
[INFO] [1639030571.016659, 244.231000]: Goal position : 0.4, 0.4
[INFO] [1639030682.468226, 355.605000]: Goal!!
[INFO] [1639030682.929499, 356.027000]: Goal position : 6.5, 4.5
[INFO] [1639030761.367020, 434.409000]: Goal!!
[INFO] [1639030761.944676, 434.946000]: Goal position : 0.4, 0.4
[INFO] [1639030829.649773, 502.605000]: Goal!!
[INFO] [1639030830.140551, 503.066000]: Goal position : 6.5, 4.5
[INFO] [1639030830.147300, 503.072000]: time out!
[INFO] [1639030831.429489, 504.350000]: Ep: 21 score: 7548.49 memory: 55022 episode_step: 2500.00 time: 3:05:27

