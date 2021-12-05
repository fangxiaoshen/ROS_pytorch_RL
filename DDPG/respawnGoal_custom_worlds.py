#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('project/src',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        #stage 1 = TCC_world_obst
        #stage 2 = TCC_world_U
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        if self.stage == 1:  
            self.init_goal_x = 0.975166
            self.init_goal_y = -0.790902
        if self.stage == 2:  
            self.init_goal_x = 2.25
            self.init_goal_y = -2.40
        if self.stage == 3:  
            self.init_goal_x = 2.25
            self.init_goal_y = -2.40
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False, running=False):
        if delete:
            self.deleteModel()

        if self.stage == 1:
            while position_check:
                # goal_x_list = [0.290287, 0.832032, 1.099705, 1.578135, 2.078463, 2.703509, 2.523060, 2.026206, 2.325653, 2.746069, 2.704726, 2.356458, 1.899472, 2.320879, 2.610759, 1.659986, 1.010895, 0.437185, 0.194019, 0.987761, 1.239078, 0.746459, 0.321399, 1.022309, 1.025044, 0.212838]
                # goal_y_list = [-0.503178, -0.668002, -0.714389, -0.624754, -0.312600, -0.334154, -0.924587, -1.409555, -1.798179, -1.419792, -1.985860, -2.170286, -2.343686, -2.569163, -2.811930, -2.732509, -2.782875, -2.755811, -2.209052, -2.185481, -1.859446, -1.587824, -1.041969, -0.971523, -0.469485, -0.601541]
                # print('\n\naqui\n\n')
                goal_x_list = [0.874355, 0.921632, 0.301330, 1.340610, 1.984565, 2.164056, 2.550608, 1.340223, 0.418784, 2.546777, 0.397608]
                goal_y_list = [-1.505722, -2.267933, -2.699439, -2.788460, -2.653711, -1.381487, -0.397855, -0.518933, -2.638746, -2.561414, -0.382616]
                if not running:
                    self.index = 0
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index < 10:
                    self.index += 1
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index >= 10:
                    aux_index = random.randrange(0, 11)
                    print(self.index, aux_index)
                    if self.last_index == aux_index:
                        position_check = True
                    else:
                        self.last_index = aux_index
                        position_check = False
                        self.goal_position.position.x = goal_x_list[aux_index]
                        self.goal_position.position.y = goal_y_list[aux_index]

                # self.index = random.randrange(0, 26)
                # #print(self.index, self.last_index)
                # if self.last_index == self.index:
                #     position_check = True
                # else:
                #     self.last_index = self.index
                #     position_check = False

                # self.goal_position.position.x = goal_x_list[self.index]
                # self.goal_position.position.y = goal_y_list[self.index]

        if self.stage == 2:
            while position_check:
                goal_x_list = [1.053979, 0.725346, 1.759492, 2.477302, 2.665976, 2.576926, 2.175901, 1.882009, 2.547499, 1.838564, 2.637267, 1.535967, 0.443954, 0.335799, 0.263102]
                goal_y_list = [-1.122977, -0.332471, -0.291238, -0.330033, -0.734655, -1.328935, -1.411599, -1.803747, -2.157146, -2.260181, -2.668954, -2.546023, -2.570976, -2.100776, -1.021737]

                if not running:
                    self.index = 0
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index < 14:
                    self.index += 1
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index >= 14:
                    aux_index = random.randrange(0, 15)
                    print(self.index, aux_index)
                    if self.last_index == aux_index:
                        position_check = True
                    else:
                        self.last_index = aux_index
                        position_check = False
                        self.goal_position.position.x = goal_x_list[aux_index]
                        self.goal_position.position.y = goal_y_list[aux_index]


        if self.stage == 3:
            while position_check:
                position_check = False
                self.goal_position.position.x = 2.25
                self.goal_position.position.y = -2.40

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
