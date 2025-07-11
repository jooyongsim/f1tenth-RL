import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers
import threading 
import csv

class DeepQNetwork:
    def __init__(self, num_actions, state_size, replay_buffer, base_dir, tensorboard_dir, args):
        
        self.num_actions = num_actions
        self.state_size = state_size
        self.replay_buffer = replay_buffer
        self.history_length = args.history_length

        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.target_model_update_freq = args.target_model_update_freq

        self.checkpoint_dir = base_dir + '/models/'

        self.lidar_to_image = args.lidar_to_image
        self.image_width = args.image_width
        self.image_height = args.image_height

        self.add_velocity = args.add_velocity
        self.add_pose = args.add_pose
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


        self.behavior_net = self.__build_q_net()
        self.target_net = self.__build_q_net()

        model_as_string = []
        self.target_net.summary(print_fn=lambda x : model_as_string.append(x))
        "\n".join(model_as_string)

        summary_writer = tf.summary.create_file_writer(tensorboard_dir)
        with summary_writer.as_default():
            tf.summary.text('model', model_as_string, step=0)

        if args.model is not None:
            self.target_net.load_weights(args.model)
            self.behavior_net.set_weights(self.target_net.get_weights())

    def __build_q_net(self):
        if self.lidar_to_image:
            return self.__build_cnn2D()
        elif self.add_velocity and self.add_pose:
            return self.__build_cnn1D_plus_velocity_and_pose()
        else:
            if self.add_velocity:
                return self.__build_cnn1D_plus_velocity()
            else:
                # select from __build_dense or build_cnn1D
                return self.__build_cnn1D()
            
    # def __build_q_net(self):
    #     if self.lidar_to_image:
    #         return self.__build_cnn2D()
    #     else:
    #         if self.add_velocity:
    #             return self.__build_cnn1D_plus_velocity()
    #         else:
    #             # select from __build_dense or build_cnn1D
    #             return self.__build_cnn1D()

    def __build_dense(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length))
        x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def __build_cnn1D(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length))
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model
    
    def __build_cnn1D_plus_velocity_and_pose(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length), name="lidar")
        input_velocity = tf.keras.Input(shape=((self.history_length)), name="velocity")
        input_x = tf.keras.Input(shape=((self.history_length)), name="x")
        input_y = tf.keras.Input(shape=((self.history_length)), name="y")
        input_yaw = tf.keras.Input(shape=((self.history_length)), name="yaw")
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        # print("[DEBUG] Concatenating lidar + velocity ONLY (pose excluded)")
        x = layers.concatenate([x, input_velocity])
        # x = layers.concatenate([x, input_velocity, input_x, input_y, input_yaw])
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=[inputs, input_velocity, input_yaw], outputs=predictions)
        # model = tf.keras.Model(inputs=[inputs, input_velocity, input_x, input_yaw], outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        # print("[DEBUG] Model input keys:", [inp.name for inp in model.inputs])
        # print("[DEBUG] Pose excluded from concatenate:", model.input_names[2:])
        return model

    def __build_cnn1D_plus_velocity(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length), name="lidar")
        input_velocity = tf.keras.Input(shape=((self.history_length)), name="velocity")
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        x = layers.concatenate([x, input_velocity])
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=[inputs, input_velocity], outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def __build_cnn2D(self):
        inputs = tf.keras.Input(shape=(self.image_width, self.image_height, self.history_length))
        x = layers.Lambda(lambda layer: layer / 255)(inputs)
        x = layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def inference(self, state):
        if self.lidar_to_image:
            state = state.reshape((-1, self.image_width, self.image_height, self.history_length))
        
        elif self.add_velocity and self.add_pose:
            state[0] = state[0].reshape((-1, self.state_size, self.history_length))
            state[1] = state[1].reshape((-1, self.history_length))
            state[2] = state[2].reshape((-1, self.history_length))
            # state[3] = state[3].reshape((-1, self.history_length))
            # state[4] = state[4].reshape((-1, self.history_length))

        elif self.add_velocity:
            state[0] = state[0].reshape((-1, self.state_size, self.history_length))
            state[1] = state[1].reshape((-1, self.history_length))
        else:
            state = state.reshape((-1, self.state_size, self.history_length))

        return np.asarray(self.behavior_predict(state)).argmax(axis=1)

    # def inference(self, state):
    #     if self.lidar_to_image:
    #         state = state.reshape((-1, self.image_width, self.image_height, self.history_length))
    #     elif self.add_velocity:
    #         state[0] = state[0].reshape((-1, self.state_size, self.history_length))
    #         state[1] = state[1].reshape((-1, self.history_length))
    #     else:
    #         state = state.reshape((-1, self.state_size, self.history_length))

    #     return np.asarray(self.behavior_predict(state)).argmax(axis=1)

    def train(self, batch, step_number):
        if self.add_velocity and self.add_pose:
            old_states_lidar = np.asarray([sample.old_state.get_data()[0] for sample in batch])
            old_states_velocity = np.asarray([sample.old_state.get_data()[1] for sample in batch]).reshape((-1, self.history_length))
            # old_states_x = np.asarray([sample.old_state.get_data()[2] for sample in batch]).reshape((-1, self.history_length))
            # old_states_y = np.asarray([sample.old_state.get_data()[3] for sample in batch]).reshape((-1, self.history_length))
            old_states_yaw = np.asarray([sample.old_state.get_data()[2] for sample in batch]).reshape((-1, self.history_length))
            new_states_lidar = np.asarray([sample.new_state.get_data()[0] for sample in batch])
            new_states_velocity = np.asarray([sample.new_state.get_data()[1] for sample in batch]).reshape((-1, self.history_length))
            # new_states_x = np.asarray([sample.new_state.get_data()[2] for sample in batch]).reshape((-1, self.history_length))
            # new_states_y = np.asarray([sample.new_state.get_data()[3] for sample in batch]).reshape((-1, self.history_length))
            new_states_yaw = np.asarray([sample.new_state.get_data()[2] for sample in batch]).reshape((-1, self.history_length))
            actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])
            # Stop action delete
            for i in range(len(actions)):
                if actions[i] == 6: 
                    rewards[i] -= 0.5

            predicted = self.target_predict({
                "lidar": new_states_lidar,
                "velocity": new_states_velocity,
                # "x": new_states_x,
                # "y": new_states_y,
                "yaw": new_states_yaw
            })
            
            q_new_state = np.max(predicted, axis=1)
            target_q = rewards + (self.gamma * q_new_state * (1 - is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)
            loss = self.gradient_train({
                "lidar": old_states_lidar,
                "velocity": old_states_velocity,
                # "x": old_states_x,
                # "y": old_states_y,
                "yaw": old_states_yaw
            }, target_q, one_hot_actions)

            # loss 기록용 csv 저장 경로
            log_path = "loss_log_sanitycheck.csv"
            # 첫 줄 헤더 작성 (처음 한 번만)
            if step_number == 0 and not os.path.exists(log_path):
                with open(log_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "loss"])
                    
            # 매 학습 스텝마다 loss 기
            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step_number, float(loss)])
            
        elif self.add_velocity:
            old_states_lidar = np.asarray([sample.old_state.get_data()[0] for sample in batch])
            old_states_velocity = np.asarray([sample.old_state.get_data()[1] for sample in batch]).reshape((-1, self.history_length))
            new_states_lidar = np.asarray([sample.new_state.get_data()[0] for sample in batch])
            new_states_velocity = np.asarray([sample.new_state.get_data()[1] for sample in batch]).reshape((-1, self.history_length))
            actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])

            predicted = self.target_predict({'lidar': new_states_lidar, 'velocity': new_states_velocity})
            q_new_state = np.max(predicted, axis=1)
            target_q = rewards + (self.gamma*q_new_state * (1-is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)# using tf.one_hot causes strange errors

            loss = self.gradient_train({'lidar': old_states_lidar, 'velocity': old_states_velocity}, target_q, one_hot_actions)
        else:
            old_states = np.asarray([sample.old_state.get_data() for sample in batch])
            new_states = np.asarray([sample.new_state.get_data() for sample in batch])
            actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])

            q_new_state = np.max(predicted, axis=1)
            target_q = rewards + (self.gamma * q_new_state * (1 - is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)
            loss = self.gradient_train({'lidar': old_states_lidar, 'velocity': old_states_velocity}, target_q, one_hot_actions)

        if step_number % self.target_model_update_freq == 0:
            self.behavior_net.set_weights(self.target_net.get_weights())
        return loss

    # def train(self, batch, step_number):
    #     if self.add_velocity:
    #         old_states_lidar = np.asarray([sample.old_state.get_data()[0] for sample in batch])
    #         old_states_velocity = np.asarray([sample.old_state.get_data()[1] for sample in batch])
    #         new_states_lidar = np.asarray([sample.new_state.get_data()[0] for sample in batch])
    #         new_states_velocity = np.asarray([sample.new_state.get_data()[1] for sample in batch])
    #         actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
    #         rewards = np.asarray([sample.reward for sample in batch])
    #         is_terminal = np.asarray([sample.terminal for sample in batch])

    #         predicted = self.target_predict({'lidar': new_states_lidar, 'velocity': new_states_velocity})
    #         q_new_state = np.max(predicted, axis=1)
    #         target_q = rewards + (self.gamma * q_new_state * (1 - is_terminal))
    #         one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)
    #         loss = self.gradient_train({'lidar': old_states_lidar, 'velocity': old_states_velocity}, target_q, one_hot_actions)
    #     else:
    #         old_states = np.asarray([sample.old_state.get_data() for sample in batch])
    #         new_states = np.asarray([sample.new_state.get_data() for sample in batch])
    #         actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
    #         rewards = np.asarray([sample.reward for sample in batch])
    #         is_terminal = np.asarray([sample.terminal for sample in batch])

    #     # Stop action delete
    #         for i in range(len(actions)):
    #             if actions[i] == 6: 
    #                 rewards[i] -= 0.5
             
    #         q_new_state = np.max(self.target_predict(new_states), axis=1)
    #         target_q = rewards + (self.gamma * q_new_state * (1 - is_terminal))
    #         one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)
    #         loss = self.gradient_train(old_states, target_q, one_hot_actions)

    #     if step_number % self.target_model_update_freq == 0:
    #         self.behavior_net.set_weights(self.target_net.get_weights())


    #     return loss
    
    def save_episode_reward(self, episode, reward):
        reward_log_path = "episode_rewards_sanitycheck.csv"
        if episode == 0 and not os.path.exists(reward_log_path):
            with open(reward_log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward"])
        with open(reward_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward])

    @tf.function
    def target_predict(self, state):
        return self.target_net(state)

    @tf.function
    def behavior_predict(self, state):
        return self.behavior_net(state)

    @tf.function
    def gradient_train(self, old_states, target_q, one_hot_actions):
        with tf.GradientTape() as tape:
            q_values = self.target_net(old_states)
            current_q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            loss = losses.Huber()(target_q, current_q)

        variables = self.target_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.target_net.optimizer.apply_gradients(zip(gradients, variables))

        return loss


    def save_network(self):
        def async_save():
            print("saving..")
            self.target_net.save_weights(self.checkpoint_dir)
            self.replay_buffer.save()
            print("saved")
        
        save_thread = threading.Thread(target=async_save)
        save_thread.start()