import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers
import threading 

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
        else:
            if self.add_velocity:
                return self.__build_cnn1D_plus_velocity()
            else:
                # select from __build_dense or build_cnn1D
                return self.__build_cnn1D()

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

    def __build_cnn1D_plus_velocity_and_pose(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length), name="lidar")
        input_velocity = tf.keras.Input(shape=(1, self.history_length), name="velocity")
        input_x = tf.keras.Input(shape=(1, self.history_length), name="x")
        input_y = tf.keras.Input(shape=(1, self.history_length), name="y")
        input_yaw = tf.keras.Input(shape=(1, self.history_length), name="yaw")
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    
        # LiDAR 데이터에 Squeeze 적용
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(x)  

        # 안전한 Squeeze 함수 정의 (history_length가 1인지 체크)
        def safe_squeeze(t):
            return tf.cond(tf.shape(t)[-1] > 1, lambda: tf.squeeze(t, axis=1), lambda: t)
        # velocity, x, y, yaw에도 Squeeze 적용
        velocity = tf.keras.layers.Lambda(safe_squeeze)(input_velocity)
        x_pos = tf.keras.layers.Lambda(safe_squeeze)(input_x)
        y_pos = tf.keras.layers.Lambda(safe_squeeze)(input_y)
        yaw = tf.keras.layers.Lambda(safe_squeeze)(input_yaw)

        # Concatenate 수행 (Flatten 없이 안전하게 차원 맞춤)
        x = layers.concatenate([x, velocity, x_pos, y_pos, yaw])
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=[inputs, input_velocity, input_x, input_y, input_yaw], outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                  loss=losses.Huber())

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

        elif self.add_velocity and self.add_pose:  # velocity와 pose를 모두 추가하는 경우
            state["lidar"] = state["lidar"].reshape((-1, self.state_size, self.history_length))
            state["velocity"] = np.asarray(state["velocity"]).reshape((-1, self.history_length))
            state["x"] = np.asarray(state["x"]).reshape((-1, self.history_length))
            state["y"] = np.asarray(state["y"]).reshape((-1, self.history_length))
            state["yaw"] = np.asarray(state["yaw"]).reshape((-1, self.history_length))

            state_dict = {
                "lidar": state["lidar"],
                "velocity": state["velocity"],
                "x": state["x"],
                "y": state["y"],
                "yaw": state["yaw"]
            }
            return np.asarray(self.behavior_predict(state_dict)).argmax(axis=1)
            
        elif self.add_velocity:  
            if isinstance(state, dict):  # 딕셔너리인 경우
                state["lidar"] = state["lidar"].reshape((-1, self.state_size, self.history_length))
                state["velocity"] = np.asarray(state["velocity"]).reshape((-1, self.history_length))
                return np.asarray(self.behavior_predict(state)).argmax(axis=1)
            elif isinstance(state, list) and len(state) == 2:  # ✅ 리스트일 경우 (기존 코드 유지)
                state[0] = state[0].reshape((-1, self.state_size, self.history_length))
                state[1] = state[1].reshape((-1, self.history_length))
                return np.asarray(self.behavior_predict(state)).argmax(axis=1)
                
        else:  
            state = state.reshape((-1, self.state_size, self.history_length))

        return np.asarray(self.behavior_predict(state)).argmax(axis=1)

    def train(self, batch, step_number):
        if self.add_velocity:
            old_states_lidar = np.asarray([sample.old_state.get_data()[0] for sample in batch])
            old_states_velocity = np.asarray([sample.old_state.get_data()[1] for sample in batch])
            new_states_lidar = np.asarray([sample.new_state.get_data()[0] for sample in batch])
            new_states_velocity = np.asarray([sample.new_state.get_data()[1] for sample in batch])
            actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])

            predicted = self.target_predict({'lidar': new_states_lidar, 'velocity': new_states_velocity})
            
        if self.add_velocity and self.add_pose:
            old_states_lidar = np.asarray([sample.old_state["lidar"] for sample in batch])
            old_states_velocity = np.asarray([sample.old_state["velocity"] for sample in batch]).reshape((-1, 1, self.history_length))
            old_states_x = np.asarray([sample.old_state["x"] for sample in batch]).reshape((-1, 1, self.history_length))
            old_states_y = np.asarray([sample.old_state["y"] for sample in batch]).reshape((-1, 1, self.history_length))
            old_states_yaw = np.asarray([sample.old_state["yaw"] for sample in batch]).reshape((-1, 1, self.history_length))
            
            new_states_lidar = np.asarray([sample.new_state["lidar"] for sample in batch])
            new_states_velocity = np.asarray([sample.new_state["velocity"] for sample in batch]).reshape((-1, 1, self.history_length))
            new_states_x = np.asarray([sample.new_state["x"] for sample in batch]).reshape((-1, 1, self.history_length))  # ✅ 수정됨
            new_states_y = np.asarray([sample.new_state["y"] for sample in batch]).reshape((-1, 1, self.history_length))  # ✅ 수정됨
            new_states_yaw = np.asarray([sample.new_state["yaw"] for sample in batch]).reshape((-1, 1, self.history_length))
            
            actions = np.asarray([sample.action[0] if isinstance(sample.action, (list, np.ndarray)) else sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])
            predicted = self.target_predict({
                "lidar": new_states_lidar,
                "velocity": new_states_velocity,
                "x": new_states_x,
                "y": new_states_y,
                "yaw": new_states_yaw
            })

            q_new_state = np.max(predicted, axis=1)
            target_q = rewards + (self.gamma * q_new_state * (1 - is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)
            
            loss = self.gradient_train({
                "lidar": old_states_lidar,
                "velocity": old_states_velocity,
                "x": old_states_x,
                "y": old_states_y,
                "yaw": old_states_yaw
            }, target_q, one_hot_actions)
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
            
        # Stop action delete
            for i in range(len(actions)):
                if actions[i] == 6: 
                    rewards[i] -= 0.5
             
            q_new_state = np.max(self.target_predict(new_states), axis=1)
            target_q = rewards + (self.gamma * q_new_state * (1 - is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)
            loss = self.gradient_train(old_states, target_q, one_hot_actions)

        if step_number % self.target_model_update_freq == 0:
            self.behavior_net.set_weights(self.target_net.get_weights())


        return loss

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
