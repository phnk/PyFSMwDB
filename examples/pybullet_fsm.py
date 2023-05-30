import gym
import wl_gym_driving

from stable_baselines3 import PPO

from PyFSMwDB.FSM import FSM
from PyFSMwDB.State import State
from PyFSMwDB.Logic import Logic

from signal import signal, SIGINT

REST_TIME = 1/500


class NormalTest:

    def __init__(self):
        self.s = 1
        env = gym.make('WL-Driving-Full', debug=True)
        reverse_model = PPO("MlpPolicy", env).load("logs/good_reversal_point/reverse/test_ppo_WL-Driving-Reverse_1675795200/best_model")
        forward_model = PPO("MlpPolicy", env).load("logs/good_reversal_point/forward/test_ppo_WL-Driving-Forward_1675877624/best_model")
        obs = env.reset()

        fsm = FSM()
        backward = State(self.drive_backward)
        forward = State(self.drive_forward)
        reseter = State(self.reseter)

        backward.add_transition(True, forward)
        backward.add_transition(False, backward)

        forward.add_transition(True, reseter)
        forward.add_transition(False, forward)

        reseter.add_transition(False, backward)
        reseter.add_transition(True, backward)

        fsm.add_states([backward, forward, reseter])
        fsm.run([reverse_model, forward_model, obs, env])

        env.close()

    @staticmethod
    def drive_backward(li):
        backward_model, forward_model, obs, env = li
        action, _ = backward_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if not env.reached:
            return False, [backward_model, forward_model, obs, env]
        return True, [backward_model, forward_model, obs, env]

    @staticmethod
    def drive_forward(li):
        backward_model, forward_model, obs, env = li
        action, _ = forward_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if not done:
            return False, [backward_model, forward_model, obs, env]
        return True, [False, backward_model, forward_model, obs, env]

    @staticmethod
    def reseter(li):
        origin, backward_model, forward_model, _,  env = li
        obs = env.reset()
        return origin, [backward_model, forward_model, obs, env]


if __name__ == "__main__":
    while True:
        test = NormalTest()
