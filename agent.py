import gym
import stable_baselines3 as sb3
import minerl

class Agent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.model = sb3.PPO('MlpPolicy', self.env, verbose=1)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def act(self, observation):
        action, _states = self.model.predict(observation)
        return action