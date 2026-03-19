import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(step, reward)

    if terminated or truncated:
        obs, info = env.reset()

env.close()