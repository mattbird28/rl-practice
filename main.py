import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(
    f"step={step} | "
    f"action={action} | "
    f"obs={obs} | "
    f"reward={reward} | "
    f"terminated={terminated} | "
    f"truncated={truncated}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()