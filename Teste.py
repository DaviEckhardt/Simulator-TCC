from Ambiente import VSS2DEnv
import pybullet as p

env = VSS2DEnv()
obs, _ = env.reset()
print("Estado inicial:", obs)

for _ in range(100000):  # Simula 10 passos
    #p.applyExternalForce(env.ball, -1, [10, 10, 0], [0, 0, 0], p.WORLD_FRAME)
    action = [10.0, 10.0, -10.0, -10.0]  # Ação manual
    obs, reward, done, _, _ = env.step(action)
    print("Estado:", obs)
    print("Recompensa:", reward)
    if done:
        break   