import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
"""DANDELİON DO ALGORİTMA YANİ KARAHİNDİBAĞI"""
"""şeytan tüyü dediğimiz karahindibağı çiçeğinin tohumlarının yayılışı mantığındadır ismi de ordan gelmektedir"""
"""belirli bir bölgeyi keşif ve öğrenme açısından gayet iyi bir optimizasyon algoritmasıdır
"""
class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.velocity = np.zeros(2)
        self.alive = True
        self.path = [self.position.copy()]

    def update_position_DO(self, global_best, step, max_steps):
        # Uçuş katsayısı zamanla azalır
        F = 0.5 * (1 - step / max_steps)  # exploration azalır
        rand = np.random.uniform(-1, 1, 2)

        # Dandelion formülü
        new_position = self.position + F * rand * (global_best - self.position)

        # Doğrudan hedefe yönlendirme (yumuşak sezgisel kuvvet)
        to_target = self.target - self.position
        distance = np.linalg.norm(to_target)
        if distance > 1e-6:
            direction = to_target / distance
            new_position += 0.2 * direction

        # Greedy yer değiştirme
        if np.linalg.norm(new_position - self.target) < np.linalg.norm(self.position - self.target):
            self.position = new_position
            self.path.append(self.position.copy())

    def check_impact(self, impact_radius=4):
        if np.linalg.norm(self.position - self.target) < impact_radius:
            self.alive = False
            print(f"Drone {self.id} hedefi imha etti!")

# Simülasyon ayarları
num_drones = 30
max_steps = 100
target_position = np.array([100, 100])
drones = [Drone(position=np.random.rand(2) * 50, target=target_position, drone_id=i) for i in range(num_drones)]

# Görselleştirme
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    ax.clear()
    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    positions = np.array([drone.position for drone in alive_drones])
    distances = np.linalg.norm(positions - target_position, axis=1)
    global_best = positions[np.argmin(distances)]

    for drone in alive_drones:
        drone.update_position_DO(global_best, step=frame, max_steps=max_steps)
        drone.check_impact()

    for drone in drones:
        path = np.array(drone.path)
        if drone.alive:
            ax.plot(path[:, 0], path[:, 1])
            ax.scatter(drone.position[0], drone.position[1], marker='o')
        else:
            ax.plot(path[:, 0], path[:, 1], '--')
            ax.scatter(drone.position[0], drone.position[1], marker='x')

    ax.scatter(target_position[0], target_position[1], c='red', marker='*', s=200)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_title(f'Dandelion Optimizer Simülasyonu - Adım {frame}')
    ax.grid()

def calculate_accuracy():
    destroyed = sum(not drone.alive for drone in drones)
    total = len(drones)
    print(f"\nAccuracy: {destroyed}/{total} = {destroyed / total:.2%}")

def on_animation_end(evt):
    calculate_accuracy()

ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)
ani._stop = on_animation_end
plt.show()
