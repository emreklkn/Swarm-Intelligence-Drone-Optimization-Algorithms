import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
"""sos algoritması yani simbiyotik algoritma mutalism commansalizm parasiztizm şeklinde """
"""sos in hızlı olmasının sebebi birbirinden faydalanıyor aslında dronler birbirinden bilgi alıyor bu şekilde hedef bilgisi hemen alınmış oluyor"""
"""bu algoritma iha drone sürüsünde belirli bölgelerdeki hedef tespiti için ve bu hedef tespiti sonucu hedefe yönelim için gayet iyi bir yöntem"""

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def fitness(self, pos):
        return np.linalg.norm(pos - self.target)

    def update_position_SOS(self, drones, step, max_steps):
        if not self.alive:
            return

        # --------- Phase 1: Mutualism ---------
        partner = np.random.choice([d for d in drones if d.id != self.id and d.alive])
        mutual_vector = (self.position + partner.position) / 2
        bf1, bf2 = np.random.uniform(1, 2, 2)

        new_pos1 = self.position + np.random.rand(2) * (self.target - mutual_vector * bf1)
        new_pos2 = partner.position + np.random.rand(2) * (self.target - mutual_vector * bf2)

        if self.fitness(new_pos1) < self.fitness(self.position):
            self.position = new_pos1
        if self.fitness(new_pos2) < partner.fitness(partner.position):
            partner.position = new_pos2

        # --------- Phase 2: Commensalism ---------
        partner2 = np.random.choice([d for d in drones if d.id != self.id and d.alive])
        new_pos = self.position + (np.random.uniform(-1, 1, 2) * (self.target - partner2.position))

        if self.fitness(new_pos) < self.fitness(self.position):
            self.position = new_pos

        # --------- Phase 3: Parasitism ---------
        host = np.random.choice([d for d in drones if d.id != self.id and d.alive])
        parasite = copy.deepcopy(self.position)
        parasite += np.random.uniform(-1, 1, 2) * np.random.rand(2) * 5  # rastgele sapma

        if self.fitness(parasite) < host.fitness(host.position):
            host.position = parasite

        self.path.append(self.position.copy())

    def check_impact(self, impact_radius=4):
        if np.linalg.norm(self.position - self.target) < impact_radius:
            self.alive = False
            print(f"Drone {self.id} hedefi imha etti!")

# Simülasyon parametreleri
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

    for drone in alive_drones:
        drone.update_position_SOS(drones, frame, max_steps)
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
    ax.set_title(f'SOS Drone Simülasyonu - Adım {frame}')
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
