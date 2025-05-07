import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## FA (Firefly Algorithm) tabanlı Kamikaze Drone Sürüsü (Hedef Odaklı)

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def move(self, best_drone=None, beta0=1.0, gamma=0.01, alpha=0.2):
        if not self.alive:
            return

        move_vector = np.zeros(2)

        # 1. Hedefe yönelme kuvveti
        r_target = np.linalg.norm(self.position - self.target)
        beta_target = beta0 * np.exp(-gamma * r_target ** 2)
        move_vector += beta_target * (self.target - self.position)

        # 2. Daha iyi drone varsa ona yönelme kuvveti
        if best_drone:
            r_best = np.linalg.norm(self.position - best_drone.position)
            beta_best = beta0 * np.exp(-gamma * r_best ** 2)
            move_vector += beta_best * (best_drone.position - self.position)

        # 3. Küçük rastgelelik
        move_vector += alpha * (np.random.rand(2) - 0.5)

        self.position += move_vector
        self.path.append(self.position.copy())

    def check_impact(self, impact_radius=2):
        if np.linalg.norm(self.position - self.target) < impact_radius:
            self.alive = False
            print(f"Drone {self.id} hedefi imha etti!")

# Başlangıç ayarları
num_drones = 30
target_position = np.array([100, 100])

# Drone sürüsü oluştur
drones = [Drone(position=np.random.rand(2) * 50, target=target_position, drone_id=i) for i in range(num_drones)]

# Simülasyon parametreleri
max_steps = 100

# Animasyon için veri
fig, ax = plt.subplots(figsize=(8,8))

def update(frame):
    ax.clear()

    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    for drone_i in alive_drones:
        # Kendinden daha iyi (hedefe daha yakın) drone'u bul
        best_drone = None
        min_distance = np.linalg.norm(drone_i.position - drone_i.target)

        for drone_j in alive_drones:
            if drone_j is drone_i:
                continue
            distance_to_target = np.linalg.norm(drone_j.position - drone_j.target)
            if distance_to_target < min_distance:
                best_drone = drone_j
                min_distance = distance_to_target

        # Hareket: hem hedef hem iyi drone'a doğru
        drone_i.move(best_drone)
        drone_i.check_impact()

    # Çizim
    for drone in drones:
        path = np.array(drone.path)
        if drone.alive:
            ax.plot(path[:,0], path[:,1], label=f"Drone {drone.id} (Alive)")
            ax.scatter(drone.position[0], drone.position[1], marker='o')
        else:
            ax.plot(path[:,0], path[:,1], '--', label=f"Drone {drone.id} (Destroyed)")
            ax.scatter(drone.position[0], drone.position[1], marker='x')

    ax.scatter(target_position[0], target_position[1], c='red', marker='*', s=200, label='Hedef')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_title(f'Firefly Kamikaze Drone Sürüsü - Adım {frame}')
    ax.legend()
    ax.grid()

ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)
plt.show()
