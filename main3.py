import pygame
import random
import math
import numpy as np
import time

# --- Game Constants ---
WIDTH, HEIGHT = 800, 600
PLAYER_SPEED = 3
SIDE_SPEED = 4
ASTEROID_COUNT = 8
ASTEROID_SPEED = 2
LIDAR_RAYS = 24
LIDAR_RANGE = 200
MAX_BLASTS = 5
BLAST_RECHARGE_TIME = 60
FPS = 60


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class AsteroidEnv:
    """
    Asteroid collision environment (manual input version)
    - Arrow keys to move
    - Space to shoot
    - Reward: +0.1 per frame alive, -10 on collision
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Asteroid Collision — Manual Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT * 0.75
        self.radius = 15
        self.blasts = MAX_BLASTS
        self.last_recharge = time.time()
        self.score = 0
        self.start_time = time.time()
        self.alive = True

        self.asteroids = [self._new_asteroid() for _ in range(ASTEROID_COUNT)]
        self.blasts_list = []
        return self._get_observation()

    def _new_asteroid(self):
        return {
            "x": random.randint(0, WIDTH),
            "y": random.randint(-HEIGHT, 0),
            "radius": random.randint(15, 40),
            "speed": ASTEROID_SPEED,
        }

    def _move_asteroids(self):
        for a in self.asteroids:
            a["y"] += a["speed"]
        self.asteroids = [
            a if a["y"] - a["radius"] <= HEIGHT else self._new_asteroid()
            for a in self.asteroids
        ]

    def _move_player(self, keys):
        # forward motion (upwards)
        self.y -= PLAYER_SPEED
        # restrict to center vertical area
        self.y = np.clip(self.y, HEIGHT * 0.5, HEIGHT * 0.75)

        # sideways motion
        if keys[pygame.K_LEFT]:
            self.x -= SIDE_SPEED
        if keys[pygame.K_RIGHT]:
            self.x += SIDE_SPEED
        self.x = np.clip(self.x, WIDTH // 2 - 150, WIDTH // 2 + 150)

    def _shoot(self):
        if self.blasts > 0:
            self.blasts -= 1
            self.blasts_list.append({"x": self.x, "y": self.y})

    def _move_blasts(self):
        for b in self.blasts_list:
            b["y"] -= 7
        self.blasts_list = [b for b in self.blasts_list if b["y"] > 0]

    def _check_collisions(self):
        for a in self.asteroids:
            if distance((self.x, self.y), (a["x"], a["y"])) <= self.radius + a["radius"]:
                return True
        return False

    def _check_blast_hits(self):
        hits = []
        for b in self.blasts_list:
            for a in self.asteroids:
                if distance((b["x"], b["y"]), (a["x"], a["y"])) <= a["radius"]:
                    hits.append(a)
                    break
        for h in hits:
            self.asteroids.remove(h)
            self.asteroids.append(self._new_asteroid())
        return len(hits)

    def _get_lidar(self):
        lidar = []
        for angle in np.linspace(0, 360, LIDAR_RAYS, endpoint=False):
            dx = math.cos(math.radians(angle))
            dy = math.sin(math.radians(angle))
            dist = LIDAR_RANGE
            for r in range(0, LIDAR_RANGE, 5):
                rx = self.x + dx * r
                ry = self.y + dy * r
                if rx < 0 or rx >= WIDTH or ry < 0 or ry >= HEIGHT:
                    dist = r
                    break
                for a in self.asteroids:
                    if distance((rx, ry), (a["x"], a["y"])) <= a["radius"]:
                        dist = r
                        break
                else:
                    continue
                break
            lidar.append(dist / LIDAR_RANGE)
        return np.array(lidar, dtype=np.float32)

    def _get_observation(self):
        return self._get_lidar()

    def step(self, action=None):
        """
        For compatibility, action can be None (manual mode)
        """
        # handle input
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, 0, True, {}
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._shoot()

        self._move_player(keys)
        self._move_asteroids()
        self._move_blasts()
        self._check_blast_hits()

        # compute reward
        done = self._check_collisions()
        reward = 0.1 if not done else -10.0
        self.score += reward

        obs = self._get_observation()
        info = {"score": self.score, "alive": not done}
        return obs, reward, done, info

    def render(self):
        self.screen.fill((10, 10, 20))

        # draw asteroids
        for a in self.asteroids:
            pygame.draw.circle(
                self.screen, (150, 150, 150), (int(a["x"]), int(a["y"])), a["radius"]
            )

        # draw player
        pygame.draw.circle(self.screen, (0, 200, 255), (int(self.x), int(self.y)), self.radius)

        # draw blasts
        for b in self.blasts_list:
            pygame.draw.circle(self.screen, (255, 100, 0), (int(b["x"]), int(b["y"])), 4)

        # draw lidar rays
        lidar = self._get_lidar()
        for i, dist in enumerate(lidar):
            angle = i * (360 / LIDAR_RAYS)
            dx = math.cos(math.radians(angle))
            dy = math.sin(math.radians(angle))
            end_x = self.x + dx * dist * LIDAR_RANGE
            end_y = self.y + dy * dist * LIDAR_RANGE
            pygame.draw.line(self.screen, (100, 100, 100), (self.x, self.y), (end_x, end_y), 1)

        # HUD
        score_text = self.font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = AsteroidEnv()
    obs = env.reset()
    done = False

    while not done:
        obs, reward, done, info = env.step()
        env.render()
        if done:
            print("Game over! Total reward:", info["score"])
            pygame.time.wait(2000)
            done = True

    env.close()
