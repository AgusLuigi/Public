import pygame
import math
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyGame Animation Framework")

# Colors
BACKGROUND = (10, 10, 30)
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)
PURPLE = (180, 70, 250)
GREEN = (50, 230, 150)
RED = (255, 70, 90)
YELLOW = (255, 255, 100)
CYAN = (0, 255, 255)

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

# Particle class for various effects
class Particle:
    def __init__(self, x, y, color, velocity_x=0, velocity_y=0, size=3, life=100):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.size = size
        self.life = life
        self.age = 0

    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += 0.05  # Gravity
        self.age += 1
        self.life -= 1

    def draw(self, surface):
        alpha = max(0, min(255, self.life * 2.55))
        color = list(self.color)
        color.append(alpha)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.size)

    def is_dead(self):
        return self.life <= 0

# Wave class for sine wave animation
class Wave:
    def __init__(self, y, amplitude, frequency, speed, color, thickness=2):
        self.y = y
        self.amplitude = amplitude
        self.frequency = frequency
        self.speed = speed
        self.color = color
        self.thickness = thickness
        self.offset = 0

    def update(self):
        self.offset += self.speed

    def draw(self, surface):
        points = []
        for x in range(0, WIDTH + 10, 10):
            y = self.y + math.sin((x + self.offset) * self.frequency) * self.amplitude
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(surface, self.color, False, points, self.thickness)

# Rotating polygon class
class RotatingPolygon:
    def __init__(self, x, y, sides, radius, color, rotation_speed):
        self.x = x
        self.y = y
        self.sides = sides
        self.radius = radius
        self.color = color
        self.rotation_speed = rotation_speed
        self.angle = 0

    def update(self):
        self.angle += self.rotation_speed

    def draw(self, surface):
        points = []
        for i in range(self.sides):
            angle = self.angle + (2 * math.pi * i / self.sides)
            px = self.x + math.cos(angle) * self.radius
            py = self.y + math.sin(angle) * self.radius
            points.append((px, py))

        pygame.draw.polygon(surface, self.color, points, 2)

# Pulsing circle class
class PulsingCircle:
    def __init__(self, x, y, min_radius, max_radius, pulse_speed, color):
        self.x = x
        self.y = y
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.pulse_speed = pulse_speed
        self.color = color
        self.radius = min_radius
        self.growing = True

    def update(self):
        if self.growing:
            self.radius += self.pulse_speed
            if self.radius >= self.max_radius:
                self.growing = False
        else:
            self.radius -= self.pulse_speed
            if self.radius <= self.min_radius:
                self.growing = True

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius), 2)

# Starfield background
class Starfield:
    def __init__(self, num_stars):
        self.stars = []
        for _ in range(num_stars):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            speed = random.uniform(0.1, 0.5)
            size = random.randint(1, 3)
            self.stars.append([x, y, speed, size])

    def update(self):
        for star in self.stars:
            star[1] += star[2]  # Move star down
            if star[1] > HEIGHT:
                star[0] = random.randint(0, WIDTH)
                star[1] = 0

    def draw(self, surface):
        for x, y, _, size in self.stars:
            pygame.draw.circle(surface, WHITE, (int(x), int(y)), size)

# Main animation controller
class AnimationController:
    def __init__(self):
        self.particles = []
        self.waves = [
            Wave(HEIGHT - 100, 20, 0.02, 1, BLUE),
            Wave(HEIGHT - 150, 15, 0.03, 1.2, PURPLE),
            Wave(HEIGHT - 200, 10, 0.04, 1.5, CYAN)
        ]
        self.polygons = [
            RotatingPolygon(WIDTH // 4, HEIGHT // 2, 5, 60, GREEN, 0.02),
            RotatingPolygon(3 * WIDTH // 4, HEIGHT // 2, 7, 80, YELLOW, -0.015),
            RotatingPolygon(WIDTH // 2, HEIGHT // 4, 6, 50, RED, 0.025)
        ]
        self.pulsing_circles = [
            PulsingCircle(WIDTH // 2, HEIGHT // 2, 30, 80, 0.5, PURPLE),
            PulsingCircle(WIDTH // 2, HEIGHT // 2, 50, 120, 0.3, CYAN),
            PulsingCircle(WIDTH // 2, HEIGHT // 2, 70, 160, 0.2, BLUE)
        ]
        self.starfield = Starfield(100)
        self.particle_timer = 0

    def update(self):
        # Update all animation elements
        for wave in self.waves:
            wave.update()

        for polygon in self.polygons:
            polygon.update()

        for circle in self.pulsing_circles:
            circle.update()

        self.starfield.update()

        # Update particles
        for particle in self.particles[:]:
            particle.update()
            if particle.is_dead():
                self.particles.remove(particle)

        # Create new particles periodically
        self.particle_timer += 1
        if self.particle_timer >= 5:
            self.particle_timer = 0
            # Create particles at random positions
            for _ in range(3):
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT // 2)
                color = random.choice([BLUE, PURPLE, GREEN, RED, YELLOW, CYAN])
                velocity_x = random.uniform(-1, 1)
                velocity_y = random.uniform(-2, 0)
                size = random.randint(2, 5)
                life = random.randint(50, 150)
                self.particles.append(Particle(x, y, color, velocity_x, velocity_y, size, life))

    def draw(self, surface):
        # Draw starfield
        self.starfield.draw(surface)

        # Draw pulsing circles
        for circle in self.pulsing_circles:
            circle.draw(surface)

        # Draw rotating polygons
        for polygon in self.polygons:
            polygon.draw(surface)

        # Draw waves
        for wave in self.waves:
            wave.draw(surface)

        # Draw particles on a transparent surface
        particle_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for particle in self.particles:
            particle.draw(particle_surface)
        surface.blit(particle_surface, (0, 0))

        # Draw UI text
        font = pygame.font.SysFont('Arial', 24)
        title = font.render("PyGame Animation Framework", True, WHITE)
        instructions = font.render("Press SPACE to toggle fullscreen, ESC to exit", True, YELLOW)
        surface.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
        surface.blit(instructions, (WIDTH // 2 - instructions.get_width() // 2, HEIGHT - 40))

# Create animation controller
controller = AnimationController()

# Main game loop
running = True
fullscreen = False

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Update animations
    controller.update()

    # Draw everything
    screen.fill(BACKGROUND)
    controller.draw(screen)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()