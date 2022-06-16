import os
import random
import neat
import pygame
pygame.font.init()

window_width = 400
window_height = 600

Generation = 0
BIRD_IMGS = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))), pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "bird2.png"))), pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "bird3.png")))
PIPE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("images", "base.png")))
# picture = pygame.image.load(filename)
# picture = pygame.transform.scale(picture, (1280, 720))
picture = pygame.image.load(os.path.join("images", "bg.png"))
BG_IMG = pygame.transform.scale(picture, (1280, 720))

STAT_FONT = pygame.font.SysFont("Arial", 30)


class Bird:
    IMGS = BIRD_IMGS
    Birds_Rotation = 25
    Birds_Rot_Velocity = 20
    Pixels_Per_Second = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bird_tilt = 0
        self.count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.count = 0
        self.height = self.y

    def move(self):
        self.count += 1
        # indicating our terminal velocity best interpreted results.
        d = self.vel * self.count + 1.5 * self.count ** 2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.bird_tilt < self.Birds_Rotation:
                self.bird_tilt = self.Birds_Rotation
        else:
            if self.bird_tilt > -90:
                self.bird_tilt -= self.Birds_Rot_Velocity

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.Pixels_Per_Second:
            self.img = self.IMGS[0]
        elif self.img_count < self.Pixels_Per_Second * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.Pixels_Per_Second * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.Pixels_Per_Second * 4:
            self.img = self.IMGS[1]
        elif self.img_count < self.Pixels_Per_Second * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.bird_tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.Pixels_Per_Second * 2

        rotated_image = pygame.transform.rotate(self.img, self.bird_tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    Gap = 200
    Vel = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 150

        self.top = 0
        self.bottom = 0
        self.Pipe_top = pygame.transform.flip(PIPE_IMG, False, True)
        self.Pipe_bottom = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 350)
        self.top = self.height - self.Pipe_top.get_height()
        self.bottom = self.height + self.gap

    def move(self):
        self.x -= self.Vel

    def draw(self, win):
        win.blit(self.Pipe_top, (self.x, self.top))
        win.blit(self.Pipe_bottom, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.Pipe_top)
        bottom_mask = pygame.mask.from_surface(self.Pipe_bottom)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True
        return False


class Base:
    VEl = 5
    Width = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.Width

    def move(self):
        self.x1 -= self.VEl
        self.x2 -= self.VEl

        if self.x1 + self.Width < 0:
            self.x1 = self.x2 + self.Width

        if self.x2 + self.Width < 0:
            self.x2 = self.x1 + self.Width

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, Generation):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (window_width - 10 - text.get_width(), 10))
    text = STAT_FONT.render("Gen: " + str(Generation), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    global Generation
    Generation += 1

    nodes = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nodes.append(net)
        birds.append(Bird(160, 250))
        g.fitness = 0
        ge.append(g)

    win = pygame.display.set_mode((window_width, window_height))

    # bird = Bird(160,250)
    base = Base(550)
    pipes = [Pipe(470)]
    clock = pygame.time.Clock()
    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # bird.move()

        pip_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].Pipe_top.get_width():
                pip_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nodes[x].activate(
                (bird.y, abs(bird.y - pipes[pip_ind].height), abs(bird.y - pipes[pip_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()
        rem = []
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    ge.pop(x)
                    nodes.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.Pipe_top.get_width() < 0:
                rem.append(pipe)

            pipe.move()
        if add_pipe:

            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(470))
        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 550 or bird.y < 0:
                birds.pop(x)
                ge.pop(x)
                nodes.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score, Generation)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 100)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    run(config_path)
