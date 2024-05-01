import random
import pixie

class Maze:
    def __init__(self, width, height, grid_size=100):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.end = (width - 1, height - 1)
        self.maze = [[0] * width for _ in range(height)]
        self.grid_size = grid_size
        self.init_image()

    def init_image(self):
        self.image = pixie.Image(width = self.grid_size * self.width, height = self.grid_size * self.height)
        self.image.fill(pixie.Color(0, 0, 0, 1))
        self.paint = pixie.Paint(pixie.SOLID_PAINT)
        self.paint.color = pixie.Color(1, 1, 1, 1)
        self.lines = pixie.Paint(pixie.SOLID_PAINT)
        self.lines.color = pixie.Color(0.3, 0.3, 0.3, 1)
        self.ctx = self.image.new_context()
        self.ctx.fill_style = self.paint
        self.ctx.stroke_style = self.lines
        self.ctx.line_width = 2
        for y in range(self.height):
            self.ctx.stroke_segment(0, y * self.grid_size, self.width * self.grid_size, y * self.grid_size)
            for x in range(self.width):
                self.ctx.stroke_segment(x * self.grid_size, 0, x * self.grid_size, self.height * self.grid_size)

    def random_tuple(self, xsize, ysize):
        return (random.randint(0, xsize), random.randint(0, ysize))

    def randomize_start_end(self):
        self.start = self.random_tuple(self.width - 1, self.height - 1)
        self.end = self.random_tuple(self.width - 1, self.height - 1)
        print(f'start: {self.start} end: {self.end}')

    def generate_maze(self):
        self.maze_dfs(*self.start)

    def maze_dfs(self, x, y):
        
        print(f'marking location {x} {y}')
        self.maze[y][x] = 1
        self.ctx.fill_rect(self.grid_size * x, self.grid_size * y, 
                           self.grid_size * x + self.grid_size, 
                           self.grid_size * y + self.grid_size)
        if ((x, y) == self.end):
            return

        # Define the directions and shuffle them to ensure randomness
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            if (0 <= new_x < self.width and 
                    0 <= new_y < self.height and
                    self.maze[new_y][new_x] == 0):
                self.maze_dfs(new_x, new_y)
                break # just use one

    # each grid location is grid_size x grid_size pixels
    def write_image(self, filepath):
        self.image.write_file(filepath)

mz = Maze(10, 10)
#mz.randomize_start_end()
mz.generate_maze()
mz.write_image('./foo.png')

