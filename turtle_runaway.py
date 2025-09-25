# This example is not working in Spyder directly (F5 or Run)
# Please type '!python turtle_runaway.py' on IPython console in your Spyder.
import tkinter as tk
import turtle, random, time, math

class RunawayGame:
    def __init__(self, canvas, player, catch_radius=50):
        self.canvas = canvas
        self.player = player
        self.catch_radius2 = catch_radius**2
        self.score = 0
        self.catch_cooldown = 1.0
        self.last_catch_time = 0

        self.player.shape('turtle')
        self.player.color('red')
        self.player.penup()

        self.runner_drawer = turtle.RawTurtle(canvas)
        self.runner_drawer.shape('turtle')
        self.runner_drawer.color('blue')
        self.runner_drawer.penup()
        self.runner_drawer.setpos(self.player.runner_pos[0], self.player.runner_pos[1])
        self.runner_drawer.setheading(self.player.runner_heading)

        self.drawer = turtle.RawTurtle(canvas)
        self.drawer.hideturtle()
        self.drawer.penup()

        self.catch_just_happened = False

    def is_catched(self):
        dx = self.player.runner_pos[0] - self.player.xcor()
        dy = self.player.runner_pos[1] - self.player.ycor()
        return dx**2 + dy**2 < self.catch_radius2

    def start(self, ai_timer_msec=100, total_time=60):
        self.player.setpos(200, 0)
        self.player.setheading(180)
        self.player.runner_pos = [-200, 0]
        self.player.runner_heading = 0

        self.start_time = time.time()
        self.ai_timer_msec = ai_timer_msec
        self.total_time = total_time
        self.step()
        self.canvas.ontimer(self.step, self.ai_timer_msec)

    def step(self):
        if not self.catch_just_happened:
            self.player.run_ai()

        self.runner_drawer.setpos(self.player.runner_pos[0], self.player.runner_pos[1])
        self.runner_drawer.setheading(self.player.runner_heading)

        if self.is_catched():
            now = time.time()
            if now - self.last_catch_time > self.catch_cooldown:
                self.score += 1
                self.last_catch_time = now
                self.catch_just_happened = True

                self.player.setpos(random.randint(-300,300), random.randint(-300,300))
                self.player.runner_pos = [random.randint(-300,300), random.randint(-300,300)]
                self.player.runner_heading = random.randint(0,360)

                self.runner_drawer.setpos(self.player.runner_pos[0], self.player.runner_pos[1])
                self.runner_drawer.setheading(self.player.runner_heading)
        else:
            self.catch_just_happened = False

        remain = max(0, math.ceil(self.total_time - (time.time() - self.start_time)))
        self.drawer.clear()
        self.drawer.setpos(200, 300)
        self.drawer.write(f"Time: {remain} / Score: {self.score}")

        if remain <= 0:
            self.drawer.clear()
            self.drawer.setpos(-100, 0)
            self.drawer.write(f"Time Over! Final Score: {self.score}", font=("Arial",16,"bold"))
            return

        self.canvas.ontimer(self.step, self.ai_timer_msec)


class ManualMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=15):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn

        self.runner_pos = [-200, 0]
        self.runner_heading = 0

        self.key_history = []

        self.x_limit = 350
        self.y_limit = 350

        canvas.onkeypress(lambda: self.move_chaser('Up'), 'Up')
        canvas.onkeypress(lambda: self.move_chaser('Down'), 'Down')
        canvas.onkeypress(lambda: self.move_chaser('Left'), 'Left')
        canvas.onkeypress(lambda: self.move_chaser('Right'), 'Right')
        canvas.listen()

    def move_chaser(self, key):
        self.key_history.append(key)
        if len(self.key_history) > 5:
            self.key_history.pop(0)

        if key == 'Up':
            self.setheading(90)
            self.sety(min(self.ycor() + self.step_move, self.y_limit))
        elif key == 'Down':
            self.setheading(270)
            self.sety(max(self.ycor() - self.step_move, -self.y_limit))
        elif key == 'Left':
            self.setheading(180)
            self.setx(max(self.xcor() - self.step_move, -self.x_limit))
        elif key == 'Right':
            self.setheading(0)
            self.setx(min(self.xcor() + self.step_move, self.x_limit))

    def run_ai(self):
        dx, dy = 0, 0
        if self.key_history:
            for k in self.key_history:
                if k == 'Up': dy += 1
                elif k == 'Down': dy -= 1
                elif k == 'Left': dx -= 1
                elif k == 'Right': dx += 1
        else:
            dx = random.choice([-1,0,1])
            dy = random.choice([-1,0,1])
            if dx == 0 and dy == 0:
                dy = 1

        angle = math.degrees(math.atan2(dy, dx))
        self.runner_heading = (angle + 180) % 360
        self.runner_pos[0] += 12 * math.cos(math.radians(self.runner_heading))
        self.runner_pos[1] += 12 * math.sin(math.radians(self.runner_heading))

        self.runner_pos[0] = max(-self.x_limit, min(self.x_limit, self.runner_pos[0]))
        self.runner_pos[1] = max(-self.y_limit, min(self.y_limit, self.runner_pos[1]))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Turtle Runaway")
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    screen = turtle.TurtleScreen(canvas)

    player = ManualMover(screen)

    game = RunawayGame(screen, player)
    game.start()
    screen.mainloop()
