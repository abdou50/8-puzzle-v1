import tkinter as tk
import numpy as np
import heapq
import time
import configurations as cf

class PuzzleSolver:
    def __init__(self, initial_state, goal_state, strategy='a_star'):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.strategy = strategy

    @staticmethod
    def manhattan_distance(state):
        distance = 0
        for i in range(1, 9):
            current_position = state.index(str(i))
            goal_position = i - 1
            current_row, current_col = divmod(current_position, 3)
            goal_row, goal_col = divmod(goal_position, 3)
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance

    @staticmethod
    def get_neighbors(state):
        neighbors = []
        zero_index = state.index("")
        zero_row, zero_col = divmod(zero_index, 3)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_state = state[:]
                new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                neighbors.append(new_state)
        return neighbors

    def a_star_search(self):
        initial_state = tuple(self.initial_state)
        goal_state = tuple(self.goal_state)
        open_set = []
        heapq.heappush(open_set, (0, initial_state))
        came_from = {initial_state: None}
        g_score = {initial_state: 0}
        f_score = {initial_state: self.manhattan_distance(initial_state)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_state:
                return self.reconstruct_path(came_from, current)

            for next_state in self.get_neighbors(list(current)):
                next_state = tuple(next_state)
                tentative_g_score = g_score[current] + 1
                if next_state not in g_score or tentative_g_score < g_score[next_state]:
                    came_from[next_state] = current
                    g_score[next_state] = tentative_g_score
                    f_score[next_state] = tentative_g_score + self.manhattan_distance(next_state)

                    heapq.heappush(open_set, (f_score[next_state], next_state))

        return None

    def hill_climbing(self):
        current_state = list(self.initial_state)
        while current_state != self.goal_state:
            neighbors = self.get_neighbors(current_state)
            current_state = min(neighbors, key=self.manhattan_distance)
            yield current_state
        yield current_state

    @staticmethod
    def reconstruct_path(came_from, current):
        path = []
        while current:
            path.append(list(current))
            current = came_from.get(current)
        return path[::-1]

    def solve(self):
        if self.strategy == 'a_star':
            return self.a_star_search()
        elif self.strategy == 'hill_climbing':
            return self.hill_climbing()

class PuzzleGame:
    def __init__(self, master):
        self.master = master
        self.master.title("8 Puzzle Game")
        self.master.geometry("800x600")
        self.tiles = {}
        self.moves = 0
        self.game_active = True
        self.start_time = time.time()
        self.status_bar = tk.Label(self.master, text="Shuffling...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.moves_label = tk.Label(self.master, text="Moves: 0 | Time: 0s", font=("Arial", 12))
        self.moves_label.pack(side=tk.TOP, fill=tk.X)
        self.puzzle = self.generate_puzzle()
        self.winning_pos = [str(i) for i in range(1, 9)] + [""]
        self.selected_strategy = 'a_star'

        self.left_frame = tk.Frame(self.master, bg='lightgrey')
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20)
        self.right_frame = tk.Frame(self.master, bg='lightgrey')
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20)

        self.start_game(self.puzzle)
        self.show_goal_state()
        self.update_time()

    def update_time(self):
        if self.game_active:
            elapsed_time = int(time.time() - self.start_time)
            self.moves_label.config(text=f"Moves: {self.moves} | Time: {elapsed_time}s")
            self.master.after(1000, self.update_time)  # Update every second

    def solve_puzzle(self):
        initial_state = [self.tiles[(row, col)]['text'] for row in range(3) for col in range(3)]
        solver = PuzzleSolver(initial_state, self.winning_pos, self.selected_strategy)
        if self.selected_strategy == 'a_star':
            solution_path = solver.solve()
            if solution_path:
                self.show_solution(solution_path)
            else:
                self.update_status_bar("No solution found.")
        elif self.selected_strategy == 'hill_climbing':
            solution_generator = solver.solve()
            self.show_hill_climbing_solution(solution_generator)

    def set_strategy(self, strategy):
        self.selected_strategy = strategy
        self.update_status_bar(f"Strategy set to {strategy.replace('_', ' ').title()}")

    def show_solution(self, solution_path, index=0):
        if index < len(solution_path):
            state = solution_path[index]
            self.update_puzzle(state)
            self.master.after(500, self.show_solution, solution_path, index + 1)
        else:
            self.update_status_bar("Solved.")
            self.game_active = False

    def show_hill_climbing_solution(self, solution_generator):
        try:
            state = next(solution_generator)
            self.update_puzzle(state)
            self.master.after(500, self.show_hill_climbing_solution, solution_generator)
        except StopIteration:
            self.update_status_bar("Solved.")
            self.game_active = False

    def find_empty_tile(self):
        for row in range(3):
            for col in range(3):
                if self.tiles[(row, col)]['text'] == "":
                    return row, col

    def swap_tiles(self, clicked_tile, empty_tile):
        self.tiles[clicked_tile]['text'], self.tiles[empty_tile]['text'] = self.tiles[empty_tile]['text'], self.tiles[clicked_tile]['text']
        self.tiles[empty_tile]['background'] = 'white'
        self.tiles[clicked_tile]['background'] = cf._from_rgb((0, 255, 100))
        self.moves += 1
        self.update_status_bar(f"Moves: {self.moves}")
        self.update_time() 

    def on_tile_click(self, row, col):
        empty_x, empty_y = self.find_empty_tile()
        if self.game_active and ((row == empty_x and abs(col - empty_y) == 1) or (col == empty_y and abs(row - empty_x) == 1)):
            self.swap_tiles((row, col), (empty_x, empty_y))
            self.check_winning_state()

    def is_solvable(self, puzzle):
        p = puzzle[puzzle != 0]
        inversions = 0
        for i, x in enumerate(p):
            for y in p[i + 1:]:
                if x > y:
                    inversions += 1
        return inversions % 2 == 0

    def check_winning_state(self):
        current_state = [self.tiles[(row, col)]['text'] for row in range(3) for col in range(3)]
        if current_state == self.winning_pos:
            self.game_active = False
            elapsed_time = int(time.time() - self.start_time)
            self.update_status_bar(f"Congratulations, You Won!! Moves: {self.moves}, Time: {elapsed_time}s")

    def update_status_bar(self, message):
        self.status_bar.config(text=message)

    def generate_puzzle(self):
        while True:
            puzzle = np.random.permutation(9)
            if self.is_solvable(puzzle):
                return puzzle

    def reset_game(self):
        self.moves = 0
        self.game_active = True
        self.start_time = time.time()
        self.update_status_bar("Shuffling...")
        self.puzzle = self.generate_puzzle()
        self.tiles[self.find_empty_tile()]["background"] = 'white'
        for c, i in enumerate(self.puzzle.flatten()):
            row, col = divmod(c, 3)
            self.tiles[(row, col)].config(text=str(i) if i != 0 else "", background=cf._from_rgb((0, 255, 100)) if i == 0 else None)
        self.update_time()

    def start_game(self, p):
        self.master.configure(background='lightgrey') 
        self.tiles_frame = tk.Frame(self.left_frame, bg='lightgrey')
        self.tiles_frame.pack(pady=(20, 0))  

        tile_font = ('Arial', 24, 'bold')

        for c, i in enumerate(p):
            row, col = divmod(c, 3)
            tile_color = 'white' if i != 0 else cf._from_rgb((0, 255, 100))
            tile_text = str(i) if i != 0 else ""
            tile = tk.Button(self.tiles_frame, background=tile_color, text=tile_text, width=5, height=2,
                             font=tile_font, command=lambda x=row, y=col: self.on_tile_click(x, y))
            tile.grid(row=row, column=col, padx=5, pady=5)  
            self.tiles[(row, col)] = tile

        button_font = ('Arial', 14)
        control_frame = tk.Frame(self.left_frame, bg='lightgrey')
        control_frame.pack(pady=(20, 0))
        reset_button = tk.Button(control_frame, text="Restart", width=10, height=1, font=button_font, command=self.reset_game)
        reset_button.grid(row=0, column=0, padx=10)
        solve_button = tk.Button(control_frame, text="Solve", width=10, height=1, font=button_font, command=self.solve_puzzle)
        solve_button.grid(row=0, column=1, padx=10)

        strategy_frame = tk.Frame(self.left_frame, bg='lightgrey')
        strategy_frame.pack(pady=(20, 0))
        a_star_button = tk.Button(strategy_frame, text="A* Search", width=10, height=1, font=button_font, command=lambda: self.set_strategy('a_star'))
        a_star_button.grid(row=0, column=0, padx=10)
        hill_climbing_button = tk.Button(strategy_frame, text="Hill Climbing", width=10, height=1, font=button_font, command=lambda: self.set_strategy('hill_climbing'))
        hill_climbing_button.grid(row=0, column=1, padx=10)

    def show_goal_state(self):
        goal_frame = tk.Frame(self.right_frame, bg='lightgrey')
        goal_frame.pack(pady=(20, 0))
        tile_font = ('Arial', 24, 'bold')

        for c, i in enumerate(self.winning_pos):
            row, col = divmod(c, 3)
            tile_color = 'white' if i != "" else cf._from_rgb((0, 255, 100))
            tile_text = i
            tile = tk.Label(goal_frame, background=tile_color, text=tile_text, width=5, height=2, font=tile_font)
            tile.grid(row=row, column=col, padx=5, pady=5)

    def update_puzzle(self, state):
        for index, tile_value in enumerate(state):
            row, col = divmod(index, 3)
            if tile_value == "":
                self.tiles[(row, col)]['background'] = cf._from_rgb((0, 255, 100))
            else:
                self.tiles[(row, col)]['background'] = 'white'
            self.tiles[(row, col)]['text'] = tile_value
        self.moves += 1
        self.update_status_bar(f"Moves: {self.moves}")
        self.update_time() 

if __name__ == '__main__':
    mainWindow = tk.Tk()
    game = PuzzleGame(mainWindow)
    mainWindow.mainloop()
