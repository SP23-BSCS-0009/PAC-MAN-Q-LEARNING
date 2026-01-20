def print_all_details(data):

    print("===================== SAVED SIMULATION DETAILS =====================")

    # Validate top-level structure
    if not isinstance(data, dict):
        print("ERROR: Invalid file format. Expected a dictionary.")
        return

    # -------------------- Parameters --------------------
    params = data.get('parameters')
    print("-------------------- PARAMETERS --------------------")

    if not isinstance(params, dict):
        print("ERROR: 'parameters' missing or corrupted.")
    else:
        if not params:
            print("PARAMETERS: Empty.")
        else:
            for key, value in params.items():
                print(f"{key:25}: {value}")

    # -------------------- Q-Table Summary --------------------
    qtable = data.get('q_table')
    print("-------------------- Q-TABLE SUMMARY --------------------")

    if not isinstance(qtable, dict):
        print("ERROR: 'q_table' missing or corrupted.")
    else:
        num_states = len(qtable)
        print(f"Total States          : {num_states}")

        if num_states > 0:
            all_values = []
            total_actions = 0
            sample_items = []
            for i, (state, actions) in enumerate(qtable.items()):
                if i < 20:
                    sample_items.append((state, actions))
                if isinstance(actions, dict):
                    total_actions += len(actions)
                    for val in actions.values():
                        if isinstance(val, (int, float)):
                            all_values.append(val)
                elif isinstance(actions, (list, tuple)):
                    total_actions += len(actions)
                    for val in actions:
                        if isinstance(val, (int, float)):
                            all_values.append(val)

            print(f"Total Actions Stored  : {total_actions}")
            if all_values:
                print(f"Min Q-value           : {min(all_values):.4f}")
                print(f"Max Q-value           : {max(all_values):.4f}")
                print(f"Avg Q-value           : {sum(all_values)/len(all_values):.4f}")
            else:
                print("No numeric Q-values found.")

            print("Sample Q-table entries (up to 20):")
            for i, (state, actions) in enumerate(sample_items):
                print(f"[{i}] State: {state} -> Actions: {actions}")
        else:
            print('Q-table is empty.')

    # -------------------- EPISODE INFO --------------------
    meta = data.get('meta', {})
    episode = None
    if isinstance(meta, dict) and isinstance(meta.get('episodes_trained'), int):
        episode = meta.get('episodes_trained')
    else:
        episode = data.get('episode')

    print("-------------------- EPISODE INFO --------------------")
    if isinstance(episode, int):
        print(f"Episode Saved         : {episode}")
    else:
        print('Episode number missing or invalid.')

    print("===================== END OF DETAILS =====================")

import os
import pickle
import time
import csv
import datetime
from collections import defaultdict, deque
import tkinter as tk
from tkinter import messagebox, filedialog
import pygame
import random
import math
import threading
import os.path

QTABLE_FILE_EXT = '.qpac'

# ----------------------------- PARAMETERS -----------------------------


# --- Safety: capture uncaught exceptions from any background thread ---
def _threading_excepthook(args):
    try:
        import time, traceback
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        print("\n" + "="*80)
        print("UNHANDLED THREAD EXCEPTION")
        print(tb)
        print("="*80 + "\n")
        with open("pacman_thread_crash.log", "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(tb)
    except Exception:
        pass

try:
    import threading as _threading_mod
    _threading_mod.excepthook = _threading_excepthook
except Exception:
    pass

DEFAULT_PARAMS = {
    'epsilon_min': 0.02,
    'epsilon_decay': 0.9995,
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'pacman_speed': 2.0,          # pixels/frame
    'ghost_base_speed': 1.5,      # pixels/frame
    'pacman_lives': 3,
    'max_steps_per_episode': 4500, # frames (~150s @ 30 FPS)
    'render': True,              # Stage-5: render pygame window (False = headless fast training)
    'eval_every': 0,             # Stage-5: run evaluation every N training episodes (0 disables)
    'eval_episodes': 5,          # Stage-5: number of evaluation episodes per evaluation run
    'metrics_enabled': True,     # Stage-5: write episode metrics CSV alongside .qpac
    'pot_food_w': 0.6,        # reduced to avoid oscillation reward dominating
    'pot_danger_w': 1.5,
    'pot_edible_w': 2.0,
    'pot_power_w': 0.8,
    'decision_step_cost': -0.1,
    'adjacent_ghost_penalty': -4.0,  # penalty when ending adjacent to a dangerous ghost (not powered)

    'revisit_penalty': -0.6,     # stronger penalty for immediate backtrack (anti-oscillation)
    'revisit_exempt_danger': 1,  # if 1: do not penalize revisits when danger is near (escape allowed)
    'block_potential_on_backtrack': 1,  # if 1: POT_* shaping cannot be positive on backtrack steps
    'stuck_penalty': -0.4,  # penalty when oscillating within a tiny tile set
    'stuck_window': 8,      # decisions to look back (longer window catches ABABAB)
    'stuck_unique_thresh': 2,  # <=2 unique tiles in window => 'stuck'
    'stuck_exempt_danger': 1,  # do not apply stuck penalty when any_danger_near==1
    'debug_print': True
}

# ----------------------------- UTILS -----------------------------
def list_simulations():
    return [f for f in os.listdir('.') if f.endswith(QTABLE_FILE_EXT)]

def save_simulation_file(name, qtable, params, episodes):
    data = {
        'q_table': dict(qtable),
        'parameters': params,
        'meta': {
            'state_version': 4,
            'reward_version': 2,
            'episodes_trained': episodes,
            'created_on': time.time(),
            'last_trained': time.time()
        }
    }
    filename = f'{name}{QTABLE_FILE_EXT}'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    messagebox.showinfo('Saved', f'Simulation saved as {filename}')

# ----------------------------- VALIDATION -----------------------------
def validate_param(value, min_val, max_val, param_type=float, show_error=True):
    try:
        if param_type is int:
            val = int(float(value))
        else:
            val = param_type(value)
        if val < min_val or val > max_val:
            raise ValueError
        return val
    except Exception:
        if show_error:
            messagebox.showerror('Invalid input', f'Value must be between {min_val} and {max_val}')
            return None
        else:
            raise ValueError(f'Invalid value: must be between {min_val} and {max_val}')

# ----------------------------- GAME CONFIGURATION -----------------------------
class GameConfig:
    TILE = 20
    WIDTH = 28
    HEIGHT = 31
    FPS = 30

    # ------------------ Stage-6: Traditional Ghost Modes ------------------
    # Mode schedule (approx classic Pac-Man). After the schedule ends, CHASE continues.
    # Times are in frames (FPS*seconds).
    GHOST_MODE_SCHEDULE = [
        ("SCATTER", 7 * FPS),
        ("CHASE",   20 * FPS),
        ("SCATTER", 7 * FPS),
        ("CHASE",   20 * FPS),
        ("SCATTER", 5 * FPS),
        ("CHASE",   20 * FPS),
        ("SCATTER", 5 * FPS),
        ("CHASE",   10**9),  # effectively "forever"
    ]

    # Power pellet frightened duration (frames)
    GHOST_FRIGHTENED_FRAMES = 7 * FPS

# ----------------------------- ENTITIES -----------------------------
class PacMan:
    HIT_RADIUS = 12

    def __init__(self, tx, ty, speed):
        self.tx = tx
        self.ty = ty
        self.px = tx * GameConfig.TILE + GameConfig.TILE/2
        self.py = ty * GameConfig.TILE + GameConfig.TILE/2
        self.speed = speed

        # QUEUED MOVEMENT
        self.current_direction = None   # 0=up,1=down,2=left,3=right
        self.queued_direction = None

        self.vx = 0.0
        self.vy = 0.0

        self.lives = 3
        self.score = 0

        self.last_logged_tile = None

    def set_queued_direction(self, action):
        self.queued_direction = action

    def apply_direction(self, direction):
        self.current_direction = direction
        if direction == 0:
            self.vx, self.vy = 0.0, -self.speed
        elif direction == 1:
            self.vx, self.vy = 0.0, self.speed
        elif direction == 2:
            self.vx, self.vy = -self.speed, 0.0
        elif direction == 3:
            self.vx, self.vy = self.speed, 0.0
        else:
            self.vx, self.vy = 0.0, 0.0

    def stop(self):
        self.current_direction = None
        self.vx = 0.0
        self.vy = 0.0

    def update_tile(self):
        self.tx = int(self.px // GameConfig.TILE)
        self.ty = int(self.py // GameConfig.TILE)

    def snap_to_center(self):
        self.px = self.tx * GameConfig.TILE + GameConfig.TILE/2
        self.py = self.ty * GameConfig.TILE + GameConfig.TILE/2

class Ghost:
    HIT_RADIUS = 12

    # 0=up,1=down,2=left,3=right
    DIRS = (0, 1, 2, 3)
    DIR_VECS = {
        0: (0, -1),
        1: (0,  1),
        2: (-1, 0),
        3: (1,  0),
    }
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self, tx, ty, speed, name, personality="BLINKY", scatter_target=(0, 0)):
        self.start_tile = (tx, ty)
        self.tx = tx
        self.ty = ty
        self.px = tx * GameConfig.TILE + GameConfig.TILE/2
        self.py = ty * GameConfig.TILE + GameConfig.TILE/2
        self.vx = 0.0
        self.vy = 0.0
        self.speed = float(speed)
        self.name = name

        # Stage-6: traditional personalities
        self.personality = personality
        self.scatter_target = scatter_target

        # Power pellet frightened state
        self.scared = False
        self.scared_timer = 0

        # Respawn flag
        self.respawn_request = False

        # Movement
        self.current_direction = None   # 0=up,1=down,2=left,3=right
        self.force_reverse = False      # set when global mode flips (scatter<->chase)

    def _legal_dirs(self, maze):
        legal = []
        for d, (dx, dy) in self.DIR_VECS.items():
            nx, ny = self.tx + dx, self.ty + dy
            if not maze.is_wall_tile(nx, ny):
                legal.append(d)
        return legal

    def _pick_random(self, maze, forbid_reverse=True):
        legal = self._legal_dirs(maze)
        if not legal:
            return None
        if forbid_reverse and self.current_direction in self.OPPOSITE and len(legal) > 1:
            rev = self.OPPOSITE[self.current_direction]
            if rev in legal:
                legal.remove(rev)
        return random.choice(legal) if legal else None

    def choose_random_direction(self, maze):
        """Backward-compatible random chooser."""
        return self._pick_random(maze, forbid_reverse=True)

    def _bfs_dist_map(self, maze, target_xy):
        """BFS distances (in tiles) from every reachable tile to target_xy."""
        from collections import deque

        tx, ty = target_xy
        if maze.is_wall_tile(tx, ty):
            return {}

        q = deque([(tx, ty)])
        dist = {(tx, ty): 0}

        while q:
            x, y = q.popleft()
            d0 = dist[(x, y)]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in dist:
                    continue
                if maze.is_wall_tile(nx, ny):
                    continue
                dist[(nx, ny)] = d0 + 1
                q.append((nx, ny))
        return dist

    def _choose_towards_target(self, maze, target_xy, forbid_reverse=True):
        """Choose a direction that minimizes BFS distance to target."""
        legal = self._legal_dirs(maze)
        if not legal:
            return None

        # Forced reversal after mode switch (classic behavior)
        if self.force_reverse and self.current_direction is not None:
            rev = self.OPPOSITE.get(self.current_direction)
            if rev in legal:
                self.force_reverse = False
                return rev
            self.force_reverse = False  # can't reverse here

        rev = self.OPPOSITE.get(self.current_direction) if self.current_direction is not None else None
        if forbid_reverse and rev in legal and len(legal) > 1:
            legal = [d for d in legal if d != rev]

        dist = self._bfs_dist_map(maze, target_xy)
        if not dist:
            return self._pick_random(maze, forbid_reverse=forbid_reverse)

        best_dirs = []
        best_val = 10**9
        for d in legal:
            dx, dy = self.DIR_VECS[d]
            nx, ny = self.tx + dx, self.ty + dy
            v = dist.get((nx, ny), 10**9)
            if v < best_val:
                best_val = v
                best_dirs = [d]
            elif v == best_val:
                best_dirs.append(d)

        if not best_dirs:
            return self._pick_random(maze, forbid_reverse=forbid_reverse)

        # Tie-break: keep going straight if possible
        if self.current_direction in best_dirs:
            return self.current_direction
        return random.choice(best_dirs)

    def apply_direction(self, direction):
        self.current_direction = direction
        if direction == 0:
            self.vx, self.vy = 0.0, -self.speed
        elif direction == 1:
            self.vx, self.vy = 0.0, self.speed
        elif direction == 2:
            self.vx, self.vy = -self.speed, 0.0
        elif direction == 3:
            self.vx, self.vy = self.speed, 0.0
        else:
            self.vx, self.vy = 0.0, 0.0

    def update_tile(self):
        self.tx = int(self.px // GameConfig.TILE)
        self.ty = int(self.py // GameConfig.TILE)

class Maze:
    def __init__(self):
        self.grid = [[0]*GameConfig.WIDTH for _ in range(GameConfig.HEIGHT)]
        self.food_count = 0
        for y in range(GameConfig.HEIGHT):
            for x in range(GameConfig.WIDTH):
                if x == 0 or y == 0 or x == GameConfig.WIDTH-1 or y == GameConfig.HEIGHT-1:
                    self.grid[y][x] = 1  # walls
                elif (x%2==0 and y%2==0):
                    self.grid[y][x] = 1  # inner walls
                else:
                    self.grid[y][x] = 2  # regular food
                    self.food_count += 1
        corners = [(1,1), (1,GameConfig.WIDTH-2), (GameConfig.HEIGHT-2,1), (GameConfig.HEIGHT-2, GameConfig.WIDTH-2)]
        for y,x in corners:
            self.grid[y][x] = 3

    def has_food(self, tx, ty):
        if tx < 0 or ty < 0 or ty >= GameConfig.HEIGHT or tx >= GameConfig.WIDTH:
            return False
        return self.grid[ty][tx] in (2, 3)

    def get_all_food_positions(self):
        positions = []
        for y in range(GameConfig.HEIGHT):
            for x in range(GameConfig.WIDTH):
                if self.grid[y][x] in (2, 3):
                    positions.append((x, y))
        return positions

    def is_wall_tile(self, tx, ty):
        if tx < 0 or ty < 0 or tx >= GameConfig.WIDTH or ty >= GameConfig.HEIGHT:
            return True
        return self.grid[ty][tx] == 1

# ----------------------------- Q-LEARNING AGENT (skeleton) -----------------------------
class QLearningAgent:
    def __init__(self, params, initial_qtable=None):
        merged = DEFAULT_PARAMS.copy()
        if params:
            merged.update(params)
        self.params = merged
        if initial_qtable is None:
            self.q_table = defaultdict(lambda: [0.0]*4)
        else:
            dq = defaultdict(lambda: [0.0]*4)
            for k, v in initial_qtable.items():
                if isinstance(k, tuple):
                    key = k
                elif isinstance(k, list):
                    key = tuple(k)
                else:
                    key = k
                dq[key] = v[:]
            self.q_table = dq
        self.epsilon = 0.9
        self.episodes_trained = 0

    def ray_until_wall(self, maze, px, py, dx, dy):
        x, y = px + dx, py + dy
        while not maze.is_wall_tile(x, y):
            yield x, y
            x += dx
            y += dy

    def get_state(self, pacman, ghosts, maze, game):
        """Stage-4 (State v4): compact, distance-bucketed, BFS-based state.

        Tuple layout (ordered):
          legal_mask(4),
          exits_count, is_junction, is_dead_end,
          power_mode, power_timer_bin,
          food_dist_bin_dir(4),
          powerpellet_dist_bin_dir(4),
          danger_ghost_dist_bin_dir(4),
          edible_ghost_dist_bin_dir(4),
          any_danger_near, any_edible_near
        """
        px, py = pacman.tx, pacman.ty
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

        # ----------------- LEGAL MASK + TOPOLOGY -----------------
        legal = []
        for dx, dy in directions:
            legal.append(int(not maze.is_wall_tile(px + dx, py + dy)))

        exits_count = int(sum(legal))
        is_junction = int(exits_count >= 3)
        is_dead_end = int(exits_count <= 1)

        # ----------------- POWER MODE -----------------
        max_timer = max((getattr(g, 'scared_timer', 0) for g in ghosts), default=0)
        power_mode = int(max_timer > 0)

        if max_timer <= 0:
            power_timer_bin = 0
        else:
            secs = float(max_timer) / float(GameConfig.FPS)
            if secs <= 1.0:
                power_timer_bin = 1
            elif secs <= 3.0:
                power_timer_bin = 2
            elif secs <= 6.0:
                power_timer_bin = 3
            else:
                power_timer_bin = 4

        # ----------------- TARGET SETS -----------------
        food_targets = set(getattr(game, 'food_positions', set()))
        power_targets = set(getattr(game, 'power_positions', set()))

        danger_targets = set()
        edible_targets = set()
        for g in ghosts:
            if getattr(g, 'scared', False) and getattr(g, 'scared_timer', 0) > 0:
                edible_targets.add((g.tx, g.ty))
            else:
                danger_targets.add((g.tx, g.ty))

        # ----------------- HELPERS -----------------
        def steps_to_bin(steps):
            """Map step-count to a compact bucket. 0 means none/unreachable."""
            if steps is None or steps <= 0:
                return 0
            if steps == 1:
                return 1
            if steps == 2:
                return 2
            if 3 <= steps <= 4:
                return 3
            if 5 <= steps <= 7:
                return 4
            return 5  # 8+

        def bfs_multi_targets(start_xy):
            """Single BFS that finds nearest distance (in tiles) to each target-set.
            Returned distances are from start_xy (0 at start)."""
            from collections import deque

            found = {'food': None, 'power': None, 'danger': None, 'edible': None}

            # Start checks (distance 0)
            if start_xy in food_targets:
                found['food'] = 0
            if start_xy in power_targets:
                found['power'] = 0
            if start_xy in danger_targets:
                found['danger'] = 0
            if start_xy in edible_targets:
                found['edible'] = 0

            if all(v is not None for v in found.values()):
                return found

            q = deque([start_xy])
            dist = {start_xy: 0}

            # Cap search depth for speed (maze is small; this is conservative)
            MAX_DEPTH = 40

            while q:
                x, y = q.popleft()
                d = dist[(x, y)]
                if d >= MAX_DEPTH:
                    continue

                for ddx, ddy in directions:
                    nx, ny = x + ddx, y + ddy
                    if maze.is_wall_tile(nx, ny):
                        continue
                    if (nx, ny) in dist:
                        continue

                    nd = d + 1
                    dist[(nx, ny)] = nd

                    if found['food'] is None and (nx, ny) in food_targets:
                        found['food'] = nd
                    if found['power'] is None and (nx, ny) in power_targets:
                        found['power'] = nd
                    if found['danger'] is None and (nx, ny) in danger_targets:
                        found['danger'] = nd
                    if found['edible'] is None and (nx, ny) in edible_targets:
                        found['edible'] = nd

                    # Early exit when all found
                    if all(v is not None for v in found.values()):
                        return found

                    q.append((nx, ny))

            return found

        # ----------------- PER-DIRECTION DISTANCE BINS -----------------
        food_bins = []
        power_bins = []
        danger_bins = []
        edible_bins = []

        min_danger_steps = None
        min_edible_steps = None

        for i, (dx, dy) in enumerate(directions):
            if legal[i] == 0:
                food_bins.append(0)
                power_bins.append(0)
                danger_bins.append(0)
                edible_bins.append(0)
                continue

            start_xy = (px + dx, py + dy)
            found = bfs_multi_targets(start_xy)

            # Total steps from CURRENT tile if choosing this direction first:
            # 1 (to move into neighbor) + distance from neighbor to target.
            food_steps = None if found['food'] is None else 1 + found['food']
            power_steps = None if found['power'] is None else 1 + found['power']
            danger_steps = None if found['danger'] is None else 1 + found['danger']
            edible_steps = None if found['edible'] is None else 1 + found['edible']

            food_bins.append(steps_to_bin(food_steps))
            power_bins.append(steps_to_bin(power_steps))
            danger_bins.append(steps_to_bin(danger_steps))
            edible_bins.append(steps_to_bin(edible_steps))

            if danger_steps is not None:
                min_danger_steps = danger_steps if min_danger_steps is None else min(min_danger_steps, danger_steps)
            if edible_steps is not None:
                min_edible_steps = edible_steps if min_edible_steps is None else min(min_edible_steps, edible_steps)

        any_danger_near = int(min_danger_steps is not None and min_danger_steps <= 3)
        any_edible_near = int(min_edible_steps is not None and min_edible_steps <= 3)

        #     labels = (
        #     ["legal_up", "legal_down", "legal_left", "legal_right"] +
        #     ["exits_count", "is_junction", "is_dead_end"] +
        #     ["power_mode", "power_timer_bin"] +
        #     ["food_dist_up", "food_dist_down", "food_dist_left", "food_dist_right"] +
        #     ["power_dist_up", "power_dist_down", "power_dist_left", "power_dist_right"] +
        #     ["danger_dist_up", "danger_dist_down", "danger_dist_left", "danger_dist_right"] +
        #     ["edible_dist_up", "edible_dist_down", "edible_dist_left", "edible_dist_right"] +
        #     ["any_danger_near", "any_edible_near"]
        # )

        return (
            *legal,
            exits_count,
            is_junction,
            is_dead_end,
            power_mode,
            power_timer_bin,
            *food_bins,
            *power_bins,
            *danger_bins,
            *edible_bins,
            any_danger_near,
            any_edible_near
        )

    def _reverse_action(self, a):
        # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        return {0:1, 1:0, 2:3, 3:2}.get(a, None)


    def choose_action(self, state):
        legal = state[0:4]

        # -------- Exploration --------
        if random.random() < self.epsilon:
            choices = [a for a in range(4) if legal[a]]
            return random.choice(choices) if choices else random.randint(0,3)

        # -------- Exploitation --------
        q_vals = self.q_table[state]
        legal_actions = [a for a in range(4) if legal[a]]

        if not legal_actions:
            return random.randint(0,3)

        max_q = max(q_vals[a] for a in legal_actions)

        # Tolerance for near-ties
        TIE_EPS = 1e-6
        best_actions = [a for a in legal_actions if abs(q_vals[a] - max_q) < TIE_EPS]

        # -------- Anti-Reversal Tie-Break --------
        if hasattr(self, "last_action") and self.last_action is not None and len(best_actions) > 1:
            rev = self._reverse_action(self.last_action)
            non_reverse = [a for a in best_actions if a != rev]
            if non_reverse:
                return random.choice(non_reverse)

        return random.choice(best_actions)


    def update_q(self, state, action, reward, next_state, terminal=False):
        if action is None or action not in (0,1,2,3):
            return
        lr = self.params.get('learning_rate', DEFAULT_PARAMS['learning_rate'])
        gamma = self.params.get('discount_factor', DEFAULT_PARAMS['discount_factor'])
        future = 0.0 if terminal else gamma * max(self.q_table[next_state])
        self.q_table[state][action] += lr * (reward + future - self.q_table[state][action])
# ----------------------------- GAME -----------------------------
class Game:
        # State v4 feature labels (must match QLearningAgent.get_state() tuple order)
    STATE_FEATURE_LABELS = [
        "legal_up", "legal_down", "legal_left", "legal_right",
        "exits_count", "is_junction", "is_dead_end",
        "power_mode", "power_timer_bin",
        "food_dist_up", "food_dist_down", "food_dist_left", "food_dist_right",
        "power_dist_up", "power_dist_down", "power_dist_left", "power_dist_right",
        "danger_dist_up", "danger_dist_down", "danger_dist_left", "danger_dist_right",
        "edible_dist_up", "edible_dist_down", "edible_dist_left", "edible_dist_right",
        "any_danger_near", "any_edible_near",
    ]
    
    def __init__(self, params, initial_qtable=None):  
        self.episode_count = 0
        self.params = params
        start_tx = 1
        start_ty = 2
        self.pacman = PacMan(start_tx, start_ty, params.get('pacman_speed', DEFAULT_PARAMS['pacman_speed']))
        self.spawn_tx = start_tx
        self.spawn_ty = start_ty
        self.pacman.lives = int(params.get('pacman_lives', DEFAULT_PARAMS['pacman_lives']))
        gspeed = params.get('ghost_base_speed', DEFAULT_PARAMS['ghost_base_speed'])
        center_x = GameConfig.WIDTH // 2
        center_y = GameConfig.HEIGHT // 2
        ghost_positions = [
            (center_x-2, center_y),
            (center_x+2, center_y),
            (center_x, center_y-2)
        ]
        # Stage-6: traditional ghost personalities (Blinky/Pinky/Inky)
        personalities = [
            ("BLINKY", (GameConfig.WIDTH-2, 0)),
            ("PINKY",  (1, 0)),
            ("INKY",   (GameConfig.WIDTH-2, GameConfig.HEIGHT-2)),
        ]
        self.ghosts = []
        for i, (tx, ty) in enumerate(ghost_positions):
            pers, corner = personalities[i % len(personalities)]
            self.ghosts.append(Ghost(tx, ty, gspeed, f"G{i}", personality=pers, scatter_target=corner))
        # ------------------ Stage-6: traditional global ghost mode ------------------
        self.ghost_mode_schedule = list(GameConfig.GHOST_MODE_SCHEDULE)
        self.ghost_mode_index = 0
        self.ghost_mode, self.ghost_mode_timer = self.ghost_mode_schedule[0]

        self.maze = Maze()
        self.agent = QLearningAgent(params, initial_qtable)

        self.last_state = None
        self.last_action = None
        # Tile-history for anti-revisit shaping (discourage immediate backtracking / dithering)
        self.last_tile = None      # tile at last decision
        self.prev_tile = None      # tile before last decision
        self.recent_tiles = deque(maxlen=int(self.params.get('stuck_window', DEFAULT_PARAMS['stuck_window'])))
        # Short history of visited decision-tiles (used for 'stuck/oscillation' penalty)
        self.recent_tiles = deque(maxlen=int(self.params.get('stuck_window', DEFAULT_PARAMS['stuck_window'])))
        self.reward_accum = 0.0
        self.illegal_penalized = False
        self.attempted_illegal = False
        self.illegal_blocked = False
        # Reward events accumulated since the last decision
        self.reward_sources = []  # list[tuple[tag:str, value:float]]

        # Debug printing
        self.debug_print = bool(params.get('debug_print', DEFAULT_PARAMS.get('debug_print', False)))
        self.decision_index = 0

        self.ticks = 0
        self.episode_reward = 0.0
        self.running = True

        # Episode control
        self.episode_steps = 0
        self.max_steps_per_episode = int(params.get('max_steps_per_episode', DEFAULT_PARAMS['max_steps_per_episode']))
        self.episode_done = False
        self.episode_end_reason = None
        self.food_positions = set()
        for y in range(GameConfig.HEIGHT):
            for x in range(GameConfig.WIDTH):
                if self.maze.grid[y][x] in (2, 3):
                    self.food_positions.add((x, y))

        # Track power pellet positions separately (tiles == 3)
        self.power_positions = set()
        for y in range(GameConfig.HEIGHT):
            for x in range(GameConfig.WIDTH):
                if self.maze.grid[y][x] == 3:
                    self.power_positions.add((x, y))

        # Stage-5: store initial counts for metrics
        self.initial_food_count = len(self.food_positions) + len(self.power_positions)

        # Stage-5: store initial counts for metrics
        self.initial_food_count = len(self.food_positions) + len(self.power_positions)

        for g in self.ghosts:
            g.vx = 0.0
            g.vy = 0.0
            g.current_direction = None
        # Stage-5: training/eval + metrics
        self.training_enabled = True
        self.save_on_finish = False
        self.save_name = None
        self.metrics_enabled = bool(self.params.get('metrics_enabled', True))
        self.metrics_path = None
        self._ensure_metrics_file()

    def sees_object_in_direction(self, start_x, start_y, dx, dy, positions):
        x, y = start_x + dx, start_y + dy
        while not self.maze.is_wall_tile(x, y):
            if (x, y) in positions:
                return 1
            x += dx
            y += dy
        return 0
    
    def add_reward(self, value, tag):
        value = float(value)
        self.reward_accum += value
        self.episode_reward += value
        self.reward_sources.append((str(tag), value))
        if getattr(self, 'debug_print', False):
            print(f"  +REWARD {tag}: {value:+.1f} (accum={self.reward_accum:+.1f})")

# ----------------------------- STAGE 4.2: POTENTIAL-BASED SHAPING -----------------------------
# We keep event rewards (FOOD/POWER/GHOST_EAT/DEATH/ILLEGAL) as the primary learning signals.
# Shaping here is potential-based: F = γ Φ(s') - Φ(s), applied once per tile-center decision.
# This preserves optimal policies while accelerating learning and keeps reward reasons distinct.

    _BIN_TO_DIST = {0: 12, 1: 1, 2: 2, 3: 4, 4: 7, 5: 12}

    def _min_dir_dist_from_bins(self, state, bins_start_idx):
        """Return an approximate min distance among the 4 directional bins, considering only legal directions."""
        if state is None:
            return 12
        try:
            legal = state[0:4]
            bins = state[bins_start_idx:bins_start_idx+4]
            best = None
            for i in range(4):
                if int(legal[i]) != 1:
                    continue
                b = int(bins[i])
                if b <= 0:
                    continue
                d = self._BIN_TO_DIST.get(b, 12)
                best = d if best is None else min(best, d)
            return best if best is not None else 12
        except Exception:
            return 12

    def _phi_components(self, state):
        """Compute scalar potentials used for shaping, derived from State v4."""
        if state is None:
            return {'food': 0.0, 'danger': 0.0, 'edible': 0.0, 'power': 0.0}

        try:
            power_mode = int(state[7])
            any_danger_near = int(state[25])
            any_edible_near = int(state[26])
        except Exception:
            power_mode = 0
            any_danger_near = 0
            any_edible_near = 0

        food_d = self._min_dir_dist_from_bins(state, 9)      # 9-12
        power_d = self._min_dir_dist_from_bins(state, 13)    # 13-16
        danger_d = self._min_dir_dist_from_bins(state, 17)   # 17-20
        edible_d = self._min_dir_dist_from_bins(state, 21)   # 21-24

        # Potentials: larger is better.
        # Food/power/edible are "closer is better" => negative distance.
        # Danger is "farther is better" => positive distance.
        food_phi = -float(food_d)
        power_phi = -float(power_d)
        danger_phi = float(danger_d)
        edible_phi = -float(edible_d)

        return {
            'food': food_phi,
            'power': power_phi,
            'danger': danger_phi,
            'edible': edible_phi,
            'power_mode': power_mode,
            'any_danger_near': any_danger_near,
            'any_edible_near': any_edible_near
        }

    def apply_potential_shaping(self, prev_state, next_state, backtrack=False):
        """Apply potential-based shaping once per decision with distinct tags."""
        gamma_shape = 1.0  # shaping gamma: 1.0 prevents loop-profit when potential doesn't change
        w_food = float(self.params.get('pot_food_w', DEFAULT_PARAMS.get('pot_food_w', 2.0)))
        w_danger = float(self.params.get('pot_danger_w', DEFAULT_PARAMS.get('pot_danger_w', 2.5)))
        w_edible = float(self.params.get('pot_edible_w', DEFAULT_PARAMS.get('pot_edible_w', 2.0)))
        w_power = float(self.params.get('pot_power_w', DEFAULT_PARAMS.get('pot_power_w', 1.5)))

        # Small per-decision time cost helps avoid dithering without per-frame penalties.
        step_cost = float(self.params.get('decision_step_cost', DEFAULT_PARAMS.get('decision_step_cost', -0.1)))
        if step_cost != 0.0:
            self.add_reward(step_cost, "STEP")

        p = self._phi_components(prev_state)
        n = self._phi_components(next_state)
        food_d = self._min_dir_dist_from_bins(prev_state, 9)  # food bins start at index 9
        power_d = self._min_dir_dist_from_bins(prev_state, 13)
        
        # Gating to keep rewards distinct and non-overlapping:
        # - When in danger (no power), prioritize danger avoidance (and optionally power pellet seeking).
        # - When powered and edible exists, prioritize edible chase (not food).
        
        prev_power = int(p.get('power_mode', 0))
        prev_danger = int(p.get('any_danger_near', 0))
        prev_edible = int(p.get('any_edible_near', 0))

        # --- FIX A: junction-gated food shaping ---
        try:
            prev_is_junction = int(prev_state[5])
        except Exception:
            prev_is_junction = 0

        # Food shaping ONLY when:
        # - not in danger
        # - not powered
        # - AT A JUNCTION (real decision point)
        enable_food = (not prev_danger) and (not prev_power) and prev_is_junction


        # DANGER shaping only when not powered
        enable_danger = (not prev_power) and bool(prev_danger)

        # EDIBLE shaping only when powered and an edible ghost is around
        enable_edible = (prev_power and prev_edible)

        # POWER PELLET shaping only when not powered and danger is near
        enable_power = (not prev_power) and prev_danger

        def _clip(x, lo=-1.0, hi=1.0):
            return lo if x < lo else hi if x > hi else x

        if enable_food and food_d < 12:
            # Potential-based shaping: positive when we reduce (binned) distance to food.
            r = w_food * (gamma_shape * float(n['food']) - float(p['food']))
            r = _clip(r)
            # Optional: never pay *positive* food shaping for immediately backtracking (prevents ABAB reward hacking)
            if backtrack and int(self.params.get('block_potential_on_backtrack', 1)) == 1 and r > 0:
                r = 0.0
            if abs(r) >= 0.05:
                self.add_reward(r, 'POT_FOOD')

        if enable_danger:
            # Danger potential is defined so moving away from danger is positive.
            r = w_danger * (gamma_shape * float(n['danger']) - float(p['danger']))
            r = _clip(r)
            if abs(r) >= 0.05:
                self.add_reward(r, "POT_DANGER")

        # Distances used for gating (based on PREV state directional bins)
        edible_d = self._min_dir_dist_from_bins(prev_state, 21)  # edible bins start at index 21
        pow_d = power_d  # alias for readability

        if enable_edible and edible_d < 12:
            r = w_edible * (gamma_shape * float(n['edible']) - float(p['edible']))
            r = _clip(r)
            if backtrack and int(self.params.get('block_potential_on_backtrack', 1)) == 1 and r > 0:
                r = 0.0
            if abs(r) >= 0.05:
                self.add_reward(r, 'POT_EDIBLE')

        if enable_power and pow_d < 12:
            r = w_power * (gamma_shape * float(n['power']) - float(p['power']))
            r = _clip(r)
            if backtrack and int(self.params.get('block_potential_on_backtrack', 1)) == 1 and r > 0:
                r = 0.0
            if abs(r) >= 0.05:
                self.add_reward(r, 'POT_POWER')

        # Hard penalty for ending up adjacent to a dangerous ghost (not powered).
        adj_pen = float(self.params.get('adjacent_ghost_penalty', DEFAULT_PARAMS.get('adjacent_ghost_penalty', 0.0)))
        if adj_pen != 0.0:
            try:
                next_power = int(next_state[7]) if len(next_state) > 7 else 0
                next_danger = int(next_state[25]) if len(next_state) > 25 else 0
                danger_d_next = self._min_dir_dist_from_bins(next_state, 17)
            except Exception:
                next_power, next_danger, danger_d_next = 0, 0, 99
            if next_power == 0 and next_danger == 1 and danger_d_next <= 1:
                self.add_reward(adj_pen, 'ADJ_DANGER')
        return

    def print_decision_summary(self, chosen_action, decision_reward, prev_state, next_state):
        """Debug helper: prints one decision transition in a human-readable form."""

        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        action_str = action_names.get(chosen_action, str(chosen_action))

        decision_idx = getattr(self, 'decision_index', None)
        print("\n================= DECISION SUMMARY =================")
        if decision_idx is not None:
            print(f"Decision: {decision_idx}")
        print(f"Action: {action_str} ({chosen_action})")
        # Reward breakdown
        sources = getattr(self, 'reward_sources', None)
        if sources:
            print("Reward events since last decision:")
            for name, val in sources:
                sign = '+' if val >= 0 else ''
                print(f"  - {name}: {sign}{val}")
        print(f"Total decision reward: {decision_reward:+.3f}" if isinstance(decision_reward, (int, float)) else f"Total decision reward: {decision_reward}")
        print("Prev State (labeled):")
        print(self._format_state(prev_state))
        print("Next State (labeled):")
        print(self._format_state(next_state))
        print("====================================================")

    def _format_state(self, state):
        """Format a state (dict/tuple/list) as labeled multi-line text."""
        if state is None:
            return "  (none)"

        # dict-like
        if isinstance(state, dict):
            items = list(state.items())

        # tuple/list state vector
        elif isinstance(state, (list, tuple)):
            labels = (
                getattr(self, "state_feature_labels", None)
                or getattr(self, "STATE_FEATURE_LABELS", None)
                or getattr(Game, "STATE_FEATURE_LABELS", None)
            )

            if labels and len(labels) == len(state):
                items = list(zip(labels, state))
            else:
                items = [(f"f{i}", v) for i, v in enumerate(state)]

        else:
            return f"  {state}"

        # pretty align
        max_k = max((len(str(k)) for k, _ in items), default=0)
        out = [f"  {str(k).ljust(max_k)} : {v}" for k, v in items]
        return "\n".join(out) if out else "  (empty)"

    def _state_get(self, state, key, default=0):
        """Safely fetch a feature by key from a state that may be dict or vector."""
        if state is None:
            return default
        if isinstance(state, dict):
            return state.get(key, default)
        # tuple/list/np-array case: use labels if available
        if not hasattr(self, '_state_label_to_idx') or self._state_label_to_idx is None:
            labels = getattr(self, 'state_feature_labels', None) or getattr(self, 'STATE_FEATURE_LABELS', None)
            if isinstance(labels, (list, tuple)) and labels:
                try:
                    self._state_label_to_idx = {str(k): i for i, k in enumerate(labels)}
                except Exception:
                    self._state_label_to_idx = {}
            else:
                self._state_label_to_idx = {}
        idx = self._state_label_to_idx.get(str(key), None)
        if idx is None:
            return default
        try:
            return state[idx]
        except Exception:
            return default
    def is_at_tile_center(self, px, py, tx, ty):
        center_x = tx * GameConfig.TILE + GameConfig.TILE/2
        center_y = ty * GameConfig.TILE + GameConfig.TILE/2
        dist = math.hypot(px - center_x, py - center_y)
        tol = max(0.5, self.pacman.speed * 0.3)
        return dist <= tol

    def is_move_legal(self, tx, ty, action):
        # action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        if action not in (0, 1, 2, 3):
            return False
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[action]
        nx, ny = tx + dx, ty + dy
        return not self.maze.is_wall_tile(nx, ny)

    # ------------------ Stage-6: Ghost AI (Traditional Scatter/Chase/Frightened) ------------------
    def update_ghost_global_mode(self):
        """Advance the global SCATTER/CHASE timer (frightened handled per-ghost)."""
        if not hasattr(self, 'ghost_mode_timer'):
            return
        self.ghost_mode_timer -= 1
        if self.ghost_mode_timer > 0:
            return

        # Advance schedule
        self.ghost_mode_index = min(self.ghost_mode_index + 1, len(self.ghost_mode_schedule) - 1)
        new_mode, new_timer = self.ghost_mode_schedule[self.ghost_mode_index]
        if new_mode != self.ghost_mode:
            # Classic behavior: reverse direction on mode flip
            for g in self.ghosts:
                g.force_reverse = True
        self.ghost_mode = new_mode
        self.ghost_mode_timer = new_timer

    @staticmethod
    def _clamp_tile(x, y):
        x = max(0, min(GameConfig.WIDTH - 1, int(x)))
        y = max(0, min(GameConfig.HEIGHT - 1, int(y)))
        return x, y

    def _pacman_ahead_tile(self, steps):
        """Tile 'steps' ahead of Pac-Man current direction (or current tile if unknown)."""
        d = getattr(self.pacman, 'current_direction', None)
        if d is None:
            return (self.pacman.tx, self.pacman.ty)
        dx, dy = Ghost.DIR_VECS[d]
        return self._clamp_tile(self.pacman.tx + dx * steps, self.pacman.ty + dy * steps)

    def get_ghost_target_tile(self, ghost):
        """Return the current target tile for a ghost under traditional logic."""
        # Frightened mode: no target (random)
        if ghost.scared and ghost.scared_timer > 0:
            return None

        mode = getattr(self, 'ghost_mode', 'SCATTER')
        if mode == 'SCATTER':
            return ghost.scatter_target

        # CHASE targets differ by personality
        pac_tile = (self.pacman.tx, self.pacman.ty)
        if ghost.personality == 'BLINKY':
            return pac_tile

        if ghost.personality == 'PINKY':
            # Classic intent: 4 tiles ahead of Pac-Man
            return self._pacman_ahead_tile(4)

        if ghost.personality == 'INKY':
            # Classic-ish: target = Blinky + 2*(tile_ahead_2 - Blinky)
            ahead2 = self._pacman_ahead_tile(2)
            blinky = None
            for g in self.ghosts:
                if getattr(g, 'personality', '') == 'BLINKY':
                    blinky = g
                    break
            if blinky is None:
                return pac_tile
            vx = ahead2[0] - blinky.tx
            vy = ahead2[1] - blinky.ty
            return self._clamp_tile(blinky.tx + 2 * vx, blinky.ty + 2 * vy)

        # Default fallback
        return pac_tile

    def choose_ghost_direction(self, ghost):
        """Choose a direction for 'ghost' when it is at tile-center."""
        # Frightened (power pellet): random at intersections
        if ghost.scared and ghost.scared_timer > 0:
            return ghost._pick_random(self.maze, forbid_reverse=False)

        target = self.get_ghost_target_tile(ghost)
        if target is None:
            return ghost._pick_random(self.maze, forbid_reverse=False)

        # Scatter/Chase: move towards target, avoid reversal unless forced
        return ghost._choose_towards_target(self.maze, target, forbid_reverse=True)

    def step_physics(self):
        moved_this_frame = False

        # Stage-6: advance global ghost mode timer
        self.update_ghost_global_mode()

        # Determine tile-center behavior for queued-direction logic
        at_center = self.is_at_tile_center(self.pacman.px, self.pacman.py, self.pacman.tx, self.pacman.ty)
        if at_center:
            # Snap to exact center before making direction changes
            self.pacman.snap_to_center()

            # Prefer queued direction if legal
            qd = self.pacman.queued_direction
            if qd is not None and self.is_move_legal(self.pacman.tx, self.pacman.ty, qd):
                self.pacman.apply_direction(qd)
                self.illegal_blocked = False
            else:
                # If current direction is illegal (blocked), stop
                cd = self.pacman.current_direction
                if cd is None or not self.is_move_legal(self.pacman.tx, self.pacman.ty, cd):
                    self.pacman.stop()
                    # If the agent attempted an illegal queued direction and it results in no legal movement,
                    # mark this decision as 'blocked by illegal'.
                    if qd is not None and not self.is_move_legal(self.pacman.tx, self.pacman.ty, qd):
                        self.illegal_blocked = True

        # Move Pac-Man along current velocity, but block axis if next tile is wall
        # Because current_direction is axis-aligned, we only need per-axis checks
        # X movement
        if abs(self.pacman.vx) > 0.0:
            next_px = self.pacman.px + self.pacman.vx
            next_tx = int(next_px // GameConfig.TILE)
            # check wall in the direction of movement using pacman's ty
            if self.maze.is_wall_tile(next_tx, self.pacman.ty):
                # Stop movement, but KEEP last_action so Q-update happens
                self.pacman.vx = 0.0

            else:
                prev_px = self.pacman.px
                self.pacman.px = next_px
                if abs(self.pacman.px - prev_px) > 0.01:
                    moved_this_frame = True

        # Y movement
        if abs(self.pacman.vy) > 0.0:
            next_py = self.pacman.py + self.pacman.vy
            next_ty = int(next_py // GameConfig.TILE)
            if self.maze.is_wall_tile(self.pacman.tx, next_ty):
                self.pacman.vy = 0.0
            else:
                prev_py = self.pacman.py
                self.pacman.py = next_py
                if abs(self.pacman.py - prev_py) > 0.01:
                    moved_this_frame = True

        # Update integer tile coords
        self.pacman.update_tile()

        for g in self.ghosts:
            if g.respawn_request:
                sx, sy = g.start_tile
                g.px = sx * GameConfig.TILE + GameConfig.TILE/2
                g.py = sy * GameConfig.TILE + GameConfig.TILE/2
                g.tx, g.ty = g.start_tile
                g.vx = 0.0
                g.vy = 0.0
                g.current_direction = None
                g.respawn_request = False
                continue

            at_center = self.is_at_tile_center(g.px, g.py, g.tx, g.ty)
            if at_center:
                g.px = g.tx * GameConfig.TILE + GameConfig.TILE/2
                g.py = g.ty * GameConfig.TILE + GameConfig.TILE/2

                new_dir = self.choose_ghost_direction(g)
                if new_dir is not None:
                    g.apply_direction(new_dir)

            # Move ghost
            g.px += g.vx
            g.py += g.vy
            g.update_tile()
        
        return moved_this_frame
    

    def check_eating(self):
        tx, ty = self.pacman.tx, self.pacman.ty
        cell = self.maze.grid[ty][tx]
        tile = (tx, ty)

        # ---------- NORMAL FOOD ----------
        if cell == 2:
            self.maze.grid[ty][tx] = 0
            self.maze.food_count -= 1

            # authoritative food tracking
            if tile in self.food_positions:
                self.food_positions.remove(tile)

            self.add_reward(10, "FOOD")

        # ---------- POWER PELLET ----------
        elif cell == 3:
            self.maze.grid[ty][tx] = 0
            self.maze.food_count -= 1

            # authoritative food tracking
            if tile in self.food_positions:
                self.food_positions.remove(tile)
            if hasattr(self, 'power_positions') and tile in self.power_positions:
                self.power_positions.remove(tile)

            self.add_reward(50, "POWER")

            for g in self.ghosts:
                g.scared = True
                g.scared_timer = GameConfig.GHOST_FRIGHTENED_FRAMES

        # Episode clear condition
        if self.maze.food_count <= 0:
            self.episode_done = True
            self.episode_end_reason = 'CLEAR'
            
    def check_ghost_collisions(self):
        for g in self.ghosts:
            dist = math.hypot(self.pacman.px - g.px, self.pacman.py - g.py)
            hit_radius = (
                getattr(self.pacman, 'HIT_RADIUS', GameConfig.TILE // 2)
                + getattr(g, 'HIT_RADIUS', GameConfig.TILE // 2)
            )

            if dist >= hit_radius:
                continue

            # ---------- EAT SCARED GHOST ----------
            if g.scared and g.scared_timer > 0:
                self.add_reward(200, "GHOST_EAT")

                g.respawn_request = True
                g.scared = False
                g.scared_timer = 0

                return  # ✅ prevent multi-hit same frame

            # ---------- DEATH ----------
            self.add_reward(-500, "DEATH")

            self.pacman.lives -= 1
            print(f'Pac-Man lost a life. Remaining lives: {self.pacman.lives}')

            if self.pacman.lives <= 0:
                self.episode_done = True
                self.episode_end_reason = 'DEATH'
                return

            # Lives remain: respawn but keep maze
            self.respawn_after_death()
            return  # ✅ critical: stop further collision checks

    def decay_scared_timers(self):
        for g in self.ghosts:
            if hasattr(g, 'scared_timer') and g.scared_timer > 0:
                g.scared_timer -= 1
                if g.scared_timer <= 0:
                    g.scared = False

    def reset_episode(self):
        # Backward-compat: keep the public name, but this now starts a fresh episode.
        self.start_new_episode()

    # RL DECISION PIPELINE (IMPORTANT):
    # - Rewards are accumulated AFTER an action is chosen
    # - State-based shaping uses last_state / last_action only
    # - Illegal moves are detected at decision-time, penalized post-physics
    # - Q-update happens only once per tile-center decision
    # DO NOT add per-frame rewards or physics-based penalties

    
    def autosave_checkpoint(self):
        """Persist Q-table/meta to the loaded/new simulation file, if enabled."""
        try:
            if getattr(self, 'save_on_finish', False):
                save_name = getattr(self, 'save_name', None)
                if not save_name:
                    return
                try:
                    with open(save_name, 'rb') as f:
                        existing = pickle.load(f)
                except Exception:
                    existing = {}
                existing['q_table'] = dict(self.agent.q_table)
                meta = existing.get('meta', {})
                meta.setdefault('state_version', 4)
                meta.setdefault('reward_version', 2)
                # Increment based on what's already stored in the file
                meta['episodes_trained'] = int(meta.get('episodes_trained', 0)) + 1
                meta['last_trained'] = time.time()
                # Persist the current epsilon so that it can be restored when loading.
                try:
                    meta['epsilon'] = float(getattr(self.agent, 'epsilon', 0.0))
                except Exception:
                    pass
                existing['meta'] = meta
                existing['parameters'] = existing.get('parameters', self.params)
                with open(save_name, 'wb') as f:
                    pickle.dump(existing, f)
                print('Autosaved updated Q-table to %s (episodes=%s)' % (save_name, meta.get('episodes_trained')))
        except Exception as e:
            print('Autosave failed:', e)

    def _ensure_metrics_file(self):
        """Create a CSV metrics file (header) if enabled and we know the save_name."""
        if not getattr(self, 'metrics_enabled', True):
            return
        save_name = getattr(self, 'save_name', None)
        if not save_name:
            # metrics path will be resolved once save_name is set (new/load simulation)
            return
        base, _ = os.path.splitext(save_name)
        self.metrics_path = base + "_metrics.csv"
        if not os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "episode",
                        "mode",
                        "end_reason",
                        "total_reward",
                        "steps",
                        "food_remaining",
                        "food_eaten",
                        "power_remaining",
                        "epsilon",
                        "timestamp"
                    ])
            except Exception as e:
                print("WARN: could not create metrics file:", e)
                self.metrics_enabled = False

    def _append_metrics_row(self, mode, end_reason):
        """Append one metrics row for the current episode."""
        if not getattr(self, 'metrics_enabled', True):
            return
        if not getattr(self, 'metrics_path', None):
            self._ensure_metrics_file()
        if not getattr(self, 'metrics_path', None):
            return

        try:
            # initial food includes power pellets (2 or 3)
            initial_food = getattr(self, 'initial_food_count', None)
            if initial_food is None:
                initial_food = len(self.food_positions) + len(self.power_positions)
            food_remaining = len(getattr(self, 'food_positions', set()))
            power_remaining = len(getattr(self, 'power_positions', set()))
            food_eaten = max(0, int(initial_food) - int(food_remaining) - int(power_remaining))
            eps = getattr(self.agent, 'epsilon', None)
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    int(getattr(self.agent, "episodes_trained", 0)),
                    mode,
                    end_reason,
                    float(getattr(self, "episode_reward", 0.0)),
                    int(getattr(self, "episode_steps", 0)),
                    int(food_remaining),
                    int(food_eaten),
                    int(power_remaining),
                    float(eps) if eps is not None else "",
                    ts
                ])
        except Exception as e:
            print("WARN: could not append metrics row:", e)
            self.metrics_enabled = False

    def run_evaluation(self, num_episodes=5):
        """Run evaluation episodes (epsilon=0, no learning updates)."""
        if num_episodes <= 0:
            return

        print(f"\n=== Evaluation: {num_episodes} episode(s), epsilon=0 (no learning) ===")
        # stash training state
        prev_eps = self.agent.epsilon
        prev_training = getattr(self, "training_enabled", True)

        self.training_enabled = False
        self.agent.epsilon = 0.0

        eval_rewards = []
        eval_clears = 0

        for _ in range(int(num_episodes)):
            self.start_new_episode()
            # evaluation episode loop (reuse same stepping logic but no rendering)
            while self.running and not self.episode_done:
                self.step_physics()
                self.episode_steps += 1
                if self.episode_steps >= int(self.params.get('max_steps_per_episode', DEFAULT_PARAMS['max_steps_per_episode'])):
                    self.episode_done = True
                    self.episode_end_reason = "TIMEOUT"

            reason = self.episode_end_reason or "END"
            if reason == "CLEAR":
                eval_clears += 1
            eval_rewards.append(float(self.episode_reward))
            self._append_metrics_row(mode="EVAL", end_reason=reason)

        avg_r = sum(eval_rewards) / max(1, len(eval_rewards))
        print(f"Eval avg reward: {avg_r:.2f} | clear rate: {eval_clears}/{len(eval_rewards)}\n")

        # restore
        self.agent.epsilon = prev_eps
        self.training_enabled = prev_training

    def respawn_after_death(self):
        """Respawn Pac-Man + reset ghosts to their start tiles, but keep the maze/food."""
        # Reset Pac-Man position
        self.pacman.tx = self.spawn_tx
        self.pacman.ty = self.spawn_ty
        self.pacman.px = self.pacman.tx * GameConfig.TILE + GameConfig.TILE / 2
        self.pacman.py = self.pacman.ty * GameConfig.TILE + GameConfig.TILE / 2
        self.pacman.stop()
        self.pacman.queued_direction = None

        # Reset ghosts to start tiles and clear scared
        for g in self.ghosts:
            gx, gy = g.start_tile
            g.tx, g.ty = gx, gy
            g.px = gx * GameConfig.TILE + GameConfig.TILE / 2
            g.py = gy * GameConfig.TILE + GameConfig.TILE / 2
            g.vx = 0.0
            g.vy = 0.0
            g.current_direction = None
            g.queued_direction = None
            g.scared = False
            g.scared_timer = 0
            g.respawn_request = False

    def start_new_episode(self):
        """Full environment reset for a new episode."""
        # Reset maze
        self.maze = Maze()

        # Rebuild food positions (authoritative)
        self.food_positions = set()
        for y in range(GameConfig.HEIGHT):
            for x in range(GameConfig.WIDTH):
                if self.maze.grid[y][x] in (2, 3):
                    self.food_positions.add((x, y))

        # Rebuild power pellet positions (tiles == 3)
        self.power_positions = set()
        for y in range(GameConfig.HEIGHT):
            for x in range(GameConfig.WIDTH):
                if self.maze.grid[y][x] == 3:
                    self.power_positions.add((x, y))

        # Stage-5: store initial counts for metrics
        self.initial_food_count = len(self.food_positions) + len(self.power_positions)

        # Reset lives for this episode
        self.pacman.lives = int(self.params.get('pacman_lives', DEFAULT_PARAMS['pacman_lives']))

        # Reset positions
        self.respawn_after_death()

        # Reset per-episode RL tracking
        self.last_state = None
        self.last_action = None
        self.last_tile = None
        self.prev_tile = None
        # Tile-history for anti-revisit shaping (discourage immediate backtracking / dithering)
        self.last_tile = None      # tile at last decision
        self.prev_tile = None      # tile before last decision
        self.reward_accum = 0.0
        self.episode_reward = 0.0
        self.reward_sources = []
        self.illegal_penalized = False
        self.attempted_illegal = False
        self.illegal_blocked = False

        self.episode_steps = 0
        self.episode_done = False
        self.episode_end_reason = None

    def finish_episode(self, reason):
        """Apply terminal update, decay epsilon, increment counters, and autosave."""
        # Terminal Q-update so last action gets credit/blame for terminal outcome
        if self.last_state is not None and self.last_action is not None:
            terminal_state = self.agent.get_state(self.pacman, self.ghosts, self.maze, self)
            if self.reward_accum != 0.0:
                if getattr(self, 'training_enabled', True):
                    self.agent.update_q(self.last_state, self.last_action, self.reward_accum, terminal_state, terminal=True)
                self.reward_accum = 0.0

        print(f'Episode finished ({reason}). Total reward: {self.episode_reward:.2f}')
        self.episode_count += 1

        decay = self.params.get('epsilon_decay', DEFAULT_PARAMS['epsilon_decay'])
        eps_min = float(self.params.get('epsilon_min', DEFAULT_PARAMS.get('epsilon_min', 0.05)))
        self.agent.epsilon = max(eps_min, self.agent.epsilon * decay)
        self.agent.episodes_trained += 1
        print(f'Epsilon after decay: {self.agent.epsilon:.6f} (Episode {self.agent.episodes_trained})')

        # Stage-5: metrics
        self._append_metrics_row(mode='TRAIN', end_reason=reason)

        self.autosave_checkpoint()

        # Stage-5: optional evaluation
        eval_every = int(self.params.get('eval_every', 0) or 0)
        if eval_every > 0 and (self.agent.episodes_trained % eval_every == 0):
            self.run_evaluation(int(self.params.get('eval_episodes', DEFAULT_PARAMS.get('eval_episodes', 5))))

    def run_episode(self):
        pygame.init()
        render = bool(self.params.get('render', True))
        screen = None
        font = None
        if render:
            screen = pygame.display.set_mode((GameConfig.WIDTH*GameConfig.TILE, GameConfig.HEIGHT*GameConfig.TILE))
            pygame.display.set_caption('Pac-Man RL - Stage 5 (Metrics + Eval + Headless)')
            font = pygame.font.SysFont(None, 24)
        clock = pygame.time.Clock()

        # Start first episode
        self.start_new_episode()

        while self.running:
            # Run one episode until terminal or quit
            while self.running and not self.episode_done:
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_s:
                                self.running = False
                else:
                    # Headless / no-window mode:
                    # Only pump events if the video subsystem is initialized.
                    # (Otherwise pygame raises: "video system not initialized")
                    try:
                        if pygame.get_init() and pygame.display.get_init():
                            pygame.event.pump()
                    except pygame.error:
                        # If the display was torn down while a training thread is still running,
                        # stop the episode cleanly instead of crashing the thread.
                        self.running = False
                        return

                # ================= DECISION TICK =================
                at_center = self.is_at_tile_center(
                    self.pacman.px, self.pacman.py,
                    self.pacman.tx, self.pacman.ty
                )

                if at_center:
                    self.pacman.snap_to_center()
                    current_state = self.agent.get_state(
                        self.pacman, self.ghosts, self.maze, self
                    )

                    # ---- LOG PREVIOUS DECISION RESULT ----
                    if self.last_state is not None and self.last_action is not None:
                        cur_tile = (self.pacman.tx, self.pacman.ty)
                        backtrack = (self.prev_tile is not None and cur_tile == self.prev_tile)
                        # Stage-4.2 shaping: potential-based, distinct tags per cause
                        self.apply_potential_shaping(self.last_state, current_state, backtrack=backtrack)

                        # Anti-revisit penalty: discourage stepping back onto the previous tile (ABAB dithering)
                        revisit_pen = float(self.params.get('revisit_penalty', DEFAULT_PARAMS.get('revisit_penalty', -0.25)))
                        if revisit_pen != 0.0 and self.prev_tile is not None:
                            if cur_tile == self.prev_tile:
                                exits = int(self._state_get(current_state, 'exits_count', 0))
                                is_dead_end = int(self._state_get(current_state, 'is_dead_end', 0))
                                danger = int(self._state_get(current_state, 'any_danger_near', 0))
                                revisit_exempt_danger = int(self.params.get('revisit_exempt_danger', DEFAULT_PARAMS.get('revisit_exempt_danger', 1)))
                                stuck_exempt_danger = int(self.params.get('stuck_exempt_danger', DEFAULT_PARAMS.get('stuck_exempt_danger', 1)))
                                # Do not penalize if backtracking is required (dead-end) or used to escape danger.
                                if (not is_dead_end) and (exits > 1) and (not (revisit_exempt_danger and danger)):
                                    self.add_reward(revisit_pen, 'REVISIT')

                        # --- Stuck/oscillation penalty (windowed) ---
                        # If we're cycling among a tiny set of tiles for several decisions, penalize to break local optima loops.
                        self.recent_tiles.append(cur_tile)
                        if len(self.recent_tiles) == self.recent_tiles.maxlen:
                            unique_tiles = len(set(self.recent_tiles))
                            thresh = int(self.params.get('stuck_unique_thresh', DEFAULT_PARAMS['stuck_unique_thresh']))
                            if unique_tiles <= thresh:
                                power_mode = int(self._state_get(current_state, 'power_mode', 0))
                                edible_near = int(self._state_get(current_state, 'any_edible_near', 0))
                                if (not (stuck_exempt_danger and danger)) and power_mode == 0 and edible_near == 0:
                                    stuck_pen = float(self.params.get('stuck_penalty', DEFAULT_PARAMS['stuck_penalty']))
                                    self.add_reward(stuck_pen, 'STUCK')
                        if getattr(self, 'debug_print', False):
                            self.print_decision_summary(self.last_action, self.reward_accum, self.last_state, current_state)

                        # Q-update for previous decision
                        if getattr(self, 'training_enabled', True):
                            self.agent.update_q(self.last_state, self.last_action, self.reward_accum, current_state)
                        self.reward_accum = 0.0
                        self.reward_sources = []  # reset per-decision reward event list
                        self.illegal_penalized = False
                        self.attempted_illegal = False
                        self.illegal_blocked = False
                    else:
                        # first decision: just clear any accumulated noise (optional but clean)
                        self.reward_accum = 0.0
                        self.reward_sources = []

                    # Choose and enqueue action at tile center
                    action = self.agent.choose_action(current_state)
                    if getattr(self, 'debug_print', False):
                        action_names = {0:'UP',1:'DOWN',2:'LEFT',3:'RIGHT'}
                        print(f"Chosen action for next decision: {action_names.get(action, action)} ({action})")
                    legal = self.is_move_legal(self.pacman.tx, self.pacman.ty, action)

                    # Track whether the agent *attempted* an illegal action.
                    # We only penalize if the attempt actually causes movement to be blocked (see step_physics).
                    self.attempted_illegal = (not legal)

                    self.pacman.set_queued_direction(action)
                    # Update tile history BEFORE storing last_state/last_action
                    self.prev_tile = self.last_tile
                    self.last_tile = (self.pacman.tx, self.pacman.ty)
                    self.last_state = current_state
                    self.last_action = action
                    self.decision_index += 1

                # ===================== PHYSICS =====================
                moved_this_frame = self.step_physics()

                # Illegal move penalty (causal)
                # Penalize only if the chosen illegal action actually caused Pac-Man to be blocked at the tile center.
                if self.attempted_illegal and self.illegal_blocked and not self.illegal_penalized:
                    self.add_reward(-5.0, "ILLEGAL")
                    self.illegal_penalized = True
                # ----------------------------------------------------------------

                self.check_eating()
                self.check_ghost_collisions()
                self.decay_scared_timers()

                # Max-step termination
                self.episode_steps += 1
                if self.episode_steps >= self.max_steps_per_episode and not self.episode_done:
                    self.episode_done = True
                    self.episode_end_reason = 'TIMEOUT'

                if render:
                    # draw
                    screen.fill((0,0,0))
                    for y in range(GameConfig.HEIGHT):
                        for x in range(GameConfig.WIDTH):
                            cell = self.maze.grid[y][x]
                            rect = pygame.Rect(x*GameConfig.TILE, y*GameConfig.TILE, GameConfig.TILE, GameConfig.TILE)
                            if cell == 1:
                                pygame.draw.rect(screen, (0,0,255), rect)
                            elif cell == 2:
                                pygame.draw.circle(screen, (255,255,0), rect.center, GameConfig.TILE//6)
                            elif cell == 3:
                                pygame.draw.circle(screen, (255,0,0), rect.center, GameConfig.TILE//3)

                    pygame.draw.circle(screen, (255,255,0), (int(self.pacman.px), int(self.pacman.py)), GameConfig.TILE//2)
                    for g in self.ghosts:
                        color = (255,0,255) if not g.scared else (0,255,255)
                        pygame.draw.circle(screen, color, (int(g.px), int(g.py)), GameConfig.TILE//2)

                    text = font.render(
                        f'Reward: {self.episode_reward:.1f}  Lives: {self.pacman.lives}  Food left: {self.maze.food_count}  Step: {self.episode_steps}/{self.max_steps_per_episode}',
                        True,
                        (255,255,255)
                    )
                    screen.blit(text, (10,10))

                    pygame.display.flip()
                    clock.tick(GameConfig.FPS)
                else:
                    # Headless: no rendering / no FPS cap
                    pass

                self.ticks += 1

            # Episode ended (but window still open)
            if not self.running:
                break

            reason = self.episode_end_reason or 'UNKNOWN'
            self.finish_episode(reason)
            self.start_new_episode()

        pygame.quit()

# ----------------------------- GUI -----------------------------

    def run_episode_safe(self):
        """Wrapper for running an episode inside a thread.
        Any uncaught exception will be printed and appended to pacman_thread_crash.log.
        """
        try:
            self.run_episode()
        except Exception:
            import time, traceback
            tb = traceback.format_exc()
            print("\n" + "="*80)
            print("UNHANDLED EXCEPTION inside episode thread")
            print(tb)
            print("="*80 + "\n")
            try:
                with open("pacman_thread_crash.log", "a", encoding="utf-8") as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    f.write(tb)
            except Exception:
                pass

class NewSimulationWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title('New Simulation')
        self.geometry('520x720')
        self.params = {}
        self.create_widgets()

    def create_widgets(self):
        # ---------- Header ----------
        tk.Label(self, text='Simulation Name:').pack(anchor='w', padx=10, pady=(10, 0))
        self.name_entry = tk.Entry(self)
        self.name_entry.pack(fill='x', padx=10, pady=(0, 10))
        # ----- Template presets (optional) -----
        self.template_var = tk.StringVar(value="(Custom)")
        self.template_desc_var = tk.StringVar(value="Select a template to auto-fill recommended values.")

        self.templates = {
            "(Custom)": {},
            "Fast Learn (Balanced)": {
                "learning_rate": 0.20,
                "discount_factor": 0.95,
                "epsilon_decay": 0.996,
                "epsilon_min": DEFAULT_PARAMS.get("epsilon_min", 0.2),
                "max_steps_per_episode": 8000,
                "render": 0,
                "metrics_enabled": 1,
                "decision_step_cost": -0.08,
                "pot_food_w": 1.0,
                "pot_danger_w": 1.8,
                "pot_power_w": 0.8,
                "pot_edible_w": 2.0,
                "revisit_penalty": -0.25,
                "revisit_exempt_danger": 1,
                "stuck_penalty": -0.60,
                "stuck_exempt_danger": 1,
                "adjacent_ghost_penalty": -4.0,
                "block_potential_on_backtrack": 1,
            },
            "Safety First (Anti-Death)": {
                "learning_rate": 0.18,
                "discount_factor": 0.96,
                "epsilon_decay": 0.996,
                "epsilon_min": DEFAULT_PARAMS.get("epsilon_min", 0.2),
                "max_steps_per_episode": 8000,
                "render": 0,
                "metrics_enabled": 1,
                "decision_step_cost": -0.10,
                "pot_food_w": 0.8,
                "pot_danger_w": 2.6,
                "pot_power_w": 1.1,
                "pot_edible_w": 1.8,
                "revisit_penalty": -0.35,
                "revisit_exempt_danger": 1,
                "stuck_penalty": -0.80,
                "stuck_exempt_danger": 1,
                "adjacent_ghost_penalty": -7.0,
                "block_potential_on_backtrack": 1,
            },
            "Aggressive Clear (Score Chase)": {
                "learning_rate": 0.22,
                "discount_factor": 0.93,
                "epsilon_decay": 0.9965,
                "epsilon_min": DEFAULT_PARAMS.get("epsilon_min", 0.2),
                "max_steps_per_episode": 8000,
                "render": 0,
                "metrics_enabled": 1,
                "decision_step_cost": -0.06,
                "pot_food_w": 1.5,
                "pot_danger_w": 1.4,
                "pot_power_w": 0.6,
                "pot_edible_w": 2.4,
                "revisit_penalty": -0.20,
                "revisit_exempt_danger": 1,
                "stuck_penalty": -0.55,
                "stuck_exempt_danger": 1,
                "adjacent_ghost_penalty": -3.0,
                "block_potential_on_backtrack": 1,
            },
        }

        def _apply_template(name: str):
            tpl = self.templates.get(name, {})
            if not tpl:
                self.template_desc_var.set("Custom values (no template applied).")
                return
            if "Safety First" in name:
                self.template_desc_var.set("Safety-first: stronger danger + adjacency penalties to reduce deaths.")
            elif "Aggressive" in name:
                self.template_desc_var.set("Aggressive clear: prioritize food/score and chase edible ghosts more.")
            else:
                self.template_desc_var.set("Balanced fast learning: good default for quick progress without heavy risk.")
            for k, v in tpl.items():
                item = self.param_widgets.get(k)
                if item is None:
                    continue

                # NewSimulationWindow stores widgets as either:
                #   - a Tk Entry-like widget, OR
                #   - a tuple: (tk.Variable, min, max, cast)
                if isinstance(item, (tuple, list)) and len(item) > 0 and hasattr(item[0], "set"):
                    try:
                        item[0].set(str(v))
                        e = getattr(self, "param_entries", {}).get(k)
                        if e is not None:
                            e.delete(0, tk.END)
                            e.insert(0, str(v))
                    except Exception:
                        pass
                    continue

                if hasattr(item, "delete") and hasattr(item, "insert"):
                    try:
                        item.delete(0, 'end')
                        item.insert(0, str(v))
                    except Exception:
                        pass
                elif hasattr(item, "set"):
                    try:
                        item.set(v)
                    except Exception:
                        pass

        tpl_frame = tk.LabelFrame(self, text="Template preset", padx=10, pady=8)
        tpl_frame.pack(fill='x', padx=10, pady=(0, 10))

        top_row = tk.Frame(tpl_frame)
        top_row.pack(fill='x')
        tk.Label(top_row, text="Preset:").pack(side='left')
        tk.OptionMenu(top_row, self.template_var, *self.templates.keys()).pack(side='left', padx=8)
        tk.Button(top_row, text="Apply", command=lambda: _apply_template(self.template_var.get())).pack(side='left')

        tk.Label(tpl_frame, textvariable=self.template_desc_var, fg="gray", wraplength=380, justify='left').pack(anchor='w', pady=(6, 0))


        # ---------- Scrollable params area (many parameters) ----------
        container = tk.Frame(self)
        container.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
        self.params_frame = tk.Frame(canvas)

        self.params_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.params_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mouse wheel scroll (Windows/macOS/Linux)
        def _on_mousewheel(event):
            try:
                # Windows: event.delta is +/-120
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except Exception:
                pass

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ---------- Param specs ----------
        # Each entry: (key, title, desc, min, max, recommended, type)
        param_specs = [
            ("learning_rate", "Learning rate (alpha)",
             "How fast Q-values update. Higher learns faster but can destabilize.",
             0.01, 0.5, 0.20, float),

            ("discount_factor", "Discount factor (gamma)",
             "How much future reward matters. Higher = more long-term planning.",
             0.80, 0.99, 0.95, float),

            ("epsilon_decay", "Exploration decay (epsilon_decay)",
             "Multiplies epsilon once per episode. Lower = faster shift to greedy policy.",
             0.990, 0.9999, 0.996, float),
            ("epsilon_min", "Exploration floor (epsilon_min)",
             "Minimum exploration probability. Lower = more stable / easier to CLEAR late-game.",
             0.0, 0.2, DEFAULT_PARAMS.get("epsilon_min", 0.2), float),

            ("max_steps_per_episode", "Max steps per episode",
             "Episode timeout (in frames). Lower speeds training cycles.",
             300, 20000, 2000, int),

            ("render", "Render (1=yes, 0=headless)",
             "If 0: headless training (no window) for much faster learning. Gameplay logic unchanged.",
             0, 1, 1, int),

            ("eval_every", "Evaluation frequency (episodes)",
             "Run evaluation (epsilon=0, no learning) every N training episodes. 0 disables.",
             0, 500, DEFAULT_PARAMS.get("eval_every", 0), int),

            ("eval_episodes", "Evaluation episodes per run",
             "How many evaluation episodes to run each time eval triggers.",
             1, 50, DEFAULT_PARAMS.get("eval_episodes", 5), int),

            ("metrics_enabled", "Write metrics CSV (1/0)",
             "If 1: write episode metrics to <save_name>_metrics.csv for plotting/tuning.",
             0, 1, 1 if DEFAULT_PARAMS.get("metrics_enabled", True) else 0, int),

            ("decision_step_cost", "Decision step cost",
             "Small penalty once per decision to reduce dithering/loops.",
             -1.0, 0.0, -0.08, float),

            ("revisit_penalty", "Revisit previous tile penalty",
             "Penalty when Pac-Man steps back onto the tile it came from (discourages left/right or up/down dithering).",
             -2.0, 0.0, DEFAULT_PARAMS.get("revisit_penalty", -0.25), float),

            ("revisit_exempt_danger", "Exempt revisit penalty when danger near (1/0)",
             "If 1: do not apply revisit penalty when any_danger_near=1 (so Pac-Man can reverse to escape ghosts).",
             0, 1, int(DEFAULT_PARAMS.get("revisit_exempt_danger", 1)), int),
            ("block_potential_on_backtrack", "Block potential shaping on reversal",
             "If 1: when Pac-Man immediately reverses direction, suppress potential-based shaping on that decision (prevents +1/-1 ping-pong).",
             0, 1, int(DEFAULT_PARAMS.get("block_potential_on_backtrack", 1)), int),

            ("stuck_penalty", "Stuck penalty (per-step)",
             "Extra penalty when Pac-Man oscillates/repeats within a short window (discourages local loops).",
             -5.0, 0.0, float(DEFAULT_PARAMS.get("stuck_penalty", -0.25)), float),
            ("stuck_window", "Stuck window (steps)",
             "Window length (in decisions) used to detect oscillation/stuck behavior.",
             4, 80, int(DEFAULT_PARAMS.get("stuck_window", 10)), int),
            ("stuck_unique_thresh", "Stuck unique positions threshold",
             "If unique tiles visited within the window is <= this, apply stuck penalty.",
             1, 40, int(DEFAULT_PARAMS.get("stuck_unique_thresh", 3)), int),
        ("stuck_exempt_danger", "Exempt stuck penalty when danger near (1/0)",
 "If 1: do not apply the stuck/oscillation penalty when any_danger_near=1 (allows escape backtracking).",
 0, 1, int(DEFAULT_PARAMS.get("stuck_exempt_danger", 1)), int),

            ("pot_food_w", "Potential weight: FOOD",
             "Shaping strength for moving closer to food when safe (not powered, no danger).",
             0.0, 5.0, 1.0, float),

            ("pot_danger_w", "Potential weight: DANGER",
             "Shaping strength for moving away from nearby dangerous ghosts (not powered).",
             0.0, 5.0, 1.5, float),

("adjacent_ghost_penalty", "Adjacent dangerous ghost penalty",
 "Extra penalty if Pac-Man ENDS a decision adjacent (distance=1) to a dangerous ghost while NOT powered.",
 -50.0, 0.0, DEFAULT_PARAMS.get("adjacent_ghost_penalty", -4.0), float),

            ("pot_power_w", "Potential weight: POWER",
             "Shaping strength for moving toward power pellets when danger is near (not powered).",
             0.0, 5.0, 0.8, float),

            ("pot_edible_w", "Potential weight: EDIBLE",
             "Shaping strength for chasing edible ghosts (powered + edible near).",
             0.0, 5.0, 2.0, float),

            ("pacman_speed", "Pac-Man speed (px/frame)",
             "Movement speed in pixels per frame (visual + physics).",
             0.5, 6.0, DEFAULT_PARAMS["pacman_speed"], float),

            ("ghost_base_speed", "Ghost base speed (px/frame)",
             "Ghost movement speed in pixels per frame (visual + physics).",
             0.5, 6.0, DEFAULT_PARAMS["ghost_base_speed"], float),

            ("pacman_lives", "Pac-Man lives per episode",
             "Lives per episode before terminal DEATH. Lower = faster episodes.",
             1, 5, DEFAULT_PARAMS["pacman_lives"], int),

            ("debug_print", "Debug print (True/False)",
             "Print detailed decision summaries + reward breakdown to console.",
             0, 1, 1 if DEFAULT_PARAMS.get("debug_print", True) else 0, int),
                                                        ]

        # ---------- Build widgets ----------
        self.param_widgets = {}
        self.param_entries = {}
        for key, title, desc, min_val, max_val, rec, param_type in param_specs:
            block = tk.LabelFrame(self.params_frame, text=title, padx=8, pady=6)
            block.pack(fill='x', expand=True, pady=6)

            hint = f"{desc}\nRange: {min_val} to {max_val}   |   Recommended: {rec}"
            tk.Label(block, text=hint, justify='left', wraplength=360).pack(anchor='w')

            var = tk.StringVar(value=str(rec))
            entry = tk.Entry(block, textvariable=var)
            entry.pack(fill='x', pady=(6, 0))

            self.param_widgets[key] = (var, min_val, max_val, param_type)

        # ---------- Buttons ----------
        btns = tk.Frame(self)
        btns.pack(fill='x', padx=10, pady=(0, 10))

        tk.Button(btns, text='Save Simulation', command=self.save_simulation).pack(side='left', expand=True, fill='x', padx=(0, 5))
        tk.Button(btns, text='Run Simulation', command=self.run_simulation).pack(side='left', expand=True, fill='x', padx=(5, 0))

    def save_simulation(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror('Error', 'Please enter a simulation name.')
            return False
        validated_params = {}
        for key, (var, min_val, max_val, param_type) in self.param_widgets.items():
            val = validate_param(var.get(), min_val, max_val, param_type)
            if val is None:
                return False
            # Convert 0/1 debug_print to bool
            if key == 'debug_print':
                validated_params[key] = bool(int(val))
            else:
                validated_params[key] = val
        self.params = validated_params
        qtable = defaultdict(lambda: [0.0]*4)
        save_simulation_file(name, qtable, self.params, 0)
        self.filename = f'{name}{QTABLE_FILE_EXT}'
        return True

    def run_simulation(self):
        ok = self.save_simulation()
        if not ok:
            return
        game = Game(self.params)
        # Ensure training persists back into the newly created .qpac file
        game.save_on_finish = True
        game.save_name = getattr(self, 'filename', f"{self.name_entry.get().strip()}{QTABLE_FILE_EXT}")
        t = threading.Thread(target=game.run_episode_safe, daemon=True)
        t.start()

class ParamOverrideDialog(tk.Toplevel):
    """Small helper dialog to override a few parameters of a loaded .qpac.
    This avoids editing pickle files manually."""

    def __init__(self, master, params, source_filename):
        super().__init__(master)
        self.title("Run with overrides")
        self.resizable(False, False)
        self.params = dict(params) if isinstance(params, dict) else {}
        self.source_filename = source_filename
        self.result = None  # (updated_params, save_as_path_or_None)

        frm = tk.Frame(self, padx=12, pady=12)
        frm.pack(fill=tk.BOTH, expand=True)

        tk.Label(frm, text="Loaded simulation:").grid(row=0, column=0, sticky="w")
        tk.Label(frm, text=str(source_filename), fg="gray").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0,10))

        def add_row(r, title, desc):
            tk.Label(frm, text=title, font=("Arial", 10, "bold")).grid(row=r, column=0, sticky="w", pady=(6,0))
            tk.Label(frm, text=desc, fg="gray", wraplength=360, justify="left").grid(row=r+1, column=0, columnspan=2, sticky="w")

        # Max steps
        add_row(2, "Max steps per episode", "Episode timeout (frames). Increase (e.g., 8000) if you want board-clears to be possible.")
        self.max_steps_var = tk.StringVar(value=str(self.params.get("max_steps_per_episode", DEFAULT_PARAMS.get("max_steps_per_episode", 2000))))
        tk.Entry(frm, textvariable=self.max_steps_var, width=12).grid(row=4, column=0, sticky="w")

        # Render
        add_row(5, "Render", "1 = show window (slow). 0 = headless (fast).")
        self.render_var = tk.IntVar(value=int(self.params.get("render", 1)))
        # Debug print (0/1)
        self.debug_var = tk.IntVar(value=1 if self.params.get("debug_print", False) else 0)
        rfrm = tk.Frame(frm)
        rfrm.grid(row=7, column=0, sticky="w")
        tk.Radiobutton(rfrm, text="Render (1)", variable=self.render_var, value=1).pack(side=tk.LEFT)
        tk.Radiobutton(rfrm, text="Headless (0)", variable=self.render_var, value=0).pack(side=tk.LEFT, padx=(10,0))

        # Debug print
        add_row(8, "Debug print", "1 = print decision summaries + reward breakdown (slow). 0 = off (fast).")
        dfrm = tk.Frame(frm)
        dfrm.grid(row=10, column=0, sticky="w")
        tk.Radiobutton(dfrm, text="On (1)", variable=self.debug_var, value=1).pack(side=tk.LEFT)
        tk.Radiobutton(dfrm, text="Off (0)", variable=self.debug_var, value=0).pack(side=tk.LEFT, padx=(10,0))

        # Buttons
        btns = tk.Frame(frm, pady=10)
        btns.grid(row=11, column=0, columnspan=2, sticky="ew")
        tk.Button(btns, text="Run", width=12, command=self._on_run).pack(side=tk.LEFT)
        tk.Button(btns, text="Save copy as... and Run", width=22, command=self._on_save_and_run).pack(side=tk.LEFT, padx=8)
        tk.Button(btns, text="Cancel", width=12, command=self._on_cancel).pack(side=tk.RIGHT)

        # Make modal-ish
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _validated_params(self):
        p = dict(self.params)
        # max_steps
        try:
            ms = int(float(self.max_steps_var.get().strip()))
            ms = max(300, min(20000, ms))
        except Exception:
            ms = int(p.get("max_steps_per_episode", DEFAULT_PARAMS.get("max_steps_per_episode", 2000)))
        p["max_steps_per_episode"] = ms
        # render
        p["render"] = int(self.render_var.get())
        p["debug_print"] = bool(int(self.debug_var.get()))
        return p

    def _on_run(self):
        self.result = (self._validated_params(), None)
        self.destroy()

    def _on_save_and_run(self):
        updated = self._validated_params()
        # Suggest a new name
        base = os.path.splitext(os.path.basename(str(self.source_filename)))[0]
        suggested = f"{base}_copy.qpac"
        save_path = filedialog.asksaveasfilename(
            title="Save simulation copy as",
            defaultextension=".qpac",
            initialfile=suggested,
            filetypes=[("Pac-Man QPAC", "*.qpac"), ("All files", "*.*")]
        )
        if not save_path:
            return
        self.result = (updated, save_path)
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

class OldSimulationWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title('Old Simulations')
        self.geometry('420x320')
        self.create_widgets()

    def create_widgets(self):
        sims = list_simulations()
        if not sims:
            tk.Label(self, text='No simulations found.').pack()
            return
        self.listbox = tk.Listbox(self)
        for f in sims:
            self.listbox.insert(tk.END, f)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        tk.Button(self, text='Load & Run Simulation', command=self.load_simulation).pack(pady=5)

    def load_simulation(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showerror('Error', 'Select a simulation.')
            return
        filename = self.listbox.get(sel[0])
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Could not load file:{e}')
            return

        # ---------- PRINT DEBUG INFO (NEW LINE ADDED HERE) ----------
        try:
            print_all_details(data)   # print loaded .qpac contents for debugging
        except Exception as e:
            print('print_all_details failed:', e)
        # ------------------------------------------------------------

        messagebox.showinfo('Loaded', f'Loaded simulation {filename}')
        params = data.get('parameters', {})
        # Clean up legacy (pre-Stage-6) proximity-based ghost tuning keys (no longer used)
        for _k in ('low_proximity','high_proximity_start','high_proximity_growth','ghost_speed_scale'):
            params.pop(_k, None)

        initial_qtable = data.get('q_table', {})
        for k in ['pacman_speed','ghost_base_speed','learning_rate','epsilon_decay','discount_factor']:
            if k in params:
                try:
                    params[k] = float(params[k])
                except Exception:
                    pass
        # Stage-4 State v4 is not compatible with older saved Q-tables.
        meta = data.get('meta', {}) if isinstance(data.get('meta', {}), dict) else {}
        try:
            state_version = int(meta.get('state_version', 3))
        except Exception:
            state_version = 3
        if state_version != 4:
            messagebox.showerror(
                'Incompatible Simulation',
                f"This simulation was saved with state_version={state_version}.\n"
                "Stage-4 State v4 requires state_version=4.\n\n"
                "Please create a NEW simulation for Stage-4 training."
            )
            return

        # Optional: override a few parameters (max steps / render) without editing the .qpac manually.
        dlg = ParamOverrideDialog(self, params, filename)
        self.wait_window(dlg)
        if dlg.result is None:
            return
        params, save_copy_path = dlg.result

        # If user asked to save a copy, write a cloned .qpac with updated parameters.
        if save_copy_path:
            try:
                cloned = dict(data)
                cloned['parameters'] = params
                meta2 = dict(meta) if isinstance(meta, dict) else {}
                meta2['cloned_from'] = filename
                meta2['timestamp'] = time.time()
                cloned['meta'] = meta2
                with open(save_copy_path, 'wb') as f:
                    pickle.dump(cloned, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved cloned simulation: {save_copy_path}")
                filename = save_copy_path
                data = cloned
                initial_qtable = cloned.get('q_table', initial_qtable)
            except Exception as e:
                messagebox.showerror('Error', f'Could not save cloned simulation: {e}')
                return

        game = Game(params, initial_qtable)
        prev_episodes = data.get('meta', {}).get('episodes_trained', 0)

        # Restore training context so exploration continues smoothly after loading
        try:
            prev_episodes = int(prev_episodes)
        except Exception:
            prev_episodes = 0
        decay = params.get('epsilon_decay', DEFAULT_PARAMS['epsilon_decay'])
        try:
            decay = float(decay)
        except Exception:
            decay = DEFAULT_PARAMS['epsilon_decay']
        # Restore the number of episodes trained so far.  Do not reset
        # the agent's epsilon when loading a saved simulation.  In the
        # original implementation epsilon was recomputed as
        # `max(0.1, 0.9 * decay**prev_episodes)`, which rapidly drives
        # epsilon towards zero after many episodes.  By leaving epsilon
        # untouched, the agent continues with its existing exploration
        # rate (the default is 0.9 on a fresh agent).  If you wish to
        # preserve an epsilon value between sessions, you can save it
        # into the `.qpac` file's meta section and read it here.
        game.agent.episodes_trained = prev_episodes
        # game.agent.epsilon remains unchanged
        game.save_on_finish = True
        game.save_name = filename
        game.loaded_episodes = prev_episodes
        t = threading.Thread(target=game.run_episode_safe, daemon=True)
        t.start()

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Pac-Man RL Menu')
        self.geometry('300x200')
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self, text='Run Simulation', width=20, command=self.run_simulation).pack(pady=20)
        tk.Button(self, text='Quit', width=20, command=self.quit).pack(pady=10)

    def run_simulation(self):
        SimMenu(self)

class SimMenu(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title('Simulation Menu')
        self.geometry('300x200')
        tk.Button(self, text='New Simulation', width=20, command=self.new_sim).pack(pady=10)
        tk.Button(self, text='Old Simulation', width=20, command=self.old_sim).pack(pady=10)
        tk.Button(self, text='Back', width=20, command=self.destroy).pack(pady=10)

    def new_sim(self):
        NewSimulationWindow(self.master)
        self.destroy()

    def old_sim(self):
        OldSimulationWindow(self.master)
        self.destroy()

if __name__ == '__main__':
    app = MainMenu()
    app.mainloop()