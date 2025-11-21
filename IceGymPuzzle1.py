import gymnasium as gym
from gymnasium import spaces
import numpy as np

class puzzles():
    # map layout, W=wall, I=fresh ice, R=rock, S=start, F=finish
    puzzles = {
        1: [
            list("WWWWW"),
            list("WIFRW"),
            list("WIIIW"),
            list("WRSIW"),
            list("WWWWW")
        ],

        2: [
            list("WWWWWWWWW"),
            list("WIIIFIIIW"),
            list("WIRIIIRIW"),
            list("WIIISIIIW"),
            list("WWWWWWWWW")
        ],

        3: [
            list("WWWWWWWWWWWWW"),
            list("WIIRIIFIIIIIW"),
            list("WIIIIIIRIIRIW"),
            list("WIRIIRIIIIIIW"),
            list("WIIIIISIIRIIW"),
            list("WWWWWWWWWWWWW")
        ]
    }

class IceEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    #all actions that the agent can take
    actions = {
        0: np.array([0, -1]), #move left
        1: np.array([1, 0]), #move down
        2: np.array([0, 1]), #move right
        3: np.array([-1, 0]) #move up
    }
    action_names = {
        0: "moved left",
        1: "moved down",
        2: "moved right",
        3: "moved up"
    }

    def __init__(self, puzzle_number=1, render_mode=None):
        super().__init__()

        if puzzle_number not in puzzles.puzzles:
            raise ValueError(f"Puzzle {puzzle_number} not defined!")

        self.map = np.array(puzzles.puzzles[puzzle_number])

        #map Height Width
        self.height, self.width = self.map.shape

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=max(self.height, self.width), shape=(2,), dtype=np.int32),
            "tiles": spaces.MultiDiscrete([3] * (self.height * self.width))
        })
        self.action_space = spaces.Discrete(4)

        #set starting position to S
        start_position = np.argwhere(self.map == "S")
        if len(start_position) == 0:
            raise ValueError("No start position")
        self.start_position = tuple(start_position[0])
        self.reset()

    #reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tiles = self._encode_tiles()
        self.agent_position = np.array(self.start_position)
        return self._get_obs(), {}

    #creating tiles
    def _encode_tiles(self):
        encoded = []
        for r in range(self.height):
            for s in range(self.width):
                if self.map[r, s] in ["I"]:
                    encoded.append(1)
                elif self.map[r, s] in ["W", "R"]:
                    encoded.append(0)
                else:
                    encoded.append(2)
        return np.array(encoded, dtype=np.int32)

    #what is a tile
    def _tile(self, r, s):
        return self.tiles[r * self.width + s]

    def _set_tiles(self, r, s, x):
        self.tiles[r * self.width + s] = x

    #actions
    def step(self, action):
        move = self.actions[action]
        new_position = self.agent_position + move

        #boundary check
        if not (0 <= new_position[0] < self.height and 0 <= new_position[1] < self.width):
            new_position = self.agent_position
        r, s = new_position

        #running into a wall or rock does not move the agent
        if self.map[r, s] in ["W", "R"]:
            new_position = self.agent_position

        #cannot step on finish line unless all tiles are cracked
        if self.map[r, s] in ["F"] and 1 in self.tiles:
            new_position = self.agent_position

        #update position
        previous_position = self.agent_position.copy()
        self.agent_position = new_position
        moved = not np.array_equal(previous_position, self.agent_position)

        r, s = self.agent_position
        tile_state = self._tile(r, s)
        reward = 0
        done = False

        #end condition
        if self.map[r, s] in ["F"]:
            if 1 not in self.tiles:
                reward+=1
                done = True
            return self._get_obs(), reward, done, False, {}

        if moved:
            #updating tiles
            if tile_state == 1:
                self._set_tiles(r, s, 2)
            #stepping on already cracked tile
            elif tile_state == 2:
                return self._get_obs(), -1, True, False, {}

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return {"agent": self.agent_position.copy(), "tiles": self.tiles.copy()}

    def render(self):
        grid = self.map.astype(str).copy()
        for r in range(self.height):
            for s in range(self.width):
                if grid[r, s] in ["I", "S"] and self._tile(r, s) == 2:
                    grid[r, s] = "C"
        ar, ac = self.agent_position
        grid[ar, ac] = "A"
        print("\n".join(" ".join(row) for row in grid))


class PuzzleTester:
    def __init__(self, puzzle_numbers, episodes_per_puzzle=100):
        self.puzzle_numbers = puzzle_numbers
        self.episodes_per_puzzle = episodes_per_puzzle
        self.successes_per_puzzle = {p: 0 for p in puzzle_numbers}

    def run(self, view=False):
        for puzzle in self.puzzle_numbers:
            env = IceEnvironment(puzzle_number=puzzle)
            successes = 0

            for ep in range(self.episodes_per_puzzle):
                obs, _ = env.reset()
                done = False
                steps = []

                while not done:
                    action = env.action_space.sample()
                    obs, reward, done, _, _ = env.step(action)
                    if view:
                        steps.append((action, env._get_obs()))

                if reward > 0:  # success
                    successes += 1
                    if view:
                        print(f"\n--- Puzzle {puzzle}, Episode {ep + 1} SUCCESS ---")
                        for a, obs_state in steps:
                            print(f"Action: {IceEnvironment.action_names[a]}")
                            env.agent_position = obs_state["agent"]
                            env.tiles = obs_state["tiles"]
                            env.render()

            self.successes_per_puzzle[puzzle] = successes
            # print(f"Puzzle {puzzle}: {successes}/{self.episodes_per_puzzle} successes")
            # print(f"Success rate: {successes / self.episodes_per_puzzle * 100:.2f}%")

    def probability_all_three(self):
        prob = 1
        for puzzle in self.puzzle_numbers:
            prob *= self.successes_per_puzzle[puzzle] / self.episodes_per_puzzle
        return prob

if __name__ == "__main__":
    puzzle_numbers = [1]
    episodes_per_puzzle = 100

    tester = PuzzleTester(puzzle_numbers, episodes_per_puzzle)
    tester.run(view=True)

    # prob = tester.probability_all_three()
    # print(f"Probability of solving all three in sequence: {prob:.12f}")