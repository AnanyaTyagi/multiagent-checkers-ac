import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils import wrappers

BOARD_SIZE = 6
N_ACTIONS = BOARD_SIZE * BOARD_SIZE * 4

EMPTY = 0
P0_MAN = 1
P0_KING = 2
P1_MAN = -1
P1_KING = -2

# 0 = up-left, 1 = up-right, 2 = down-left, 3 = down-right
DIRS = {
    0: (-1, -1),
    1: (-1, 1),
    2: (1, -1),
    3: (1, 1),
}


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = raw_env(render_mode=internal_render_mode)

    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)

    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "mycheckers_v0"}

    def __init__(self, render_mode=None, max_moves=200):
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }

        self.render_mode = render_mode
        self.max_moves = max_moves

        self.board = None
        self.move_count = 0
        self.must_continue_jump = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict(
            {
                "observation": Box(
                    low=-2, high=2, shape=(BOARD_SIZE * BOARD_SIZE,), dtype=np.int8
                ),
                "action_mask": Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(N_ACTIONS)

    def reset(self, seed=None, options=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.board = self._init_board()
        self.move_count = 0
        self.must_continue_jump = None

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        obs_board = self._board_from_perspective(agent).reshape(-1).astype(np.int8)
        mask = self._legal_action_mask(agent)
        return {
            "observation": obs_board,
            "action_mask": mask,
        }

    def close(self):
        pass

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render() without specifying a render_mode."
            )
            return

        symbol_map = {
            EMPTY: ".",
            P0_MAN: "r",
            P0_KING: "R",
            P1_MAN: "b",
            P1_KING: "B",
        }

        print("\n  0 1 2 3 4 5")
        for r in range(BOARD_SIZE):
            row_str = " ".join(symbol_map[int(x)] for x in self.board[r])
            print(f"{r} {row_str}")
        print(f"Current agent: {self.agent_selection}")
        print(f"Move count: {self.move_count}")
        if self.must_continue_jump is not None:
            print(f"Must continue jump from: {self.must_continue_jump}")
        print()

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()

        legal_actions = self._legal_actions(agent)

        if action is None or action not in legal_actions:
            other = self._other_agent(agent)
            self.rewards[agent] = -1.0
            self.rewards[other] = 1.0
            self.terminations = {a: True for a in self.agents}
            self.infos[agent]["illegal_move"] = True
            self._accumulate_rewards()

            if self.render_mode == "human":
                self.render()
            return

        same_turn = self._apply_action(agent, action)
        self._check_game_over()

        self.move_count += 1
        if self.move_count >= self.max_moves and not any(self.terminations.values()):
            self.truncations = {a: True for a in self.agents}

        if not any(self.terminations.values()) and not any(self.truncations.values()):
            if same_turn:
                self.agent_selection = agent
            else:
                self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _init_board(self):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        # player_1 on top two rows
        for r in range(2):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r, c] = P1_MAN

        # player_0 on bottom two rows
        for r in range(BOARD_SIZE - 2, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r, c] = P0_MAN

        return board

    def _other_agent(self, agent):
        return self.possible_agents[1 - self.agent_name_mapping[agent]]

    def _agent_sign(self, agent):
        return 1 if agent == "player_0" else -1

    def _board_from_perspective(self, agent):
        # Makes current player's own pieces positive
        sign = self._agent_sign(agent)
        return (self.board * sign).astype(np.int8)

    def _in_bounds(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _encode_action(self, r, c, direction_idx):
        return (r * BOARD_SIZE + c) * 4 + direction_idx

    def _decode_action(self, action):
        from_sq, direction_idx = divmod(int(action), 4)
        r, c = divmod(from_sq, BOARD_SIZE)
        dr, dc = DIRS[direction_idx]
        return r, c, dr, dc

    def _belongs_to_agent(self, piece, agent):
        if agent == "player_0":
            return piece > 0
        return piece < 0

    def _is_opponent_piece(self, piece, agent):
        if piece == EMPTY:
            return False
        return not self._belongs_to_agent(piece, agent)

    def _is_king(self, piece):
        return abs(int(piece)) == 2

    def _forward_dirs(self, agent):
        if agent == "player_0":
            return [0, 1]
        return [2, 3]

    def _allowed_dirs_for_piece(self, piece, agent):
        if self._is_king(piece):
            return [0, 1, 2, 3]
        return self._forward_dirs(agent)

    def _all_pieces_for_agent(self, agent):
        coords = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._belongs_to_agent(self.board[r, c], agent):
                    coords.append((r, c))
        return coords

    def _get_piece_moves(self, r, c, agent):
        piece = self.board[r, c]
        if piece == EMPTY or not self._belongs_to_agent(piece, agent):
            return []

        actions = []
        for direction_idx in self._allowed_dirs_for_piece(piece, agent):
            dr, dc = DIRS[direction_idx]
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc) and self.board[nr, nc] == EMPTY:
                actions.append(self._encode_action(r, c, direction_idx))
        return actions

    def _get_piece_jumps(self, r, c, agent):
        piece = self.board[r, c]
        if piece == EMPTY or not self._belongs_to_agent(piece, agent):
            return []

        actions = []
        for direction_idx in self._allowed_dirs_for_piece(piece, agent):
            dr, dc = DIRS[direction_idx]
            mid_r, mid_c = r + dr, c + dc
            land_r, land_c = r + 2 * dr, c + 2 * dc

            if not self._in_bounds(mid_r, mid_c) or not self._in_bounds(land_r, land_c):
                continue

            if (
                self._is_opponent_piece(self.board[mid_r, mid_c], agent)
                and self.board[land_r, land_c] == EMPTY
            ):
                actions.append(self._encode_action(r, c, direction_idx))
        return actions

    def _legal_actions(self, agent):
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            return []

        pieces = self._all_pieces_for_agent(agent)

        # multi-jump: only same piece can continue
        if self.must_continue_jump is not None:
            r, c = self.must_continue_jump
            if not self._in_bounds(r, c):
                return []
            if not self._belongs_to_agent(self.board[r, c], agent):
                return []
            return self._get_piece_jumps(r, c, agent)

        all_jumps = []
        for r, c in pieces:
            all_jumps.extend(self._get_piece_jumps(r, c, agent))

        if all_jumps:
            return all_jumps

        all_moves = []
        for r, c in pieces:
            all_moves.extend(self._get_piece_moves(r, c, agent))
        return all_moves

    def _legal_action_mask(self, agent):
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        for a in self._legal_actions(agent):
            mask[a] = 1
        return mask

    def _promote_if_needed(self, r, c):
        piece = self.board[r, c]
        if piece == P0_MAN and r == 0:
            self.board[r, c] = P0_KING
            return True
        if piece == P1_MAN and r == BOARD_SIZE - 1:
            self.board[r, c] = P1_KING
            return True
        return False

    def _apply_action(self, agent, action):
        """
        Returns True if same player must move again because of multi-jump.
        """
        r, c, dr, dc = self._decode_action(action)
        piece = self.board[r, c]

        normal_r, normal_c = r + dr, c + dc
        jump_r, jump_c = r + 2 * dr, c + 2 * dc

        other = self._other_agent(agent)

        is_jump = (
            self._in_bounds(jump_r, jump_c)
            and self._in_bounds(normal_r, normal_c)
            and self._is_opponent_piece(self.board[normal_r, normal_c], agent)
            and self.board[jump_r, jump_c] == EMPTY
        )

        if is_jump:
            captured_r, captured_c = normal_r, normal_c
            self.board[r, c] = EMPTY
            self.board[captured_r, captured_c] = EMPTY
            self.board[jump_r, jump_c] = piece

            self.rewards[agent] += 0.2
            self.rewards[other] -= 0.2

            promoted = self._promote_if_needed(jump_r, jump_c)
            if promoted:
                self.rewards[agent] += 0.3

            more_jumps = self._get_piece_jumps(jump_r, jump_c, agent)
            if more_jumps:
                self.must_continue_jump = (jump_r, jump_c)
                return True

            self.must_continue_jump = None
            return False

        # normal move
        self.board[r, c] = EMPTY
        self.board[normal_r, normal_c] = piece

        promoted = self._promote_if_needed(normal_r, normal_c)
        if promoted:
            self.rewards[agent] += 0.3

        self.must_continue_jump = None
        return False

    def _count_pieces(self, agent):
        total = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._belongs_to_agent(self.board[r, c], agent):
                    total += 1
        return total

    def _check_game_over(self):
        p0 = "player_0"
        p1 = "player_1"

        p0_pieces = self._count_pieces(p0)
        p1_pieces = self._count_pieces(p1)

        if p0_pieces == 0:
            self.terminations = {a: True for a in self.agents}
            self.rewards[p0] -= 1.0
            self.rewards[p1] += 1.0
            return

        if p1_pieces == 0:
            self.terminations = {a: True for a in self.agents}
            self.rewards[p0] += 1.0
            self.rewards[p1] -= 1.0
            return

        p0_legal = self._legal_actions(p0)
        p1_legal = self._legal_actions(p1)

        if len(p0_legal) == 0:
            self.terminations = {a: True for a in self.agents}
            self.rewards[p0] -= 1.0
            self.rewards[p1] += 1.0
            return

        if len(p1_legal) == 0:
            self.terminations = {a: True for a in self.agents}
            self.rewards[p0] += 1.0
            self.rewards[p1] -= 1.0
            return