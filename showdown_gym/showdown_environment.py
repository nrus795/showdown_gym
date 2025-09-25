import os
from typing import Any, Dict

import numpy as np
from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv


# --- helpers inside ShowdownEnvironment ---
_STATUS = ["BRN", "PAR", "PSN", "SLP", "FRZ"]


class ShowdownEnvironment(BaseShowdownEnv):
    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

    def _one_hot_status(self, mon) -> list[float]:
        st = [0.0] * 5
        if getattr(mon, "status", None) in _STATUS:
            st[_STATUS.index(mon.status)] = 1.0
        return st

    def _boost_sum_norm(self, mon) -> float:
        # boosts is dict { "atk":int, "def":int, "spa":int, "spd":int, "spe":int, ... } in [-6,6]
        if getattr(mon, "boosts", None):
            s = sum(
                mon.boosts.get(k, 0)
                for k in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
            )
            return max(-1.0, min(1.0, s / 18.0))  # crude, stable
        return 0.0

    def _field_weather_one_hot(self, battle) -> list[float]:
        # names depend on poke-env; fall back to 'none'
        names = ["none", "raindance", "sunnyday", "sandstorm", "snow"]
        cur = getattr(battle, "weather", None)
        out = [0.0] * 5
        idx = 0
        if cur is not None and getattr(cur, "name", None) in names:
            idx = names.index(cur.name)
        out[idx] = 1.0
        return out

    def _screens_flags(self, side_conditions: dict) -> list[float]:
        # keys like "reflect", "lightscreen", "auroraveil" (poke-env SideCondition names)
        keys = ["reflect", "lightscreen", "auroraveil"]
        return [1.0 if k in side_conditions else 0.0 for k in keys]

    def _hazards_vec(self, side_conditions: dict) -> list[float]:
        # Scale layers to [0,1]; spikes (0..3) /3, tspikes (0..2) /2; rocks/web are booleans.
        spikes = min(side_conditions.get("spikes", 0), 3) / 3.0
        tsp = min(side_conditions.get("toxicspikes", 0), 2) / 2.0
        rocks = 1.0 if "stealthrock" in side_conditions else 0.0
        web = 1.0 if "stickyweb" in side_conditions else 0.0
        return [rocks, spikes, tsp, web]

    def _move_feats(self, mv, self_mon, opp_mon) -> list[float]:
        if mv is None:
            return [0.0, 0.0, 0.0, 0.0]
        bp = float(getattr(mv, "base_power", 0.0)) / 200.0
        stab = (
            1.0
            if (
                mv.type is not None
                and self_mon is not None
                and mv.type in getattr(self_mon, "types", [])
            )
            else 0.0
        )
        # crude effectiveness estimate if types are known
        eff = getattr(mv, "type", None)
        if eff is None or opp_mon is None or not getattr(opp_mon, "types", None):
            effv = 0.66  # “unknown” ~ neutral-ish
        else:
            try:
                effx = mv.type.damage_multiplier(*opp_mon.types)  # poke-env type chart
                effv = {0: 0.0, 0.5: 0.33, 1: 0.66, 2: 1.0, 4: 1.0}.get(effx, 0.66)
            except Exception:
                effv = 0.66
        ppf = 0.0
        if hasattr(mv, "current_pp") and getattr(mv, "max_pp", 0):
            ppf = mv.current_pp / mv.max_pp
        return [bp, stab, effv, ppf]

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)

        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # If the opponent has less than 6 Pokémon, fill the missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        prior_health_opponent = []
        prior_health_team = []
        if prior_battle is not None:
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]
            prior_health_team = [
                mon.current_hp_fraction for mon in prior_battle.team.values()
            ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend(
                [1.0] * (len(health_team) - len(prior_health_opponent))
            )

        if len(prior_health_team) < len(health_team):
            prior_health_team.extend(
                [1.0] * (len(health_team) - len(prior_health_team))
            )

        diff_health_opponent = np.array(prior_health_opponent) - np.array(
            health_opponent
        )
        diff_health_team = np.array(prior_health_team) - np.array(health_team)

        # Reward for reducing the opponent's health
        reward += np.sum(diff_health_opponent)
        # Penalty for losing your own health
        reward -= np.sum(diff_health_team)

        faint_team = [mon.fainted for mon in battle.team.values()]
        faint_opponent = [mon.fainted for mon in battle.opponent_team.values()]

        if len(faint_opponent) < len(faint_team):
            faint_opponent.extend([False] * (len(faint_team) - len(faint_opponent)))

        prior_faint_team = []
        prior_faint_opponent = []

        if prior_battle is not None:
            prior_faint_opponent = [
                mon.fainted for mon in prior_battle.opponent_team.values()
            ]
            prior_faint_team = [mon.fainted for mon in prior_battle.team.values()]

        if len(prior_faint_opponent) < len(faint_team):
            prior_faint_opponent.extend(
                [False] * (len(faint_team) - len(prior_faint_opponent))
            )

        if len(prior_faint_team) < len(faint_team):
            prior_faint_team.extend([False] * (len(faint_team) - len(prior_faint_team)))

        diff_faint_team = np.array(prior_faint_team) - np.array(faint_team)
        diff_faint_opponent = np.array(prior_faint_opponent) - np.array(faint_opponent)

        reward += np.sum(diff_faint_opponent)
        reward -= np.sum(diff_faint_team)

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set observation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 40

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        # health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        # health_opponent = [
        #     mon.current_hp_fraction for mon in battle.opponent_team.values()
        # ]

        # # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        # if len(health_opponent) < len(health_team):
        #     health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # #########################################################################################################
        # # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        # #########################################################################################################

        # # Final vector - single array with health of both teams
        # final_vector = np.concatenate(
        #     [
        #         health_team,  # N components for the health of each pokemon
        #         health_opponent,  # N components for the health of opponent pokemon
        #     ]
        # ).astype(np.float32, copy=False)

        # return final_vector

        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        # Self active (8)
        self_block = (
            [getattr(me, "current_hp_fraction", 0.0)]
            + self._one_hot_status(me)
            + [self._boost_sum_norm(me)]
        )
        # Opp active (7)
        known_opp_moves = len(getattr(opp, "moves", {}))
        opp_block = (
            [getattr(opp, "current_hp_fraction", 0.0)]
            + self._one_hot_status(opp)
            + [min(4, known_opp_moves) / 4.0]
        )
        # Field + counts (9)
        my_alive = sum(1 for m in battle.team.values() if m.fainted is False)
        opp_alive = sum(1 for m in battle.opponent_team.values() if m.fainted is False)
        field_block = (
            self._field_weather_one_hot(battle)
            + self._screens_flags(battle.side_conditions)
            + [my_alive / 6.0, opp_alive / 6.0]
        )
        # Hazards (ours 4 + theirs 4 = 8)
        haz_block = self._hazards_vec(battle.side_conditions) + self._hazards_vec(
            battle.opponent_side_conditions
        )
        # Moves (2 x 4 = 8)
        legal = list(battle.available_moves) if battle.available_moves else []
        m0 = self._move_feats(legal[0] if len(legal) > 0 else None, me, opp)
        m1 = self._move_feats(legal[1] if len(legal) > 1 else None, me, opp)
        moves_block = m0 + m1
        vec = np.array(
            self_block + opp_block + field_block + haz_block + moves_block,
            dtype=np.float32,
        )
        return vec


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer()
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer()
        elif opponent_type == "random":
            opponent = RandomPlayer()
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "train_one" if not evaluation else "eval_one"
        account_name_two: str = "train_two" if not evaluation else "eval_two"

        account_name_one = f"{account_name_one}_{opponent_type}"
        account_name_two = f"{account_name_two}_{opponent_type}"

        team = self._load_team(team_type)

        battle_fomat = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_fomat,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
