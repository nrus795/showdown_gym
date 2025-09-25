import os
from typing import Any, Dict

import numpy as np
from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

STATUSES = {"BRN", "PAR", "PSN", "SLP", "FRZ"}


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

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 36

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

        # --- your original health blocks ---
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # --- hazards on each side (rocks, spikes/3, tspikes/2, web) ---
        side_conditions_team = getattr(battle, "side_conditions", {}) or {}
        side_conditions_opponent = getattr(battle, "opponent_side_conditions", {}) or {}

        def _hazards(sc: dict) -> list[float]:
            rocks = 1.0 if "stealthrock" in sc else 0.0
            spikes = min(sc.get("spikes", 0), 3) / 3.0
            tspikes = min(sc.get("toxicspikes", 0), 2) / 2.0
            web = 1.0 if "stickyweb" in sc else 0.0
            return [rocks, spikes, tspikes, web]

        hazards_team = _hazards(side_conditions_team)  # 4
        hazards_opponent = _hazards(side_conditions_opponent)  # 4

        # --- screens on each side (reflect, lightscreen, auroraveil) ---
        def _screens(sc: dict) -> list[float]:
            keys = ["reflect", "lightscreen", "auroraveil"]
            return [1.0 if k in sc else 0.0 for k in keys]

        screens_team = _screens(side_conditions_team)  # 3
        screens_opponent = _screens(side_conditions_opponent)  # 3

        # --- weather one-hot (none, rain, sun, sand, snow) ---
        weather_names = ["none", "raindance", "sunnyday", "sandstorm", "snow"]
        weather = [0.0] * 5
        cur_weather = getattr(battle, "weather", None)
        idx = 0
        if (
            cur_weather is not None
            and getattr(cur_weather, "name", None) in weather_names
        ):
            idx = weather_names.index(cur_weather.name)
        weather[idx] = 1.0  # 5

        # --- simple counts (alive + status) for each side, normalised by 6 ---

        team_alive = sum(1 for m in battle.team.values() if not m.fainted) / 6.0
        opponent_alive = (
            sum(1 for m in battle.opponent_team.values() if not m.fainted) / 6.0
        )
        team_status = (
            sum(
                1
                for m in battle.team.values()
                if getattr(m, "status", None) in STATUSES
            )
            / 6.0
        )
        opponent_status = (
            sum(
                1
                for m in battle.opponent_team.values()
                if getattr(m, "status", None) in STATUSES
            )
            / 6.0
        )

        counts_team = [team_alive, team_status]  # 2
        counts_opponent = [opponent_alive, opponent_status]  # 2

        # --- opponent revealed moves fraction (0..1) ---
        opponent_active = getattr(battle, "opponent_active_pokemon", None)
        opponent_revealed_moves = 0.0
        if (
            opponent_active is not None
            and getattr(opponent_active, "moves", None) is not None
        ):
            opponent_revealed_moves = min(4, len(opponent_active.moves)) / 4.0  # 1

        # --- final vector (keep your variable name) ---
        final_vector = np.concatenate(
            [
                health_team,  # ~6
                health_opponent,  # ~6 (padded)
                hazards_team,  # 4
                hazards_opponent,  # 4
                screens_team,  # 3
                screens_opponent,  # 3
                weather,  # 5
                counts_team,  # 2
                counts_opponent,  # 2
                [opponent_revealed_moves],  # 1
            ]
        ).astype(np.float32)

        return final_vector


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
