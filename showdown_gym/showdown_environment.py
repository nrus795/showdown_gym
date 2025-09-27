import os
import time
from typing import Any

import numpy as np
from poke_env import (
	AccountConfiguration,
	MaxBasePowerPlayer,
	RandomPlayer,
	SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

STATUSES = ("BRN", "PAR", "PSN", "SLP", "FRZ")
BOOSTS = ("atk", "def", "spa", "spd", "spe")

KO_WEIGHT = 3.0
WIN_BONUS = 25.0
LOSS_PENALTY = 25.0
STEP_PENALTY = -0.01
REWARD_CLIP = 50.0
SWITCH_PENALTY = 0.02
HAZARD_SWITCH_PENALTY = 0.02
STAY_BONUS = 0.005


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

	def get_additional_info(self) -> dict[str, dict[str, Any]]:
		info = super().get_additional_info()

		# Add any additional information you want
		# to include in the info dictionary that is saved in logs
		# For example, you can add the win status

		if self.battle1 is not None:
			agent = self.possible_agents[0]
			info[agent]["win"] = self.battle1.won
			info[agent]["turns"] = self.battle1.turn
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
		health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]

		# If the opponent has less than 6 Pokémon,
		# fill the missing values with 1.0 (fraction of health)

		if len(health_opponent) < len(health_team):
			health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))
		prior_health_opponent = []
		prior_health_team = []
		if prior_battle is not None:
			prior_health_opponent = [
				mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
			]
			prior_health_team = [mon.current_hp_fraction for mon in prior_battle.team.values()]
		# Ensure prior_health_opponent has 6 components, filling missing values with 1.0

		if len(prior_health_opponent) < len(health_team):
			prior_health_opponent.extend([1.0] * (len(health_team) - len(prior_health_opponent)))
		# If no prior state yet, use current so diffs are zero on the first step

		if prior_battle is None:
			prior_health_team = health_team.copy()
			prior_health_opponent = health_opponent.copy()
		diff_health_opponent = np.array(prior_health_opponent) - np.array(health_opponent)
		diff_health_team = np.array(prior_health_team) - np.array(health_team)

		# Reward for reducing the opponent's health; penalty for our health loss

		reward += np.sum(diff_health_opponent) * 1.0
		reward -= np.sum(diff_health_team) * 1.0

		faints_team = [sum(1 for mon in battle.team.values() if mon.fainted)]
		faints_opponent = [sum(1 for mon in battle.opponent_team.values() if mon.fainted)]

		prior_faints_opponent = []
		prior_faints_team = []
		if prior_battle is not None:
			prior_faints_opponent = [
				sum(1 for mon in prior_battle.opponent_team.values() if mon.fainted)
			]
			prior_faints_team = [sum(1 for mon in prior_battle.team.values() if mon.fainted)]
		if prior_battle is None:
			prior_faints_opponent = faints_opponent.copy()
			prior_faints_team = faints_team.copy()
		diff_faints_opponent = np.array(prior_faints_opponent) - np.array(faints_opponent)
		diff_faints_team = np.array(prior_faints_team) - np.array(faints_team)

		# Make KOs matter more (opponent KO gained -> positive; our KO suffered -> negative)

		reward += (-np.sum(diff_faints_opponent)) * KO_WEIGHT
		reward -= (-np.sum(diff_faints_team)) * KO_WEIGHT

		# Detect a voluntary switch (active changed, we didn't faint to force it)

		voluntary_switch = False
		if (
			prior_battle is not None
			and prior_battle.active_pokemon is not None
			and battle.active_pokemon is not None
		):
			if prior_battle.active_pokemon.species != battle.active_pokemon.species:
				if (np.sum(diff_faints_team) == 0) and (not prior_battle.active_pokemon.fainted):
					voluntary_switch = True
		# Save for state embedding (next step sees what we just did)

		self._last_voluntary_switch = 1.0 if voluntary_switch else 0.0

		# Penalise switch spam and hazard entries

		if voluntary_switch:
			reward -= SWITCH_PENALTY
			sc = battle.side_conditions  # our side
			if SideCondition.STEALTH_ROCK in sc:
				reward -= HAZARD_SWITCH_PENALTY
			spikes_layers = sc.get(SideCondition.SPIKES, 0)
			if spikes_layers:
				reward -= 0.01 * float(spikes_layers)
			tox_layers = sc.get(SideCondition.TOXIC_SPIKES, 0)
			if tox_layers:
				reward -= 0.01 * float(tox_layers)
			if SideCondition.STICKY_WEB in sc:
				reward -= 0.005
		else:
			# tiny nudge for staying in when not forced

			if (
				prior_battle is not None
				and prior_battle.active_pokemon is not None
				and battle.active_pokemon is not None
			):
				if (
					prior_battle.active_pokemon.species == battle.active_pokemon.species
					and np.sum(diff_faints_team) == 0
				):
					reward += STAY_BONUS
		# Small per-step nudge to end games sooner

		reward += STEP_PENALTY

		if battle.finished:
			if battle.won:
				reward += WIN_BONUS
			elif battle.lost:
				reward -= LOSS_PENALTY
		return float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

	def _observation_size(self) -> int:
		"""
		Returns the size of the observation size to create the observation
		space for all possible agents in the environment.

		You need to set observation size to the number of
		features you want to include in the observation.
		Annoyingly, you need to set this manually based on the features you want
		to include in the observation from embed_battle.

		Returns:
				int: The size of the observation space.
		"""

		# Simply change this number to the number
		# of features you want to include in the observation from embed_battle.
		# If you find a way to automate this, please let me know!

		return 45

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

		health_team = [mon.current_hp_fraction for mon in battle.team.values()]
		health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
		if len(health_opponent) < len(health_team):
			health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))
		# faint counts

		faints_team_count = float(sum(1 for mon in battle.team.values() if mon.fainted))
		faints_opponent_count = float(
			sum(1 for mon in battle.opponent_team.values() if mon.fainted)
		)

		# status one-hots for active mons

		status_self = [0.0] * len(STATUSES)
		status_opp = [0.0] * len(STATUSES)
		if battle.active_pokemon is not None:
			for i, s in enumerate(STATUSES):
				status_self[i] = 1.0 if battle.active_pokemon.status == s else 0.0
		if getattr(battle, "opponent_active_pokemon", None) is not None:
			for i, s in enumerate(STATUSES):
				status_opp[i] = 1.0 if battle.opponent_active_pokemon.status == s else 0.0
		# boosts for active mons (normalised by 6)

		boosts_self = [0.0] * len(BOOSTS)
		boosts_opp = [0.0] * len(BOOSTS)
		if battle.active_pokemon is not None:
			for i, k in enumerate(BOOSTS):
				boosts_self[i] = float(battle.active_pokemon.boosts.get(k, 0)) / 6.0
		if getattr(battle, "opponent_active_pokemon", None) is not None:
			for i, k in enumerate(BOOSTS):
				boosts_opp[i] = float(battle.opponent_active_pokemon.boosts.get(k, 0)) / 6.0
		# hazards on each side

		sc_self = battle.side_conditions
		sc_opp = battle.opponent_side_conditions

		hazards_self = [
			1.0 if SideCondition.STEALTH_ROCK in sc_self else 0.0,
			float(sc_self.get(SideCondition.SPIKES, 0)) / 3.0,
			float(sc_self.get(SideCondition.TOXIC_SPIKES, 0)) / 2.0,
			1.0 if SideCondition.STICKY_WEB in sc_self else 0.0,
		]
		hazards_opp = [
			1.0 if SideCondition.STEALTH_ROCK in sc_opp else 0.0,
			float(sc_opp.get(SideCondition.SPIKES, 0)) / 3.0,
			float(sc_opp.get(SideCondition.TOXIC_SPIKES, 0)) / 2.0,
			1.0 if SideCondition.STICKY_WEB in sc_opp else 0.0,
		]

		# a few scalars that help policy

		last_voluntary_switch = float(getattr(self, "_last_voluntary_switch", 0.0))
		available_switches_norm = float(len(getattr(battle, "available_switches", []))) / 5.0
		turn_norm = float(min(getattr(battle, "turn", 0), 100)) / 100.0

		final_vector = np.concatenate(
			[
				np.array(health_team, dtype=np.float32),  # 6
				np.array(health_opponent, dtype=np.float32),  # 6
				np.array([faints_team_count, faints_opponent_count], dtype=np.float32),  # 2
				np.array(status_self, dtype=np.float32),  # 5
				np.array(status_opp, dtype=np.float32),  # 5
				np.array(boosts_self, dtype=np.float32),  # 5
				np.array(boosts_opp, dtype=np.float32),  # 5
				np.array(hazards_self, dtype=np.float32),  # 4
				np.array(hazards_opp, dtype=np.float32),  # 4
				np.array(
					[last_voluntary_switch, available_switches_norm, turn_norm],
					dtype=np.float32,
				),  # 3
			]
		)

		return final_vector.astype(np.float32)


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
		unique_id = time.strftime("%H%M%S")

		opponent_account = "ot" if not evaluation else "oe"
		opponent_account = f"{opponent_account}_{unique_id}"

		opponent_configuration = AccountConfiguration(opponent_account, None)
		if opponent_type == "simple":
			opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
		elif opponent_type == "max":
			opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
		elif opponent_type == "random":
			opponent = RandomPlayer(account_configuration=opponent_configuration)
		else:
			raise ValueError(f"Unknown opponent type: {opponent_type}")
		account_name_one: str = "t1" if not evaluation else "e1"
		account_name_two: str = "t2" if not evaluation else "e2"

		account_name_one = f"{account_name_one}_{unique_id}"
		account_name_two = f"{account_name_two}_{unique_id}"

		team = self._load_team(team_type)

		battle_format = "gen9randombattle" if team is None else "gen9ubers"

		primary_env = ShowdownEnvironment(
			battle_format=battle_format,
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
