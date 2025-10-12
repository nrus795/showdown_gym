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
from poke_env.data import GenData, to_id_str
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

# from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

STATUSES = ("BRN", "PAR", "PSN", "SLP", "FRZ")
BOOSTS = ("atk", "def", "spa", "spd", "spe")

# Main reward weights
KO_WEIGHT = 3.0
STATUS_WEIGHT = 0.5
WIN_BONUS = 5.0
LOSS_PENALTY = 5.0
REWARD_CLIP = 50.0

# Dense HP shaping (we take helper hp_value=0 and do it ourselves)
HP_WEIGHT = 1.0  # reward += HP_WEIGHT * (opp_hp_loss - own_hp_loss)

# Switch shaping
SWITCH_PENALTY = 0.02
HAZARD_SWITCH_PENALTY = 0.02
STAY_BONUS = 0.005
_RECENT_SWITCH_EXTRA = 0.05
_ATTACK_READY_EXTRA = 0.04
_SWITCH_COOLDOWN_TURNS = 2

# Attack quality shaping
_DECENT_BP = 70
_MAX_BP_NORM = 150.0

# Tactical nudges
TYPE_HIT_BONUS = 0.03
INEFFECTIVE_PENALTY = 0.02
THREAT_SWITCH_BONUS = 0.05
SE_THRESHOLD = 2.0
NO_PROGRESS_EPS = 1e-4

# Anti-stall (soft)
NO_PROGRESS_TURNS = 3
NO_PROGRESS_PENALTY = 0.02

# Hazard progress
HAZARD_BONUS = 0.01  # per effective new hazard "unit" applied to opponent

# --- Simple Elo tracking (internal, not PS ladder) ---
ELO_K = 32.0
ELO_AGENT_INIT = 1000.0
ELO_OPP_INIT = 1200.0  # if using RandomPlayer, set this to 1000.0; SimpleHeuristics ~1100.0
# -----------------------------------------------------


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
		self._turns_since_switch = 10
		self._last_voluntary_switch = 0.0
		self._type_chart = GenData.from_format(battle_format).type_chart

		# Elo state
		self._elo_agent_rating = float(ELO_AGENT_INIT)
		self._elo_opponent_rating = float(ELO_OPP_INIT)
		self._elo_updated_this_battle = False
		self._last_battle_tag = None
		self.rl_agent = account_name_one

		# Learning aids
		self._turns_since_progress = 0
		self._illegal_action_count = 0

	def _to_id(self, x) -> str:
		s = getattr(x, "name", x)
		return to_id_str(str(s)) if s is not None else ""

	def _types_of(self, mon) -> tuple[str, ...]:
		if mon is None:
			return tuple()
		ts = getattr(mon, "types", ()) or ()
		return tuple(self._to_id(t) for t in ts if t is not None)

	def _type_multiplier(self, atk_type: Any, defender_types: tuple[str, ...]) -> float:
		atk = self._to_id(atk_type)
		if not atk or not defender_types:
			return 1.0
		m = 1.0
		row = self._type_chart.get(atk, {})
		for dt in defender_types:
			m *= row.get(dt, 1.0)
		return float(m)

	def _best_offense_multiplier(self, prior_battle: AbstractBattle | None) -> float:
		if prior_battle is None or not getattr(prior_battle, "available_moves", None):
			return 1.0
		opp = getattr(prior_battle, "opponent_active_pokemon", None)
		opp_types = self._types_of(opp)
		best = 1.0
		for mv in prior_battle.available_moves:
			try:
				mul_ = self._type_multiplier(getattr(mv, "type", None), opp_types)
			except Exception:
				mul_ = 1.0
			if mul_ > best:
				best = mul_
		return best

	def _threat_from_opp(self, prior_battle: AbstractBattle | None) -> float:
		if prior_battle is None:
			return 1.0
		me = getattr(prior_battle, "active_pokemon", None)
		opp = getattr(prior_battle, "opponent_active_pokemon", None)
		my_types = self._types_of(me)
		opp_types = self._types_of(opp)
		if not my_types or not opp_types:
			return 1.0
		threat = 1.0
		for ot in opp_types:
			row = self._type_chart.get(ot, {})
			m = 1.0
			for mt in my_types:
				m *= row.get(mt, 1.0)
			threat = max(threat, m)
		return float(threat)

	# ------------------------ helpers for Elo tracking ------------------------

	def _elo_expected(self, ra: float, rb: float) -> float:
		return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

	def _elo_update_once(self, battle: AbstractBattle) -> None:
		# Score: 1 win, 0 loss, 0.5 tie (rare)
		if getattr(battle, "won", False):
			score = 1.0
		elif getattr(battle, "lost", False):
			score = 0.0
		else:
			score = 0.5

		ra, rb = self._elo_agent_rating, self._elo_opponent_rating
		ea = self._elo_expected(ra, rb)
		self._elo_agent_rating = ra + ELO_K * (score - ea)
		self._elo_opponent_rating = rb + ELO_K * ((1.0 - score) - (1.0 - ea))

	def _get_action_size(self) -> int | None:
		"""
		None just uses the default number of actions as laid out in process_action - 26 actions.

		This defines the size of the action space
		for the agent - e.g. the output of the RL agent.

		This should return the number of actions you
		wish to use if not using the default action scheme.
		"""
		return None  # Return None if action size is default

	# ------------------------ action helpers/masking ------------------------

	def _legal_action_ids(self, battle: AbstractBattle) -> set[int]:
		legal: set[int] = set()

		moves = getattr(battle, "available_moves", []) or []
		switches = getattr(battle, "available_switches", []) or []

		# Base moves 6..9
		for i in range(min(len(moves), 4)):
			legal.add(6 + i)

		# Mega 10..13
		if bool(getattr(battle, "can_mega_evolve", False)):
			for i in range(min(len(moves), 4)):
				legal.add(10 + i)

		# Z-moves 14..17 (only if Z is available and the move can be Z)
		if bool(getattr(battle, "can_z_move", False)):
			for i, mv in enumerate(moves[:4]):
				is_z_ok = bool(getattr(mv, "is_z_move", False)) or bool(
					getattr(mv, "can_z_move", False)
				)
				# If we can't detect per-move, allow but we'll fallback below if rejected.
				if is_z_ok or True:
					legal.add(14 + i)

		# Dynamax 18..21
		if bool(getattr(battle, "can_dynamax", False)):
			for i in range(min(len(moves), 4)):
				legal.add(18 + i)

		# Terastallize 22..25
		if bool(getattr(battle, "can_tera", False)) or bool(
			getattr(battle, "can_terastallize", False)
		):
			for i in range(min(len(moves), 4)):
				legal.add(22 + i)

		# Switches 0..5
		for i in range(min(len(switches), 6)):
			legal.add(i)

		# default (-2) and forfeit (-1)
		legal.update({-2, -1})
		return legal

	def _best_move_index(self, battle: AbstractBattle) -> int | None:
		moves = getattr(battle, "available_moves", []) or []
		if not moves:
			return None
		opp = getattr(battle, "opponent_active_pokemon", None)
		opp_types = self._types_of(opp)
		best_idx = 0
		best_score = -1.0
		for i, mv in enumerate(moves[:4]):
			bp = float(getattr(mv, "base_power", 0) or 0)
			mul_ = self._type_multiplier(getattr(mv, "type", None), opp_types)
			score = bp * max(mul_, 1.0)  # prefer SE, but don't punish NVE too hard
			if score > best_score:
				best_score = score
				best_idx = i
		return int(best_idx)

	def _best_switch_index(self, battle: AbstractBattle) -> int | None:
		switches = getattr(battle, "available_switches", []) or []
		if not switches:
			return None
		opp = getattr(battle, "opponent_active_pokemon", None)
		opp_types = self._types_of(opp)
		best_i = 0
		best_threat = float("inf")
		for i, cand in enumerate(switches[:6]):
			my_types = self._types_of(cand)
			if not my_types or not opp_types:
				threat = 1.0
			else:
				threat = 1.0
				for ot in opp_types:
					row = self._type_chart.get(ot, {})
					m = 1.0
					for mt in my_types:
						m *= row.get(mt, 1.0)
					threat = max(threat, m)
			if threat < best_threat:
				best_threat = threat
				best_i = i
		return int(best_i)

	def process_action(self, action: np.int64) -> np.int64:
		"""
		Returns the np.int64 relative to the given action.

		The action mapping is as follows:
		action = -2: default
		action = -1: forfeit
		0 <= action <= 5: switch
		6 <= action <= 9: move
		10 <= action <= 13: move and mega evolve
		14 <= action <= 17: move and z-move
		18 <= action <= 21: move and dynamax
		22 <= action <= 25: move and terastallize

		:param action: The action to take.
		:type action: int64

		:return: The battle order ID for the given action in context of the current battle.
		:rtype: np.Int64
		"""
		b = self.battle1
		if b is None:
			return action

		legal = self._legal_action_ids(b)
		a = int(action)

		if a in legal:
			return np.int64(a)

		# Illegal -> smart fallback
		self._illegal_action_count += 1

		# Prefer good move
		m_idx = self._best_move_index(b)
		if m_idx is not None:
			base_move_id = 6 + m_idx
			if base_move_id in legal:
				return np.int64(base_move_id)

		# Otherwise safest switch
		s_idx = self._best_switch_index(b)
		if s_idx is not None:
			switch_id = s_idx  # 0..5
			if switch_id in legal:
				return np.int64(switch_id)

		# Last resort default
		return np.int64(-2)

	def get_additional_info(self) -> dict[str, dict[str, Any]]:
		info = super().get_additional_info()

		# Add any additional information you want to include
		# in the info dictionary that is saved in logs
		# For example, you can add the win status

		if self.battle1 is not None:
			agent = self.possible_agents[0]
			info[agent]["win"] = self.battle1.won
			info[agent]["turns"] = self.battle1.turn
			# expose internal Elo for logging
			info[agent]["elo_agent"] = round(self._elo_agent_rating, 1)
			info[agent]["elo_opponent"] = round(self._elo_opponent_rating, 1)
			info[agent]["illegal_actions"] = int(self._illegal_action_count)
			# info[agent]["win_rate"] = self.agent1.win_rate

		# (belt-and-suspenders) ensure Elo is applied if logger reads after finish
		b = self.battle1
		if b is not None and getattr(b, "finished", False):
			if (
				getattr(b, "battle_tag", None) == self._last_battle_tag
				and not self._elo_updated_this_battle
			):
				self._elo_update_once(b)
				self._elo_updated_this_battle = True

		return info

	def _hazard_score(self, sc: dict) -> float:
		score = 0.0
		if SideCondition.STEALTH_ROCK in sc:
			score += 1.0
		score += 0.5 * float(sc.get(SideCondition.SPIKES, 0))
		score += 0.5 * float(sc.get(SideCondition.TOXIC_SPIKES, 0))
		if SideCondition.STICKY_WEB in sc:
			score += 1.0
		return score

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

		# Track battle boundaries for Elo application
		curr_tag = getattr(battle, "battle_tag", None)
		if curr_tag != self._last_battle_tag:
			self._last_battle_tag = curr_tag
			self._elo_updated_this_battle = False
			self._turns_since_progress = 0
			self._illegal_action_count = 0

		# Minimal diffs we still use for shaping (switch detection and damage check)
		health_team = [mon.current_hp_fraction for mon in battle.team.values()]
		health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
		if len(health_opponent) < len(health_team):
			health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

		if prior_battle is not None:
			prior_health_opponent = [
				mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
			]
			prior_health_team = [mon.current_hp_fraction for mon in prior_battle.team.values()]
			if len(prior_health_opponent) < len(health_team):
				prior_health_opponent.extend(
					[1.0] * (len(health_team) - len(prior_health_opponent))
				)
		else:
			prior_health_opponent = health_opponent.copy()
			prior_health_team = health_team.copy()

		diff_health_opponent = np.array(prior_health_opponent) - np.array(health_opponent)
		diff_health_team = np.array(prior_health_team) - np.array(health_team)

		# Faints diffs for switch detection
		faints_team = [sum(1 for mon in battle.team.values() if mon.fainted)]
		faints_opponent = [sum(1 for mon in battle.opponent_team.values() if mon.fainted)]
		if prior_battle is not None:
			prior_faints_opponent = [
				sum(1 for mon in prior_battle.opponent_team.values() if mon.fainted)
			]
			prior_faints_team = [sum(1 for mon in prior_battle.team.values() if mon.fainted)]
		else:
			prior_faints_opponent = faints_opponent.copy()
			prior_faints_team = faints_team.copy()

		diff_faints_team = np.array(prior_faints_team) - np.array(faints_team)

		# Base reward: rely on poke-env helper for KOs, status, victory (HP via our dense shaping)
		base = float(
			self.reward_computing_helper(
				battle,
				fainted_value=KO_WEIGHT,
				hp_value=0.0,  # we do HP shaping below
				victory_value=WIN_BONUS,
				status_value=STATUS_WEIGHT,
			)
		)
		reward = float(base)

		# Dense HP shaping (progress signal every turn)
		opp_hp_loss = float(np.sum(diff_health_opponent))
		own_hp_loss = float(np.sum(diff_health_team))
		reward += HP_WEIGHT * (opp_hp_loss - own_hp_loss)

		# Detect voluntary switch (not from faint)
		voluntary_switch = False
		if (
			prior_battle is not None
			and prior_battle.active_pokemon is not None
			and battle.active_pokemon is not None
		):
			if prior_battle.active_pokemon.species != battle.active_pokemon.species:
				if (np.sum(diff_faints_team) == 0) and (not prior_battle.active_pokemon.fainted):
					voluntary_switch = True
		self._last_voluntary_switch = 1.0 if voluntary_switch else 0.0

		# Attack strength check
		best_attack_bp = 0.0
		if getattr(battle, "available_moves", None):
			for m in battle.available_moves:
				bp = float(getattr(m, "base_power", 0) or 0)
				if bp > best_attack_bp:
					best_attack_bp = bp
		had_decent_attack = best_attack_bp >= _DECENT_BP

		# Gentle anti-switch-spam + hazard cost
		if voluntary_switch:
			pen = SWITCH_PENALTY
			if had_decent_attack:
				pen += _ATTACK_READY_EXTRA
			if self._turns_since_switch <= _SWITCH_COOLDOWN_TURNS:
				pen += _RECENT_SWITCH_EXTRA

			sc = battle.side_conditions  # our side
			if SideCondition.STEALTH_ROCK in sc:
				pen += HAZARD_SWITCH_PENALTY
			pen += 0.005 * float(sc.get(SideCondition.SPIKES, 0))
			pen += 0.005 * float(sc.get(SideCondition.TOXIC_SPIKES, 0))
			if SideCondition.STICKY_WEB in sc:
				pen += 0.003

			reward -= pen
			self._turns_since_switch = 0
		else:
			# tiny bonus for not churning switches
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
			self._turns_since_switch = min(self._turns_since_switch + 1, 10)

		# Hazard progress (reward when we add hazards on opponent side)
		# sc_self = battle.side_conditions
		sc_opp = battle.opponent_side_conditions
		if prior_battle is not None:
			prior_sc_opp = prior_battle.opponent_side_conditions
		else:
			prior_sc_opp = sc_opp
		hazard_delta = self._hazard_score(sc_opp) - self._hazard_score(prior_sc_opp)
		if hazard_delta > 0:
			reward += HAZARD_BONUS * float(hazard_delta)

		# Small nudges for tactical quality
		best_effectiveness_vs_opponent_prev = self._best_offense_multiplier(prior_battle)
		opponent_threat_prev = self._threat_from_opp(prior_battle)
		did_damage = bool((diff_health_opponent > NO_PROGRESS_EPS).any())

		if voluntary_switch and opponent_threat_prev >= SE_THRESHOLD:
			reward += THREAT_SWITCH_BONUS * (1.0 if opponent_threat_prev < 4.0 else 1.5)

		if did_damage and best_effectiveness_vs_opponent_prev >= SE_THRESHOLD:
			reward += TYPE_HIT_BONUS * (1.0 if best_effectiveness_vs_opponent_prev < 4.0 else 1.5)

		if (not voluntary_switch) and opponent_threat_prev >= SE_THRESHOLD and not did_damage:
			reward -= INEFFECTIVE_PENALTY

		# Anti-stall if we made no progress (no damage, no faints on either side)
		made_progress = did_damage or bool((diff_health_team > NO_PROGRESS_EPS).any())
		made_progress = made_progress or (
			int(np.sum(prior_faints_opponent)) != int(np.sum(faints_opponent))
		)
		made_progress = made_progress or (
			int(np.sum(prior_faints_team)) != int(np.sum(faints_team))
		)
		if made_progress:
			self._turns_since_progress = 0
		else:
			self._turns_since_progress += 1
			if self._turns_since_progress >= NO_PROGRESS_TURNS:
				reward -= NO_PROGRESS_PENALTY

		# Elo update (not part of reward)
		if battle.finished and not self._elo_updated_this_battle:
			self._elo_update_once(battle)
			self._elo_updated_this_battle = True

		reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
		return reward

	def _observation_size(self) -> int:
		"""
		Returns the size of the observation size to create
		the observation space for all possible agents in the environment.

		You need to set observation size to the
		number of features you want to include in the observation.
		Annoyingly, you need to set this manually based
		on the features you want to include in the observation from embed_battle.

		Returns:
			int: The size of the observation space.
		"""

		# 47 original + 2 tactical scalars (best offense multiplier, opp threat) = 49
		return 49

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

		best_attack_bp = 0.0
		if getattr(battle, "available_moves", None):
			for m in battle.available_moves:
				bp = float(getattr(m, "base_power", 0) or 0)
				if bp > best_attack_bp:
					best_attack_bp = bp

		turns_since_switch_norm = float(min(getattr(self, "_turns_since_switch", 10), 10)) / 10.0
		best_bp_norm = float(min(best_attack_bp, _MAX_BP_NORM)) / _MAX_BP_NORM

		# NEW: tactical scalars
		best_effectiveness_vs_opponent_curr = self._best_offense_multiplier(battle)
		opponent_threat_curr = self._threat_from_opp(battle)

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
				np.array([turns_since_switch_norm, best_bp_norm], dtype=np.float32),  # 2
				np.array(
					[best_effectiveness_vs_opponent_curr, opponent_threat_curr], dtype=np.float32
				),  # 2 (new)
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
