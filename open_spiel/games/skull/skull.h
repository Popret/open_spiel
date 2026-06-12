// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_SKULL_H_
#define OPEN_SPIEL_GAMES_SKULL_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Skull & Roses
// https://en.wikipedia.org/wiki/Skull_(card_game)
//
// A bluffing and deduction game for 3-6 players. Each player starts with
// a hand of Rose and Skull cards. Players take turns placing cards face-down
// on their mats, then bidding on how many total cards they can flip without
// revealing a Skull. The last remaining bidder becomes the challenger and
// must flip that many cards, starting from their own mat. Flipping a Skull
// costs the challenger a card permanently; flipping the full bid with only
// Roses wins the round. First player to win two rounds wins the game.

namespace open_spiel {
namespace skull {

enum class GamePhase {
  kPlacement,
  kBidding,
  kFlipping,
  kCardLoss,
  kChooseStarter,
};
inline constexpr int kGamePhaseCount = 5;

enum class CardType {
  kRose = 0,
  kSkull = 1,
};
inline constexpr int kCardTypeCount = 2;

class SkullGame;

class SkullState : public State {
public:
  explicit SkullState(std::shared_ptr<const Game> game);
  SkullState(const SkullState &) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ActionToStringPlayerless(Action action) const;
  std::string ToString() const override;
  std::string HistoryString() const;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;

  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State>
  ResampleFromInfostate(int player_id, std::function<double()> rng) const;
  std::vector<Action> ActionsConsistentWithInformationFrom(Action action) const;
  std::unique_ptr<State> Clone() const override;

  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalActions() const override;
  std::vector<Action> LegalBids() const;

  GamePhase current_phase() const { return current_phase_; }
  int current_bid() const { return current_bid_; }
  Player challenger() const { return challenger_; }
  Player last_flipped() const { return last_flipped_; }
  int cards_flipped() const { return cards_flipped_; }
  int score(Player p) const { return scores_[p]; }
  bool is_active(Player p) const { return active_[p]; }
  bool is_known_to_have_rose(Player p) const { return known_to_have_rose_[p]; }
  bool is_known_to_have_skull(Player p) const {
    return known_to_have_skull_[p];
  }
  bool is_known_to_have_only_roses(Player p) const {
    return known_to_have_only_roses_[p];
  }
  bool is_known_to_have_only_skull(Player p) const {
    return known_to_have_only_skull_[p];
  }
  int stack_size(Player p) const { return static_cast<int>(stacks_[p].size()); }
  int hand_size(Player p) const { return static_cast<int>(hands_[p].size()); }
  int flipped_stack_count(Player p) const { return flipped_[p]; }

  bool has_skull(Player p) const;
  int total_cards_on_table() const;
  int num_active_players() const;

protected:
  void DoApplyAction(Action action) override;

private:
  const std::vector<CardType> InitialHand();
  void StartNewRound();
  void AdvanceToNextPlayer();
  void ResolveWonBet();
  void ResolveLostBet();
  void CheckForWin();
  void ApplyFlipAction(Action action);
  void DiscardCard(Player p, CardType type);
  void ResolveCardLoss();
  void UpdateHandInfoCertainties(Player p);

  const Action FlipTargetFromAction(Action action) const;
  const Action StarterTargetFromAction(Action action) const;
  const Action ActionDiscardFromChanceDiscard(Action action) const;

  // These are all derived from SkullGame, put here for easier access.
  const int max_hand_size_;
  const int wins_needed_;
  const Action flip_base_;
  const Action choose_starter_base_;
  const Action action_discard_rose_;
  const Action action_discard_skull_;
  const int max_total_cards_;

  // hands_[p]    : cards in player p's hand (private to p).
  //                Permanently shrinks when a card is lost after a failed
  //                bet. Reaching zero triggers EliminatePlayer.
  // stacks_[p]   : cards placed face-down on player p's mat, bottom-to-top.
  //                Returned to hands_ at the start of each new round.
  // flipped_[p] :  number of cards flipped per pile
  std::vector<std::vector<CardType>> hands_;
  std::vector<std::vector<CardType>> stacks_;
  std::vector<int> flipped_;

  std::vector<int> scores_;  // Bets won per player. Winning condition: 2.
  std::vector<bool> active_; // False once eliminated.
  std::vector<bool> passed_;
  std::vector<bool> known_to_have_rose_;
  std::vector<bool> known_to_have_only_roses_;
  std::vector<bool> known_to_have_skull_;
  std::vector<bool> known_to_have_only_skull_;

  GamePhase current_phase_;
  Player current_player_; // Current acting player, or kChancePlayerId.
  Player first_player_;   // First to act this round (challenger of previous).
  Player challenger_;     // Highest bidder entering kFlipping; kInvalidPlayer
                          // until kBidding resolves.
  Player last_flipped_;

  Action current_bid_; // Highest bid so far this round; 0 before any bid.
  int cards_flipped_;  // Cards flipped by the challenger so far.

  int winner_; // kInvalidPlayer until the game ends.
};

class SkullGame : public Game {
public:
  explicit SkullGame(const GameParameters &params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;

  absl::optional<double> UtilitySum() const override { return 0; }

  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override;

  const int MaxHandSize() const { return max_hand_size_; }
  const int WinsNeeded() const { return wins_needed_; }

  const Action flip_base() const { return flip_base_; }
  const Action discard_base() const { return discard_base_; }
  const int choose_starter_base() const { return choose_starter_base_; }

  // std::shared_ptr<Observer> MakeObserver(
  // absl::optional<IIGObservationType> iig_obs_type, const GameParameters&
  // params) const override;

private:
  const int num_players_;
  const int max_hand_size_;
  const int wins_needed_;
  const Action flip_base_;
  const Action discard_base_;
  const Action choose_starter_base_;
  static const int CalcMaxGameLength(int num_players, int max_hand_size,
                                     int wins_needed);
  const int max_game_length_;
};

} // namespace skull
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_SKULL_H_
