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

#include "open_spiel/games/skull/skull.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_globals.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace skull {

namespace {

constexpr int kMinPlayers = 3;
constexpr int kMaxPlayers = 6;

constexpr int kDefaultPlayers = 4;
constexpr int kDefaultInitialHandSize = 4;
constexpr int kDefaultWinningScore = 2;

// ---------------------------------------------------------------------------
// Action encoding
//
// The flat action space is laid out as follows (offsets computed at runtime
// from num_players and max_total_cards = num_players * MaxHandSize()):
//
//   [0]                              PlaceRose   (placement phase)
//   [1]                              PlaceSkull  (placement phase)
//   [kBidBase]                       Pass        (bidding phase only)
//   [kBidBase +1, kBidBase + max_total_cards]    Bid N cards (min 1)
//   [kFlipBase, kFlipBase + num_players - 1]  Flip player n's card (reveal)
//   [kDiscardBase, kDiscardBase + kCardTypeCount - 1] Discard Rose, Skull
//
//  Note: Chance_Discard is a "seperate" action because ChanceOutcomes must
//  start at 0 to conform to the API so they start at 0 rather than kDiscardBase
//  but are treated identically.
//
// Total actions = 2 + 1 + max_bid + players + card_types
//               = 2 + 1 + (max_hand * players) + players + card_types
//
// ---------------------------------------------------------------------------

constexpr Action kActionPlaceRose = 0;
constexpr Action kActionPlaceSkull = 1;
constexpr Action kActionPass = 2;
constexpr Action kActionBidBase = kActionPass;
constexpr Action kActionChanceDiscardRose = 0;
constexpr Action kActionChanceDiscardSkull = 1;

const GameType kGameType{/*short_name=*/"skull",
                         /*long_name=*/"Skull",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kMaxPlayers,
                         /*min_num_players=*/kMinPlayers,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"handsize", GameParameter(kDefaultInitialHandSize)},
                          {"winsneeded", GameParameter(kDefaultWinningScore)}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::make_shared<const SkullGame>(params);
}
std::string CardTypeToString(CardType c) {
  return c == CardType::kRose ? "R" : "S";
};
std::string VecStr(const std::vector<CardType> &v) {
  std::string s = "[";
  for (int i = 0; i < static_cast<int>(v.size()); ++i) {
    if (i > 0)
      s += ",";
    s += CardTypeToString(v[i]);
  }
  return s + "]";
}

std::string PhaseToString(GamePhase p) {
  switch (p) {
  case GamePhase::kPlacement:
    return "Placement";
  case GamePhase::kBidding:
    return "Bidding";
  case GamePhase::kFlipping:
    return "Flipping";
  case GamePhase::kCardLoss:
    return "CardLoss";
  case GamePhase::kChooseStarter:
    return "ChooseStarter";
  default:
    return "Unknown";
  }
};

const SkullGame *UnwrapGame(const Game *game) {
  return down_cast<const SkullGame *>(game);
}
REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);
} // namespace

SkullState::SkullState(std::shared_ptr<const Game> game)
    : State(game), max_hand_size_(UnwrapGame(game_.get())->MaxHandSize()),
      wins_needed_(UnwrapGame(game_.get())->WinsNeeded()),
      flip_base_(UnwrapGame(game_.get())->flip_base()),
      choose_starter_base_(UnwrapGame(game_.get())->choose_starter_base()),
      action_discard_rose_(UnwrapGame(game_.get())->discard_base()),
      action_discard_skull_(UnwrapGame(game_.get())->discard_base() + 1),
      max_total_cards_(num_players_ * max_hand_size_),
      hands_(num_players_, InitialHand()), stacks_(num_players_),
      flipped_(num_players_, 0), scores_(num_players_, 0),
      active_(num_players_, true), passed_(num_players_, false),
      known_to_have_rose_(num_players_, true),
      known_to_have_only_roses_(num_players_, false),
      known_to_have_skull_(num_players_, true),
      known_to_have_only_skull_(num_players_, false),
      current_phase_(GamePhase::kPlacement), current_player_(kDefaultPlayerId),
      first_player_(kDefaultPlayerId), challenger_(kInvalidPlayer),
      last_flipped_(kInvalidPlayer), current_bid_(0), cards_flipped_(0),
      winner_(kInvalidPlayer) {}

const std::vector<CardType> SkullState::InitialHand() {
  std::vector<CardType> hand;
  const int handsize = max_hand_size_;
  hand.reserve(handsize);
  hand.assign(handsize - 1, CardType::kRose);
  hand.push_back(CardType::kSkull);
  return hand;
}

const Action SkullState::FlipTargetFromAction(Action action) const {
  return action - flip_base_;
}
const Action SkullState::ActionDiscardFromChanceDiscard(Action action) const {
  return action_discard_rose_ + action;
}
const Action SkullState::StarterTargetFromAction(Action action) const {
  return action - choose_starter_base_;
}
Player SkullState::CurrentPlayer() const {
  if (IsTerminal()) {
    SPIEL_CHECK_EQ(current_player_, kTerminalPlayerId);
  }
  return current_player_;
}

std::string SkullState::ActionToStringPlayerless(Action action) const {
  if (action == kActionPlaceRose)
    return "Place Rose";
  if (action == kActionPlaceSkull)
    return "Place Skull";
  if (action == kActionPass)
    return "Pass";
  if (action > kActionBidBase && action <= kActionBidBase + max_total_cards_)
    return absl::StrCat("Bid ", action - kActionBidBase);
  if (action >= flip_base_ && action < action_discard_rose_)
    return absl::StrCat("Flip player ", FlipTargetFromAction(action));
  if (action == action_discard_rose_)
    return absl::StrCat("Discard Rose");
  if (action == action_discard_skull_)
    return absl::StrCat("Discard Skull");
  if (action >= choose_starter_base_ &&
      action < choose_starter_base_ + num_players_)
    return absl::StrCat("Choosing Starter player:",
                        FlipTargetFromAction(action));
  return absl::StrCat("Unknown action ", action);
}

std::string SkullState::ActionToString(Player player, Action action) const {
  return absl::StrCat(player, ": ", ActionToStringPlayerless(action));
}

std::string SkullState::ToString() const {
  std::string out = absl::StrCat(
      "SkullState{phase=", PhaseToString(current_phase_),
      " move=", MoveNumber(), " cur=",
      (current_player_ == kChancePlayerId ? "chance"
                                          : std::to_string(current_player_)),
      " challenger=", challenger_, " bid=", current_bid_,
      " first=", first_player_, " last_flipped=", last_flipped_,
      " flipped=", cards_flipped_, " winner=", winner_, "\n");

  for (Player p = 0; p < num_players_; ++p) {
    out += absl::StrCat(
        "  P", p, ": hand=", VecStr(hands_[p]), " stack=", VecStr(stacks_[p]),
        " revealed=", flipped_[p], " score=", scores_[p],
        " passed=", passed_[p], " active=", (active_[p] ? "T" : "F"), "\n");
  }

  return out + "}";
}

std::string SkullState::HistoryString() const {
  std::vector<std::string> short_history;
  const std::vector<PlayerAction> &history = FullHistory();
  for (const PlayerAction &player_action : history) {
    const Action action = player_action.action;
    if (action == kActionPlaceRose) {
      short_history.push_back("r");
    } else if (action == kActionPlaceSkull) {
      short_history.push_back("s");
    } else if (action == kActionPass) {
      short_history.push_back("p");
    } else if (action > kActionBidBase &&
               action <= kActionBidBase + max_total_cards_) {
      short_history.push_back(absl::StrCat("b", action - kActionBidBase));
    } else if (action >= flip_base_ && action < action_discard_rose_) {
      short_history.push_back(absl::StrCat("f", FlipTargetFromAction(action)));
    } else if (action == action_discard_rose_) {
      short_history.push_back("dr");
    } else if (action == action_discard_skull_) {
      short_history.push_back("ds");
    } else if (action >= choose_starter_base_ &&
               action < choose_starter_base_ + num_players_) {
      short_history.push_back(absl::StrCat("cs", FlipTargetFromAction(action)));
    } else {
      short_history.push_back(absl::StrCat("?", action));
    }
  }
  return absl::StrCat("[", absl::StrJoin(short_history, ","), "]");
}

bool SkullState::IsTerminal() const { return winner_ != kInvalidPlayer; }
void SkullState::CheckForWin() {
  if (scores_[challenger_] >= wins_needed_) {
    winner_ = challenger_;
    current_player_ = kTerminalPlayerId;
    return;
  }
  if (num_active_players() == 1) {
    for (Player q = 0; q < num_players_; ++q) {
      if (active_[q]) {
        winner_ = q;
        current_player_ = kTerminalPlayerId;
        return;
      }
    }
    SPIEL_CHECK_TRUE_WSI(winner_ != kInvalidPlayer,
                         "Eliminated, found no surviving player", *game_,
                         *this);
  }
}

std::vector<double> SkullState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  if (IsTerminal()) {

    int winner_return = 0;
    for (Player p = 0; p < num_players_; ++p) {
      if (winner_ != p) {
        winner_return += wins_needed_ - score(p);
        returns[p] = score(p) - wins_needed_;
      }
    }
    returns[winner_] = winner_return;
  }
  return returns;
}

std::unique_ptr<State> SkullState::Clone() const {
  return std::unique_ptr<State>(new SkullState(*this));
}

std::vector<std::pair<Action, double>> SkullState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(current_phase() == GamePhase::kCardLoss);
  SPIEL_CHECK_EQ(CurrentPlayer(), kChancePlayerId);
  const Player p = challenger();
  SPIEL_CHECK_TRUE(p != kInvalidPlayer);
  if (hand_size(p) > 1 && has_skull(p)) {
    return {{kActionChanceDiscardRose, 1.0 - (1.0 / (double)hand_size(p))},
            {kActionChanceDiscardSkull, (1.0 / (double)hand_size(p))}};
  }
  if (has_skull(p)) {
    return {{kActionChanceDiscardSkull, 1.0}};
  }
  return {{kActionChanceDiscardRose, 1.0}};
}

std::vector<Action> SkullState::LegalBids() const {
  std::vector<Action> movelist;
  for (int i = current_bid() + 1; i <= total_cards_on_table(); i++) {
    movelist.push_back(kActionBidBase + i);
  }
  return movelist;
}

std::vector<Action> SkullState::LegalActions() const {
  if (IsTerminal())
    return {};
  if (IsChanceNode())
    return LegalChanceOutcomes();
  switch (current_phase()) {
  case GamePhase::kPlacement: {
    const Player p = CurrentPlayer();
    std::vector<Action> movelist;

    bool skull_on_stack = false;
    for (const auto &card : stacks_[p]) {
      if (card == CardType::kSkull) {
        skull_on_stack = true;
        break;
      }
    }
    bool can_place_skull = has_skull(p) && !skull_on_stack;
    int cards_in_hand_count = hand_size(p) - stack_size(p);
    bool can_place_rose = (cards_in_hand_count > 1) || !can_place_skull;

    if (cards_in_hand_count > 0) {
      if (can_place_rose)
        movelist.push_back(kActionPlaceRose);
      if (can_place_skull)
        movelist.push_back(kActionPlaceSkull);
    }

    if (stack_size(p) > 0) {
      movelist = LegalBids();
    }
    return movelist;
  }
  case GamePhase::kBidding: {
    std::vector<Action> movelist = LegalBids();
    movelist.insert(movelist.begin(), kActionPass);
    return movelist;
  }
  case GamePhase::kFlipping: {
    const Player p = CurrentPlayer();
    if (cards_flipped_ < stack_size(p)) {
      return {flip_base_ + p};
    } else {
      std::vector<Action> movelist;
      for (Player target = 0; target < num_players_; target++) {
        if (is_active(target) &&
            (flipped_stack_count(target) < stack_size(target))) {
          movelist.push_back(flip_base_ + target);
        }
      }
      return movelist;
    }
  }
  case GamePhase::kCardLoss: {
    const Player p = challenger();
    if (hand_size(p) > 1 && has_skull(p)) {
      return {action_discard_rose_, action_discard_skull_};
    } else if (has_skull(p)) {
      return {action_discard_skull_};
    } else {
      return {action_discard_rose_};
    }
  }
  case GamePhase::kChooseStarter: {
    SPIEL_CHECK_TRUE_WSI(!is_active(challenger()),
                         "eliminated challengers cannot choose starters",
                         *game_, *this);
    std::vector<Action> movelist;
    for (Player target = 0; target < num_players_; target++) {
      if (is_active(target)) {
        movelist.push_back(choose_starter_base_ + target);
      }
    }
    return movelist;
  }
  default:
    SpielFatalError("Invalid GamePhase in Skull");
  }
}

void SkullState::DoApplyAction(Action action) {
  if (CurrentPlayer() == kChancePlayerId)
    action = ActionDiscardFromChanceDiscard(action);

  if (action == kActionPlaceRose) {
    stacks_[CurrentPlayer()].push_back(CardType::kRose);
    AdvanceToNextPlayer();
  } else if (action == kActionPlaceSkull) {
    stacks_[CurrentPlayer()].push_back(CardType::kSkull);
    AdvanceToNextPlayer();
  } else if (action > kActionBidBase &&
             action <= kActionBidBase + max_total_cards_) {
    passed_[CurrentPlayer()] = false;
    challenger_ = CurrentPlayer();
    current_bid_ = action - kActionBidBase;
    if (action == kActionBidBase + max_total_cards_) { // passing is forced
      current_phase_ = GamePhase::kFlipping;
    } else {
      current_phase_ = GamePhase::kBidding;
      AdvanceToNextPlayer();
    }
  } else if (action == kActionPass) {
    passed_[CurrentPlayer()] = true;
    AdvanceToNextPlayer();
    if (CurrentPlayer() == challenger()) {
      current_phase_ = GamePhase::kFlipping;
    }
  } else if (action >= flip_base_ && action < action_discard_rose_) {
    Player flip_target = FlipTargetFromAction(action);
    last_flipped_ = flip_target;
    flipped_[flip_target]++;
    int depth = flipped_[flip_target];
    cards_flipped_++;
    UpdateHandInfoCertainties(flip_target);
    if (stacks_[flip_target].at(stack_size(flip_target) - depth) ==
        CardType::kSkull) {
      ResolveLostBet();
    } else if (cards_flipped_ >= current_bid_) {
      ResolveWonBet();
      if (!IsTerminal())
        StartNewRound();
    }
  } else if (action == action_discard_rose_) {
    DiscardCard(challenger(), CardType::kRose);
    UpdateHandInfoCertainties(challenger());
    ResolveCardLoss();
  } else if (action == action_discard_skull_) {
    DiscardCard(challenger(), CardType::kSkull);
    UpdateHandInfoCertainties(challenger());
    ResolveCardLoss();
  } else if (action >= choose_starter_base_ &&
             action < choose_starter_base_ + num_players_) {
    Player starter_target = StarterTargetFromAction(action);
    first_player_ = starter_target;
    StartNewRound();
  } else {
    SpielFatalError("Invalid Action");
  }
}

void SkullState::StartNewRound() {
  SPIEL_CHECK_TRUE_WSI(
      first_player_ != kInvalidPlayer && is_active(first_player_),
      absl::StrCat("StartNewRound with invalid first_player_=", first_player_),
      *game_, *this);
  for (Player p = 0; p < num_players_; ++p) {
    stacks_[p].clear();
    flipped_[p] = 0;
    passed_[p] = false;
  }

  current_bid_ = 0;
  cards_flipped_ = 0;
  challenger_ = kInvalidPlayer;
  last_flipped_ = kInvalidPlayer;
  current_player_ = first_player_;
  current_phase_ = GamePhase::kPlacement;
}

void SkullState::AdvanceToNextPlayer() {
  SPIEL_CHECK_GE(CurrentPlayer(), 0);
  const Player start = current_player_;
  do {
    current_player_ = (current_player_ + 1) % num_players_;
    if (!active_[current_player_])
      continue;
    return;
  } while (current_player_ != start);

  SpielFatalError("AdvanceToNextPlayer: no eligible player found.");
}

void SkullState::ResolveWonBet() {
  SPIEL_CHECK_EQ(CurrentPlayer(), challenger());
  ++scores_[challenger()];
  first_player_ = challenger();
  CheckForWin();
}

void SkullState::ResolveLostBet() {
  SPIEL_CHECK_NE(last_flipped(), kInvalidPlayer);
  SPIEL_CHECK_EQ(CurrentPlayer(), challenger());
  current_phase_ = GamePhase::kCardLoss;
  if (last_flipped() != challenger()) {
    current_player_ = kChancePlayerId;
  }
}

void SkullState::DiscardCard(Player p, CardType type) {
  SPIEL_CHECK_TRUE(current_phase() == GamePhase::kCardLoss);
  SPIEL_CHECK_GE(p, 0);
  auto &hand = hands_[p];
  auto it = std::find(hand.begin(), hand.end(), type);
  SPIEL_CHECK_TRUE_WSI(it != hand.end(),
                       absl::StrCat("Card not found to discard: ", ToString()),
                       *game_, *this);

  hand.erase(it);

  if (hand_size(p) <= 0) {
    active_[p] = false;
    CheckForWin();
  }
}

void SkullState::UpdateHandInfoCertainties(Player p) {
  if (hand_size(p) <= 0)
    return;
  if (is_known_to_have_only_roses(p) || is_known_to_have_only_skull(p))
    return;
  if (current_phase() == GamePhase::kCardLoss) {
    known_to_have_skull_[p] = false;
    known_to_have_rose_[p] = false;
    return;
  }

  SPIEL_CHECK_TRUE(current_phase() == GamePhase::kFlipping);
  if (is_known_to_have_rose(p) || is_known_to_have_skull(p))
    return;
  bool rose_revealed = false;
  bool skull_revealed = false;
  int depth;
  for (depth = 1; depth < flipped_stack_count(p); ++depth) {
    CardType card = stacks_[p].at(stack_size(p) - depth);
    if (card == CardType::kRose) {
      rose_revealed = true;
      known_to_have_rose_[p] = true;
    }
    if (card == CardType::kSkull) {
      skull_revealed = true;
      known_to_have_skull_[p] = true;
    }
  }
  if (depth == hand_size(p)) {
    if (!rose_revealed)
      known_to_have_only_skull_[p] = true;
    if (!skull_revealed)
      known_to_have_only_roses_[p] = true;
  }
}

void SkullState::ResolveCardLoss() {
  SPIEL_CHECK_TRUE(current_phase() == GamePhase::kCardLoss);
  SPIEL_CHECK_NE(challenger_, kInvalidPlayer);
  if (IsTerminal())
    return;

  if (!is_active(challenger_)) {
    current_phase_ = GamePhase::kChooseStarter;
    current_player_ = challenger_;
  } else {
    first_player_ = challenger_;
    StartNewRound();
  }
}

bool SkullState::has_skull(Player p) const {
  return is_active(p) &&
         hands_[p].end() !=
             find(hands_[p].begin(), hands_[p].end(), CardType::kSkull);
}
int SkullState::total_cards_on_table() const {
  int total = 0;
  for (Player p = 0; p < num_players_; ++p) {
    if (active_[p])
      total += static_cast<int>(stacks_[p].size());
  }
  return total;
}

int SkullState::num_active_players() const {
  int count = 0;
  for (Player p = 0; p < num_players_; ++p) {
    if (active_[p])
      ++count;
  }
  return count;
}

SkullGame::SkullGame(const GameParameters &params)
    : Game(kGameType, params), num_players_(ParameterValue<int>("players")),
      max_hand_size_(ParameterValue<int>("handsize")),
      wins_needed_(ParameterValue<int>("winsneeded")),
      max_game_length_(
          CalcMaxGameLength(num_players_, max_hand_size_, wins_needed_)),
      flip_base_(kActionBidBase + num_players_ * max_hand_size_ + 1),
      discard_base_(flip_base_ + num_players_),
      choose_starter_base_(discard_base_ + kCardTypeCount) {
  SPIEL_CHECK_GE(num_players_, kMinPlayers);
  SPIEL_CHECK_LE(num_players_, kMaxPlayers);
}

double SkullGame::MinUtility() const { return -wins_needed_; }
double SkullGame::MaxUtility() const {
  return (num_players_ - 1) * wins_needed_;
}
std::unique_ptr<State> SkullGame::NewInitialState() const {
  return std::make_unique<SkullState>(shared_from_this());
}

int SkullGame::NumDistinctActions() const {
  return choose_starter_base() + num_players_;
}

int SkullGame::MaxChanceOutcomes() const { return kCardTypeCount; }

const int SkullGame::CalcMaxGameLength(int num_players, int max_hand_size,
                                       int wins_needed) {
  auto max_round_length = [](int max_t, int num_p, bool has_card_loss_phase) {
    int card_loss_actions = has_card_loss_phase ? 2 : 0; // choose card, starter

    return max_t + // every player plays every card they have, then bid:
           (max_t - 1) * (num_p - 1) + // bid1 pass pass bid2 pass pass ...
           1 +                         // until bid_max (not followed by passes)
           max_t +                     // max flip actions
           card_loss_actions;          // if present.
  };

  int max_total_cards = num_players * max_hand_size;
  int max_rounds_without_card_loss = num_players * (wins_needed - 1);
  int max_round_one_length =
      max_round_length(max_total_cards, num_players, false);
  int early_rounds_length = max_round_one_length * max_rounds_without_card_loss;

  int max_total_turns;
  int card_loss_rounds_length = 0;
  for (max_total_turns = max_total_cards; max_total_turns > num_players; max_total_turns--) {
    card_loss_rounds_length += max_round_length(max_total_turns, num_players, true);
  }
  int players_left = num_players;
  while (max_total_turns > 1) {
    SPIEL_CHECK_TRUE(max_total_turns == players_left);
    card_loss_rounds_length += max_round_length(max_total_turns, players_left, true);
    max_total_turns--;
    players_left--;
  }
  return early_rounds_length + card_loss_rounds_length;
}

int SkullGame::MaxGameLength() const {
  // cached, see CalcMaxGameLength()
  return max_game_length_;
}

int SkullGame::MaxChanceNodesInHistory() const {
  return num_players_ * MaxHandSize();
}

std::vector<int> SkullGame::InformationStateTensorShape() const {
  int private_info = MaxHandSize() * 2;
  int action_info = NumDistinctActions();

  return {private_info + MaxGameLength() * action_info};
}

std::string SkullState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  int hand_roses = 0;
  bool hand_skull = false;
  for (auto card : hands_[player]) {
    if (card == CardType::kSkull)
      hand_skull = true;
    else
      hand_roses++;
  }
  int stack_roses = 0;
  bool stack_skull = false;
  for (auto card : stacks_[player]) {
    if (card == CardType::kSkull)
      stack_skull = true;
    else
      stack_roses++;
  }

  std::vector<std::string> short_history;
  const std::vector<PlayerAction> &history = FullHistory();
  for (const PlayerAction &player_action : history) {
    const Action action = player_action.action;
    if (action == kActionPlaceRose) {
      short_history.push_back("r");
    } else if (action == kActionPlaceSkull) {
      short_history.push_back("s");
    } else if (action == kActionPass) {
      short_history.push_back("p");
    } else if (action > kActionBidBase &&
               action <= kActionBidBase + max_total_cards_) {
      short_history.push_back(absl::StrCat("b", action - kActionBidBase));
    } else if (action >= flip_base_ && action < action_discard_rose_) {
      short_history.push_back(absl::StrCat("f", FlipTargetFromAction(action)));
    } else if (action == action_discard_rose_) {
      short_history.push_back("dr");
    } else if (action == action_discard_skull_) {
      short_history.push_back("ds");
    } else if (action >= choose_starter_base_ &&
               action < choose_starter_base_ + num_players_) {
      short_history.push_back(absl::StrCat("cs", FlipTargetFromAction(action)));
    } else {
      short_history.push_back(absl::StrCat("?", action));
    }
  }

  return absl::StrCat("P", player, " H:R", hand_roses, "S",
                      (hand_skull ? "1" : "0"), " ST:R", stack_roses, "S",
                      (stack_skull ? "1" : "0"), " Hist:[",
                      absl::StrJoin(short_history, ","), "]");
}

void SkullState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {

  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(GetGame()->InformationStateTensorShape()[0], values.size());
  std::fill(values.begin(), values.end(), 0.0);
  int offset = 0;

  int hand_roses = 0;
  bool hand_skull = false;
  for (auto card : hands_[player]) {
    if (card == CardType::kSkull)
      hand_skull = true;
    else
      hand_roses++;
  }
  if (hand_skull)
    values[offset] = 1.0;
  for (int j = 1; j <= hand_roses && j < max_hand_size_; ++j) {
    values[offset + j] = 1.0;
  }
  offset += max_hand_size_;
  int stack_roses = 0;
  bool stack_skull = false;
  for (auto card : stacks_[player]) {
    if (card == CardType::kSkull)
      stack_skull = true;
    else
      stack_roses++;
  }
  if (stack_skull)
    values[offset] = 1.0;
  for (int j = 1; j <= stack_roses && j < max_hand_size_; ++j) {
    values[offset + j] = 1.0;
  }
  offset += max_hand_size_;

  // 2. Write the historic action sequence
  const std::vector<Action> &history = History();
  for (const Action &action : history) {
    values[offset + action] = 1.0;
    offset += NumDistinctActions();
  }
}

std::vector<int> SkullGame::ObservationTensorShape() const {
  // one-hot phase 0001
  //
  // one-hot bid = num_players * MaxHandSize()
  //
  // one-hot player 0001 num_players_
  // one-hot challenger 0001 num_players_
  // one-hot passed 0001 num_players_
  // one-hot wins 0101 num_players_
  //
  // one-hot hand and stack starting with self (skull, rose rose rose)
  // followed by next player in turn oder as just one-hot hand count
  return {kGamePhaseCount + (num_players_ * max_hand_size_) +
          (num_players_ * 4) + (num_players_ * max_hand_size_ * 2)};
}
void SkullState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.0);

  int offset = 0;

  // 1. One-hot Phase (4 bins: Playing, Bidding, Revealing, Game Over)
  // Mapping: 0: Playing, 1: Bidding, 2: Revealing, 3: GameOver
  values[offset + static_cast<int>(current_phase_)] = 1.0;
  offset += kGamePhaseCount;

  // 2. One-hot Bid (num_players * MaxHandSize())
  // Represents current bid from 0 to Max Cards
  if (current_bid_ > 0) {
    values[offset + current_bid_ - 1] = 1.0;
  }
  offset += (num_players_ * max_hand_size_);

  // 3. One-hot Player Stats (Current, Challenger, Passed, Wins)
  // Total size: num_players * 4
  for (int p = 0; p < num_players_; ++p) {
    // Current player's turn
    if (current_player_ == p)
      values[offset + p] = 1.0;
    // The challenger (highest bidder)
    if (challenger_ == p)
      values[offset + num_players_ + p] = 1.0;
    // Has passed the bid
    if (passed_[p])
      values[offset + 2 * num_players_ + p] = 1.0;
    // Has won 1 round (0101 logic)
    if (scores_[p] > 0)
      values[offset + 3 * num_players_ + p] = 1.0;
  }
  offset += (num_players_ * 4);

  // 4. One-hot Hand/Stack starting with self
  // Size: num_players * MaxHandSize()
  for (int i = 0; i < num_players_; ++i) {
    int p = (player + i) % num_players_;
    int player_offset = offset + (i * max_hand_size_ * 2);

    if (p == player) {
      // For the observer: encode specific hand contents
      // (Simplified: count of roses vs skull remaining)
      int roses = 0;
      bool has_skull = false;
      for (auto card : hands_[p]) {
        if (card == CardType::kSkull)
          has_skull = true;
        else
          roses++;
      }
      if (has_skull)
        values[player_offset] = 1.0;
      for (int j = 1; j <= roses && j < max_hand_size_; ++j) {
        values[player_offset + j] = 1.0;
      }
    } else {
      // For others: one-hot count of remaining cards
      int hand_count = hands_[p].size();
      if (hand_count > 0) {
        values[player_offset + hand_count - 1] = 1.0;
      }
      int stack_count = stacks_[p].size();
      if (stack_count > 0) {
        values[player_offset + stack_count - 1] = 1.0;
      }
    }
  }
}

std::string SkullState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string s = absl::StrCat("Phase: ", PhaseToString(current_phase_));
  absl::StrAppend(&s, " | Bid: ", current_bid_);
  absl::StrAppend(&s, " | Current Player: ", current_player_);

  if (challenger_ != kInvalidPlayer) {
    absl::StrAppend(&s, " | Challenger: ", challenger_);
  }

  absl::StrAppend(&s, "\nPlayer Stats (Wins/Passed/Cards):");
  for (int p = 0; p < num_players_; ++p) {
    absl::StrAppend(&s, "\n P", p, ": ", scores_[p],
                    " wins | Passed: ", passed_[p] ? "Y" : "N",
                    " | Cards: ", hands_[p].size());

    // Show cards only for the observer
    if (p == player) {
      absl::StrAppend(&s, " (Your Hand: ");
      for (auto card : hands_[p]) {
        absl::StrAppend(&s, card == CardType::kSkull ? "S " : "R ");
      }
      absl::StrAppend(&s, ")");
    }
  }
  return s;
}

std::unique_ptr<State>
SkullState::ResampleFromInfostate(int player_id,
                                  std::function<double()> rng) const {
  auto cloned_state = std::make_unique<SkullState>(*this);

  for (Player p = 0; p < num_players_; ++p) {
    if (p == player_id)
      continue;

    int num_revealed = cloned_state->flipped_[p];
    int num_unrevealed = cloned_state->stacks_[p].size() - num_revealed;
    if (num_unrevealed <= 0)
      continue;

    std::vector<CardType> pool = cloned_state->hands_[p];

    for (int i = 0; i < num_revealed; ++i) {
      int revealed_idx = cloned_state->stacks_[p].size() - 1 - i;
      CardType revealed_card = cloned_state->stacks_[p][revealed_idx];

      auto it = std::find(pool.begin(), pool.end(), revealed_card);
      SPIEL_CHECK_TRUE(it != pool.end());
      pool.erase(it);
    }

    for (int i = 0; i < num_unrevealed; ++i) {
      int idx = static_cast<int>(rng() * pool.size());
      cloned_state->stacks_[p][i] = pool[idx];
      pool.erase(pool.begin() + idx);
    }
  }

  return cloned_state;
}

std::vector<Action>
SkullState::ActionsConsistentWithInformationFrom(Action action) const {
  if (IsChanceNode())
    return {kActionChanceDiscardRose, kActionChanceDiscardSkull};

  if (action == kActionPlaceRose || action == kActionPlaceSkull) {
    return {kActionPlaceRose, kActionPlaceSkull};
  }
  if (action == action_discard_rose_ || action == action_discard_skull_) {
    return {kActionPlaceRose, kActionPlaceSkull};
  }
  return {action};
}

} // namespace skull
} // namespace open_spiel
