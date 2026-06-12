#ifndef OPEN_SPIEL_GAMES_SKULL_POLICIES_H_
#define OPEN_SPIEL_GAMES_SKULL_POLICIES_H_

// Skull & Roses – named policy factories
//
// Each function returns one of the three concrete types already in policy.h:
//
//   TabularPolicy            – full game-tree traversal, deterministic per info-state
//   PartialTabularPolicy     – covers only selected info-states; delegates the rest
//                              to a fallback Policy (default: UniformPolicy)
//   PreferredActionPolicy    – picks the first legal action from a preference list
//                              (usable directly when all relevant action indices are
//                              static; wrap in GetPrefActionPolicy(game,…) to get
//                              a safe TabularPolicy covering the whole game tree)
//
// Action-space layout (offsets are game-instance constants, not compile-time):
//
//   0                              PlaceRose
//   1                              PlaceSkull
//   2  (kActionPass/kActionBidBase) Pass
//   [3 .. 2+max_total_cards]       Bid N  (N = action - kActionBidBase, N >= 1)
//   [flip_base .. flip_base+P-1]   Flip player p's top unflipped card
//   [discard_base]                 DiscardRose   (challenger choosing their card loss)
//   [discard_base+1]               DiscardSkull
//   [choose_starter_base .. +P-1]  Choose the player who starts the next round
//
// Composition pattern
// -------------------
// PartialTabularPolicy(table, fallback) is the composition primitive:
//   • build a TabularPolicy via traversal for the phases you care about
//   • pass another named policy as the fallback for everything else
//
// Example:
//   // Bury the skull at position 2, then max-call, flip greedily everywhere else:
//   TabularPolicy placement = GetPlaceSkullAtPositionNPolicy(game, 2);
//   TabularPolicy bidding   = GetAlwaysMaxCallPolicy(game);
//   // Keep only placement-phase entries, fall through to bidding policy for the rest:
//   auto composed = PartialTabularPolicy(
//       ExtractPhaseEntries(placement, game, GamePhase::kPlacement),
//       std::make_shared<TabularPolicy>(bidding));

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "open_spiel/games/skull/skull.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace skull {

// ---------------------------------------------------------------------------
// Internal helpers (in anonymous namespace in the .cc; declared here for
// header-only use).
// ---------------------------------------------------------------------------
namespace {

constexpr Action kActionPlaceRose  = 0;
constexpr Action kActionPlaceSkull = 1;
constexpr Action kActionPass       = 2;  // == kActionBidBase

// Downcast helper – returns nullptr if state is not a SkullState.
const SkullState* AsSkullState(const State& state) {
  return dynamic_cast<const SkullState*>(&state);
}

// Returns the smallest bid action among legal actions, or kInvalidAction.
Action MinBidAction(const std::vector<Action>& legal) {
  Action best = kInvalidAction;
  for (Action a : legal) {
    if (a > kActionPass && (best == kInvalidAction || a < best))
      best = a;
  }
  return best;
}

// Returns the largest bid action among legal actions, or kInvalidAction.
Action MaxBidAction(const std::vector<Action>& legal) {
  Action best = kInvalidAction;
  for (Action a : legal)
    if (a > kActionPass && (best == kInvalidAction || a > best))
      best = a;
  return best;
}

// Standard BFS used by every traversal-based builder below.
// Visits every non-terminal, non-chance state and calls visitor(state).
// The visitor fills `policy` directly.
void TraverseGame(
    const Game& game,
    std::unordered_map<std::string, ActionsAndProbs>& policy,
    std::function<void(const State&,
                       std::unordered_map<std::string, ActionsAndProbs>&)>
        visitor) {
  SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSequential);
  std::list<std::unique_ptr<State>> frontier;
  frontier.push_back(game.NewInitialState());
  while (!frontier.empty()) {
    std::unique_ptr<State> state = std::move(frontier.back());
    frontier.pop_back();
    if (state->IsTerminal()) continue;
    if (state->IsChanceNode()) {
      for (const auto& oc : state->ChanceOutcomes())
        frontier.emplace_back(state->Child(oc.first));
      continue;
    }
    visitor(*state, policy);
    for (Action a : state->LegalActions())
      frontier.push_back(state->Child(a));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Utility: extract only the entries belonging to a given phase.
// Useful when composing a PartialTabularPolicy from two full TabularPolicies.
// ---------------------------------------------------------------------------
inline std::unordered_map<std::string, ActionsAndProbs>
ExtractPhaseEntries(const TabularPolicy& full_policy,
                   const Game& game,
                   GamePhase phase) {
  // We need to know which info-state strings belong to the requested phase.
  // The simplest way is to re-traverse and collect them.
  std::unordered_map<std::string, ActionsAndProbs> result;
  std::list<std::unique_ptr<State>> frontier;
  frontier.push_back(game.NewInitialState());
  while (!frontier.empty()) {
    std::unique_ptr<State> state = std::move(frontier.back());
    frontier.pop_back();
    if (state->IsTerminal()) continue;
    if (state->IsChanceNode()) {
      for (const auto& oc : state->ChanceOutcomes())
        frontier.emplace_back(state->Child(oc.first));
      continue;
    }
    const SkullState* s = AsSkullState(*state);
    if (s && s->current_phase() == phase) {
      const std::string key = state->InformationStateString();
      auto it = full_policy.PolicyTable().find(key);
      if (it != full_policy.PolicyTable().end())
        result[key] = it->second;
    }
    for (Action a : state->LegalActions())
      frontier.push_back(state->Child(a));
  }
  return result;
}

// ===========================================================================
// 1. GetNeverSkullPolicy
//
//    Returns a TabularPolicy (full game-tree traversal).
//
//    Intent: never voluntarily place a Skull.
//    • Placement phase:
//        – Prefer PlaceRose if legal.
//        – If PlaceRose is NOT legal (player is down to one card which is the
//          Skull AND nothing is on the stack yet, so the only move is
//          PlaceSkull) then place the Skull – there is no other option.
//        – If bids are available alongside PlaceSkull (stack non-empty),
//          prefer the minimum bid over placing the Skull.
//    • All other phases: uniform over legal actions.
// ===========================================================================
inline TabularPolicy GetNeverSkullPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [](const State& state,
       std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (s && s->current_phase() == GamePhase::kPlacement) {
        // Prefer: Rose > MinBid > Skull (last resort only when truly forced).
        bool rose_legal = false;
        for (Action a : legal) if (a == kActionPlaceRose) { rose_legal = true; break; }

        if (rose_legal) {
          entry = GetDeterministicPolicy(legal, kActionPlaceRose);
        } else {
          Action min_bid = MinBidAction(legal);
          if (min_bid != kInvalidAction) {
            entry = GetDeterministicPolicy(legal, min_bid);
          } else {
            // Only PlaceSkull is legal – no choice.
            entry = GetDeterministicPolicy(legal, kActionPlaceSkull);
          }
        }
      } else {
        entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// ===========================================================================
// 2. GetAlwaysFlipPolicy
//
//    Returns a TabularPolicy.
//
//    Intent: during the flipping phase always choose the first legal flip
//    target (which the game already constrains: own stack first, then
//    lowest active player index). Uniform elsewhere.
//
//    Note: the game's LegalActions in kFlipping already enforces "must flip
//    your own cards before anyone else's", so legal.front() is always the
//    correct greedy choice.
// ===========================================================================
inline TabularPolicy GetAlwaysFlipPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [](const State& state,
       std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (s && s->current_phase() == GamePhase::kFlipping) {
        // legal is already ordered: own remaining cards first, then others
        // by player index.  Greedily pick the front.
        entry = GetDeterministicPolicy(legal, legal.front());
      } else {
        entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// ===========================================================================
// 3. GetAlwaysMinCallPolicy
//
//    Returns a TabularPolicy.
//
//    Intent: whenever a bid is possible (Placement or Bidding phase), always
//    make the smallest legal raise.  Never pass.  Uniform in other phases.
// ===========================================================================
inline TabularPolicy GetAlwaysMinCallPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [](const State& state,
       std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (s && (s->current_phase() == GamePhase::kPlacement ||
                s->current_phase() == GamePhase::kBidding)) {
        Action min_bid = MinBidAction(legal);
        if (min_bid != kInvalidAction) {
          entry = GetDeterministicPolicy(legal, min_bid);
        } else {
          // No bids available yet (nothing on table): place a card uniformly.
          entry = UniformStatePolicy(state);
        }
      } else {
        entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// ===========================================================================
// 4. GetAlwaysMaxCallPolicy
//
//    Returns a TabularPolicy.
//
//    Intent: whenever a bid is possible always immediately commit to the
//    maximum bid (all cards on the table).  Never pass.  Uniform elsewhere.
// ===========================================================================
inline TabularPolicy GetAlwaysMaxCallPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [](const State& state,
       std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (s && (s->current_phase() == GamePhase::kPlacement ||
                s->current_phase() == GamePhase::kBidding)) {
        Action max_bid = MaxBidAction(legal);
        if (max_bid != kInvalidAction) {
          entry = GetDeterministicPolicy(legal, max_bid);
        } else {
          entry = UniformStatePolicy(state);
        }
      } else {
        entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// ===========================================================================
// 5. GetAlwaysCallAfterSkullPolicy
//
//    Returns a TabularPolicy.
//
//    Intent: place cards normally (Rose preferred) until the player's own
//    Skull is on the stack, then immediately make the minimum bid.  In the
//    Bidding phase, always raise to the minimum legal bid (never pass).
//    Uniform in all other phases.
//
//    Rationale: once your Skull is buried you know exactly where the danger
//    is; commit to a bid rather than let someone else take the round.
// ===========================================================================
inline TabularPolicy GetAlwaysCallAfterSkullPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [](const State& state,
       std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (s && s->current_phase() == GamePhase::kPlacement) {
        const Player p = state.CurrentPlayer();
        // has_skull(p) is true when the Skull is still in hand.
        // If it's NOT in hand, it must be on the stack → bid now.
        const bool skull_still_in_hand = s->has_skull(p);
        if (!skull_still_in_hand) {
          Action min_bid = MinBidAction(legal);
          if (min_bid != kInvalidAction) {
            entry = GetDeterministicPolicy(legal, min_bid);
          } else {
            // Nothing on table yet somehow – fall back to placing Rose.
            entry = GetDeterministicPolicy(legal, kActionPlaceRose);
          }
        } else {
          // Skull still in hand: prefer Rose, then Skull.
          bool rose_legal = false;
          for (Action a : legal)
            if (a == kActionPlaceRose) { rose_legal = true; break; }
          entry = GetDeterministicPolicy(
              legal, rose_legal ? kActionPlaceRose : kActionPlaceSkull);
        }
      } else if (s && s->current_phase() == GamePhase::kBidding) {
        // Never pass; always raise to the minimum.
        Action min_bid = MinBidAction(legal);
        entry = (min_bid != kInvalidAction)
                    ? GetDeterministicPolicy(legal, min_bid)
                    : UniformStatePolicy(state);
      } else {
        entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// ===========================================================================
// 6. GetPlaceSkullAtPositionNPolicy
//
//    Returns a TabularPolicy.
//
//    Intent: in the Placement phase, build a stack with exactly (n-1) Roses
//    before placing the Skull, so the Skull sits at depth n from the top.
//      n=1  → place Skull immediately (top of stack = first card placed).
//      n=MaxHandSize → fill with Roses first, Skull comes last.
//    Once the Skull is placed (no longer in hand), play Roses if available.
//    If the desired Rose count has been reached but the Skull can't be placed
//    (already one Skull in the stack from a prior round), fall back to a Rose
//    or the minimum bid.  Uniform outside the Placement phase.
//
//    n is clamped to [1, MaxHandSize] on entry.
// ===========================================================================
inline TabularPolicy GetPlaceSkullAtPositionNPolicy(const Game& game, int n) {
  const SkullGame* skull_game = dynamic_cast<const SkullGame*>(&game);
  SPIEL_CHECK_TRUE(skull_game != nullptr);
  n = std::max(1, std::min(n, skull_game->MaxHandSize()));

  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [n](const State& state,
        std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (s && s->current_phase() == GamePhase::kPlacement) {
        const Player p = state.CurrentPlayer();
        const bool skull_in_hand = s->has_skull(p);
        const int stack_sz       = s->stack_size(p);

        if (!skull_in_hand) {
          // Skull already placed. Place a Rose if we still can; else min-bid.
          bool rose_legal = false;
          for (Action a : legal)
            if (a == kActionPlaceRose) { rose_legal = true; break; }
          if (rose_legal) {
            entry = GetDeterministicPolicy(legal, kActionPlaceRose);
          } else {
            Action min_bid = MinBidAction(legal);
            entry = (min_bid != kInvalidAction)
                        ? GetDeterministicPolicy(legal, min_bid)
                        : UniformStatePolicy(state);
          }
        } else if (stack_sz < n - 1) {
          // Still need more roses before burying the skull.
          bool rose_legal = false;
          for (Action a : legal)
            if (a == kActionPlaceRose) { rose_legal = true; break; }
          if (rose_legal) {
            entry = GetDeterministicPolicy(legal, kActionPlaceRose);
          } else {
            // Can't place a Rose right now – place Skull or min-bid.
            bool skull_legal = false;
            for (Action a : legal)
              if (a == kActionPlaceSkull) { skull_legal = true; break; }
            if (skull_legal) {
              entry = GetDeterministicPolicy(legal, kActionPlaceSkull);
            } else {
              Action min_bid = MinBidAction(legal);
              entry = (min_bid != kInvalidAction)
                          ? GetDeterministicPolicy(legal, min_bid)
                          : UniformStatePolicy(state);
            }
          }
        } else {
          // Target depth reached (or exceeded): place the Skull now.
          bool skull_legal = false;
          for (Action a : legal)
            if (a == kActionPlaceSkull) { skull_legal = true; break; }
          if (skull_legal) {
            entry = GetDeterministicPolicy(legal, kActionPlaceSkull);
          } else {
            // Skull can't be placed (e.g. already on stack from earlier):
            // put a Rose or take the minimum bid.
            bool rose_legal = false;
            for (Action a : legal)
              if (a == kActionPlaceRose) { rose_legal = true; break; }
            if (rose_legal) {
              entry = GetDeterministicPolicy(legal, kActionPlaceRose);
            } else {
              Action min_bid = MinBidAction(legal);
              entry = (min_bid != kInvalidAction)
                          ? GetDeterministicPolicy(legal, min_bid)
                          : UniformStatePolicy(state);
            }
          }
        }
      } else {
        entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// ===========================================================================
// 7. GetNeverRiskBluffPolicy  ("safe / passive" policy)
//
//    Returns a TabularPolicy.
//
//    Intent: never Skull, never bid, never take any unnecessary risk.
//    • Placement:    Rose > Skull (Skull only when forced – same logic as
//                   GetNeverSkullPolicy, but here we also never initiate a bid
//                   even when bids would be available alongside PlaceSkull).
//    • Bidding:      Always Pass.
//    • Flipping:     First legal target (own cards first – forced by the game).
//    • CardLoss:     DiscardRose first (preserve the Skull for future rounds);
//                   discard_base() is the Rose action, discard_base()+1 is Skull,
//                   so legal.front() is always the Rose when both are present.
//    • ChooseStarter: First active player (legal.front()).
// ===========================================================================
inline TabularPolicy GetNeverRiskBluffPolicy(const Game& game) {
  std::unordered_map<std::string, ActionsAndProbs> policy;

  TraverseGame(game, policy,
    [](const State& state,
       std::unordered_map<std::string, ActionsAndProbs>& pol) {
      const SkullState* s = AsSkullState(state);
      const std::vector<Action> legal = state.LegalActions();
      ActionsAndProbs entry;

      if (!s) { pol[state.InformationStateString()] = UniformStatePolicy(state); return; }

      switch (s->current_phase()) {
        case GamePhase::kPlacement: {
          // Rose first; if not legal only then Skull (never bid here).
          bool rose_legal = false;
          for (Action a : legal) if (a == kActionPlaceRose) { rose_legal = true; break; }
          entry = rose_legal
                      ? GetDeterministicPolicy(legal, kActionPlaceRose)
                      : GetDeterministicPolicy(legal, kActionPlaceSkull);
          break;
        }
        case GamePhase::kBidding:
          // Always pass – let someone else be the challenger.
          entry = GetDeterministicPolicy(legal, kActionPass);
          break;
        case GamePhase::kFlipping:
          // Greedy: own cards first (enforced by LegalActions already), then
          // lowest player index.  legal.front() is always correct.
          entry = GetDeterministicPolicy(legal, legal.front());
          break;
        case GamePhase::kCardLoss:
          // discard_base() < discard_base()+1, so front == Rose when both present.
          entry = GetDeterministicPolicy(legal, legal.front());
          break;
        case GamePhase::kChooseStarter:
          entry = GetDeterministicPolicy(legal, legal.front());
          break;
        default:
          entry = UniformStatePolicy(state);
      }

      pol[state.InformationStateString()] = std::move(entry);
    });

  return TabularPolicy(policy);
}

// Conservative bidder: never Skull in placement, then minimum bid, flip greedy.
// -> placement table from NeverSkull, bidding/flipping from AlwaysMinCall.
inline PartialTabularPolicy GetConservativeBidderPolicy(const Game& game) {
  TabularPolicy never_skull   = GetNeverSkullPolicy(game);
  TabularPolicy always_min    = GetAlwaysMinCallPolicy(game);
  auto min_call_shared = std::make_shared<TabularPolicy>(std::move(always_min));
  return PartialTabularPolicy(
      ExtractPhaseEntries(never_skull, game, GamePhase::kPlacement),
      min_call_shared);
}

// Aggressive bluffer: bury skull at position n, then max-call immediately.
// -> placement table from PlaceSkullAtPositionN, rest from AlwaysMaxCall.
inline PartialTabularPolicy GetAggressiveBlufferPolicy(const Game& game,
                                                        int skull_position = 2) {
  TabularPolicy skull_at_n  = GetPlaceSkullAtPositionNPolicy(game, skull_position);
  TabularPolicy always_max  = GetAlwaysMaxCallPolicy(game);
  auto max_call_shared = std::make_shared<TabularPolicy>(std::move(always_max));
  return PartialTabularPolicy(
      ExtractPhaseEntries(skull_at_n, game, GamePhase::kPlacement),
      max_call_shared);
}

// Commit-after-skull: bury skull at position n, then min-call, flip greedy.
// -> placement from PlaceSkullAtPositionN, rest from AlwaysCallAfterSkull.
inline PartialTabularPolicy GetCommitAfterSkullPolicy(const Game& game,
                                                       int skull_position = 1) {
  TabularPolicy skull_at_n = GetPlaceSkullAtPositionNPolicy(game, skull_position);
  TabularPolicy call_after = GetAlwaysCallAfterSkullPolicy(game);
  auto call_after_shared   = std::make_shared<TabularPolicy>(std::move(call_after));
  return PartialTabularPolicy(
      ExtractPhaseEntries(skull_at_n, game, GamePhase::kPlacement),
      call_after_shared);
}

}  // namespace skull
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SKULL_POLICIES_H_
