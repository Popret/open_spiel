
// Extends the basic CFR example to evaluate the trained policy by:
//   1. Playing CFR average policy vs. a uniform-random opponent (N games)
//   2. Playing CFR average policy vs. itself            (N games)
//   3. Printing a full move-by-move game log for one representative game
//
// Build alongside the rest of open_spiel; link against:
//   open_spiel, algorithms_cfr, tabular_exploitability

#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, game_name, "skull",
          "Game to run CFR on (default: skull).");
ABSL_FLAG(int, num_iters, 1000,
          "CFR training iterations.");
ABSL_FLAG(int, report_every, 100,
          "How often (in iters) to print exploitability.");
ABSL_FLAG(int, num_eval_games, 200,
          "Games to play per evaluation match-up.");
ABSL_FLAG(int, seed, 42,
          "RNG seed for evaluation play-outs.");

// Sample an action from a probability distribution over legal actions.
// `z` is a uniform random number in [0, 1).
open_spiel::Action SampleAction(
    const open_spiel::ActionsAndProbs& actions_and_probs, double z) {
  double cumulative = 0.0;
  for (const auto& [action, prob] : actions_and_probs) {
    cumulative += prob;
    if (z < cumulative) return action;
  }
  // Fallback: return the last action (handles floating-point edge cases).
  return actions_and_probs.back().first;
}

std::vector<double> PlayGame(
    const open_spiel::Game& game,
    const std::vector<const open_spiel::Policy*>& policies,
    std::mt19937& rng,
    bool verbose = false) {

  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  if (verbose) {
    std::cout << "\n=== New Game ===\n";
    std::cout << state->ToString() << "\n";
  }

  while (!state->IsTerminal()) {
    open_spiel::ActionsAndProbs actions_and_probs;

    if (state->IsChanceNode()) {
      actions_and_probs = state->ChanceOutcomes();
    } else {
      const open_spiel::Player cur = state->CurrentPlayer();
      actions_and_probs =
          policies[cur]->GetStatePolicy(*state, cur);

      // If the policy returns nothing (e.g. info state not in table),
      // fall back to uniform over legal actions.
      if (actions_and_probs.empty()) {
        const auto legal = state->LegalActions();
        const double p = 1.0 / static_cast<double>(legal.size());
        for (open_spiel::Action a : legal) {
          actions_and_probs.push_back({a, p});
        }
      }
    }

    open_spiel::Action chosen = SampleAction(actions_and_probs, dist(rng));

    if (verbose) {
      if (state->IsChanceNode()) {
        std::cout << "  [Chance]  " << state->ActionToString(
            open_spiel::kChancePlayerId, chosen) << "\n";
      } else {
        std::cout << "  [P" << state->CurrentPlayer() << "]  "
                  << state->ActionToString(state->CurrentPlayer(), chosen)
                  << "\n";
      }
    }

    state->ApplyAction(chosen);

    if (verbose && !state->IsTerminal()) {
      std::cout << state->ToString() << "\n";
    }
  }

  const std::vector<double> returns = state->Returns();

  if (verbose) {
    std::cout << "\n--- Game Over ---\n";
    std::cout << state->ToString() << "\n";
    std::cout << "Returns: ";
    for (int i = 0; i < static_cast<int>(returns.size()); ++i) {
      std::cout << "P" << i << "=" << returns[i];
      if (i + 1 < static_cast<int>(returns.size())) std::cout << "  ";
    }
    std::cout << "\n";
  }

  return returns;
}

// ---------------------------------------------------------------------------
// Evaluation runner
// ---------------------------------------------------------------------------

struct MatchStats {
  int num_games = 0;
  std::vector<double> total_returns;   // summed over all games, per player
  std::vector<int>    wins;            // times each player had the max return

  explicit MatchStats(int num_players)
      : total_returns(num_players, 0.0), wins(num_players, 0) {}
};

MatchStats RunMatch(
    const open_spiel::Game& game,
    const std::vector<const open_spiel::Policy*>& policies,
    int num_games,
    std::mt19937& rng,
    bool verbose_first_game = false) {

  const int np = game.NumPlayers();
  MatchStats stats(np);

  for (int g = 0; g < num_games; ++g) {
    bool verbose = (verbose_first_game && g == 0);
    auto returns = PlayGame(game, policies, rng, verbose);

    stats.num_games++;
    double best = *std::max_element(returns.begin(), returns.end());
    for (int p = 0; p < np; ++p) {
      stats.total_returns[p] += returns[p];
      if (returns[p] == best) stats.wins[p]++;
    }
  }
  return stats;
}

void PrintMatchStats(const std::string& label,
                     const MatchStats& s,
                     const open_spiel::Game& game) {
  const int np = game.NumPlayers();
  std::cout << "\n--- " << label << " (" << s.num_games << " games) ---\n";
  std::cout << std::fixed << std::setprecision(3);
  for (int p = 0; p < np; ++p) {
    double avg = s.total_returns[p] / s.num_games;
    double win_pct = 100.0 * s.wins[p] / s.num_games;
    std::cout << "  P" << p
              << "  avg_return=" << std::setw(7) << avg
              << "  wins=" << s.wins[p]
              << " (" << std::setprecision(1) << win_pct << "%)\n";
    std::cout << std::fixed << std::setprecision(3);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game_name));
  std::cerr << "Loaded game: " << game->GetType().long_name
            << "  (players=" << game->NumPlayers() << ")\n";

  open_spiel::algorithms::OutcomeSamplingMCCFRSolver solver(*game);
  std::cerr << "Training CFR for " << absl::GetFlag(FLAGS_num_iters)
            << " iterations...\n";

  const int num_iters   = absl::GetFlag(FLAGS_num_iters);
  const int report_every = absl::GetFlag(FLAGS_report_every);

  for (int i = 0; i < num_iters; ++i) {
    solver.EvaluateAndUpdatePolicy();
    if (i % report_every == 0 || i == num_iters - 1) {
      double expl = open_spiel::algorithms::Exploitability(
          *game, *solver.AveragePolicy());
      std::cerr << "  iter=" << std::setw(6) << i
                << "  exploitability=" << expl << "\n";
    }
  }

  // ---- Build evaluation policies ----
  std::shared_ptr<open_spiel::Policy> cfr_policy  = solver.AveragePolicy();
  open_spiel::UniformPolicy               rand_policy;

  const int np = game->NumPlayers();
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));
  const int eval_games = absl::GetFlag(FLAGS_num_eval_games);

  // ---- Match 1: CFR (P0) vs Random (all other players) ----
  {
    std::vector<const open_spiel::Policy*> policies(np, &rand_policy);
    policies[0] = cfr_policy.get();

    std::cout << "\n====================================================\n";
    std::cout << "MATCH 1: CFR (P0) vs Random opponents\n";
    std::cout << "====================================================";

    // Verbose trace of the very first game so we can see move-by-move play.
    auto stats = RunMatch(*game, policies, eval_games, rng,
                          /*verbose_first_game=*/true);
    PrintMatchStats("CFR(P0) vs Random", stats, *game);
  }

  // ---- Match 2: CFR vs CFR (self-play) ----
  {
    std::vector<const open_spiel::Policy*> policies(np, cfr_policy.get());

    std::cout << "\n====================================================\n";
    std::cout << "MATCH 2: CFR self-play (all players use CFR policy)\n";
    std::cout << "====================================================";

    auto stats = RunMatch(*game, policies, eval_games, rng,
                          /*verbose_first_game=*/false);
    PrintMatchStats("CFR self-play", stats, *game);
  }

  // ---- Match 3: Random vs Random (baseline) ----
  {
    std::vector<const open_spiel::Policy*> policies(np, &rand_policy);

    std::cout << "\n====================================================\n";
    std::cout << "MATCH 3: Random vs Random (baseline)\n";
    std::cout << "====================================================";

    auto stats = RunMatch(*game, policies, eval_games, rng,
                          /*verbose_first_game=*/false);
    PrintMatchStats("Random vs Random (baseline)", stats, *game);
  }

  std::cout << "\nDone.\n";
  return 0;
}
