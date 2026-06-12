// mccfr_skull_eval.cc
//
// Trains an Outcome-Sampling MCCFR policy on Skull with configurable player
// count, serializes a checkpoint, then evaluates by sampling directly from
// the solver's info-state table (no Policy object materialized).
//
// Link against:
//   open_spiel, algorithms_cfr, algorithms_outcome_sampling_mccfr

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h" // CFRInfoStateValuesTable
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, game, "skull(players=3,handsize=4,winsneeded=1)",
          "Game string, e.g. skull(players=2) or skull(players=4).");
ABSL_FLAG(std::string, file_prefix, "/tmp",
          "Directory to write the serialised average policy into.");
ABSL_FLAG(int, seed, 39827891, "Master RNG seed.");
ABSL_FLAG(int, num_iters, 5000000, "Outcome-Sampling MCCFR iterations.");
ABSL_FLAG(double, epsilon, 0.6,
          "Exploration epsilon (higher = more uniform sampling).");
ABSL_FLAG(int, num_eval_games, 100,
          "Self-play games to run for win-rate evaluation.");

// ---------------------------------------------------------------------------
// Action sampling
// ---------------------------------------------------------------------------
open_spiel::Action
SampleAction(const open_spiel::ActionsAndProbs &actions_and_probs, double z) {
  double cumulative = 0.0;
  for (const auto &[action, prob] : actions_and_probs) {
    cumulative += prob;
    if (z < cumulative)
      return action;
  }
  return actions_and_probs.back().first;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, false);
  absl::ParseCommandLine(argc, argv);

  // ---- Load game ----
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  const int np = game->NumPlayers();
  std::cerr << "Loaded: " << game->GetType().long_name << "  players=" << np
            << "\n";

  // ---- Build solver ----
  // RunIteration() uses the solver's own internal RNG; we seed it here.
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));
  absl::uniform_int_distribution<int> dist;
  const int solver_seed = dist(rng);

  open_spiel::algorithms::OutcomeSamplingMCCFRSolver solver(
      *game, absl::GetFlag(FLAGS_epsilon), solver_seed);

  // ---- Evaluate ----
  const auto &table = solver.InfoStateValuesTable();

  const double epsilon = 0.05;

  auto SampleFromTable = [&table,
                          epsilon](const open_spiel::State &state,
                                   open_spiel::Player player,
                                   std::mt19937 &rng) -> open_spiel::Action {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    const std::vector<open_spiel::Action> legal = state.LegalActions();

    if (d(rng) < epsilon) {
      std::uniform_int_distribution<int> pick(
          0, static_cast<int>(legal.size()) - 1);
      return legal[pick(rng)];
    }

    const std::string key = state.InformationStateString(player);
    auto it = table.find(key);

    if (it == table.end() || it->second.current_policy.empty()) {
      std::uniform_int_distribution<int> pick(
          0, static_cast<int>(legal.size()) - 1);
      return legal[pick(rng)];
    }

    const auto &policy = it->second.current_policy;
    double z = d(rng);
    double cumulative = 0.0;
    for (int a = 0; a < static_cast<int>(legal.size()); ++a) {
      cumulative += policy[a];
      if (z < cumulative)
        return legal[a];
    }
    return legal.back();
  };

  auto UniformAction = [](const open_spiel::State &state,
                          std::mt19937 &rng) -> open_spiel::Action {
    const std::vector<open_spiel::Action> legal = state.LegalActions();
    std::uniform_int_distribution<int> pick(0,
                                            static_cast<int>(legal.size()) - 1);
    return legal[pick(rng)];
  };

  // Move the RNG outside so it does not reset every game.
  std::mt19937 eval_rng(absl::GetFlag(FLAGS_seed) ^ 0xDEADBEEF);

  using PolicyFn = std::function<open_spiel::Action(
      const open_spiel::State &, open_spiel::Player, std::mt19937 &)>;

  auto PlayMatch = [&](const PolicyFn &policy,
                       bool verbose) -> std::vector<double> {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    if (verbose)
      std::cout << "\n=== Game Trace ===\n" << state->ToString() << "\n";

    while (!state->IsTerminal()) {
      open_spiel::Action action;
      if (state->IsChanceNode()) {
        action = SampleAction(state->ChanceOutcomes(), d(eval_rng));
      } else {
        action = policy(*state, state->CurrentPlayer(), eval_rng);
        if (verbose) {
          std::cout << "  [P" << state->CurrentPlayer() << "] "
                    << state->ActionToString(state->CurrentPlayer(), action)
                    << "\n";
        }
      }
      state->ApplyAction(action);
      if (verbose && !state->IsTerminal())
        std::cout << state->ToString() << "\n";
    }

    if (verbose) {
      std::cout << "Returns:";
      for (int p = 0; p < np; ++p)
        std::cout << "  P" << p << "=" << state->Returns()[p];
      std::cout << "\n";
    }
    return state->Returns();
  };

  auto TablePolicy = [&](const open_spiel::State &state,
                         open_spiel::Player player,
                         std::mt19937 &rng) -> open_spiel::Action {
    return SampleFromTable(state, player, rng);
  };

  auto MakeVsUniformPolicy = [&](int trained_player) -> PolicyFn {
    return [&, trained_player](const open_spiel::State &state,
                               open_spiel::Player player,
                               std::mt19937 &rng) -> open_spiel::Action {
      if (player == trained_player)
        return SampleFromTable(state, player, rng);

      return UniformAction(state, rng);
    };
  };

  auto RunEval = [&](const std::string &label, const PolicyFn &policy,
                     int eval_games) {
    std::vector<double> total(np, 0.0);
    std::vector<int> wins(np, 0);

    for (int g = 0; g < eval_games; ++g) {
      auto ret = PlayMatch(policy, /*verbose=*/false);
      double best = *std::max_element(ret.begin(), ret.end());
      for (int p = 0; p < np; ++p) {
        total[p] += ret[p];
        if (ret[p] == best)
          wins[p]++;
      }
    }

    std::cout << "\n--- " << label << " (" << eval_games << " games) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int p = 0; p < np; ++p) {
      std::cout << "  P" << p << "  avg_return=" << std::setw(7)
                << total[p] / eval_games << "  wins=" << wins[p] << " ("
                << std::setprecision(1) << 100.0 * wins[p] / eval_games
                << "%)\n";
      std::cout << std::fixed << std::setprecision(3);
    }
  };
  PlayMatch(TablePolicy, /*verbose=*/true);

  const int eval_games = 100;
  const int report_every_eval = 10; // every 10th report checkpoint
  int report_count = 0;

  // ---- Train ----
  const int num_iters = absl::GetFlag(FLAGS_num_iters);
  const int report_every = 10000;
  const double starting_epsilon = absl::GetFlag(FLAGS_epsilon);

  std::cerr << "Training MCCFR for " << num_iters
            << " iterations  (epsilon=" << absl::GetFlag(FLAGS_epsilon)
            << ")...\n";
  uint size = 1;
  uint last_size = 1;
  for (int i = 0; i < num_iters; ++i) {
    solver.RunIteration();
    if (report_every > 0 && (i % report_every == 0 || i == num_iters - 1)) {
      last_size = size;
      size = solver.InfoStateValuesTable().size();
      double diff = size - last_size;
      double percent_diff = (diff / double(last_size));

      // Dynamic Epsilon Scaling Logic
      double current_epsilon = starting_epsilon;

      if (percent_diff < 0.00005) {
        break;
      } else if (percent_diff < 0.05) {
        double t = (percent_diff - 0.00005) / (0.05 - 0.00005);
        t = std::clamp(t, 0.0, 0.98);
        current_epsilon =
            0.01 + t * (starting_epsilon - 0.01);
        solver.SetEpsilon(current_epsilon);
      }

      std::cerr << "iter=" << std::setw(8) << i << " table_size=" << size
                << " abs_diff=" << diff << " rel_diff=" << percent_diff
                << " current_eps=" << current_epsilon << "\n";

      ++report_count;
      if (report_count % report_every_eval == 0 || i == num_iters - 1) {
        RunEval("MCCFR self-play", TablePolicy, eval_games);
        for (int seat = 0; seat < np; ++seat) {
          auto policy = MakeVsUniformPolicy(seat);

          RunEval("MCCFR vs uniform RNG (trained seat P" +
                      std::to_string(seat) + ")",
                  policy, eval_games);
        }
      }
    }
  }
}
