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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace skull {
namespace {

void BasicSkullTests() {
  open_spiel::testing::LoadGameTest("skull");
  open_spiel::testing::ChanceOutcomesTest(*LoadGame("skull"));
  open_spiel::testing::RandomSimTest(*LoadGame("skull"), 50);
  open_spiel::testing::ResampleInfostateTest(*LoadGame("skull"), 10);
/*
  open_spiel::testing::TestPoliciesCanPlay(TabularPolicyGenerator policy_generator,
                           const Game &game, int numSims = 10);
  open_spiel::testing::TestEveryInfostateInPolicy(TabularPolicyGenerator policy_generator,
                                  const Game &game);
*/
}

} // namespace
} // namespace skull
} // namespace open_spiel

int main(int argc, char **argv) { open_spiel::skull::BasicSkullTests(); }
