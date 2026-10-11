#!/usr/bin/env bash

set -euo pipefail

ENV_SH="$(find /work -maxdepth 3 -type f -path '*/master-thesis/env.sh' -print -quit)"
test -n "$ENV_SH"
source "$ENV_SH"
cd "$ALPHA_STP"

# The original STP workspace uses Lean 4.9.0-rc1 and the DeepSeek REPL fork.
mkdir -p lean
git clone https://github.com/xinhjBrant/mathlib4.git lean/mathlib4
cd lean/mathlib4
git checkout 2f65ba7f1a9144b20c8e7358513548e317d26de1
elan toolchain install "$(cat lean-toolchain)"
sed 's/@ "deepseek"/@ "c6199a81de2a7e16cb27d6f85f56cff7043cd27f"/' \
    "$ALPHA_STP/assets/setup/lakefile.lean" > lakefile.lean
cp "$ALPHA_STP/assets/setup/miniF2F.lean" .
lake update REPL
test "$(git -C .lake/packages/REPL rev-parse HEAD)" = c6199a81de2a7e16cb27d6f85f56cff7043cd27f
cp "$ALPHA_STP/assets/setup/Main.lean" .lake/packages/REPL/REPL/Main.lean
lake build Mathlib miniF2F repl
lake exe repl < .lake/packages/REPL/test/aime_1983_p9.in
