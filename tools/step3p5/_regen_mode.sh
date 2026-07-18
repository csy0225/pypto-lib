#!/usr/bin/env bash
# Env-controlled regen switch for whole_decode_faithful_real dispatch/combine mode.
#
#   N1_DISPATCH = push | pull   (default: pull)   -- MoE dispatch collective
#   N1_COMBINE  = push | pull   (default: push)   -- MoE combine collective
#
# push/pull is chosen at CODE-GEN time (pypto compiles per run anyway, so this is
# cost-equivalent to a runtime switch but single-source + lower risk). After this
# script, decode_layer.py holds the selected mode; run the canonical test as usual.
#
# Modes:
#   pull  / push  (DEFAULT, VALIDATED argmax==303) : moe.py fixed-slot pull dispatch + push combine
#   pull  / pull                                    : + fixed-slot pull combine
#   push  / push  (GOODKEEP baseline)               : original push dispatch + push combine
#   push  / pull   -> unsupported (rejected)        : no use case
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> pypto-lib root

DISPATCH="${N1_DISPATCH:-pull}"
COMBINE="${N1_COMBINE:-push}"
GEN=tools/step3p5/_gen_faithful_real.py
DL=models/step3p5/decode_layer.py
FULLPULL_GEN=tools/step3p5/_gen_faithful_real.py.FULLPULL_20260715_054234
PREPULL_GEN=tools/step3p5/_gen_faithful_real.py.bak.pre_pulldispatch_20260715_123346

echo "[regen_mode] DISPATCH=$DISPATCH COMBINE=$COMBINE"

if [ "$DISPATCH" = "push" ] && [ "$COMBINE" = "push" ]; then
    # GOODKEEP push+push: use the pre-pull (push-baseline) generator as-is.
    cp "$PREPULL_GEN" "$GEN"
elif [ "$DISPATCH" = "pull" ]; then
    # FULLPULL generator + fixed-slot dispatch patch (-> pull dispatch + push combine).
    cp "$FULLPULL_GEN" "$GEN"
    python tools/step3p5/_patch_moepy_dispatch.py
    if [ "$COMBINE" = "pull" ]; then
        python tools/step3p5/_patch_combine_pull.py      # + fixed-slot pull combine
    fi
else
    echo "[regen_mode] ERROR: unsupported combo DISPATCH=$DISPATCH COMBINE=$COMBINE (push-dispatch requires push-combine)" >&2
    exit 2
fi

python -m py_compile "$GEN"
python tools/step3p5/_strip_real_builder.py
python "$GEN"
find models/step3p5 -name '*.pyc' -delete 2>/dev/null || true
python -m py_compile "$DL"

DP=$(grep -c 'def _dispatch_pull' "$DL")
SR=$(sed -n '24000,$p' "$DL" | grep -c 'self._stage_routed_src(' || true)
echo "[regen_mode] DONE: decode_layer _dispatch_pull-defs=$DP real-builder _stage_routed_src-calls=$SR"
echo "[regen_mode] mode = DISPATCH:$DISPATCH COMBINE:$COMBINE  (now run the canonical test)"
