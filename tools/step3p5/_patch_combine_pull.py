#!/usr/bin/env python3
"""Patch combine -> fixed-slot PULL (symmetric reverse of the fixed-slot dispatch).

Runs AFTER _patch_moepy_dispatch.py on the active generator. Rewrites the two
FRESH_COMBINE_PULL methods to the fixed-slot scheme (compound-scalar offset,
AtomicAdd barrier, static bounds; NO cross-rank pub_counts read in the remote_load
loop) and re-enables the combine-pull splice (undo the R10 push-revert).
Per-edit asserts fail loudly on anchor mismatch.
"""
from pathlib import Path

G = Path("tools/step3p5/_gen_faithful_real.py")
t = G.read_text()


def rep(old, new, n=1):
    global t
    c = t.count(old)
    assert c == n, f"[FAIL] expected {n} got {c} for anchor:\n{old[:120]!r}"
    t = t.replace(old, new, n)


# C1 — _stage_routed_src sig: add pub_counts + my_rank
rep(
    "        def _stage_routed_src(\n"
    "            self,\n"
    "            local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],\n"
    "            routed_src_buf: pld.DistributedTensor[\n"
    "                [local_recv_max, HIDDEN], pl.BF16\n"
    "            ],\n"
    "        ):\n",
    "        def _stage_routed_src(  # noqa: PLR0913\n"
    "            self,\n"
    "            local_routed_y: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],\n"
    "            pub_counts: pld.DistributedTensor[\n"
    "                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32\n"
    "            ],\n"
    "            routed_src_buf: pld.DistributedTensor[\n"
    "                [local_recv_max, HIDDEN], pl.BF16\n"
    "            ],\n"
    "            my_rank: pl.Scalar[pl.INT32],\n"
    "        ):\n",
)

# C2 — _stage_routed_src body: straight-copy -> reverse re-pack (expert-major
# local_routed_y -> PEER-MAJOR routed_src_buf[src*MAX + within], via pub_counts).
rep(
    "            # Combine task 1 (PULL): copy the expert holder's routed output into\n"
    "            # its OWN peer-readable window (local writes). The InCore task boundary\n"
    "            # drains these before the pull rendezvous, so peers read landed data.\n"
    "            for row in pl.range(0, local_recv_max, stage_rows):\n"
    "                tile = pl.load(local_routed_y, [row, 0], [stage_rows, HIDDEN])\n"
    "                pl.store(tile, [row, 0], routed_src_buf)\n",
    "            # Combine task 1 (PULL, fixed-slot): reverse of _dispatch_stage re-pack.\n"
    "            # expert-major local_routed_y -> PEER-MAJOR routed_src_buf[src*MAX+within]\n"
    "            # via pub_counts (LOCAL pl.load/store), so each source pulls its own\n"
    "            # block at a STATIC my_rank*MAX offset. within matches the source's\n"
    "            # within-dst pack position (src_e_off + arrival).\n"
    "            running = pl.cast(0, pl.INT32)\n"
    "            for e in pl.range(n_local_experts):\n"
    "                for src in pl.range(n_ranks):\n"
    "                    rn = pl.cast(pl.read(pub_counts, [src * n_ranks + my_rank, e]), pl.INDEX)\n"
    "                    src_base = pl.cast(src * n_routes_per_rank, pl.INDEX)\n"
    "                    src_e_off = pl.cast(0, pl.INT32)\n"
    "                    for prev_e in pl.range(n_local_experts):\n"
    "                        if prev_e < e:\n"
    "                            src_e_off = src_e_off + pl.read(pub_counts, [src * n_ranks + my_rank, prev_e])\n"
    "                    for row in pl.range(rn):\n"
    "                        dst_row = src_base + pl.cast(src_e_off, pl.INDEX) + row\n"
    "                        src_row = pl.cast(running, pl.INDEX) + row\n"
    "                        tile = pl.load(local_routed_y, [src_row, 0], [1, HIDDEN])\n"
    "                        pl.store(tile, [dst_row, 0], routed_src_buf)\n"
    "                    running = running + pl.cast(rn, pl.INT32)\n",
)

# C3 — _pull_routed_y sig: drop pub_counts (offset is local now)
rep(
    "            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],\n"
    "            pub_counts: pld.DistributedTensor[\n"
    "                [n_ranks * n_ranks, n_local_experts_pad], pl.INT32\n"
    "            ],\n"
    "            routed_y_buf: pld.DistributedTensor[\n"
    "                [n_routes_per_rank, HIDDEN], pl.BF16\n"
    "            ],\n"
    "            combine_done: pld.DistributedTensor[[n_ranks, 1], pl.INT32],\n"
    "            my_rank: pl.Scalar[pl.INT32],\n"
    "        ):\n",
    "            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],\n"
    "            routed_y_buf: pld.DistributedTensor[\n"
    "                [n_routes_per_rank, HIDDEN], pl.BF16\n"
    "            ],\n"
    "            combine_done: pld.DistributedTensor[[n_ranks, 1], pl.INT32],\n"
    "            my_rank: pl.Scalar[pl.INT32],\n"
    "        ):\n",
)

# C4 — _pull_routed_y barrier: Set -> AtomicAdd (target=combine_done unique)
rep(
    "                    pld.system.notify(\n"
    "                        target=combine_done, peer=peer,\n"
    "                        offsets=[my_rank, 0], value=1,\n"
    "                        op=pld.NotifyOp.Set,\n"
    "                    )",
    "                    pld.system.notify(\n"
    "                        target=combine_done, peer=peer,\n"
    "                        offsets=[my_rank, 0], value=1,\n"
    "                        op=pld.NotifyOp.AtomicAdd,\n"
    "                    )",
)

# C5 — _pull_routed_y body: inverse_map (variable offset) -> local within-recompute
# + compound-scalar offset my_rank*MAX + within (NO cross-rank pub_counts).
rep(
    "            inverse_map = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)\n"
    "            self._build_inverse_map(\n"
    "                expert_indices, pub_counts, inverse_map, my_rank,\n"
    "            )\n"
    "            for t in pl.range(BATCH):\n"
    "                for k in pl.range(TOPK):\n"
    "                    packed = pl.read(inverse_map, [t, k])\n"
    "                    dst = packed // pl.cast(local_recv_max, pl.INT32)\n"
    "                    dst_row = pl.cast(\n"
    "                        packed - dst * pl.cast(local_recv_max, pl.INT32),\n"
    "                        pl.INDEX,\n"
    "                    )\n"
    "                    r_route = pl.cast(t * TOPK + k, pl.INDEX)\n"
    "                    # Always remote_load (peer=dst may == my_rank; _dispatch_pull\n"
    "                    # does the same self-read and it compiled+ran on device). A\n"
    "                    # local-vs-remote device-if would give `tile` two different\n"
    "                    # Tile types (Mem.Vec vs plain) -> SSA reassign reject.\n"
    "                    tile = pld.tile.remote_load(\n"
    "                        routed_src_buf, peer=dst,\n"
    "                        offsets=[dst_row, 0], shape=[1, HIDDEN],\n"
    "                    )\n"
    "                    pl.store(tile, [r_route, 0], routed_y_buf)\n",
    "            # Recompute this source's within-dst pack position (relative, base 0\n"
    "            # per dst-block) — same cursor logic as the fixed-slot pack. Purely\n"
    "            # LOCAL (own expert_indices); no cross-rank read.\n"
    "            send_counts_bkt = pl.create_tensor([per_rank_buckets], dtype=pl.INT32)\n"
    "            send_counts_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)\n"
    "            send_offsets_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)\n"
    "            self._histogram_and_prefix_sum(\n"
    "                expert_indices, send_counts_bkt, send_counts_rank, send_offsets_rank,\n"
    "            )\n"
    "            cursor_rel = pl.create_tensor([per_rank_buckets], dtype=pl.INT32)\n"
    "            for r in pl.range(n_ranks):\n"
    "                pl.write(cursor_rel, [r * n_local_experts], pl.cast(0, pl.INT32))\n"
    "                for e in pl.range(1, n_local_experts):\n"
    "                    prev = pl.read(cursor_rel, [r * n_local_experts + e - 1])\n"
    "                    prev_cnt = pl.read(send_counts_bkt, [r * n_local_experts + e - 1])\n"
    "                    pl.write(cursor_rel, [r * n_local_experts + e], pl.cast(prev + prev_cnt, pl.INT32))\n"
    "            for t in pl.range(BATCH):\n"
    "                for k in pl.range(TOPK):\n"
    "                    eid = pl.read(expert_indices, [t, k])\n"
    "                    dst = eid // n_local_experts\n"
    "                    loc_e = eid - dst * n_local_experts\n"
    "                    bkt = dst * n_local_experts + loc_e\n"
    "                    within = pl.read(cursor_rel, [bkt])\n"
    "                    off = pl.cast(my_rank * n_routes_per_rank, pl.INDEX) + pl.cast(within, pl.INDEX)\n"
    "                    r_route = pl.cast(t * TOPK + k, pl.INDEX)\n"
    "                    tile = pld.tile.remote_load(\n"
    "                        routed_src_buf, peer=dst,\n"
    "                        offsets=[off, 0], shape=[1, HIDDEN],\n"
    "                    )\n"
    "                    pl.store(tile, [r_route, 0], routed_y_buf)\n"
    "                    pl.write(cursor_rel, [bkt], pl.cast(within + 1, pl.INT32))\n",
)

# C6 — combine_step body call: update to new sigs
rep(
    "    _cs_body_new = (\"            self._zero_routed_y_buf(routed_y_buf)\\n\"\n"
    "                    \"            self._stage_routed_src(local_routed_y, routed_src_buf)\\n\"\n"
    "                    \"            self._pull_routed_y(\\n\"\n"
    "                    \"                routed_src_buf, expert_indices, pub_counts,\\n\"\n"
    "                    \"                routed_y_buf, combine_done_sig, my_rank,\\n\"\n"
    "                    \"            )\\n\"\n",
    "    _cs_body_new = (\"            self._zero_routed_y_buf(routed_y_buf)\\n\"\n"
    "                    \"            self._stage_routed_src(local_routed_y, pub_counts, routed_src_buf, my_rank)\\n\"\n"
    "                    \"            self._pull_routed_y(\\n\"\n"
    "                    \"                routed_src_buf, expert_indices,\\n\"\n"
    "                    \"                routed_y_buf, combine_done_sig, my_rank,\\n\"\n"
    "                    \"            )\\n\"\n",
)

# C7 — re-enable combine-pull splice (undo R10 push-revert)
rep(
    "    head_and_setA = head_and_setA.replace(_cs_body, _cs_body, 1)  # REVERT: combine stays PUSH (moe.py combo)\n",
    "    head_and_setA = head_and_setA.replace(_cs_body, _cs_body_new, 1)  # combine = fixed-slot PULL\n",
)

G.write_text(t)
print("COMBINE_PULL_PATCH_OK fixed-slot combine pull applied")
