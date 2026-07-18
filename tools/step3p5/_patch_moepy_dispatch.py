#!/usr/bin/env python3
"""Patch the FULLPULL generator -> moe.py fixed-slot pull dispatch.

Applies coordinated edits to tools/step3p5/_gen_faithful_real.py with per-edit
asserts (fails loudly on anchor mismatch). Dispatch only; combine left as-is.
"""
from pathlib import Path

G = Path("tools/step3p5/_gen_faithful_real.py")
t = G.read_text()


def rep(old, new, n=1):
    global t
    c = t.count(old)
    assert c == n, f"[FAIL] expected {n} got {c} for anchor:\n{old[:120]!r}"
    t = t.replace(old, new, n)


# R1 — pack base: prefix-sum -> fixed dst-block slot r*n_routes_per_rank
rep(
    "                rank_off = pl.read(send_offsets_rank, [r])\n",
    "                rank_off = pl.cast(r * n_routes_per_rank, pl.INT32)  # moe.py fixed dst-block base\n",
)

# R2-R4 — send window sigs 128 -> local_recv_max (fixed-slot needs n_ranks blocks)
rep("[[n_routes_per_rank, HIDDEN], pl.INT8]", "[[local_recv_max, HIDDEN], pl.INT8]", 5)
rep("[[n_routes_per_rank, 8], pl.FP32]", "[[local_recv_max, 8], pl.FP32]", 5)
rep("[[n_routes_per_rank, idx_pad], pl.INT32]", "[[local_recv_max, idx_pad], pl.INT32]", 3)

# R5 — host_orch send buffer byte sizes
rep('("send_x_buf", "n_routes_per_rank * HIDDEN * 1")', '("send_x_buf", "local_recv_max * HIDDEN * 1")')
rep('("send_scale_buf", "n_routes_per_rank * 8 * 4")', '("send_scale_buf", "local_recv_max * 8 * 4")')
rep('("send_route_buf", "n_routes_per_rank * idx_pad * 4")', '("send_route_buf", "local_recv_max * idx_pad * 4")')

# R6 — host_orch send window shapes
rep("send_x = pld.window(send_x_buf_{sfx}, [n_routes_per_rank, HIDDEN]", "send_x = pld.window(send_x_buf_{sfx}, [local_recv_max, HIDDEN]")
rep("send_scale = pld.window(send_scale_buf_{sfx}, [n_routes_per_rank, 8]", "send_scale = pld.window(send_scale_buf_{sfx}, [local_recv_max, 8]")
rep("send_route = pld.window(send_route_buf_{sfx}, [n_routes_per_rank, idx_pad]", "send_route = pld.window(send_route_buf_{sfx}, [local_recv_max, idx_pad]")

# R7 — _dispatch_pull rendezvous barrier: Set -> AtomicAdd (align moe.py/ep_all_to_all)
rep(
    "                    pld.system.notify(\n"
    "                        target=pack_done_sig, peer=peer,\n"
    "                        offsets=[my_rank, 0], value=1,\n"
    "                        op=pld.NotifyOp.Set,\n"
    "                    )",
    "                    pld.system.notify(\n"
    "                        target=pack_done_sig, peer=peer,\n"
    "                        offsets=[my_rank, 0], value=1,\n"
    "                        op=pld.NotifyOp.AtomicAdd,\n"
    "                    )",
)

# R8 — gather: fused-CSR (runtime pub_counts bound + cross-rank offset) -> static
# fixed-slot peer-major pull (static pl.range(n_routes_per_rank), compound offset).
_GATHER_OLD = (
    "            recv_cursor = pl.cast(0, pl.INT32)\n"
    "            for loc_e in pl.range(n_local_experts):\n"
    "                for s in pl.range(n_ranks):\n"
    "                    off_s = pl.cast(0, pl.INT32)\n"
    "                    for d2 in pl.range(n_ranks):\n"
    "                        if d2 < my_rank:\n"
    "                            for e2 in pl.range(n_local_experts):\n"
    "                                off_s = off_s + pl.read(\n"
    "                                    pub_counts, [s * n_ranks + d2, e2],\n"
    "                                )\n"
    "                    for e3 in pl.range(n_local_experts):\n"
    "                        if e3 < loc_e:\n"
    "                            off_s = off_s + pl.read(\n"
    "                                pub_counts, [s * n_ranks + my_rank, e3],\n"
    "                            )\n"
    "                    n = pl.cast(\n"
    "                        pl.read(pub_counts, [s * n_ranks + my_rank, loc_e]),\n"
    "                        pl.INDEX,\n"
    "                    )\n"
    "                    off_i = pl.cast(off_s, pl.INDEX)\n"
    "                    for row in pl.range(n):\n"
    "                        dst_row = pl.cast(recv_cursor, pl.INDEX) + row\n"
    "                        xt = pld.tile.remote_load(\n"
    "                            send_x, peer=s,\n"
    "                            offsets=[off_i + row, 0], shape=[1, HIDDEN],\n"
    "                        )\n"
    "                        pl.store(xt, [dst_row, 0], recv_x)\n"
    "                        st = pld.tile.remote_load(\n"
    "                            send_scale, peer=s,\n"
    "                            offsets=[off_i + row, 0], shape=[1, 8],\n"
    "                        )\n"
    "                        pl.store(st, [dst_row, 0], recv_scale)\n"
    "                        rt = pld.tile.remote_load(\n"
    "                            send_route, peer=s,\n"
    "                            offsets=[off_i + row, 0], shape=[1, idx_pad],\n"
    "                        )\n"
    "                        pl.store(rt, [dst_row, 0], recv_r_route)\n"
    "                    recv_cursor = recv_cursor + pl.cast(n, pl.INT32)\n"
    "            return local_expert_offset, local_expert_count\n"
)
_GATHER_NEW = (
    "            # moe.py ep_all_to_all static fixed-slot pull -> recv_x PEER-MAJOR\n"
    "            # (peer block at peer*n_routes_per_rank). Self block copied locally;\n"
    "            # peer blocks pulled via remote_load at compound-scalar my_rank*MAX.\n"
    "            # NO cross-rank offset, NO runtime pub_counts bound in the pull loop.\n"
    "            # _dispatch_stage re-packs peer-major -> expert-major.\n"
    "            _self_base = pl.cast(my_rank * n_routes_per_rank, pl.INDEX)\n"
    "            for r in pl.range(n_routes_per_rank):\n"
    "                sxt = pl.load(send_x, [_self_base + r, 0], [1, HIDDEN])\n"
    "                pl.store(sxt, [_self_base + r, 0], recv_x)\n"
    "                sst = pl.load(send_scale, [_self_base + r, 0], [1, 8])\n"
    "                pl.store(sst, [_self_base + r, 0], recv_scale)\n"
    "                srt = pl.load(send_route, [_self_base + r, 0], [1, idx_pad])\n"
    "                pl.store(srt, [_self_base + r, 0], recv_r_route)\n"
    "            for peer in pl.range(n_ranks):\n"
    "                if peer != my_rank:\n"
    "                    _peer_base = pl.cast(peer * n_routes_per_rank, pl.INDEX)\n"
    "                    for r in pl.range(n_routes_per_rank):\n"
    "                        xt = pld.tile.remote_load(\n"
    "                            send_x, peer=peer,\n"
    "                            offsets=[_self_base + r, 0], shape=[1, HIDDEN],\n"
    "                        )\n"
    "                        pl.store(xt, [_peer_base + r, 0], recv_x)\n"
    "                        st = pld.tile.remote_load(\n"
    "                            send_scale, peer=peer,\n"
    "                            offsets=[_self_base + r, 0], shape=[1, 8],\n"
    "                        )\n"
    "                        pl.store(st, [_peer_base + r, 0], recv_scale)\n"
    "                        rt = pld.tile.remote_load(\n"
    "                            send_route, peer=peer,\n"
    "                            offsets=[_self_base + r, 0], shape=[1, idx_pad],\n"
    "                        )\n"
    "                        pl.store(rt, [_peer_base + r, 0], recv_r_route)\n"
    "            return local_expert_offset, local_expert_count\n"
)
rep(_GATHER_OLD, _GATHER_NEW)

# R9 — dispatch_step: pass pub_counts to _dispatch_stage (edit the _scall_new
# STRING LITERAL in the generator source; \n is literal backslash-n there).
rep(
    '                local_routed_x_out, local_routed_x_scale_out, recv_r_route_out, my_rank,\\n"',
    '                local_routed_x_out, local_routed_x_scale_out, recv_r_route_out,\\n"\n'
    '                      "                pub_counts, my_rank,\\n"',
)

# R11 — _stage_edits: add pub_counts to _dispatch_stage sig + straight-copy -> re-pack
_STAGE_TAIL_OLD = (
    "        _ret = \"            return local_routed_x_out, recv_r_route_out\\n\"\n"
    "        _scale_stage = (\n"
    "            \"            for _sr in pl.range(local_recv_max):\\n\"\n"
    "            \"                pl.write(\\n\"\n"
    "            \"                    local_routed_x_scale_out, [0, _sr],\\n\"\n"
    "            \"                    pl.read(recv_scale, [_sr, 0]),\\n\"\n"
    "            \"                )\\n\"\n"
    "            \"            return local_routed_x_out, local_routed_x_scale_out, recv_r_route_out\\n\")\n"
    "        assert seg.count(_ret) == 1, (\"stage ret\", seg.count(_ret))\n"
    "        return seg.replace(_ret, _scale_stage, 1)\n"
)
_STAGE_TAIL_NEW = (
    "        _sig = (\"            recv_r_route_out: pl.Out[pl.Tensor[[local_recv_max], pl.INT32]],\\n\"\n"
    "                \"            my_rank: pl.Scalar[pl.INT32],\\n\")\n"
    "        _sig_new = (\"            recv_r_route_out: pl.Out[pl.Tensor[[local_recv_max], pl.INT32]],\\n\"\n"
    "                    \"            pub_counts: pld.DistributedTensor[[n_ranks * n_ranks, n_local_experts_pad], pl.INT32],\\n\"\n"
    "                    \"            my_rank: pl.Scalar[pl.INT32],\\n\")\n"
    "        assert seg.count(_sig) == 1, (\"stage sig\", seg.count(_sig))\n"
    "        seg = seg.replace(_sig, _sig_new, 1)\n"
    "        _body = (\"            for row in pl.range(0, local_recv_max, stage_rows):\\n\"\n"
    "                 \"                tile = pl.load(recv_x, [row, 0], [stage_rows, HIDDEN])\\n\"\n"
    "                 \"                pl.store(tile, [row, 0], local_routed_x_out)\\n\"\n"
    "                 \"            for e in pl.range(n_local_experts):\\n\"\n"
    "                 \"                off = pl.cast(pl.read(local_expert_offset, [e]), pl.INDEX)\\n\"\n"
    "                 \"                n = pl.cast(pl.read(local_expert_count, [e]), pl.INDEX)\\n\"\n"
    "                 \"                for s in pl.range(n):\\n\"\n"
    "                 \"                    pl.write(\\n\"\n"
    "                 \"                        recv_r_route_out, [off + s],\\n\"\n"
    "                 \"                        pl.read(recv_r_route, [off + s, 0]),\\n\"\n"
    "                 \"                    )\\n\"\n"
    "                 \"            return local_routed_x_out, recv_r_route_out\\n\")\n"
    "        _body_new = (\"            running = pl.cast(0, pl.INT32)\\n\"\n"
    "                     \"            for e in pl.range(n_local_experts):\\n\"\n"
    "                     \"                for src in pl.range(n_ranks):\\n\"\n"
    "                     \"                    rn = pl.cast(pl.read(pub_counts, [src * n_ranks + my_rank, e]), pl.INDEX)\\n\"\n"
    "                     \"                    src_base = pl.cast(src * n_routes_per_rank, pl.INDEX)\\n\"\n"
    "                     \"                    src_e_off = pl.cast(0, pl.INT32)\\n\"\n"
    "                     \"                    for prev_e in pl.range(n_local_experts):\\n\"\n"
    "                     \"                        if prev_e < e:\\n\"\n"
    "                     \"                            src_e_off = src_e_off + pl.read(pub_counts, [src * n_ranks + my_rank, prev_e])\\n\"\n"
    "                     \"                    for row in pl.range(rn):\\n\"\n"
    "                     \"                        src_row = src_base + pl.cast(src_e_off, pl.INDEX) + row\\n\"\n"
    "                     \"                        dst_row = pl.cast(running, pl.INDEX) + row\\n\"\n"
    "                     \"                        tile = pl.load(recv_x, [src_row, 0], [1, HIDDEN])\\n\"\n"
    "                     \"                        pl.store(tile, [dst_row, 0], local_routed_x_out)\\n\"\n"
    "                     \"                        pl.write(local_routed_x_scale_out, [0, dst_row], pl.read(recv_scale, [src_row, 0]))\\n\"\n"
    "                     \"                        pl.write(recv_r_route_out, [dst_row], pl.read(recv_r_route, [src_row, 0]))\\n\"\n"
    "                     \"                    running = running + pl.cast(rn, pl.INT32)\\n\"\n"
    "                     \"            return local_routed_x_out, local_routed_x_scale_out, recv_r_route_out\\n\")\n"
    "        assert seg.count(_body) == 1, (\"stage body\", seg.count(_body))\n"
    "        return seg.replace(_body, _body_new, 1)\n"
)
rep(_STAGE_TAIL_OLD, _STAGE_TAIL_NEW)

# R10 — REVERT combine to PUSH (device-proven moe.py combo = pull-dispatch +
# push-combine). Make the combine_step body-swap a no-op so combine keeps the
# base _push_routed_y_to_sources push; the FRESH_COMBINE_PULL methods stay
# defined-but-dead and the extra sig params (expert_indices/routed_src_buf) are
# harmless unused. Combine-push = jitter only (never flips greedy argmax), not stall.
rep(
    "    head_and_setA = head_and_setA.replace(_cs_body, _cs_body_new, 1)\n",
    "    head_and_setA = head_and_setA.replace(_cs_body, _cs_body, 1)  # REVERT: combine stays PUSH (moe.py combo)\n",
)

G.write_text(t)
print("PATCH_OK moepy fixed-slot dispatch applied")
