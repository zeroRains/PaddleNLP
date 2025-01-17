# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddlenlp_ops as _ops


def append_attention(
    qkv,
    key_cache,
    value_cache,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    padding_offsets,
    cum_offsets,
    block_tables,
    encoder_batch_ids,
    encoder_tile_ids_per_batch,
    encoder_num_blocks,
    kv_batch_ids,
    kv_tile_ids_per_batch,
    kv_num_blocks,
    decoder_batch_ids,
    decoder_tile_ids_per_batch,
    decoder_num_blocks,
    max_enc_len_this_time,
    max_dec_len_this_time,
    max_len_kv,
    rotary_embs,
    attn_mask,
    qkv_bias,
    qkv_out_scales,
    cache_k_quant_scales,
    cache_v_quant_scales,
    cache_k_dequant_scales,
    cache_v_dequant_scales,
    cache_k_zp,
    cache_v_zp,
    out_linear_shifts,
    out_linear_smooths,
    compute_type,
    cache_quant_type,
    use_neox_rotary_style,
    max_input_length,
    quant_max_bound,
    quant_min_bound,
    out_linear_in_scale,
    speculate_max_draft_token_num,
    causal,
    speculate_decoder,
):
    return _ops.append_attention(
        qkv,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        block_tables,
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        kv_batch_ids,
        kv_tile_ids_per_batch,
        kv_num_blocks,
        decoder_batch_ids,
        decoder_tile_ids_per_batch,
        decoder_num_blocks,
        max_enc_len_this_time,
        max_dec_len_this_time,
        max_len_kv,
        rotary_embs,
        attn_mask,
        qkv_bias,
        qkv_out_scales,
        cache_k_quant_scales,
        cache_v_quant_scales,
        cache_k_dequant_scales,
        cache_v_dequant_scales,
        cache_k_zp,
        cache_v_zp,
        out_linear_shifts,
        out_linear_smooths,
        compute_type,
        cache_quant_type,
        use_neox_rotary_style,
        max_input_length,
        quant_max_bound,
        quant_min_bound,
        out_linear_in_scale,
        speculate_max_draft_token_num,
        causal,
        speculate_decoder,
    )


def avx_weight_only(x, weight, alog, trans):
    return _ops.avx_weight_only(x, weight, alog, trans)


def dequant_int8(intput, out_scale, dtype):
    return _ops.dequant_int8(intput, out_scale, dtype)


def encode_rotary_qk(q, kv, rotary_emb, seq_lens, rotary_emb_dims, use_neox):
    return _ops.encode_rotary_qk(q, kv, rotary_emb, seq_lens, rotary_emb_dims, use_neox)


def flash_attn_bwd(q, k, v, out, softmax_lse, seed_offset, attn_mask, out_grad, dropout, causal):
    return _ops.flash_attn_bwd(q, k, v, out, softmax_lse, seed_offset, attn_mask, out_grad, dropout, causal)


def cutlass_fp8_fp8_fp8_dual_gemm_fused(
    x, y0, y1, bias0, bias1, transpose_x, transpose_y, scale0, scale1, scale_out, act
):
    return _ops.cutlass_fp8_fp8_fp8_dual_gemm_fused(
        x, y0, y1, bias0, bias1, transpose_x, transpose_y, scale0, scale1, scale_out, act
    )


def cutlass_fp8_fp8_half_gemm_fused(x, y, bias, transpose_x, transpose_y, scale, output_type, act):
    return _ops.cutlass_fp8_fp8_half_gemm_fused(x, y, bias, transpose_x, transpose_y, scale, output_type, act)


def fused_get_rotary_embedding(input_ids, position_ids, head_dim_shape_tensor, prompt_num, theta, use_neox):
    return _ops.fused_get_rotary_embedding(input_ids, position_ids, head_dim_shape_tensor, prompt_num, theta, use_neox)


def gemm_dequant(x, y, scale, out_dtype):
    return _ops.gemm_dequant(x, y, scale, out_dtype)


def get_block_shape_and_split_kv_block(
    seq_lens_encoder,
    seq_lens_decoder,
    max_enc_len_this_time,
    max_dec_len_this_time,
    seq_lens_this_time,
    cum_offsets,
    group_size,
    block_size,
    decoder_step_token_num,
):
    return _ops.get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        max_enc_len_this_time,
        max_dec_len_this_time,
        seq_lens_this_time,
        cum_offsets,
        group_size,
        block_size,
        decoder_step_token_num,
    )


def get_output(x, rank_id, wait_flag):
    return _ops.get_output(x, rank_id, wait_flag)


def get_padding_offset(input_ids, cum_offsets, token_num, seq_len):
    return _ops.get_padding_offset(input_ids, cum_offsets, token_num, seq_len)


def get_padding_offset_v2(input_ids, cum_offsets, token_num, seq_len, draft_tokens, seq_lens_encoder):
    return _ops.get_padding_offset_v2(input_ids, cum_offsets, token_num, seq_len, draft_tokens, seq_lens_encoder)


def get_token_penalty_multi_scores(
    pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id
):
    return _ops.get_token_penalty_multi_scores(
        pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id
    )


def ngram_match(
    input_ids,
    input_ids_len,
    pre_ids,
    step_idx,
    draft_token_num,
    draft_tokens,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    real_batch_size,
    max_ngram_size,
    max_draft_tokens,
):
    return _ops.ngram_match(
        input_ids,
        input_ids_len,
        pre_ids,
        step_idx,
        draft_token_num,
        draft_tokens,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        real_batch_size,
        max_ngram_size,
        max_draft_tokens,
    )


def qkv_transpose_split(qkv, padding_offset, seq_lens, input_ids, num_head, head_size):
    return _ops.qkv_transpose_split(qkv, padding_offset, seq_lens, input_ids, num_head, head_size)


def quant_int8(intput, shift, smooth, scale, round_type, max_bound, min_bound):
    return _ops.quant_int8(intput, shift, smooth, scale, round_type, max_bound, min_bound)


def rebuild_padding(tmp_out, padding_offset, seq_lens, input_ids):
    return _ops.rebuild_padding(tmp_out, padding_offset, seq_lens, input_ids)


def rebuild_padding_v2(
    tmp_out, cum_offsets, seq_lens_decoder, seq_lens_encoder, output_padding_offset, max_input_length
):
    return _ops.rebuild_padding_v2(
        tmp_out, cum_offsets, seq_lens_decoder, seq_lens_encoder, output_padding_offset, max_input_length
    )


def save_output(x, not_need_stop, rank_id):
    return _ops.save_output(x, not_need_stop, rank_id)


def save_with_output(x, batch_idx, step_idx, file_path, rank_id):
    return _ops.save_with_output(x, batch_idx, step_idx, file_path, rank_id)


def set_preids_token_penalty_multi_scores(
    pre_ids,
    input_ids,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    stop_flags,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    temperatures,
    bad_tokens,
    cur_len,
    min_len,
    eos_token_id,
):
    return _ops.set_preids_token_penalty_multi_scores(
        pre_ids,
        input_ids,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        stop_flags,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
    )


def set_stop_value_multi_ends(topk_ids, stop_flags, end_ids, mode):
    return _ops.set_stop_value_multi_ends(topk_ids, stop_flags, end_ids, mode)


def set_value_by_flags_and_idx(pre_ids_all, pre_ids_now, step_idx, stop_flags):
    return _ops.set_value_by_flags_and_idx(pre_ids_all, pre_ids_now, step_idx, stop_flags)


def speculate_get_output(x, rank_id, wait_flag):
    return _ops.speculate_get_output(x, rank_id, wait_flag)


def speculate_get_output_padding_offset(output_cum_offsets_tmp, out_token_num, seq_lens_output, max_seq_len):
    return _ops.speculate_get_output_padding_offset(
        output_cum_offsets_tmp, out_token_num, seq_lens_output, max_seq_len
    )


def speculate_get_seq_lens_output(seq_lens_this_time, seq_lens_encoder, seq_lens_decoder):
    return _ops.speculate_get_seq_lens_output(seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)


def speculate_get_token_penalty_multi_scores(
    pre_ids,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    temperatures,
    bad_tokens,
    cur_len,
    min_len,
    eos_token_id,
    seq_lens_this_time,
    output_padding_offset,
    output_cum_offsets,
    max_seq_len,
):
    return _ops.speculate_get_token_penalty_multi_scores(
        pre_ids,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id,
        seq_lens_this_time,
        output_padding_offset,
        output_cum_offsets,
        max_seq_len,
    )


def speculate_save_output(accept_tokens, accept_num, not_need_stop, rank_id):
    return _ops.speculate_save_output(accept_tokens, accept_num, not_need_stop, rank_id)


def speculate_set_value_by_flags_and_idx(
    pre_ids_all,
    accept_tokens,
    accept_num,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
):
    return _ops.speculate_set_value_by_flags_and_idx(
        pre_ids_all,
        accept_tokens,
        accept_num,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
    )


def speculate_verify_and_update(
    accept_tokens,
    accept_num,
    step_idx,
    seq_lens_encoder,
    seq_lens_decoder,
    stop_flags,
    not_need_stop,
    draft_tokens,
    seq_lens_this_time,
    verify_tokens,
    verify_scores,
    max_dec_len,
    end_tokens,
    is_block_step,
    output_cum_offsets,
    actual_candidate_len,
    actual_draft_token_nums,
    topp,
    max_seq_len,
    verify_window,
    enable_topp,
):
    return _ops.speculate_verify_and_update(
        accept_tokens,
        accept_num,
        step_idx,
        seq_lens_encoder,
        seq_lens_decoder,
        stop_flags,
        not_need_stop,
        draft_tokens,
        seq_lens_this_time,
        verify_tokens,
        verify_scores,
        max_dec_len,
        end_tokens,
        is_block_step,
        output_cum_offsets,
        actual_candidate_len,
        actual_draft_token_nums,
        topp,
        max_seq_len,
        verify_window,
        enable_topp,
    )


def top_p_candidates(probs, top_p, output_padding_offset, candidates_len, max_seq_len):
    return _ops.top_p_candidates(probs, top_p, output_padding_offset, candidates_len, max_seq_len)


def top_p_sampling_reject(probs, top_p, seed):
    return _ops.top_p_sampling_reject(probs, top_p, seed)


def transpose_remove_padding(input, seq_lens, padding_offset):
    return _ops.transpose_remove_padding(input, seq_lens, padding_offset)


def update_inputs_v2(
    stop_flags,
    step_idx,
    not_need_stop,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    max_dec_len,
    input_ids,
    stop_nums,
    next_tokens,
    is_block_step,
    end_ids,
    kwargs_next_tokens,
):
    return _ops.update_inputs_v2(
        stop_flags,
        step_idx,
        not_need_stop,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        max_dec_len,
        input_ids,
        stop_nums,
        next_tokens,
        is_block_step,
        end_ids,
        kwargs_next_tokens,
    )


def write_cache_kv(input_k, input_v, cache_kv, sequence_lengths):
    return _ops.write_cache_kv(input_k, input_v, cache_kv, sequence_lengths)


def xft_greedy_search(probs):
    return _ops.xft_greedy_search(probs)


def xft_transformer(
    input,
    ln1Gamma,
    qkvWeight,
    attnOutWeight,
    ln2Gamma,
    gateWeight,
    upWeight,
    downWeight,
    pastSeqLen,
    currentSeqLen,
    step,
    hiddensize,
    totalLayer,
    computeType,
    cacheDtype,
    activation,
    normType,
    attHeadDim,
    attHeadNum,
    kvHeadNum,
    maxPositions,
    maxPosEmbed,
    intermediateSize,
):
    return _ops.xft_transformer(
        input,
        ln1Gamma,
        qkvWeight,
        attnOutWeight,
        ln2Gamma,
        gateWeight,
        upWeight,
        downWeight,
        pastSeqLen,
        currentSeqLen,
        step,
        hiddensize,
        totalLayer,
        computeType,
        cacheDtype,
        activation,
        normType,
        attHeadDim,
        attHeadNum,
        kvHeadNum,
        maxPositions,
        maxPosEmbed,
        intermediateSize,
    )
