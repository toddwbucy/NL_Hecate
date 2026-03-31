//! Binary checkpoint I/O: HuggingFace safetensors format.
//!
//! Replaces serde_json text checkpoints (5GB, 60s load) with binary safetensors
//! (~1.7GB, <1s mmap load) for 433M param SwiGLU models.
//!
//! File format (spec: https://github.com/huggingface/safetensors):
//!   [8 bytes LE u64: padded header length N]
//!   [N bytes: JSON header, space-padded to 8-byte alignment]
//!     {
//!       "__metadata__": { "version": "2", "format": "nl_hecate_v2",
//!                         "created_at": "epoch:...",
//!                         "config": "<MAGConfig as JSON string>",
//!                         "build_state": "<BuildResumeState JSON | \"null\">" },
//!       "embed.weight": { "dtype": "F32", "shape": [n], "data_offsets": [s, e] },
//!       ...
//!     }
//!   [raw tensor bytes, concatenated, little-endian f32]
//!
//! Python interop (for weight inspection):
//!   from safetensors import safe_open
//!   with safe_open("checkpoint.safetensors", framework="np") as f:
//!       embed = f.get_tensor("embed.weight")  # flat [vocab*d_model] f32

use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::bf16::Bf16Storage;
use crate::model::{BuildResumeState, MAGConfig, MAGParams, MemoryLevelParams, SWAParams};
use crate::stacked_model::{BlockParams, StackedMAGParams};

// ── Save ─────────────────────────────────────────────────────────────

/// Save MAGParams + config as a safetensors binary checkpoint.
///
/// The config and optional build_state are stored as JSON strings in `__metadata__`.
/// Tensor data is raw LE f32 bytes — exact bitwise round-trip.
/// Bf16Storage fields are serialized from their fp32 master copy (source of truth).
pub fn save_safetensors(
    path: &Path,
    params: &MAGParams,
    config: &MAGConfig,
    build_state: Option<&BuildResumeState>,
) -> io::Result<()> {
    // 1. Collect tensors: (name, raw_le_f32_bytes)
    let mut tensors: Vec<(String, Vec<u8>)> = Vec::new();

    fn enc(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // SWA attention branch
    tensors.push(("embed.weight".into(), enc(&params.swa.w_embed)));
    tensors.push(("swa.w_q".into(),      enc(&params.swa.w_q)));
    tensors.push(("swa.w_k".into(),      enc(&params.swa.w_k)));
    tensors.push(("swa.w_v".into(),      enc(&params.swa.w_v)));
    tensors.push(("swa.w_o".into(),      enc(&params.swa.w_o)));
    tensors.push(("lm_head.weight".into(), enc(&params.swa.w_unembed)));

    // LayerNorm params (only save if non-trivial to save space on old models)
    if !params.swa.ln_attn_gamma.is_empty() {
        tensors.push(("ln_attn.gamma".into(), enc(&params.swa.ln_attn_gamma)));
        tensors.push(("ln_attn.beta".into(),  enc(&params.swa.ln_attn_beta)));
        tensors.push(("ln_mem.gamma".into(),   enc(&params.swa.ln_mem_gamma)));
        tensors.push(("ln_mem.beta".into(),    enc(&params.swa.ln_mem_beta)));
    }

    // CMS aggregation logits
    if !params.alpha_mem.is_empty() {
        tensors.push(("alpha_mem".into(), enc(&params.alpha_mem)));
    }
    if !params.alpha_refl.is_empty() {
        tensors.push(("alpha_refl".into(), enc(&params.alpha_refl)));
    }
    if !params.persistent_tokens.is_empty() {
        tensors.push(("persistent_tokens".into(), enc(&params.persistent_tokens)));
    }

    // Per-level memory weights
    for (i, lp) in params.levels.iter().enumerate() {
        let p = format!("level.{i}");

        // Memory projections: Bf16Storage — serialize fp32 master (source of truth)
        tensors.push((format!("{p}.w_k"), enc(lp.w_k_mem.master())));
        tensors.push((format!("{p}.w_v"), enc(lp.w_v_mem.master())));
        tensors.push((format!("{p}.w_q"), enc(lp.w_q_mem.master())));

        // Gate weights (always present)
        tensors.push((format!("{p}.gate.alpha"),   enc(&lp.w_alpha)));
        tensors.push((format!("{p}.gate.b_alpha"), enc(&lp.b_alpha)));
        tensors.push((format!("{p}.gate.theta"),   enc(&lp.w_theta)));
        tensors.push((format!("{p}.gate.b_theta"), enc(&lp.b_theta)));
        tensors.push((format!("{p}.gate.eta"),     enc(&lp.w_eta)));
        tensors.push((format!("{p}.gate.b_eta"),   enc(&lp.b_eta)));

        // Atlas Omega projection (always present; zero-init for non-Atlas rules)
        if !lp.w_omega.is_empty() {
            tensors.push((format!("{p}.w_omega"), enc(&lp.w_omega)));
        }

        // Learned frequency gate (empty for Fixed schedule)
        if !lp.w_freq.is_empty() {
            tensors.push((format!("{p}.w_freq"), enc(&lp.w_freq)));
            tensors.push((format!("{p}.b_freq"), enc(&lp.b_freq)));
        }

        // Conv1D key/query projections (empty when kernel_size=0)
        if !lp.w_k_conv.is_empty() {
            tensors.push((format!("{p}.w_k_conv"), enc(&lp.w_k_conv)));
            tensors.push((format!("{p}.b_k_conv"), enc(&lp.b_k_conv)));
            tensors.push((format!("{p}.w_q_conv"), enc(&lp.w_q_conv)));
            tensors.push((format!("{p}.b_q_conv"), enc(&lp.b_q_conv)));
        }

        // Self-referential projection init states (empty for Static)
        if !lp.m_k_init.is_empty() {
            tensors.push((format!("{p}.m_state.k"),     enc(&lp.m_k_init)));
            tensors.push((format!("{p}.m_state.v"),     enc(&lp.m_v_init)));
            tensors.push((format!("{p}.m_state.q"),     enc(&lp.m_q_init)));
            tensors.push((format!("{p}.m_state.eta"),   enc(&lp.m_eta_init)));
            tensors.push((format!("{p}.m_state.alpha"), enc(&lp.m_alpha_init)));
            tensors.push((format!("{p}.m_state.mem"),   enc(&lp.m_mem_init)));
        }

        // SwiGluMlp projections (empty for all other rules)
        if !lp.gate_proj.is_empty() {
            tensors.push((format!("{p}.mlp.gate_proj"), enc(&lp.gate_proj)));
            tensors.push((format!("{p}.mlp.up_proj"),   enc(&lp.up_proj)));
            tensors.push((format!("{p}.mlp.down_proj"), enc(&lp.down_proj)));
        }

        // Feature map frozen weights (empty for Identity).
        // Enforce pair integrity: both must be present or both absent.
        let has_w = !lp.w_rand.is_empty();
        let has_b = !lp.b_rand.is_empty();
        if has_w != has_b {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{p}.fm.w_rand and {p}.fm.b_rand must both be non-empty or both be empty \
                         (got w_rand.len()={}, b_rand.len()={})",
                        lp.w_rand.len(), lp.b_rand.len()),
            ));
        }
        if has_w {
            tensors.push((format!("{p}.fm.w_rand"), enc(&lp.w_rand)));
            tensors.push((format!("{p}.fm.b_rand"), enc(&lp.b_rand)));
        }
    }

    // 2. Serialize config + build_state into __metadata__
    let config_json = serde_json::to_string(config)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let build_state_json = match build_state {
        Some(bs) => serde_json::to_string(bs)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        None => "null".to_string(),
    };

    // 3. Build safetensors JSON header
    let mut header = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    meta.insert("version".into(),     serde_json::Value::String("2".into()));
    meta.insert("format".into(),      serde_json::Value::String("nl_hecate_v2".into()));
    meta.insert("created_at".into(),  serde_json::Value::String(now_epoch()));
    meta.insert("config".into(),      serde_json::Value::String(config_json));
    meta.insert("build_state".into(), serde_json::Value::String(build_state_json));
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));

    let mut data_offset: usize = 0;
    for (name, bytes) in &tensors {
        let n = bytes.len();
        header.insert(name.clone(), serde_json::json!({
            "dtype": "F32",
            "shape": [n / 4],   // flat [n_elems] shape; exact shape reconstructed from config
            "data_offsets": [data_offset, data_offset + n],
        }));
        data_offset += n;
    }

    // 4. Serialize header to JSON, pad to 8-byte alignment (safetensors spec)
    let header_bytes = serde_json::to_vec(&header)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let raw_len = header_bytes.len();
    let padded_len = (raw_len + 7) & !7usize;

    // 5. Write: [u64 LE header_len][header JSON][space padding][tensor bytes]
    let mut file = std::fs::File::create(path)?;
    file.write_all(&(padded_len as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    if padded_len > raw_len {
        file.write_all(&vec![0x20u8; padded_len - raw_len])?;
    }
    for (_, bytes) in &tensors {
        file.write_all(bytes)?;
    }

    Ok(())
}

// ── Load ─────────────────────────────────────────────────────────────

/// Load a safetensors binary checkpoint.
///
/// Returns (MAGParams, MAGConfig, Option<BuildResumeState>).
/// Optional tensor fields absent from the file are returned as empty Vec — this
/// preserves backward compat when loading older checkpoints that predate optional
/// fields like w_freq, gate_proj, or m_k_init.
pub fn load_safetensors(
    path: &Path,
) -> io::Result<(MAGParams, MAGConfig, Option<BuildResumeState>)> {
    let file = std::fs::File::open(path)?;
    // SAFETY: The file is read-only after training completes; no concurrent writes
    // while the mmap is live. Pages are loaded on-demand by the OS — avoids the
    // ~3.4 GB peak heap that std::fs::read would require for a 1.7 GB checkpoint.
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    // ── Parse header ──────────────────────────────────────────────────
    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "safetensors: file too small"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "safetensors: truncated header"));
    }
    // Trim space padding (our saver pads with 0x20 to 8-byte alignment).
    // Use trim_end() on str rather than trim_ascii_end() on &[u8] to stay
    // compatible with Rust < 1.80 (trim_ascii_end stabilised in 1.80.0).
    let header_slice = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let header_json = header_slice.trim_end();
    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // ── Extract __metadata__ ─────────────────────────────────────────
    let meta = header["__metadata__"].as_object().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "safetensors: missing __metadata__")
    })?;

    let config_str = meta.get("config")
        .and_then(|v| v.as_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "safetensors: missing config"))?;
    let config: MAGConfig = serde_json::from_str(config_str)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let build_state: Option<BuildResumeState> = match meta.get("build_state").and_then(|v| v.as_str()) {
        Some("null") | None => None,
        Some(s) => Some(
            serde_json::from_str(s)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        ),
    };

    // ── Tensor extraction ────────────────────────────────────────────
    // Data region follows the padded header.
    let data_region = &bytes[8 + header_len..];

    // Extract tensor by name using the header's data_offsets descriptor.
    // Returns empty Vec if tensor is absent (backward compat for optional fields).
    let get = |name: &str| -> Vec<f32> {
        let offsets = header.get(name)
            .and_then(|v| v.get("data_offsets"))
            .and_then(|v| v.as_array());
        if let Some(offs) = offsets {
            if offs.len() == 2 {
                let s = offs[0].as_u64().unwrap_or(0) as usize;
                let e = offs[1].as_u64().unwrap_or(0) as usize;
                if e <= data_region.len() && e >= s {
                    return data_region[s..e].chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                }
            }
        }
        vec![]
    };

    // Reconstruct SWAParams
    let d = config.swa.d_model;
    let ln_attn_gamma = { let v = get("ln_attn.gamma"); if v.is_empty() { vec![1.0f32; d] } else { v } };
    let ln_attn_beta  = { let v = get("ln_attn.beta");  if v.is_empty() { vec![0.0f32; d] } else { v } };
    let ln_mem_gamma  = { let v = get("ln_mem.gamma");   if v.is_empty() { vec![1.0f32; d] } else { v } };
    let ln_mem_beta   = { let v = get("ln_mem.beta");    if v.is_empty() { vec![0.0f32; d] } else { v } };
    let swa = SWAParams {
        w_embed:   get("embed.weight"),
        w_q:       get("swa.w_q"),
        w_k:       get("swa.w_k"),
        w_v:       get("swa.w_v"),
        w_o:       get("swa.w_o"),
        w_unembed: get("lm_head.weight"),
        ln_attn_gamma,
        ln_attn_beta,
        ln_mem_gamma,
        ln_mem_beta,
    };

    // Reconstruct per-level MemoryLevelParams
    let k = config.k;
    let mut levels = Vec::with_capacity(k);
    for i in 0..k {
        let p = format!("level.{i}");
        // Bf16Storage: reconstruct from fp32 master (both stored and master copies)
        let w_k_mem = Bf16Storage::from_f32_vec(get(&format!("{p}.w_k")));
        let w_v_mem = Bf16Storage::from_f32_vec(get(&format!("{p}.w_v")));
        let w_q_mem = Bf16Storage::from_f32_vec(get(&format!("{p}.w_q")));

        levels.push(MemoryLevelParams {
            w_k_mem,
            w_v_mem,
            w_q_mem,
            w_alpha:      get(&format!("{p}.gate.alpha")),
            b_alpha:      get(&format!("{p}.gate.b_alpha")),
            w_theta:      get(&format!("{p}.gate.theta")),
            b_theta:      get(&format!("{p}.gate.b_theta")),
            w_eta:        get(&format!("{p}.gate.eta")),
            b_eta:        get(&format!("{p}.gate.b_eta")),
            w_omega:      get(&format!("{p}.w_omega")),
            w_freq:       get(&format!("{p}.w_freq")),
            b_freq:       get(&format!("{p}.b_freq")),
            w_k_conv:     get(&format!("{p}.w_k_conv")),
            b_k_conv:     get(&format!("{p}.b_k_conv")),
            w_q_conv:     get(&format!("{p}.w_q_conv")),
            b_q_conv:     get(&format!("{p}.b_q_conv")),
            m_k_init:     get(&format!("{p}.m_state.k")),
            m_v_init:     get(&format!("{p}.m_state.v")),
            m_q_init:     get(&format!("{p}.m_state.q")),
            m_eta_init:   get(&format!("{p}.m_state.eta")),
            m_alpha_init: get(&format!("{p}.m_state.alpha")),
            m_mem_init:   get(&format!("{p}.m_state.mem")),
            gate_proj:    get(&format!("{p}.mlp.gate_proj")),
            up_proj:      get(&format!("{p}.mlp.up_proj")),
            down_proj:    get(&format!("{p}.mlp.down_proj")),
            w_rand:       get(&format!("{p}.fm.w_rand")),
            b_rand:       get(&format!("{p}.fm.b_rand")),
        });
        // Pair integrity: both fm weights must be present or both absent.
        let lp = levels.last().unwrap();
        let has_w = !lp.w_rand.is_empty();
        let has_b = !lp.b_rand.is_empty();
        if has_w != has_b {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{p}.fm.w_rand and {p}.fm.b_rand must both be present or both absent \
                         (got w_rand.len()={}, b_rand.len()={}). \
                         Checkpoint may be corrupt.",
                        lp.w_rand.len(), lp.b_rand.len()),
            ));
        }
    }

    let params = MAGParams {
        swa,
        levels,
        alpha_mem:         get("alpha_mem"),
        alpha_refl:        get("alpha_refl"),
        persistent_tokens: get("persistent_tokens"),
    };

    Ok((params, config, build_state))
}

// ── Stacked checkpoint save/load ────────────────────────────────────
// Spec: specs/infrastructure/22_stacked_extend_k_per_block.md
// Hierarchical keys: shared.* for embed/unembed/ln_final, block.{n}.* for per-block.

/// Save StackedMAGParams + config as a safetensors binary checkpoint.
///
/// Uses hierarchical keys: shared.embed.weight, block.0.swa.w_q, block.0.level.0.w_k, etc.
/// The metadata includes "stacked":"true" and "n_blocks":"N" for format detection.
pub fn save_stacked_safetensors(
    path: &Path,
    params: &StackedMAGParams,
    config: &MAGConfig,
    build_state: Option<&BuildResumeState>,
) -> io::Result<()> {
    let n_blocks = params.blocks.len();
    let mut tensors: Vec<(String, Vec<u8>)> = Vec::new();

    fn enc(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // Shared parameters
    tensors.push(("shared.embed.weight".into(), enc(&params.w_embed)));
    tensors.push(("shared.lm_head.weight".into(), enc(&params.w_unembed)));
    tensors.push(("shared.ln_final.gamma".into(), enc(&params.ln_final_gamma)));
    tensors.push(("shared.ln_final.beta".into(), enc(&params.ln_final_beta)));

    // Per-block parameters
    for (b, block) in params.blocks.iter().enumerate() {
        let bp = format!("block.{b}");

        // SWA projections
        tensors.push((format!("{bp}.swa.w_q"), enc(&block.w_q)));
        tensors.push((format!("{bp}.swa.w_k"), enc(&block.w_k)));
        tensors.push((format!("{bp}.swa.w_v"), enc(&block.w_v)));
        tensors.push((format!("{bp}.swa.w_o"), enc(&block.w_o)));

        // LayerNorms
        tensors.push((format!("{bp}.ln_attn.gamma"), enc(&block.ln_attn_gamma)));
        tensors.push((format!("{bp}.ln_attn.beta"), enc(&block.ln_attn_beta)));
        tensors.push((format!("{bp}.ln_mem.gamma"), enc(&block.ln_mem_gamma)));
        tensors.push((format!("{bp}.ln_mem.beta"), enc(&block.ln_mem_beta)));

        // CMS aggregation logits
        if !block.alpha_mem.is_empty() {
            tensors.push((format!("{bp}.alpha_mem"), enc(&block.alpha_mem)));
        }
        if !block.alpha_refl.is_empty() {
            tensors.push((format!("{bp}.alpha_refl"), enc(&block.alpha_refl)));
        }

        // Per-level memory weights
        for (i, lp) in block.levels.iter().enumerate() {
            let p = format!("{bp}.level.{i}");
            encode_level_tensors(&mut tensors, &p, lp)?;
        }
    }

    // Metadata
    let config_json = serde_json::to_string(config)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let build_state_json = match build_state {
        Some(bs) => serde_json::to_string(bs)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        None => "null".to_string(),
    };

    let mut header = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    meta.insert("version".into(),     serde_json::Value::String("2".into()));
    meta.insert("format".into(),      serde_json::Value::String("nl_hecate_v2_stacked".into()));
    meta.insert("stacked".into(),     serde_json::Value::String("true".into()));
    meta.insert("n_blocks".into(),    serde_json::Value::String(n_blocks.to_string()));
    meta.insert("created_at".into(),  serde_json::Value::String(now_epoch()));
    meta.insert("config".into(),      serde_json::Value::String(config_json));
    meta.insert("build_state".into(), serde_json::Value::String(build_state_json));
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));

    let mut data_offset: usize = 0;
    for (name, bytes) in &tensors {
        let n = bytes.len();
        header.insert(name.clone(), serde_json::json!({
            "dtype": "F32",
            "shape": [n / 4],
            "data_offsets": [data_offset, data_offset + n],
        }));
        data_offset += n;
    }

    let header_bytes = serde_json::to_vec(&header)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let raw_len = header_bytes.len();
    let padded_len = (raw_len + 7) & !7usize;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&(padded_len as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    if padded_len > raw_len {
        file.write_all(&vec![0x20u8; padded_len - raw_len])?;
    }
    for (_, bytes) in &tensors {
        file.write_all(bytes)?;
    }

    Ok(())
}

/// Load a stacked safetensors checkpoint.
///
/// Returns (StackedMAGParams, MAGConfig, n_blocks, Option<BuildResumeState>).
/// Detects n_blocks by scanning header keys for the highest block.{n} prefix.
pub fn load_stacked_safetensors(
    path: &Path,
) -> io::Result<(StackedMAGParams, MAGConfig, usize, Option<BuildResumeState>)> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "safetensors: file too small"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "safetensors: truncated header"));
    }
    let header_slice = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let header_json = header_slice.trim_end();
    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let meta = header["__metadata__"].as_object().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "safetensors: missing __metadata__")
    })?;

    let config_str = meta.get("config")
        .and_then(|v| v.as_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "safetensors: missing config"))?;
    let config: MAGConfig = serde_json::from_str(config_str)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let build_state: Option<BuildResumeState> = match meta.get("build_state").and_then(|v| v.as_str()) {
        Some("null") | None => None,
        Some(s) => Some(
            serde_json::from_str(s)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        ),
    };

    // Scan keys for actual block count (authoritative)
    let scanned_n_blocks = {
        let mut max_b = 0usize;
        if let Some(obj) = header.as_object() {
            for key in obj.keys() {
                if let Some(rest) = key.strip_prefix("block.") {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(b) = rest[..dot].parse::<usize>() {
                            max_b = max_b.max(b + 1);
                        }
                    }
                }
            }
        }
        max_b
    };

    // Use metadata n_blocks if present, but validate against scanned keys
    let n_blocks = match meta.get("n_blocks").and_then(|v| v.as_str()).and_then(|s| s.parse::<usize>().ok()) {
        Some(meta_n) => {
            if scanned_n_blocks > 0 && meta_n != scanned_n_blocks {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("stacked safetensors: metadata n_blocks={meta_n} but found {scanned_n_blocks} blocks in keys"),
                ));
            }
            meta_n
        }
        None => scanned_n_blocks,
    };

    if n_blocks == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stacked safetensors: could not determine n_blocks (no block.* keys found)",
        ));
    }

    let data_region = &bytes[8 + header_len..];

    let get = |name: &str| -> Vec<f32> {
        let offsets = header.get(name)
            .and_then(|v| v.get("data_offsets"))
            .and_then(|v| v.as_array());
        if let Some(offs) = offsets {
            if offs.len() == 2 {
                let s = offs[0].as_u64().unwrap_or(0) as usize;
                let e = offs[1].as_u64().unwrap_or(0) as usize;
                if e <= data_region.len() && e >= s {
                    return data_region[s..e].chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                }
            }
        }
        vec![]
    };

    // Helper: load required tensor, fail fast if missing or wrong size
    let get_required = |name: &str, expected_len: usize| -> io::Result<Vec<f32>> {
        let v = get(name);
        if v.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("stacked safetensors: required tensor '{}' missing", name),
            ));
        }
        if v.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("stacked safetensors: tensor '{}' has {} elements, expected {}",
                        name, v.len(), expected_len),
            ));
        }
        Ok(v)
    };

    // Shared params
    let d = config.swa.d_model;
    let v = config.swa.vocab_size;
    let w_embed = get_required("shared.embed.weight", v * d)?;
    let w_unembed = get_required("shared.lm_head.weight", d * v)?;
    let ln_final_gamma = { let t = get("shared.ln_final.gamma"); if t.is_empty() { vec![1.0f32; d] } else { t } };
    let ln_final_beta = { let t = get("shared.ln_final.beta"); if t.is_empty() { vec![0.0f32; d] } else { t } };

    // Per-block params
    let k = config.k;
    let dd = d * d;
    let mut blocks = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let bp = format!("block.{b}");

        let w_q = get_required(&format!("{bp}.swa.w_q"), dd)?;
        let w_k = get_required(&format!("{bp}.swa.w_k"), dd)?;
        let w_v = get_required(&format!("{bp}.swa.w_v"), dd)?;
        let w_o = get_required(&format!("{bp}.swa.w_o"), dd)?;
        let ln_attn_gamma = { let t = get(&format!("{bp}.ln_attn.gamma")); if t.is_empty() { vec![1.0f32; d] } else { t } };
        let ln_attn_beta = { let t = get(&format!("{bp}.ln_attn.beta")); if t.is_empty() { vec![0.0f32; d] } else { t } };
        let ln_mem_gamma = { let t = get(&format!("{bp}.ln_mem.gamma")); if t.is_empty() { vec![1.0f32; d] } else { t } };
        let ln_mem_beta = { let t = get(&format!("{bp}.ln_mem.beta")); if t.is_empty() { vec![0.0f32; d] } else { t } };
        let alpha_mem = get(&format!("{bp}.alpha_mem"));
        let alpha_refl = get(&format!("{bp}.alpha_refl"));

        let mut levels = Vec::with_capacity(k);
        for i in 0..k {
            let p = format!("{bp}.level.{i}");
            levels.push(decode_level_params(&get, &p));
        }

        blocks.push(BlockParams {
            w_q, w_k, w_v, w_o,
            ln_attn_gamma, ln_attn_beta,
            ln_mem_gamma, ln_mem_beta,
            levels, alpha_mem, alpha_refl,
        });
    }

    let params = StackedMAGParams {
        w_embed, w_unembed,
        ln_final_gamma, ln_final_beta,
        blocks,
    };

    Ok((params, config, n_blocks, build_state))
}

/// Check if a safetensors file is a stacked checkpoint (has shared.embed.weight key).
pub fn is_stacked_checkpoint(path: &Path) -> io::Result<bool> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    if bytes.len() < 8 {
        return Ok(false);
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        return Ok(false);
    }
    let header_slice = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(header_slice.contains("shared.embed.weight"))
}

/// Encode per-level tensors into the tensor list (shared by single-block and stacked).
fn encode_level_tensors(
    tensors: &mut Vec<(String, Vec<u8>)>,
    prefix: &str,
    lp: &MemoryLevelParams,
) -> io::Result<()> {
    fn enc(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    tensors.push((format!("{prefix}.w_k"), enc(lp.w_k_mem.master())));
    tensors.push((format!("{prefix}.w_v"), enc(lp.w_v_mem.master())));
    tensors.push((format!("{prefix}.w_q"), enc(lp.w_q_mem.master())));

    tensors.push((format!("{prefix}.gate.alpha"),   enc(&lp.w_alpha)));
    tensors.push((format!("{prefix}.gate.b_alpha"), enc(&lp.b_alpha)));
    tensors.push((format!("{prefix}.gate.theta"),   enc(&lp.w_theta)));
    tensors.push((format!("{prefix}.gate.b_theta"), enc(&lp.b_theta)));
    tensors.push((format!("{prefix}.gate.eta"),     enc(&lp.w_eta)));
    tensors.push((format!("{prefix}.gate.b_eta"),   enc(&lp.b_eta)));

    if !lp.w_omega.is_empty() {
        tensors.push((format!("{prefix}.w_omega"), enc(&lp.w_omega)));
    }
    if !lp.w_freq.is_empty() {
        tensors.push((format!("{prefix}.w_freq"), enc(&lp.w_freq)));
        tensors.push((format!("{prefix}.b_freq"), enc(&lp.b_freq)));
    }
    if !lp.w_k_conv.is_empty() {
        tensors.push((format!("{prefix}.w_k_conv"), enc(&lp.w_k_conv)));
        tensors.push((format!("{prefix}.b_k_conv"), enc(&lp.b_k_conv)));
        tensors.push((format!("{prefix}.w_q_conv"), enc(&lp.w_q_conv)));
        tensors.push((format!("{prefix}.b_q_conv"), enc(&lp.b_q_conv)));
    }
    if !lp.m_k_init.is_empty() {
        tensors.push((format!("{prefix}.m_state.k"),     enc(&lp.m_k_init)));
        tensors.push((format!("{prefix}.m_state.v"),     enc(&lp.m_v_init)));
        tensors.push((format!("{prefix}.m_state.q"),     enc(&lp.m_q_init)));
        tensors.push((format!("{prefix}.m_state.eta"),   enc(&lp.m_eta_init)));
        tensors.push((format!("{prefix}.m_state.alpha"), enc(&lp.m_alpha_init)));
        tensors.push((format!("{prefix}.m_state.mem"),   enc(&lp.m_mem_init)));
    }
    if !lp.gate_proj.is_empty() {
        tensors.push((format!("{prefix}.mlp.gate_proj"), enc(&lp.gate_proj)));
        tensors.push((format!("{prefix}.mlp.up_proj"),   enc(&lp.up_proj)));
        tensors.push((format!("{prefix}.mlp.down_proj"), enc(&lp.down_proj)));
    }
    let has_w = !lp.w_rand.is_empty();
    let has_b = !lp.b_rand.is_empty();
    if has_w != has_b {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{prefix}.fm.w_rand and {prefix}.fm.b_rand must both be non-empty or both be empty"),
        ));
    }
    if has_w {
        tensors.push((format!("{prefix}.fm.w_rand"), enc(&lp.w_rand)));
        tensors.push((format!("{prefix}.fm.b_rand"), enc(&lp.b_rand)));
    }
    Ok(())
}

/// Decode per-level params from a get-tensor closure (shared by single-block and stacked).
fn decode_level_params(get: &dyn Fn(&str) -> Vec<f32>, prefix: &str) -> MemoryLevelParams {
    let w_k_mem = Bf16Storage::from_f32_vec(get(&format!("{prefix}.w_k")));
    let w_v_mem = Bf16Storage::from_f32_vec(get(&format!("{prefix}.w_v")));
    let w_q_mem = Bf16Storage::from_f32_vec(get(&format!("{prefix}.w_q")));

    MemoryLevelParams {
        w_k_mem,
        w_v_mem,
        w_q_mem,
        w_alpha:      get(&format!("{prefix}.gate.alpha")),
        b_alpha:      get(&format!("{prefix}.gate.b_alpha")),
        w_theta:      get(&format!("{prefix}.gate.theta")),
        b_theta:      get(&format!("{prefix}.gate.b_theta")),
        w_eta:        get(&format!("{prefix}.gate.eta")),
        b_eta:        get(&format!("{prefix}.gate.b_eta")),
        w_omega:      get(&format!("{prefix}.w_omega")),
        w_freq:       get(&format!("{prefix}.w_freq")),
        b_freq:       get(&format!("{prefix}.b_freq")),
        w_k_conv:     get(&format!("{prefix}.w_k_conv")),
        b_k_conv:     get(&format!("{prefix}.b_k_conv")),
        w_q_conv:     get(&format!("{prefix}.w_q_conv")),
        b_q_conv:     get(&format!("{prefix}.b_q_conv")),
        m_k_init:     get(&format!("{prefix}.m_state.k")),
        m_v_init:     get(&format!("{prefix}.m_state.v")),
        m_q_init:     get(&format!("{prefix}.m_state.q")),
        m_eta_init:   get(&format!("{prefix}.m_state.eta")),
        m_alpha_init: get(&format!("{prefix}.m_state.alpha")),
        m_mem_init:   get(&format!("{prefix}.m_state.mem")),
        gate_proj:    get(&format!("{prefix}.mlp.gate_proj")),
        up_proj:      get(&format!("{prefix}.mlp.up_proj")),
        down_proj:    get(&format!("{prefix}.mlp.down_proj")),
        w_rand:       get(&format!("{prefix}.fm.w_rand")),
        b_rand:       get(&format!("{prefix}.fm.b_rand")),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Epoch-based timestamp (no chrono dependency — same convention as model.rs).
fn now_epoch() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("epoch:{}", d.as_secs()),
        Err(_) => "epoch:0".to_string(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MAGConfig;

    #[test]
    fn test_safetensors_roundtrip_serving() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let dir = std::env::temp_dir().join("hecate_test_st_serving");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("serving.safetensors");

        save_safetensors(&path, &params, &cfg, None).unwrap();
        let (loaded, loaded_cfg, build_state) = load_safetensors(&path).unwrap();

        assert_eq!(loaded.swa.w_q, params.swa.w_q, "swa.w_q round-trip");
        assert_eq!(loaded.swa.w_embed, params.swa.w_embed, "embed.weight round-trip");
        assert_eq!(loaded.swa.w_unembed, params.swa.w_unembed, "lm_head round-trip");
        assert_eq!(loaded_cfg.swa.d_model, cfg.swa.d_model);
        assert_eq!(loaded_cfg.k, cfg.k);
        assert!(build_state.is_none());

        // Bf16Storage master must round-trip exactly (stored as fp32)
        assert_eq!(
            loaded.levels[0].w_k_mem.master(),
            params.levels[0].w_k_mem.master(),
            "w_k_mem master round-trip"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_safetensors_roundtrip_build() {
        use crate::conductor::{ConductorState, ContextState};
        use crate::context_stream::StreamCursor;

        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 99);
        let d = cfg.swa.d_model;

        let build_state = BuildResumeState {
            conductor: ConductorState { k: 1, chunk_sizes: vec![1], step: 77 },
            stream_cursor: StreamCursor {
                position: 500, chunk_id: 10, pulse_id: 10, rng_state: None, content_hash: 0,
            },
            context: ContextState::new(1, d),
            global_step: 77,
            stream_cursors: Vec::new(),
            total_tokens_seen: 0,
        };

        let dir = std::env::temp_dir().join("hecate_test_st_build");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("build.safetensors");

        save_safetensors(&path, &params, &cfg, Some(&build_state)).unwrap();
        let (loaded, loaded_cfg, loaded_bs) = load_safetensors(&path).unwrap();

        assert_eq!(loaded.swa.w_q, params.swa.w_q);
        assert_eq!(loaded_cfg.k, cfg.k);
        let bs = loaded_bs.expect("build_state should be present");
        assert_eq!(bs.global_step, 77);
        assert_eq!(bs.conductor.step, 77);
        assert_eq!(bs.stream_cursor.position, 500);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_safetensors_file_is_binary() {
        // Verify the file is not JSON text (binary format was written),
        // and that the safetensors crate can parse what we produce (Python interop).
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 1);
        let dir = std::env::temp_dir().join("hecate_test_st_binary");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("check.safetensors");

        save_safetensors(&path, &params, &cfg, None).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 8, "file must have at least 8 bytes");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert!(header_len > 0, "header length must be nonzero");
        assert!(bytes.len() > 8 + header_len, "data region must follow header");

        // Header JSON should contain __metadata__ and tensor descriptors
        let header_slice = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
        let header: serde_json::Value = serde_json::from_str(header_slice.trim_end()).unwrap();
        assert!(header.get("__metadata__").is_some());
        assert!(header.get("embed.weight").is_some());

        // Verify the safetensors crate can parse our output (proves Python interop).
        // If this panics, the file format is not spec-compliant.
        let _st = safetensors::SafeTensors::deserialize(&bytes)
            .expect("safetensors crate must be able to parse our output");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_stacked_safetensors_roundtrip() {
        let cfg = MAGConfig::test_config();
        let n_blocks = 3;
        let params = StackedMAGParams::init(&cfg, n_blocks, 42);
        let dir = std::env::temp_dir().join("hecate_test_stacked_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stacked.safetensors");

        save_stacked_safetensors(&path, &params, &cfg, None).unwrap();
        let (loaded, loaded_cfg, loaded_nb, build_state) = load_stacked_safetensors(&path).unwrap();

        assert_eq!(loaded_nb, n_blocks);
        assert_eq!(loaded_cfg.k, cfg.k);
        assert!(build_state.is_none());

        // Shared params round-trip
        assert_eq!(loaded.w_embed, params.w_embed, "embed round-trip");
        assert_eq!(loaded.w_unembed, params.w_unembed, "unembed round-trip");
        assert_eq!(loaded.ln_final_gamma, params.ln_final_gamma);
        assert_eq!(loaded.ln_final_beta, params.ln_final_beta);

        // Per-block round-trip
        assert_eq!(loaded.blocks.len(), n_blocks);
        for b in 0..n_blocks {
            assert_eq!(loaded.blocks[b].w_q, params.blocks[b].w_q, "block {b} w_q");
            assert_eq!(loaded.blocks[b].w_o, params.blocks[b].w_o, "block {b} w_o");
            assert_eq!(loaded.blocks[b].alpha_mem, params.blocks[b].alpha_mem, "block {b} alpha_mem");
            assert_eq!(loaded.blocks[b].levels.len(), cfg.k);
            for l in 0..cfg.k {
                assert_eq!(
                    loaded.blocks[b].levels[l].w_k_mem.master(),
                    params.blocks[b].levels[l].w_k_mem.master(),
                    "block {b} level {l} w_k_mem"
                );
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_stacked_safetensors_with_build_state() {
        use crate::conductor::{ConductorState, ContextState};
        use crate::context_stream::StreamCursor;

        let cfg = MAGConfig::test_config();
        let n_blocks = 2;
        let params = StackedMAGParams::init(&cfg, n_blocks, 99);
        let d = cfg.swa.d_model;

        let build_state = BuildResumeState {
            conductor: ConductorState { k: cfg.k, chunk_sizes: cfg.chunk_sizes.clone(), step: 500 },
            stream_cursor: StreamCursor {
                position: 10000, chunk_id: 20, pulse_id: 20, rng_state: None, content_hash: 0,
            },
            context: ContextState::new(cfg.k, d),
            global_step: 500,
            stream_cursors: Vec::new(),
            total_tokens_seen: 0,
        };

        let dir = std::env::temp_dir().join("hecate_test_stacked_bs");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stacked_bs.safetensors");

        save_stacked_safetensors(&path, &params, &cfg, Some(&build_state)).unwrap();
        let (_, _, loaded_nb, loaded_bs) = load_stacked_safetensors(&path).unwrap();

        assert_eq!(loaded_nb, n_blocks);
        let bs = loaded_bs.expect("build_state should be present");
        assert_eq!(bs.global_step, 500);
        assert_eq!(bs.stream_cursor.position, 10000);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_is_stacked_checkpoint() {
        let cfg = MAGConfig::test_config();
        let dir = std::env::temp_dir().join("hecate_test_is_stacked");
        std::fs::create_dir_all(&dir).unwrap();

        // Single-block checkpoint
        let single_path = dir.join("single.safetensors");
        let single_params = MAGParams::init(&cfg, 42);
        save_safetensors(&single_path, &single_params, &cfg, None).unwrap();
        assert!(!is_stacked_checkpoint(&single_path).unwrap());

        // Stacked checkpoint
        let stacked_path = dir.join("stacked.safetensors");
        let stacked_params = StackedMAGParams::init(&cfg, 2, 42);
        save_stacked_safetensors(&stacked_path, &stacked_params, &cfg, None).unwrap();
        assert!(is_stacked_checkpoint(&stacked_path).unwrap());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_stacked_safetensors_python_interop() {
        let cfg = MAGConfig::test_config();
        let params = StackedMAGParams::init(&cfg, 2, 42);
        let dir = std::env::temp_dir().join("hecate_test_stacked_interop");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stacked_interop.safetensors");

        save_stacked_safetensors(&path, &params, &cfg, None).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let _st = safetensors::SafeTensors::deserialize(&bytes)
            .expect("safetensors crate must parse stacked checkpoint");

        std::fs::remove_dir_all(&dir).ok();
    }
}
