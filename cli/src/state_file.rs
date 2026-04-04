//! Spec 02: JSON state file — the model's lifecycle record.
//!
//! The state file IS the model. Identity, history, health, and references
//! to immutable weight checkpoints. Given the state file and its referenced
//! safetensors, any operation (resume, serve, diff, fork) is possible.
//!
//! See specs/infrastructure/checkpoint/02_state_file_schema.md for the full
//! v1.0 schema definition.

use std::path::Path;

use serde::{Serialize, Deserialize};
use nl_hecate_core::context_stream::StreamCursor;
use nl_hecate_core::model::MAGConfig;

// ── Schema types ────────────────────────────────────────────────────

/// Top-level state file (v1.0).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateFile {
    pub schema_version: String,
    pub format: String,
    pub identity: Identity,
    pub tokens_total: u64,
    #[serde(default)]
    pub checkpoints: Vec<CheckpointEntry>,
    pub current_checkpoint: Option<CurrentCheckpoint>,
    pub cursor: CursorState,
    /// Reserved for future behavioral summary (null in v1.0).
    pub behavioral: Option<serde_json::Value>,
}

/// Model identity — generated once at init, never changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Identity {
    pub instance_id: String,
    pub name: String,
    pub created_at: String,
    pub parent: Option<ParentRef>,
    pub architecture: MAGConfig,
}

/// Lineage reference to parent model (for fork operations).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParentRef {
    pub instance_id: String,
    pub checkpoint: String,
    pub tokens_at_fork: u64,
}

/// Immutable record of a persisted checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointEntry {
    pub path: String,
    pub tokens: u64,
    pub content_hash: String,
    pub timestamp: String,
    pub health: HealthSnapshot,
    pub session: SessionInfo,
}

/// Health snapshot at checkpoint time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub loss: f32,
    #[serde(default)]
    pub m_norm_per_level: Vec<f32>,
    #[serde(default)]
    pub gate_alpha_mean_per_level: Vec<f32>,
    #[serde(default)]
    pub gate_theta_mean_per_level: Vec<f32>,
    #[serde(default)]
    pub cms_activations_since_restore: Vec<u64>,
}

/// Session context that produced a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionInfo {
    Build {
        label: String,
        #[serde(default)]
        dataset: String,
        #[serde(default)]
        dataset_hash: String,
        #[serde(default)]
        config_snapshot: String,
    },
    Serving {
        label: String,
        #[serde(default)]
        domain_tags: Vec<String>,
        #[serde(default)]
        date_start: String,
    },
}

/// Convenience pointer to the most recent checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurrentCheckpoint {
    pub path: String,
    pub tokens: u64,
    pub content_hash: String,
}

/// Stream cursor state, inlined in the state file (no sidecar).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CursorState {
    #[serde(default)]
    pub slots: Vec<StreamCursor>,
}

// ── Operations ──────────────────────────────────────────────────────

/// Create a new state file for a fresh model instance.
pub fn init_state_file(
    name: &str,
    architecture: &MAGConfig,
) -> StateFile {
    let instance_id = uuid_v4();
    let created_at = iso8601_now();

    StateFile {
        schema_version: "1.0".into(),
        format: "nl_hecate_state".into(),
        identity: Identity {
            instance_id,
            name: name.into(),
            created_at,
            parent: None,
            architecture: architecture.clone(),
        },
        tokens_total: 0,
        checkpoints: Vec::new(),
        current_checkpoint: None,
        cursor: CursorState { slots: Vec::new() },
        behavioral: None,
    }
}

/// Create a state file forked from an existing model's checkpoint.
pub fn fork_state_file(
    name: &str,
    architecture: &MAGConfig,
    parent_instance_id: &str,
    parent_checkpoint: &str,
    parent_tokens: u64,
) -> StateFile {
    let mut state = init_state_file(name, architecture);
    state.identity.parent = Some(ParentRef {
        instance_id: parent_instance_id.into(),
        checkpoint: parent_checkpoint.into(),
        tokens_at_fork: parent_tokens,
    });
    state
}

/// Record a new checkpoint in the state file. Updates tokens_total,
/// appends to checkpoints array, sets current_checkpoint, writes cursor.
pub fn record_checkpoint(
    state: &mut StateFile,
    entry: CheckpointEntry,
    cursors: &[StreamCursor],
) {
    state.tokens_total = entry.tokens;
    state.current_checkpoint = Some(CurrentCheckpoint {
        path: entry.path.clone(),
        tokens: entry.tokens,
        content_hash: entry.content_hash.clone(),
    });
    state.cursor.slots = cursors.to_vec();
    state.checkpoints.push(entry);
}

/// Write the state file to disk atomically (write to .tmp, then rename).
pub fn save_state_file(path: &Path, state: &StateFile) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a state file from disk. Validates format and schema_version.
pub fn load_state_file(path: &Path) -> std::io::Result<StateFile> {
    let text = std::fs::read_to_string(path)?;
    let state: StateFile = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    if state.format != "nl_hecate_state" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown state file format: {:?} (expected \"nl_hecate_state\")", state.format),
        ));
    }
    if state.schema_version != "1.0" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported schema_version: {:?} (expected \"1.0\")", state.schema_version),
        ));
    }
    Ok(state)
}

/// Derive the state file path from the save_path base.
/// `checkpoints/model.safetensors` → `model.state.json` (in run_dir).
pub fn state_file_path(run_dir: &str, name: &str) -> std::path::PathBuf {
    Path::new(run_dir).join(format!("{name}.state.json"))
}

/// Attempt to load a legacy `.cursor.json` sidecar from a checkpoint path.
///
/// Sidecar path: `<checkpoint_path>.cursor.json`
/// Returns cursors mapped from the sidecar's position/chunk_id/content_hash fields.
/// Returns empty vec if sidecar doesn't exist or can't be parsed.
pub fn load_legacy_cursor(checkpoint_path: &str) -> Vec<StreamCursor> {
    let sidecar = format!("{checkpoint_path}.cursor.json");
    let path = Path::new(&sidecar);
    if !path.exists() {
        return Vec::new();
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    // Legacy sidecar schema: {"position": N, "total_tokens": N, "content_hash": N, ...}
    let val: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let position = val.get("position").and_then(|v| v.as_u64()).unwrap_or(0);
    let chunk_id = val.get("chunk_id").and_then(|v| v.as_u64()).unwrap_or(0);
    let content_hash = val.get("content_hash").and_then(|v| v.as_u64()).unwrap_or(0);
    if position == 0 {
        return Vec::new();
    }
    vec![StreamCursor {
        position,
        chunk_id,
        pulse_id: 0,
        rng_state: None,
        content_hash,
    }]
}

// ── Helpers ─────────────────────────────────────────────────────────

fn uuid_v4() -> String {
    // Simple UUID v4 from /dev/urandom (no external crate dependency).
    let mut buf = [0u8; 16];
    if let Ok(mut file) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        let _ = file.read_exact(&mut buf);
    } else {
        // Fallback: use system time as entropy source
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for i in 0..16 {
            buf[i] = ((t >> (i * 8)) & 0xFF) as u8;
        }
    }
    // Set version (4) and variant (RFC 4122)
    buf[6] = (buf[6] & 0x0F) | 0x40;
    buf[8] = (buf[8] & 0x3F) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        buf[0], buf[1], buf[2], buf[3],
        buf[4], buf[5],
        buf[6], buf[7],
        buf[8], buf[9],
        buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    )
}

pub fn iso8601_now() -> String {
    // Use UNIX epoch seconds + manual UTC formatting (no chrono dependency).
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple UTC timestamp (good enough for state files)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since epoch to Y-M-D (simplified Gregorian)
    let (year, month, day) = epoch_days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal MAGConfig via JSON — avoids maintaining 40+ required fields by hand.
    fn test_mag_config() -> MAGConfig {
        serde_json::from_str(r#"{
            "swa": {"d_model": 64, "num_heads": 2, "head_dim": 32, "vocab_size": 256, "seq_len": 32, "window_size": 32},
            "memory_enabled": true,
            "composition": "MAG",
            "memory_rule": "TitansLMM",
            "k": 1,
            "chunk_sizes": [1],
            "d_hidden": 0, "lp_p": 2.0, "lq_q": 2.0,
            "lambda_local": 0.01, "lambda_2": 0.01, "delta": 1.0,
            "m_slots": 0, "d_compress": 0, "lambda_k": 0.0, "lambda_v": 0.0,
            "parallel": null,
            "retention": "L2WeightDecay",
            "m3": null,
            "frequency_schedule": "Fixed"
        }"#).expect("test MAGConfig should parse")
    }

    #[test]
    fn test_init_state_file() {
        let cfg = test_mag_config();
        let state = init_state_file("test_model", &cfg);

        assert_eq!(state.schema_version, "1.0");
        assert_eq!(state.format, "nl_hecate_state");
        assert_eq!(state.identity.name, "test_model");
        assert_eq!(state.tokens_total, 0);
        assert!(state.checkpoints.is_empty());
        assert!(state.current_checkpoint.is_none());
        assert!(state.cursor.slots.is_empty());
        assert!(state.behavioral.is_none());
        assert!(state.identity.parent.is_none());

        // UUID v4 format check
        let id = &state.identity.instance_id;
        assert_eq!(id.len(), 36);
        assert_eq!(&id[8..9], "-");
        assert_eq!(&id[13..14], "-");
        assert_eq!(&id[14..15], "4"); // version nibble
    }

    #[test]
    fn test_fork_state_file() {
        let cfg = test_mag_config();
        let state = fork_state_file(
            "child_model", &cfg,
            "parent-uuid-1234", "checkpoints/model_150M_tok.safetensors", 150_000_000,
        );

        assert_eq!(state.identity.name, "child_model");
        assert_eq!(state.tokens_total, 0);
        let parent = state.identity.parent.as_ref().unwrap();
        assert_eq!(parent.instance_id, "parent-uuid-1234");
        assert_eq!(parent.checkpoint, "checkpoints/model_150M_tok.safetensors");
        assert_eq!(parent.tokens_at_fork, 150_000_000);
        // Instance ID should be different from parent
        assert_ne!(state.identity.instance_id, "parent-uuid-1234");
    }

    #[test]
    fn test_record_checkpoint() {
        let cfg = test_mag_config();
        let mut state = init_state_file("test_model", &cfg);

        let entry = CheckpointEntry {
            path: "checkpoints/model_512000tok.safetensors".into(),
            tokens: 512_000,
            content_hash: "sha256:abc123".into(),
            timestamp: "2026-04-04T12:00:00Z".into(),
            health: HealthSnapshot {
                loss: 3.8,
                m_norm_per_level: vec![8.1],
                gate_alpha_mean_per_level: vec![0.91],
                gate_theta_mean_per_level: vec![0.012],
                cms_activations_since_restore: vec![512_000],
            },
            session: SessionInfo::Build {
                label: "foundations".into(),
                dataset: "test_data".into(),
                dataset_hash: "sha256:def456".into(),
                config_snapshot: "configs/test.json".into(),
            },
        };

        let cursors = vec![StreamCursor {
            position: 512_000,
            chunk_id: 100,
            pulse_id: 1000,
            rng_state: None,
            content_hash: 12345,
        }];

        record_checkpoint(&mut state, entry, &cursors);

        assert_eq!(state.tokens_total, 512_000);
        assert_eq!(state.checkpoints.len(), 1);
        assert_eq!(state.checkpoints[0].path, "checkpoints/model_512000tok.safetensors");
        assert_eq!(state.checkpoints[0].health.loss, 3.8);

        let cur = state.current_checkpoint.as_ref().unwrap();
        assert_eq!(cur.path, "checkpoints/model_512000tok.safetensors");
        assert_eq!(cur.tokens, 512_000);

        assert_eq!(state.cursor.slots.len(), 1);
        assert_eq!(state.cursor.slots[0].position, 512_000);
    }

    #[test]
    fn test_multiple_checkpoints_accumulate() {
        let cfg = test_mag_config();
        let mut state = init_state_file("test_model", &cfg);

        for i in 1..=3 {
            let tokens = i * 100_000;
            let entry = CheckpointEntry {
                path: format!("checkpoints/model_{tokens}tok.safetensors"),
                tokens: tokens as u64,
                content_hash: format!("sha256:hash{i}"),
                timestamp: format!("2026-04-04T{i:02}:00:00Z"),
                health: HealthSnapshot {
                    loss: 4.0 - i as f32 * 0.5,
                    m_norm_per_level: vec![],
                    gate_alpha_mean_per_level: vec![],
                    gate_theta_mean_per_level: vec![],
                    cms_activations_since_restore: vec![],
                },
                session: SessionInfo::Build {
                    label: format!("phase_{i}"),
                    dataset: "data".into(),
                    dataset_hash: "".into(),
                    config_snapshot: "".into(),
                },
            };
            record_checkpoint(&mut state, entry, &[]);
        }

        assert_eq!(state.checkpoints.len(), 3);
        assert_eq!(state.tokens_total, 300_000);
        assert_eq!(state.current_checkpoint.as_ref().unwrap().tokens, 300_000);
    }

    #[test]
    fn test_round_trip_json() {
        let cfg = test_mag_config();
        let mut state = init_state_file("round_trip_model", &cfg);

        let entry = CheckpointEntry {
            path: "checkpoints/model_1000tok.safetensors".into(),
            tokens: 1000,
            content_hash: "sha256:test".into(),
            timestamp: "2026-04-04T00:00:00Z".into(),
            health: HealthSnapshot {
                loss: 5.0,
                m_norm_per_level: vec![1.0, 2.0],
                gate_alpha_mean_per_level: vec![0.9, 0.95],
                gate_theta_mean_per_level: vec![0.01, 0.005],
                cms_activations_since_restore: vec![1000, 125],
            },
            session: SessionInfo::Build {
                label: "test".into(),
                dataset: "test_data".into(),
                dataset_hash: "hash".into(),
                config_snapshot: "config.json".into(),
            },
        };
        let cursor = StreamCursor {
            position: 1000,
            chunk_id: 10,
            pulse_id: 50,
            rng_state: Some(42),
            content_hash: 99999,
        };
        record_checkpoint(&mut state, entry, &[cursor]);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&state).unwrap();

        // Verify it's valid JSON and human-readable
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["schema_version"], "1.0");
        assert_eq!(parsed["tokens_total"], 1000);
        assert_eq!(parsed["identity"]["name"], "round_trip_model");

        // Deserialize back to StateFile
        let loaded: StateFile = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.identity.instance_id, state.identity.instance_id);
        assert_eq!(loaded.tokens_total, 1000);
        assert_eq!(loaded.checkpoints.len(), 1);
        assert_eq!(loaded.checkpoints[0].health.loss, 5.0);
        assert_eq!(loaded.cursor.slots.len(), 1);
        assert_eq!(loaded.cursor.slots[0].rng_state, Some(42));
    }

    #[test]
    fn test_save_and_load_file() {
        let cfg = test_mag_config();
        let state = init_state_file("file_test", &cfg);

        let dir = std::env::temp_dir().join("nl_hecate_state_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test_model.state.json");

        save_state_file(&path, &state).unwrap();
        let loaded = load_state_file(&path).unwrap();

        assert_eq!(loaded.identity.instance_id, state.identity.instance_id);
        assert_eq!(loaded.identity.name, "file_test");
        assert_eq!(loaded.tokens_total, 0);

        // Cleanup
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_state_file_path() {
        let p = state_file_path("/runs/experiment_1", "legal_v1");
        assert_eq!(p.to_str().unwrap(), "/runs/experiment_1/legal_v1.state.json");
    }

    #[test]
    fn test_uuid_v4_format() {
        let id = uuid_v4();
        assert_eq!(id.len(), 36);
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
        // Version nibble = 4
        assert_eq!(&parts[2][0..1], "4");
        // Variant: first nibble of part 3 must be 8, 9, a, or b
        let variant = u8::from_str_radix(&parts[3][0..1], 16).unwrap();
        assert!((8..=11).contains(&variant));
    }

    #[test]
    fn test_iso8601_now_format() {
        let ts = iso8601_now();
        // Should match YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20);
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
        assert_eq!(&ts[19..20], "Z");
        // Year should be reasonable
        let year: u64 = ts[0..4].parse().unwrap();
        assert!(year >= 2026);
    }

    #[test]
    fn test_serving_session_serialization() {
        let session = SessionInfo::Serving {
            label: "prod_legal".into(),
            domain_tags: vec!["legal".into(), "contracts".into()],
            date_start: "2026-04-04T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains("\"type\":\"serving\""));
        assert!(json.contains("\"domain_tags\""));

        let parsed: SessionInfo = serde_json::from_str(&json).unwrap();
        match parsed {
            SessionInfo::Serving { label, domain_tags, .. } => {
                assert_eq!(label, "prod_legal");
                assert_eq!(domain_tags.len(), 2);
            }
            _ => panic!("expected Serving session"),
        }
    }

    #[test]
    fn test_load_rejects_wrong_format() {
        let dir = std::env::temp_dir().join("nl_hecate_state_test_fmt");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("bad_format.state.json");

        let cfg = test_mag_config();
        let mut state = init_state_file("test", &cfg);
        state.format = "some_other_format".into();
        // Write directly (bypass save_state_file which doesn't validate)
        let json = serde_json::to_string_pretty(&state).unwrap();
        std::fs::write(&path, &json).unwrap();

        let err = load_state_file(&path).unwrap_err();
        assert!(err.to_string().contains("unknown state file format"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_load_rejects_wrong_schema_version() {
        let dir = std::env::temp_dir().join("nl_hecate_state_test_ver");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("bad_version.state.json");

        let cfg = test_mag_config();
        let mut state = init_state_file("test", &cfg);
        state.schema_version = "2.0".into();
        let json = serde_json::to_string_pretty(&state).unwrap();
        std::fs::write(&path, &json).unwrap();

        let err = load_state_file(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported schema_version"));

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_load_legacy_cursor_valid() {
        let dir = std::env::temp_dir().join("nl_hecate_cursor_test");
        std::fs::create_dir_all(&dir).ok();
        let ckpt_path = dir.join("model.safetensors");
        let sidecar_path = dir.join("model.safetensors.cursor.json");

        let sidecar = r#"{"position": 512000, "total_tokens": 1000000, "content_hash": 12345, "chunk_id": 100}"#;
        std::fs::write(&sidecar_path, sidecar).unwrap();

        let cursors = load_legacy_cursor(ckpt_path.to_str().unwrap());
        assert_eq!(cursors.len(), 1);
        assert_eq!(cursors[0].position, 512000);
        assert_eq!(cursors[0].chunk_id, 100);
        assert_eq!(cursors[0].content_hash, 12345);
        assert_eq!(cursors[0].pulse_id, 0);
        assert!(cursors[0].rng_state.is_none());

        std::fs::remove_file(&sidecar_path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_load_legacy_cursor_missing() {
        let cursors = load_legacy_cursor("/nonexistent/model.safetensors");
        assert!(cursors.is_empty());
    }

    #[test]
    fn test_load_legacy_cursor_zero_position() {
        let dir = std::env::temp_dir().join("nl_hecate_cursor_test_zero");
        std::fs::create_dir_all(&dir).ok();
        let ckpt_path = dir.join("model.safetensors");
        let sidecar_path = dir.join("model.safetensors.cursor.json");

        let sidecar = r#"{"position": 0, "total_tokens": 0}"#;
        std::fs::write(&sidecar_path, sidecar).unwrap();

        let cursors = load_legacy_cursor(ckpt_path.to_str().unwrap());
        assert!(cursors.is_empty()); // position=0 means no meaningful cursor

        std::fs::remove_file(&sidecar_path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
