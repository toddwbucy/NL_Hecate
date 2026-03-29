use std::path::Path;
use std::io::{Read, Seek, SeekFrom};
use std::fs::File;

use memmap2::Mmap;

/// Threshold for mmap vs read-into-memory (500 MB).
const MMAP_THRESHOLD: u64 = 500_000_000;

/// Storage backend: either a heap-allocated Vec or a memory-mapped file.
enum TokenStorage {
    Vec(Vec<u32>),
    Mmap { mmap: Mmap, offset: usize, len: usize },
}

impl TokenStorage {
    fn len(&self) -> usize {
        match self {
            TokenStorage::Vec(v) => v.len(),
            TokenStorage::Mmap { len, .. } => *len,
        }
    }

    fn get_u32(&self, idx: usize) -> u32 {
        match self {
            TokenStorage::Vec(v) => v[idx],
            TokenStorage::Mmap { mmap, offset, .. } => {
                let byte_off = *offset + idx * 4;
                let bytes = &mmap[byte_off..byte_off + 4];
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
        }
    }
}

enum TargetStorage {
    Vec(Vec<i32>),
    Mmap { mmap: Mmap, offset: usize, len: usize },
}

impl TargetStorage {
    fn len(&self) -> usize {
        match self {
            TargetStorage::Vec(v) => v.len(),
            TargetStorage::Mmap { len, .. } => *len,
        }
    }

    fn get_i32(&self, idx: usize) -> i32 {
        match self {
            TargetStorage::Vec(v) => v[idx],
            TargetStorage::Mmap { mmap, offset, .. } => {
                let byte_off = *offset + idx * 4;
                let bytes = &mmap[byte_off..byte_off + 4];
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
        }
    }
}

/// BPE token stream that reads flat numpy arrays.
/// Compatible with prepare_dolmino.py / prepare_gutenberg_deck.py output format.
///
/// For files > 500 MB, uses mmap to avoid loading the entire file into memory.
/// The OS page cache handles paging; RSS stays near zero for large corpora.
pub struct BpeTokenStream {
    tokens: TokenStorage,
    targets: TargetStorage,
    pub vocab_size: usize,
    pub position: usize,
    pub total_tokens: usize,
}

impl BpeTokenStream {
    /// Load from data directory containing train_tokens.npy, train_targets.npy, meta.json.
    pub fn load(data_dir: &str) -> Result<Self, String> {
        let dir = Path::new(data_dir);

        // Load meta.json for vocab_size
        let meta_path = dir.join("meta.json");
        let meta_text = std::fs::read_to_string(&meta_path)
            .map_err(|e| format!("Failed to read {}: {e}", meta_path.display()))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_text)
            .map_err(|e| format!("Failed to parse meta.json: {e}"))?;
        let vocab_size = meta["vocab_size"].as_u64()
            .ok_or("meta.json missing vocab_size")? as usize;

        let tokens_path = dir.join("train_tokens.npy");
        let targets_path = dir.join("train_targets.npy");

        let tokens_size = std::fs::metadata(&tokens_path)
            .map_err(|e| format!("Failed to stat {}: {e}", tokens_path.display()))?.len();
        let targets_size = std::fs::metadata(&targets_path)
            .map_err(|e| format!("Failed to stat {}: {e}", targets_path.display()))?.len();

        let use_mmap = tokens_size > MMAP_THRESHOLD || targets_size > MMAP_THRESHOLD;

        let (tokens, targets) = if use_mmap {
            eprintln!("  [mmap: tokens={}MB targets={}MB]",
                tokens_size / 1_000_000, targets_size / 1_000_000);
            let (tok_mmap, tok_offset, tok_len) = mmap_npy_u32(&tokens_path)?;
            let (tgt_mmap, tgt_offset, tgt_len) = mmap_npy_i32(&targets_path)?;
            (
                TokenStorage::Mmap { mmap: tok_mmap, offset: tok_offset, len: tok_len },
                TargetStorage::Mmap { mmap: tgt_mmap, offset: tgt_offset, len: tgt_len },
            )
        } else {
            let tok_vec = load_npy_u32(&tokens_path)?;
            let tgt_vec = load_npy_i32(&targets_path)?;
            (TokenStorage::Vec(tok_vec), TargetStorage::Vec(tgt_vec))
        };

        if tokens.len() != targets.len() {
            return Err(format!("Token/target length mismatch: {} vs {}",
                tokens.len(), targets.len()));
        }

        let total_tokens = tokens.len();
        Ok(BpeTokenStream { tokens, targets, vocab_size, position: 0, total_tokens })
    }

    /// Get next chunk of seq_len tokens. Wraps at end.
    pub fn next_chunk(&mut self, seq_len: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        if self.total_tokens < seq_len {
            return None;
        }
        if self.position + seq_len > self.total_tokens {
            self.position = 0; // wrap
        }

        let input_ids: Vec<usize> = (self.position..self.position + seq_len)
            .map(|i| self.tokens.get_u32(i) as usize)
            .collect();

        let target_ids: Vec<usize> = (self.position..self.position + seq_len)
            .map(|i| {
                let t = self.targets.get_i32(i);
                if t < 0 { self.vocab_size } else { t as usize }
            })
            .collect();

        self.position += seq_len;
        Some((input_ids, target_ids))
    }

    /// Seek to a specific position.
    pub fn seek(&mut self, pos: usize) {
        self.position = pos.min(self.total_tokens);
    }

    /// Get cursor state for sidecar serialization.
    pub fn cursor(&self) -> DataCursor {
        DataCursor {
            position: self.position as u64,
            total_tokens: self.total_tokens as u64,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct DataCursor {
    pub position: u64,
    pub total_tokens: u64,
}

// ── Mmap loaders ─────────────────────────────────────────────────────

/// Memory-map a numpy .npy u32 file. Returns (mmap, data_offset, n_elements).
fn mmap_npy_u32(path: &Path) -> Result<(Mmap, usize, usize), String> {
    let mut f = File::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let header_len = parse_npy_header(&mut f, path)?;
    let file_len = f.metadata()
        .map_err(|e| format!("metadata failed: {e}"))?.len() as usize;
    let data_bytes = file_len - header_len;
    if data_bytes % 4 != 0 {
        return Err(format!("{}: data section ({data_bytes} bytes) not aligned to 4-byte u32",
            path.display()));
    }
    let mmap = unsafe { Mmap::map(&f) }
        .map_err(|e| format!("mmap failed for {}: {e}", path.display()))?;
    Ok((mmap, header_len, data_bytes / 4))
}

/// Memory-map a numpy .npy i32 file. Returns (mmap, data_offset, n_elements).
fn mmap_npy_i32(path: &Path) -> Result<(Mmap, usize, usize), String> {
    let mut f = File::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let header_len = parse_npy_header(&mut f, path)?;
    let file_len = f.metadata()
        .map_err(|e| format!("metadata failed: {e}"))?.len() as usize;
    let data_bytes = file_len - header_len;
    if data_bytes % 4 != 0 {
        return Err(format!("{}: data section ({data_bytes} bytes) not aligned to 4-byte i32",
            path.display()));
    }
    let mmap = unsafe { Mmap::map(&f) }
        .map_err(|e| format!("mmap failed for {}: {e}", path.display()))?;
    Ok((mmap, header_len, data_bytes / 4))
}

// ── Vec loaders (small files) ────────────────────────────────────────

/// Load a 1D numpy .npy file as Vec<u32>.
fn load_npy_u32(path: &Path) -> Result<Vec<u32>, String> {
    let mut f = File::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;

    let header_len = parse_npy_header(&mut f, path)?;

    let file_len = f.metadata()
        .map_err(|e| format!("metadata failed: {e}"))?.len() as usize;
    if header_len > file_len {
        return Err(format!("{}: header_len ({header_len}) exceeds file size ({file_len})",
            path.display()));
    }
    let data_bytes = file_len - header_len;
    if data_bytes % 4 != 0 {
        return Err(format!("{}: data section ({data_bytes} bytes) not aligned to 4-byte u32",
            path.display()));
    }
    let n_elements = data_bytes / 4;

    f.seek(SeekFrom::Start(header_len as u64))
        .map_err(|e| format!("Seek failed: {e}"))?;

    let mut buf = vec![0u8; data_bytes];
    f.read_exact(&mut buf)
        .map_err(|e| format!("Read failed: {e}"))?;

    let mut result = vec![0u32; n_elements];
    for (i, chunk) in buf.chunks_exact(4).enumerate() {
        result[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(result)
}

/// Load a 1D numpy .npy file as Vec<i32>.
fn load_npy_i32(path: &Path) -> Result<Vec<i32>, String> {
    let mut f = File::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;

    let header_len = parse_npy_header(&mut f, path)?;

    let file_len = f.metadata()
        .map_err(|e| format!("metadata failed: {e}"))?.len() as usize;
    if header_len > file_len {
        return Err(format!("{}: header_len ({header_len}) exceeds file size ({file_len})",
            path.display()));
    }
    let data_bytes = file_len - header_len;
    if data_bytes % 4 != 0 {
        return Err(format!("{}: data section ({data_bytes} bytes) not aligned to 4-byte i32",
            path.display()));
    }
    let n_elements = data_bytes / 4;

    f.seek(SeekFrom::Start(header_len as u64))
        .map_err(|e| format!("Seek failed: {e}"))?;

    let mut buf = vec![0u8; data_bytes];
    f.read_exact(&mut buf)
        .map_err(|e| format!("Read failed: {e}"))?;

    let mut result = vec![0i32; n_elements];
    for (i, chunk) in buf.chunks_exact(4).enumerate() {
        result[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(result)
}

/// Parse numpy .npy header and return total header size (magic + version + header).
fn parse_npy_header(f: &mut File, path: &Path) -> Result<usize, String> {
    // Magic: \x93NUMPY
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic)
        .map_err(|e| format!("Failed to read npy magic from {}: {e}", path.display()))?;
    if &magic != b"\x93NUMPY" {
        return Err(format!("{}: not a numpy file", path.display()));
    }

    // Version
    let mut version = [0u8; 2];
    f.read_exact(&mut version)
        .map_err(|e| format!("Failed to read npy version: {e}"))?;

    // Header length
    let header_data_len = match version[0] {
        1 => {
            let mut len_bytes = [0u8; 2];
            f.read_exact(&mut len_bytes)
                .map_err(|e| format!("Failed to read header len: {e}"))?;
            u16::from_le_bytes(len_bytes) as usize
        }
        2 => {
            let mut len_bytes = [0u8; 4];
            f.read_exact(&mut len_bytes)
                .map_err(|e| format!("Failed to read header len: {e}"))?;
            u32::from_le_bytes(len_bytes) as usize
        }
        v => return Err(format!("{}: unsupported npy version {v}", path.display())),
    };

    // Total header = magic(6) + version(2) + len_field(2 or 4) + header_data
    let len_field_size = if version[0] == 1 { 2 } else { 4 };
    Ok(6 + 2 + len_field_size + header_data_len)
}
