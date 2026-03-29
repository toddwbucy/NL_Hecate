use std::path::Path;
use std::io::{self, Read, Seek, SeekFrom};
use std::fs::File;

/// BPE token stream that reads flat numpy arrays.
/// Compatible with prepare_dolmino.py / prepare_gutenberg_deck.py output format.
pub struct BpeTokenStream {
    tokens: Vec<u32>,    // uint32 input tokens
    targets: Vec<i32>,   // int32 target tokens (-1 = masked)
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

        // Load numpy arrays
        let tokens = load_npy_u32(&dir.join("train_tokens.npy"))?;
        let targets = load_npy_i32(&dir.join("train_targets.npy"))?;

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

        let input_ids: Vec<usize> = self.tokens[self.position..self.position + seq_len]
            .iter()
            .map(|&t| t as usize)
            .collect();

        let target_ids: Vec<usize> = self.targets[self.position..self.position + seq_len]
            .iter()
            .map(|&t| {
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

/// Load a 1D numpy .npy file as Vec<u32>.
/// Supports numpy format 1.0: magic + version + header + raw data.
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

    // Reinterpret as u32 (little-endian, same as numpy default)
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
