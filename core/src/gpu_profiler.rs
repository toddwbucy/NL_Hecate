//! Per-step GPU profiling via cudaEvent timing.
//!
//! `GpuProfiler` wraps cudaEvent pairs around named pipeline regions to produce
//! a wall-clock breakdown of each training step. Zero overhead when disabled
//! (single branch on `enabled` flag — no cudaEvent allocation, no sync barriers).
//!
//! Usage:
//!   let mut prof = GpuProfiler::new(true);
//!   prof.start("swa_fwd", ProfileCategory::Attention, Some(0), None);
//!   swa_forward_dd(...);
//!   prof.stop();
//!   // ... more regions ...
//!   let profile = prof.collect();  // synchronizes + computes elapsed times
//!
//! Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventElapsedTime(
        ms: *mut f32,
        start: *mut std::ffi::c_void,
        stop: *mut std::ffi::c_void,
    ) -> i32;
}

/// Category tag for grouping related profiling regions.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileCategory {
    Embedding,
    LayerNorm,
    Projection,
    Attention,
    Precision,
    MemoryForward,
    MemoryBackward,
    GateCompute,
    Composition,
    Residual,
    Loss,
    MNormClamp,
    Optimizer,
    OptimizerNS,
    GradClip,
    Sync,
}

#[cfg(feature = "cuda")]
impl ProfileCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embedding => "Embedding",
            Self::LayerNorm => "LayerNorm",
            Self::Projection => "Projection",
            Self::Attention => "Attention",
            Self::Precision => "Precision",
            Self::MemoryForward => "MemoryForward",
            Self::MemoryBackward => "MemoryBackward",
            Self::GateCompute => "GateCompute",
            Self::Composition => "Composition",
            Self::Residual => "Residual",
            Self::Loss => "Loss",
            Self::MNormClamp => "MNormClamp",
            Self::Optimizer => "Optimizer",
            Self::OptimizerNS => "OptimizerNS",
            Self::GradClip => "GradClip",
            Self::Sync => "Sync",
        }
    }
}

/// One recorded profiling region with start/stop cudaEvent handles.
#[cfg(feature = "cuda")]
struct ProfileEvent {
    name: &'static str,
    category: ProfileCategory,
    block_idx: Option<usize>,
    level_idx: Option<usize>,
    start: *mut std::ffi::c_void,
    stop: *mut std::ffi::c_void,
}

/// Collected timing for one region.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct ComponentTiming {
    pub name: String,
    pub category: ProfileCategory,
    pub block_idx: Option<usize>,
    pub level_idx: Option<usize>,
    pub ms: f32,
    pub pct: f32,
}

/// Aggregated timing for one category.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CategoryTiming {
    pub category: ProfileCategory,
    pub ms: f32,
    pub pct: f32,
}

/// Per-block timing summary.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct BlockTiming {
    pub block_idx: usize,
    pub fwd_ms: f32,
    pub bwd_ms: f32,
    pub opt_ms: f32,
}

/// Complete profile for one training step.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct StepProfile {
    pub total_ms: f32,
    pub components: Vec<ComponentTiming>,
    pub by_category: Vec<CategoryTiming>,
    pub per_block: Vec<BlockTiming>,
}

/// GPU profiler using cudaEvent pairs. Zero overhead when disabled.
#[cfg(feature = "cuda")]
pub struct GpuProfiler {
    enabled: bool,
    events: Vec<ProfileEvent>,
    /// Tracks the overall step start/stop for total wall-clock.
    step_start: *mut std::ffi::c_void,
    step_stop: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
fn create_event() -> *mut std::ffi::c_void {
    let mut event: *mut std::ffi::c_void = std::ptr::null_mut();
    let rc = unsafe { cudaEventCreate(&mut event) };
    assert_eq!(rc, 0, "cudaEventCreate failed: error code {rc}");
    event
}

#[cfg(feature = "cuda")]
fn record_event(event: *mut std::ffi::c_void) {
    let rc = unsafe { cudaEventRecord(event, std::ptr::null_mut()) };
    assert_eq!(rc, 0, "cudaEventRecord failed: error code {rc}");
}

#[cfg(feature = "cuda")]
impl GpuProfiler {
    /// Create a new profiler. When `enabled=false`, all methods are no-ops.
    pub fn new(enabled: bool) -> Self {
        let (step_start, step_stop) = if enabled {
            (create_event(), create_event())
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        GpuProfiler {
            enabled,
            events: Vec::with_capacity(if enabled { 128 } else { 0 }),
            step_start,
            step_stop,
        }
    }

    /// Whether profiling is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record the step start timestamp. Call once before forward pass.
    pub fn step_start(&self) {
        if !self.enabled {
            return;
        }
        record_event(self.step_start);
    }

    /// Record the step stop timestamp. Call once after cuda_sync at end of step.
    pub fn step_stop(&self) {
        if !self.enabled {
            return;
        }
        record_event(self.step_stop);
    }

    /// Begin a named profiling region.
    #[inline]
    pub fn start(
        &mut self,
        name: &'static str,
        category: ProfileCategory,
        block_idx: Option<usize>,
        level_idx: Option<usize>,
    ) {
        if !self.enabled {
            return;
        }
        let start = create_event();
        let stop = create_event();
        record_event(start);
        self.events.push(ProfileEvent {
            name,
            category,
            block_idx,
            level_idx,
            start,
            stop,
        });
    }

    /// End the current profiling region.
    #[inline]
    pub fn stop(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(ev) = self.events.last() {
            record_event(ev.stop);
        }
    }

    /// Synchronize GPU, compute elapsed times, and return the step profile.
    /// Destroys all cudaEvents — call `reset()` to reuse the profiler.
    pub fn collect(&mut self, n_blocks: usize) -> StepProfile {
        if !self.enabled || self.events.is_empty() {
            return StepProfile {
                total_ms: 0.0,
                components: Vec::new(),
                by_category: Vec::new(),
                per_block: Vec::new(),
            };
        }

        // Synchronize on the last stop event to ensure all work is complete.
        let rc = unsafe { cudaEventSynchronize(self.step_stop) };
        assert_eq!(rc, 0, "cudaEventSynchronize failed: error code {rc}");

        // Total step time.
        let mut total_ms = 0.0f32;
        let rc = unsafe {
            cudaEventElapsedTime(&mut total_ms, self.step_start, self.step_stop)
        };
        assert_eq!(rc, 0, "cudaEventElapsedTime (total) failed: error code {rc}");

        // Per-region timings.
        let mut components = Vec::with_capacity(self.events.len());
        for ev in &self.events {
            let mut ms = 0.0f32;
            let rc = unsafe { cudaEventElapsedTime(&mut ms, ev.start, ev.stop) };
            assert_eq!(rc, 0, "cudaEventElapsedTime failed: error code {rc}");
            let pct = if total_ms > 0.0 { ms / total_ms * 100.0 } else { 0.0 };
            components.push(ComponentTiming {
                name: ev.name.to_string(),
                category: ev.category,
                block_idx: ev.block_idx,
                level_idx: ev.level_idx,
                ms,
                pct,
            });
        }

        // Aggregate by category.
        let mut cat_map: std::collections::HashMap<ProfileCategory, f32> =
            std::collections::HashMap::new();
        for c in &components {
            *cat_map.entry(c.category).or_insert(0.0) += c.ms;
        }
        let mut by_category: Vec<CategoryTiming> = cat_map
            .into_iter()
            .map(|(cat, ms)| CategoryTiming {
                category: cat,
                ms,
                pct: if total_ms > 0.0 { ms / total_ms * 100.0 } else { 0.0 },
            })
            .collect();
        by_category.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap_or(std::cmp::Ordering::Equal));

        // Per-block breakdown.
        let fwd_cats = [
            ProfileCategory::LayerNorm,
            ProfileCategory::Projection,
            ProfileCategory::Attention,
            ProfileCategory::Precision,
            ProfileCategory::MemoryForward,
            ProfileCategory::GateCompute,
            ProfileCategory::Composition,
            ProfileCategory::Residual,
            ProfileCategory::MNormClamp,
        ];
        let bwd_cats = [ProfileCategory::MemoryBackward];
        let opt_cats = [ProfileCategory::Optimizer, ProfileCategory::OptimizerNS];

        let mut per_block = Vec::with_capacity(n_blocks);
        for bi in 0..n_blocks {
            let mut fwd = 0.0f32;
            let mut bwd = 0.0f32;
            let mut opt = 0.0f32;
            for c in &components {
                if c.block_idx == Some(bi) {
                    if fwd_cats.contains(&c.category) {
                        fwd += c.ms;
                    } else if bwd_cats.contains(&c.category) {
                        bwd += c.ms;
                    } else if opt_cats.contains(&c.category) {
                        opt += c.ms;
                    }
                }
            }
            per_block.push(BlockTiming {
                block_idx: bi,
                fwd_ms: fwd,
                bwd_ms: bwd,
                opt_ms: opt,
            });
        }

        // Sort components by time descending for the top-N display.
        components.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap_or(std::cmp::Ordering::Equal));

        StepProfile {
            total_ms,
            components,
            by_category,
            per_block,
        }
    }

    /// Reset the profiler for the next step — destroys all cudaEvents.
    pub fn reset(&mut self) {
        if !self.enabled {
            return;
        }
        for ev in self.events.drain(..) {
            unsafe {
                cudaEventDestroy(ev.start);
                cudaEventDestroy(ev.stop);
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuProfiler {
    fn drop(&mut self) {
        for ev in &self.events {
            unsafe {
                cudaEventDestroy(ev.start);
                cudaEventDestroy(ev.stop);
            }
        }
        if self.enabled {
            unsafe {
                cudaEventDestroy(self.step_start);
                cudaEventDestroy(self.step_stop);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Convenience macros — zero boilerplate for instrumentation sites.
// Usage:  prof_start!(profiler, "swa_fwd", Attention, Some(b), None);
//         ...work...
//         prof_stop!(profiler);
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! prof_start {
    ($prof:expr, $name:expr, $cat:ident, $block:expr, $level:expr) => {
        if let Some(ref mut _p) = $prof {
            _p.start($name, $crate::gpu_profiler::ProfileCategory::$cat, $block, $level);
        }
    };
}

#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! prof_stop {
    ($prof:expr) => {
        if let Some(ref mut _p) = $prof {
            _p.stop();
        }
    };
}
