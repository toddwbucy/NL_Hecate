#![feature(autodiff)]

pub mod tensor;
pub mod swa;
pub mod model;
pub mod forward;
pub mod backward;
pub mod delta_rule;
pub mod titans_lmm;
pub mod hebbian_rule;
pub mod moneta;
pub mod yaad;
pub mod memora;
pub mod lattice_osr;
pub mod trellis;
pub mod mag;
pub mod mal;
pub mod mac;
pub mod parallel;
pub mod chunkwise_gd;
pub mod associative_scan;
pub mod tnt;
pub mod lattice_gla;
pub mod atlas_parallel;
#[cfg(feature = "internal")]
pub mod gradient;
#[cfg(not(feature = "internal"))]
pub(crate) mod gradient;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_ffi;
pub mod conductor;
pub mod context_stream;
pub mod dispatch;
#[cfg(feature = "distributed")]
pub mod distributed;
#[cfg(feature = "serving")]
pub mod serving;
