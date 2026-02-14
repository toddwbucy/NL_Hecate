#![feature(autodiff)]

pub mod tensor;
pub mod swa;
pub mod model;
pub mod forward;
pub mod backward;
pub mod delta_rule;
pub mod titans_lmm;
pub mod mag;
#[cfg(feature = "internal")]
pub mod gradient;
#[cfg(not(feature = "internal"))]
pub(crate) mod gradient;

#[cfg(feature = "cuda")]
pub(crate) mod cuda_ffi;
pub mod conductor;
pub mod dispatch;
