pub mod base;
pub mod wgsl {
    pub use upstream_wgpu::WgslCompiler;
    pub type ComputeShader = <WgslCompiler as cubecl_core::Compiler>::Representation;
}
