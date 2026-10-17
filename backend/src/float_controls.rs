use cubecl_spirv::SpirvKernel;
use rspirv::{
    binary::Assemble,
    dr::{Instruction, Operand},
    spirv::{Capability, ExecutionMode, Op},
};
use std::sync::Arc;

pub(crate) fn preserve_denormals(kernel: &mut SpirvKernel, version: (u8, u8)) {
    let module = Arc::make_mut(
        kernel
            .module
            .as_mut()
            .expect("fresh SPIR-V compilation includes its module"),
    );
    let mut declared = false;
    for width in [32, 64] {
        if !module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == Op::TypeFloat
                && instruction.operands == [Operand::LiteralBit32(width)]
        }) {
            continue;
        }
        if !declared {
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(Capability::DenormPreserve)],
            ));
            if version < (1, 4) {
                module.extensions.push(Instruction::new(
                    Op::Extension,
                    None,
                    None,
                    vec![Operand::LiteralString("SPV_KHR_float_controls".into())],
                ));
            }
            declared = true;
        }
        for entry in &module.entry_points {
            module.execution_modes.push(Instruction::new(
                Op::ExecutionMode,
                None,
                None,
                vec![
                    entry.operands[1].clone(),
                    Operand::ExecutionMode(ExecutionMode::DenormPreserve),
                    Operand::LiteralBit32(width),
                ],
            ));
        }
    }
    kernel.assembled_module = module.assemble();
}
