use super::ast::Script;
use super::lower;
use super::Result;
use crate::Program;

/// Lower JIT AST into core ferrite `Program` IR.
pub fn lower_script(ast: Script) -> Result<Program> {
    lower::lower(ast)
}
