/// AST validation pass — runs between parse and lower.
///
/// Catches structural errors that the parser's grammar permits but that
/// the lowering pass would reject in confusing ways.  Errors are
/// reported with stable error codes and remediation hints.

use super::ast::*;
use super::JitError;

/// Validate a parsed script for structural correctness before lowering.
///
/// Returns `Ok(())` when the script is well-formed, or the first error
/// with a stable code and hint.
pub fn validate(script: &Script) -> Result<(), JitError> {
    let mut top_has_return = false;
    let mut defined_fns: Vec<&str> = Vec::new();

    for stmt in &script.stmts {
        match stmt {
            Stmt::FnDef { name, params, body } => {
                validate_fn_def(name, params, body)?;
                defined_fns.push(name);
            }
            Stmt::Tile { output, inputs, body } => {
                validate_tile(output, inputs, body)?;
            }
            Stmt::Let { value, .. } => {
                validate_expr(value)?;
            }
            Stmt::Return(_) => {
                top_has_return = true;
            }
        }
    }

    if !top_has_return {
        return Err(JitError::Validate {
            code: "V001",
            message: "script has no return statement".into(),
            hint: "add `return <variable>` as the last statement".into(),
        });
    }

    Ok(())
}

fn validate_fn_def(name: &str, params: &[String], body: &[Stmt]) -> Result<(), JitError> {
    if body.is_empty() {
        return Err(JitError::Validate {
            code: "V002",
            message: format!("function '{}' has an empty body", name),
            hint: "add at least one statement and a return to the function body".into(),
        });
    }

    // Check for duplicate parameter names
    let mut seen_params: Vec<&str> = Vec::new();
    for p in params {
        if seen_params.contains(&p.as_str()) {
            return Err(JitError::Validate {
                code: "V003",
                message: format!("function '{}' has duplicate parameter '{}'", name, p),
                hint: "use distinct parameter names".into(),
            });
        }
        seen_params.push(p);
    }

    // Check that the body ends with a return
    let has_return = body.iter().any(|s| matches!(s, Stmt::Return(_)));
    if !has_return {
        return Err(JitError::Validate {
            code: "V004",
            message: format!("function '{}' has no return statement", name),
            hint: "add `return <variable>` at the end of the function body".into(),
        });
    }

    // Validate expressions in body
    for stmt in body {
        match stmt {
            Stmt::Let { value, .. } => validate_expr(value)?,
            Stmt::FnDef { .. } => {
                return Err(JitError::Validate {
                    code: "V005",
                    message: "nested function definitions are not supported".into(),
                    hint: "move the inner function to the top level".into(),
                });
            }
            Stmt::Tile { .. } => {
                return Err(JitError::Validate {
                    code: "V006",
                    message: "tile blocks are not supported inside function definitions".into(),
                    hint: "move the tile block outside the function".into(),
                });
            }
            Stmt::Return(_) => {}
        }
    }

    Ok(())
}

fn validate_tile(output: &str, inputs: &[String], body: &[Stmt]) -> Result<(), JitError> {
    if inputs.is_empty() {
        return Err(JitError::Validate {
            code: "V007",
            message: "tile block has no input variables".into(),
            hint: "specify at least one input: `tile y over (x): ... end`".into(),
        });
    }

    if body.is_empty() {
        return Err(JitError::Validate {
            code: "V008",
            message: format!("tile block for '{}' has an empty body", output),
            hint: "add statements to compute the output".into(),
        });
    }

    // Check for duplicate input names
    let mut seen: Vec<&str> = Vec::new();
    for inp in inputs {
        if seen.contains(&inp.as_str()) {
            return Err(JitError::Validate {
                code: "V009",
                message: format!("tile block has duplicate input '{}'", inp),
                hint: "use distinct input names".into(),
            });
        }
        seen.push(inp);
    }

    // Output name must not collide with input names
    if inputs.iter().any(|i| i == output) {
        return Err(JitError::Validate {
            code: "V010",
            message: format!(
                "tile output '{}' shadows an input of the same name",
                output
            ),
            hint: "use a distinct name for the tile output".into(),
        });
    }

    // Validate body statements
    for stmt in body {
        match stmt {
            Stmt::Let { value, .. } => validate_expr(value)?,
            Stmt::FnDef { .. } => {
                return Err(JitError::Validate {
                    code: "V005",
                    message: "function definitions not allowed inside tile block".into(),
                    hint: "move the function to the top level".into(),
                });
            }
            Stmt::Tile { .. } => {
                return Err(JitError::Validate {
                    code: "V011",
                    message: "nested tile blocks are not supported".into(),
                    hint: "flatten the tile blocks into sequential statements".into(),
                });
            }
            Stmt::Return(_) => {
                return Err(JitError::Validate {
                    code: "V012",
                    message: "return not allowed inside tile block".into(),
                    hint: "assign the result to the tile output variable instead".into(),
                });
            }
        }
    }

    Ok(())
}

/// Validate an expression tree for structural correctness.
fn validate_expr(expr: &Expr) -> Result<(), JitError> {
    match expr {
        Expr::Shape(dims) => {
            if dims.is_empty() {
                return Err(JitError::Validate {
                    code: "V013",
                    message: "empty shape literal `[]`".into(),
                    hint: "shapes must have at least one dimension, e.g. `[4]`".into(),
                });
            }
            for (i, &d) in dims.iter().enumerate() {
                if d == 0 {
                    return Err(JitError::Validate {
                        code: "V014",
                        message: format!(
                            "shape literal has zero at dimension {} — `{:?}`",
                            i, dims
                        ),
                        hint: "all shape dimensions must be >= 1".into(),
                    });
                }
            }
        }
        Expr::Call { args, .. } => {
            for arg in args {
                match arg {
                    Arg::Positional(e) => validate_expr(e)?,
                    Arg::Named { value, .. } => validate_expr(value)?,
                }
            }
        }
        Expr::BinOp { left, right, .. } => {
            validate_expr(left)?;
            validate_expr(right)?;
        }
        Expr::Neg(inner) => validate_expr(inner)?,
        Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::lexer::tokenize;
    use crate::jit::parser;

    fn parse_script(src: &str) -> Script {
        let toks = tokenize(src).unwrap();
        parser::parse(toks).unwrap()
    }

    #[test]
    fn validate_ok_simple() {
        let script = parse_script("x = input([1, 4])\ny = relu(x)\nreturn y");
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_missing_return() {
        let script = parse_script("x = input([1, 4])\ny = relu(x)");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V001"));
    }

    #[test]
    fn validate_empty_fn_body() {
        // Parser allows fn with empty body; validation catches it.
        let script = parse_script("fn noop(): end\nx = input([1])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V002"));
    }

    #[test]
    fn validate_fn_missing_return() {
        let script = parse_script("fn bad(x): y = relu(x) end\nx = input([4])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V004"));
    }

    #[test]
    fn validate_duplicate_fn_params() {
        let script = parse_script("fn dup(x, x): return x end\na = input([4])\nreturn a");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V003"));
    }

    #[test]
    fn validate_empty_shape() {
        let script = parse_script("x = input([])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V013"));
    }

    #[test]
    fn validate_zero_dim_shape() {
        let script = parse_script("x = input([1, 0, 4])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V014"));
    }

    #[test]
    fn validate_tile_no_inputs() {
        // Can't construct this with the parser (grammar requires at least one input),
        // but the validator handles it defensively.
    }

    #[test]
    fn validate_tile_empty_body() {
        let script = parse_script("x = input([4])\ntile y over x: end\nreturn y");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V008"));
    }

    #[test]
    fn validate_tile_output_shadows_input() {
        let script = parse_script("x = input([4])\ntile x over x: x = x * 2.0 end\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V010"));
    }

    #[test]
    fn validate_ok_with_fn() {
        let script = parse_script(
            "fn activate(x):\n  h = relu(x)\n  return h\nend\na = input([4])\nb = activate(a)\nreturn b",
        );
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_ok_with_tile() {
        let script = parse_script(
            "x = input([4])\ntile y over (x):\n  y = x * 2.0 + 1.0\nend\nreturn y",
        );
        assert!(validate(&script).is_ok());
    }
}
