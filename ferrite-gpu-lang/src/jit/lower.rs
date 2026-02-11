/// Lowers a parsed AST into a `Program` (the existing graph-builder IR).
///
/// Builtin ops (`relu`, `add`, `input`, …) map 1:1 to `Program`
/// methods.  User-defined functions are **inlined** — the function
/// body is spliced into the caller's graph with parameters bound to
/// the caller's `ValueId`s.  No runtime call overhead.
///
/// Float literals and integer literals in expression position are
/// tracked as `Scalar(f64)` values until they meet a tensor operand,
/// at which point a `FillLike` op broadcasts them to the tensor's shape.

use std::collections::HashMap;

use super::ast::*;
use super::JitError;
use crate::{Program, ValueId};

/// Result of lowering an expression — either a concrete IR tensor or a
/// scalar constant that hasn't been materialised yet.
enum LowerResult {
    Tensor(ValueId),
    Scalar(f64),
}

struct Ctx {
    program: Program,
    names: HashMap<String, ValueId>,
    functions: HashMap<String, (Vec<String>, Vec<Stmt>)>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            program: Program::new(),
            names: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    // ── top-level ────────────────────────────────────────────────

    fn lower(mut self, script: Script) -> Result<Program, JitError> {
        // First pass: collect function definitions
        for stmt in &script.stmts {
            if let Stmt::FnDef { name, params, body } = stmt {
                self.functions
                    .insert(name.clone(), (params.clone(), body.clone()));
            }
        }

        // Second pass: process statements
        for stmt in &script.stmts {
            self.lower_stmt(stmt)?;
        }

        Ok(self.program)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), JitError> {
        match stmt {
            Stmt::FnDef { .. } => {} // already collected
            Stmt::Let { name, value } => {
                let vid = self.lower_expr_to_tensor(value)?;
                self.names.insert(name.clone(), vid);
            }
            Stmt::Return(name) => {
                let vid = self.resolve(name)?;
                self.program.set_output(vid);
            }
            Stmt::Tile { output, inputs, body } => {
                self.lower_tile(output, inputs, body)?;
            }
        }
        Ok(())
    }

    // ── tile lowering ───────────────────────────────────────────

    fn lower_tile(
        &mut self,
        output: &str,
        inputs: &[String],
        body: &[Stmt],
    ) -> Result<(), JitError> {
        // Verify all inputs exist
        for name in inputs {
            if !self.names.contains_key(name) {
                return Err(JitError::Lower(format!(
                    "tile: undefined input variable: {}",
                    name
                )));
            }
        }

        // Process body statements (same as top-level, but disallow fn/return/nested tile)
        for stmt in body {
            match stmt {
                Stmt::Let { name, value } => {
                    let vid = self.lower_expr_to_tensor(value)?;
                    self.names.insert(name.clone(), vid);
                }
                Stmt::FnDef { .. } => {
                    return Err(JitError::Lower(
                        "function definitions not allowed inside tile block".into(),
                    ));
                }
                Stmt::Return(_) => {
                    return Err(JitError::Lower(
                        "return not allowed inside tile block".into(),
                    ));
                }
                Stmt::Tile { .. } => {
                    return Err(JitError::Lower(
                        "nested tile blocks are not supported".into(),
                    ));
                }
            }
        }

        // Verify output was assigned
        if !self.names.contains_key(output) {
            return Err(JitError::Lower(format!(
                "tile: output variable '{}' was not assigned in body",
                output
            )));
        }

        Ok(())
    }

    // ── expressions ──────────────────────────────────────────────

    /// Lower an expression, requiring a tensor result.
    fn lower_expr_to_tensor(&mut self, expr: &Expr) -> Result<ValueId, JitError> {
        match self.lower_expr(expr)? {
            LowerResult::Tensor(vid) => Ok(vid),
            LowerResult::Scalar(_) => Err(JitError::Lower(
                "scalar literal cannot be used alone — it needs a tensor operand to infer shape"
                    .into(),
            )),
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<LowerResult, JitError> {
        match expr {
            Expr::Var(name) => Ok(LowerResult::Tensor(self.resolve(name)?)),
            Expr::Call { func, args } => {
                Ok(LowerResult::Tensor(self.lower_call(func, args)?))
            }
            Expr::Float(v) => Ok(LowerResult::Scalar(*v)),
            Expr::Int(v) => Ok(LowerResult::Scalar(*v as f64)),
            Expr::Neg(inner) => {
                match self.lower_expr(inner)? {
                    LowerResult::Scalar(v) => Ok(LowerResult::Scalar(-v)),
                    LowerResult::Tensor(vid) => {
                        // -x  →  FillLike(0.0, x) then Sub(fill, x)
                        let zero = self.program.fill_like(0.0, vid);
                        let neg = self.program.sub(zero, vid);
                        Ok(LowerResult::Tensor(neg))
                    }
                }
            }
            Expr::BinOp { op, left, right } => {
                let lhs = self.lower_expr(left)?;
                let rhs = self.lower_expr(right)?;
                let (l, r) = self.resolve_binop_operands(lhs, rhs)?;
                let vid = match op {
                    BinOper::Add => self.program.add(l, r),
                    BinOper::Sub => self.program.sub(l, r),
                    BinOper::Mul => self.program.mul(l, r),
                    BinOper::Div => self.program.div(l, r),
                };
                Ok(LowerResult::Tensor(vid))
            }
            other => Err(JitError::Lower(format!(
                "unexpected expression in statement position: {:?}",
                other
            ))),
        }
    }

    /// Resolve two LowerResult operands into (ValueId, ValueId),
    /// materializing scalars with FillLike when paired with a tensor.
    fn resolve_binop_operands(
        &mut self,
        lhs: LowerResult,
        rhs: LowerResult,
    ) -> Result<(ValueId, ValueId), JitError> {
        match (lhs, rhs) {
            (LowerResult::Tensor(l), LowerResult::Tensor(r)) => Ok((l, r)),
            (LowerResult::Tensor(t), LowerResult::Scalar(s)) => {
                let fill = self.program.fill_like(s, t);
                Ok((t, fill))
            }
            (LowerResult::Scalar(s), LowerResult::Tensor(t)) => {
                let fill = self.program.fill_like(s, t);
                Ok((fill, t))
            }
            (LowerResult::Scalar(_), LowerResult::Scalar(_)) => Err(JitError::Lower(
                "cannot perform binary op on two scalars — no tensor to infer shape from".into(),
            )),
        }
    }

    fn lower_call(&mut self, func: &str, args: &[Arg]) -> Result<ValueId, JitError> {
        match func {
            // ── builtins ─────────────────────────────────────────
            "input" => {
                let shape = self.positional_shape(args, 0)?;
                self.program.input(&shape).map_err(JitError::from)
            }
            "relu" => {
                let x = self.positional_expr(args, 0)?;
                Ok(self.program.relu(x))
            }
            "tanh" => {
                let x = self.positional_expr(args, 0)?;
                Ok(self.program.tanh(x))
            }
            "sigmoid" => {
                let x = self.positional_expr(args, 0)?;
                Ok(self.program.sigmoid(x))
            }
            "add" => {
                let lhs = self.positional_expr(args, 0)?;
                let rhs = self.positional_expr(args, 1)?;
                Ok(self.program.add(lhs, rhs))
            }
            "mul" => {
                let lhs = self.positional_expr(args, 0)?;
                let rhs = self.positional_expr(args, 1)?;
                Ok(self.program.mul(lhs, rhs))
            }
            "cumsum" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                Ok(self.program.cumsum(x, dim as usize))
            }
            "topk" => {
                let x = self.positional_expr(args, 0)?;
                let k = self.named_int(args, "k")?;
                let dim = self.named_int(args, "dim")?;
                let largest = self.named_bool(args, "largest").unwrap_or(true);
                Ok(self.program.topk(x, k as usize, dim as usize, largest))
            }

            // ── user-defined function → inline ───────────────────
            name => {
                let (params, body) = self
                    .functions
                    .get(name)
                    .ok_or_else(|| JitError::Lower(format!("unknown function: {}", name)))?
                    .clone();
                self.inline_call(&params, &body, args)
            }
        }
    }

    // ── function inlining ────────────────────────────────────────

    fn inline_call(
        &mut self,
        params: &[String],
        body: &[Stmt],
        args: &[Arg],
    ) -> Result<ValueId, JitError> {
        // Resolve positional arguments to ValueIds
        let mut arg_vids = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                Arg::Positional(expr) => arg_vids.push(self.lower_expr_to_tensor(expr)?),
                Arg::Named { .. } => {
                    return Err(JitError::Lower(
                        "named arguments not supported for user-defined functions".into(),
                    ));
                }
            }
        }

        if params.len() != arg_vids.len() {
            return Err(JitError::Lower(format!(
                "function expects {} arguments, got {}",
                params.len(),
                arg_vids.len(),
            )));
        }

        // Save outer scope, create function scope
        let outer = self.names.clone();
        for (param, vid) in params.iter().zip(&arg_vids) {
            self.names.insert(param.clone(), *vid);
        }

        // Process function body
        let mut return_vid = None;
        for stmt in body {
            match stmt {
                Stmt::Let { name, value } => {
                    let vid = self.lower_expr_to_tensor(value)?;
                    self.names.insert(name.clone(), vid);
                }
                Stmt::Return(name) => {
                    return_vid = Some(self.resolve(name)?);
                }
                Stmt::FnDef { .. } => {
                    return Err(JitError::Lower(
                        "nested function definitions are not supported".into(),
                    ));
                }
                Stmt::Tile { .. } => {
                    return Err(JitError::Lower(
                        "tile blocks not supported inside function definitions".into(),
                    ));
                }
            }
        }

        // Restore outer scope
        self.names = outer;

        return_vid.ok_or_else(|| JitError::Lower("function body missing return statement".into()))
    }

    // ── argument helpers ─────────────────────────────────────────

    fn resolve(&self, name: &str) -> Result<ValueId, JitError> {
        self.names
            .get(name)
            .copied()
            .ok_or_else(|| JitError::Lower(format!("undefined variable: {}", name)))
    }

    /// Lower a positional argument as an arbitrary expression to a ValueId.
    fn positional_expr(&mut self, args: &[Arg], index: usize) -> Result<ValueId, JitError> {
        match args.get(index) {
            Some(Arg::Positional(expr)) => self.lower_expr_to_tensor(expr),
            Some(other) => Err(JitError::Lower(format!(
                "expected positional argument at {}, got {:?}",
                index, other
            ))),
            None => Err(JitError::Lower(format!(
                "missing argument at position {}",
                index
            ))),
        }
    }

    fn positional_shape(&self, args: &[Arg], index: usize) -> Result<Vec<usize>, JitError> {
        match args.get(index) {
            Some(Arg::Positional(Expr::Shape(dims))) => Ok(dims.clone()),
            Some(other) => Err(JitError::Lower(format!(
                "expected shape at argument {}, got {:?}",
                index, other
            ))),
            None => Err(JitError::Lower(format!(
                "missing shape argument at position {}",
                index
            ))),
        }
    }

    fn named_int(&self, args: &[Arg], name: &str) -> Result<i64, JitError> {
        for arg in args {
            if let Arg::Named {
                name: n,
                value: Expr::Int(v),
            } = arg
            {
                if n == name {
                    return Ok(*v);
                }
            }
        }
        Err(JitError::Lower(format!(
            "missing required named argument: {}",
            name
        )))
    }

    fn named_bool(&self, args: &[Arg], name: &str) -> Option<bool> {
        for arg in args {
            if let Arg::Named {
                name: n,
                value: Expr::Bool(v),
            } = arg
            {
                if n == name {
                    return Some(*v);
                }
            }
        }
        None
    }
}

pub fn lower(script: Script) -> Result<Program, JitError> {
    Ctx::new().lower(script)
}
