/// Abstract syntax tree for ferrite JIT scripts.
///
/// Scripts are flat sequences of statements that map 1:1 to the
/// `Program` builder ops.  Functions (`fn ... end`) are graph
/// templates that are inlined during lowering — no runtime call
/// overhead.

use super::lexer::Span;

// ── Spanned wrapper ────────────────────────────────────────────

/// A value paired with its source span.
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// ── Operators ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOper {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
}

// ── Expressions ────────────────────────────────────────────────

/// Convenience alias for a spanned expression.
pub type SExpr = Spanned<Expr>;

#[derive(Debug, Clone)]
pub enum Expr {
    /// Variable reference: `x`
    Var(String),
    /// Integer literal: `42`
    Int(i64),
    /// Float literal: `2.0`, `0.5`, `3.14`
    Float(f64),
    /// Boolean literal: `true` / `false`
    Bool(bool),
    /// Shape literal: `[1, 1, 1, 4096]`
    Shape(Vec<usize>),
    /// Function / builtin call: `relu(x)` or `topk(x, k=10, dim=3)`
    Call { func: String, args: Vec<Arg> },
    /// Binary operation: `x + y`, `a * b`
    BinOp { op: BinOper, left: Box<SExpr>, right: Box<SExpr> },
    /// Unary negation: `-x`
    Neg(Box<SExpr>),
    /// Comparison: `a < b`, `x == y`
    Cmp { op: CmpOp, left: Box<SExpr>, right: Box<SExpr> },
    /// Logical: `a and b`, `x or y`
    Logic { op: LogicOp, left: Box<SExpr>, right: Box<SExpr> },
    /// Logical negation: `not x`
    LogicNot(Box<SExpr>),
}

#[derive(Debug, Clone)]
pub enum Arg {
    /// Positional argument: `x` or `[1, 2, 3]`
    Positional(SExpr),
    /// Named argument: `dim=3`
    Named { name: String, value: SExpr },
}

// ── Statements ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Stmt {
    /// Variable binding: `x = relu(y)`
    Let { name: String, value: SExpr, span: Span },
    /// Return statement: `return x`
    Return { name: String, span: Span },
    /// Function definition: `fn name(params): body end`
    FnDef {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
        span: Span,
    },
    /// Tile block: `tile output over (inputs): body end`
    Tile {
        output: String,
        inputs: Vec<String>,
        body: Vec<Stmt>,
        span: Span,
    },
    /// Conditional: `if cond then body (elif cond then body)* (else body)? end`
    If {
        branches: Vec<(SExpr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    /// Bounded loop: `for var in start..end: body end`
    For {
        var: String,
        start: i64,
        end: i64,
        body: Vec<Stmt>,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct Script {
    pub stmts: Vec<Stmt>,
}
