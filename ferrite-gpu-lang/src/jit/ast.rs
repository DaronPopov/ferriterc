/// Abstract syntax tree for ferrite JIT scripts.
///
/// Scripts are flat sequences of statements that map 1:1 to the
/// `Program` builder ops.  Functions (`fn ... end`) are graph
/// templates that are inlined during lowering — no runtime call
/// overhead.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOper {
    Add,
    Sub,
    Mul,
    Div,
}

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
    BinOp { op: BinOper, left: Box<Expr>, right: Box<Expr> },
    /// Unary negation: `-x`
    Neg(Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum Arg {
    /// Positional argument: `x` or `[1, 2, 3]`
    Positional(Expr),
    /// Named argument: `dim=3`
    Named { name: String, value: Expr },
}

#[derive(Debug, Clone)]
pub enum Stmt {
    /// Variable binding: `x = relu(y)`
    Let { name: String, value: Expr },
    /// Return statement: `return x`
    Return(String),
    /// Function definition: `fn name(params): body end`
    FnDef {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    /// Tile block: `tile output over (inputs): body end`
    Tile {
        output: String,
        inputs: Vec<String>,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone)]
pub struct Script {
    pub stmts: Vec<Stmt>,
}
