/// Recursive-descent parser for ferrite JIT scripts.
///
/// Grammar (informal):
/// ```text
/// script     := stmt*
/// stmt       := fn_def | tile_stmt | return_stmt | let_stmt
/// fn_def     := 'fn' IDENT '(' params ')' ':' stmt* 'end'
/// params     := (IDENT (',' IDENT)*)?
/// return_stmt:= 'return' IDENT
/// let_stmt   := IDENT '=' expr
/// tile_stmt  := 'tile' IDENT 'over' input_list ':' stmt* 'end'
/// input_list := IDENT | '(' IDENT (',' IDENT)* ')'
///
/// expr       := add_expr
/// add_expr   := mul_expr (('+' | '-') mul_expr)*
/// mul_expr   := unary_expr (('*' | '/') unary_expr)*
/// unary_expr := '-' unary_expr | atom
/// atom       := '(' expr ')' | call | shape | float | int | bool | var
/// call       := IDENT '(' args ')'
/// args       := (arg (',' arg)*)?
/// arg        := IDENT '=' expr   (named)
///             | expr              (positional)
/// shape      := '[' INT (',' INT)* ']'
/// ```

use super::ast::*;
use super::lexer::{Span, Spanned, Token};
use super::JitError;

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Spanned>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ── public entry ─────────────────────────────────────────────

    pub fn parse(mut self) -> Result<Script, JitError> {
        let mut stmts = Vec::new();
        while !self.at_eof() {
            stmts.push(self.stmt()?);
        }
        Ok(Script { stmts })
    }

    // ── statements ───────────────────────────────────────────────

    fn stmt(&mut self) -> Result<Stmt, JitError> {
        match self.peek() {
            Token::Fn => self.fn_def(),
            Token::Tile => self.tile_stmt(),
            Token::Return => self.return_stmt(),
            Token::Ident(_) => self.let_stmt(),
            other => Err(self.err(format!("expected statement, got {:?}", other))),
        }
    }

    fn fn_def(&mut self) -> Result<Stmt, JitError> {
        self.eat(&Token::Fn)?;
        let name = self.ident()?;
        self.eat(&Token::LParen)?;
        let params = self.params()?;
        self.eat(&Token::RParen)?;
        self.eat(&Token::Colon)?;

        let mut body = Vec::new();
        while !self.check(&Token::End) && !self.at_eof() {
            body.push(self.stmt()?);
        }
        self.eat(&Token::End)?;

        Ok(Stmt::FnDef { name, params, body })
    }

    fn tile_stmt(&mut self) -> Result<Stmt, JitError> {
        self.eat(&Token::Tile)?;
        let output = self.ident()?;
        self.eat(&Token::Over)?;

        // input_list: IDENT | '(' IDENT (',' IDENT)* ')'
        let inputs = if self.check(&Token::LParen) {
            self.advance(); // eat '('
            let mut names = Vec::new();
            if !self.check(&Token::RParen) {
                names.push(self.ident()?);
                while self.check(&Token::Comma) {
                    self.advance();
                    if self.check(&Token::RParen) {
                        break; // trailing comma
                    }
                    names.push(self.ident()?);
                }
            }
            self.eat(&Token::RParen)?;
            names
        } else {
            vec![self.ident()?]
        };

        self.eat(&Token::Colon)?;

        let mut body = Vec::new();
        while !self.check(&Token::End) && !self.at_eof() {
            body.push(self.stmt()?);
        }
        self.eat(&Token::End)?;

        Ok(Stmt::Tile { output, inputs, body })
    }

    fn params(&mut self) -> Result<Vec<String>, JitError> {
        let mut out = Vec::new();
        if self.check(&Token::RParen) {
            return Ok(out);
        }
        out.push(self.ident()?);
        while self.check(&Token::Comma) {
            self.advance();
            out.push(self.ident()?);
        }
        Ok(out)
    }

    fn return_stmt(&mut self) -> Result<Stmt, JitError> {
        self.advance(); // eat 'return'
        let name = self.ident()?;
        Ok(Stmt::Return(name))
    }

    fn let_stmt(&mut self) -> Result<Stmt, JitError> {
        let name = self.ident()?;
        self.eat(&Token::Eq)?;
        let value = self.expr()?;
        Ok(Stmt::Let { name, value })
    }

    // ── expressions (precedence climbing) ─────────────────────────

    fn expr(&mut self) -> Result<Expr, JitError> {
        self.add_expr()
    }

    fn add_expr(&mut self) -> Result<Expr, JitError> {
        let mut left = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOper::Add,
                Token::Minus => BinOper::Sub,
                _ => break,
            };
            self.advance();
            let right = self.mul_expr()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> Result<Expr, JitError> {
        let mut left = self.unary_expr()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOper::Mul,
                Token::Slash => BinOper::Div,
                _ => break,
            };
            self.advance();
            let right = self.unary_expr()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn unary_expr(&mut self) -> Result<Expr, JitError> {
        if self.check(&Token::Minus) {
            self.advance();
            let inner = self.unary_expr()?;
            Ok(Expr::Neg(Box::new(inner)))
        } else {
            self.atom()
        }
    }

    fn atom(&mut self) -> Result<Expr, JitError> {
        match self.peek() {
            Token::LParen => {
                self.advance(); // eat '('
                let inner = self.expr()?;
                self.eat(&Token::RParen)?;
                Ok(inner)
            }
            Token::LBracket => self.shape(),
            Token::Float(v) => {
                self.advance();
                Ok(Expr::Float(v))
            }
            Token::Int(_) => self.int_lit(),
            Token::True => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Expr::Bool(false))
            }
            Token::Ident(_) => {
                let name = self.ident()?;
                if self.check(&Token::LParen) {
                    self.call(name)
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(self.err(format!("expected expression, got {:?}", other))),
        }
    }

    fn call(&mut self, func: String) -> Result<Expr, JitError> {
        self.eat(&Token::LParen)?;
        let args = self.args()?;
        self.eat(&Token::RParen)?;
        Ok(Expr::Call { func, args })
    }

    fn args(&mut self) -> Result<Vec<Arg>, JitError> {
        let mut out = Vec::new();
        if self.check(&Token::RParen) {
            return Ok(out);
        }
        out.push(self.arg()?);
        while self.check(&Token::Comma) {
            self.advance();
            out.push(self.arg()?);
        }
        Ok(out)
    }

    fn arg(&mut self) -> Result<Arg, JitError> {
        // Lookahead: IDENT followed by '=' → named argument
        if matches!(self.peek(), Token::Ident(_)) {
            if self.lookahead(1).map(|t| t == &Token::Eq).unwrap_or(false) {
                let name = self.ident()?;
                self.advance(); // eat '='
                let value = self.expr()?;
                return Ok(Arg::Named { name, value });
            }
        }
        let value = self.expr()?;
        Ok(Arg::Positional(value))
    }

    fn shape(&mut self) -> Result<Expr, JitError> {
        self.eat(&Token::LBracket)?;
        let mut dims = Vec::new();
        if !self.check(&Token::RBracket) {
            dims.push(self.expect_int()? as usize);
            while self.check(&Token::Comma) {
                self.advance();
                if self.check(&Token::RBracket) {
                    break; // trailing comma
                }
                dims.push(self.expect_int()? as usize);
            }
        }
        self.eat(&Token::RBracket)?;
        Ok(Expr::Shape(dims))
    }

    fn int_lit(&mut self) -> Result<Expr, JitError> {
        let v = self.expect_int()?;
        Ok(Expr::Int(v))
    }

    // ── helpers ──────────────────────────────────────────────────

    fn peek(&self) -> Token {
        self.tokens
            .get(self.pos)
            .map(|s| s.token.clone())
            .unwrap_or(Token::Eof)
    }

    fn lookahead(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|s| &s.token)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len() || self.tokens[self.pos].token == Token::Eof
    }

    fn check(&self, token: &Token) -> bool {
        self.peek() == *token
    }

    fn eat(&mut self, expected: &Token) -> Result<(), JitError> {
        let actual = self.peek();
        if actual == *expected {
            self.advance();
            Ok(())
        } else {
            Err(self.err(format!("expected {:?}, got {:?}", expected, actual)))
        }
    }

    fn ident(&mut self) -> Result<String, JitError> {
        match self.peek() {
            Token::Ident(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(self.err(format!("expected identifier, got {:?}", other))),
        }
    }

    fn expect_int(&mut self) -> Result<i64, JitError> {
        match self.peek() {
            Token::Int(v) => {
                self.advance();
                Ok(v)
            }
            other => Err(self.err(format!("expected integer, got {:?}", other))),
        }
    }

    fn span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|s| s.span)
            .unwrap_or(Span { line: 0, col: 0 })
    }

    fn err(&self, message: String) -> JitError {
        let s = self.span();
        JitError::Parse {
            line: s.line,
            col: s.col,
            message,
        }
    }
}

pub fn parse(tokens: Vec<Spanned>) -> Result<Script, JitError> {
    Parser::new(tokens).parse()
}
