// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

//! Shared validation for the text and binary input of `rabitq8` and `rabitq4`.
//!
//! The text form is `(sum_of_x2, norm_of_lattice, sum_of_code, sum_of_abs_x)[c, c, ...]`.
//! Whitespace may appear around every token but ends a number, and nothing
//! but whitespace may follow `]`. Everything the output functions print is
//! accepted, so dumps restore.

use pgrx::PgSqlErrorCode;
use std::ffi::CStr;

/// The largest dimension the on-disk header (`dim: u16`) can represent.
pub const MAX_DIM: u32 = u16::MAX as u32;

#[derive(Debug, Clone, Copy)]
pub struct Metadata {
    pub sum_of_x2: f32,
    pub norm_of_lattice: f32,
    pub sum_of_code: f32,
    pub sum_of_abs_x: f32,
}

fn is_space(c: u8) -> bool {
    // The same set as PostgreSQL's `scanner_isspace` / `array_isspace`.
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn is_token(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'.' | b'+' | b'-')
}

fn syntax_error(type_name: &str, input: &CStr) -> ! {
    pgrx::ereport!(
        ERROR,
        PgSqlErrorCode::ERRCODE_INVALID_TEXT_REPRESENTATION,
        format!(
            "invalid input syntax for type {type_name}: \"{}\"",
            String::from_utf8_lossy(input.to_bytes())
        )
    );
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn skip_space(&mut self) {
        while self.pos < self.bytes.len() && is_space(self.bytes[self.pos]) {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn token(&mut self) -> &'a [u8] {
        let start = self.pos;
        while self.pos < self.bytes.len() && is_token(self.bytes[self.pos]) {
            self.pos += 1;
        }
        &self.bytes[start..self.pos]
    }
    /// Parses `open item (',' item)* close`, with whitespace around items.
    /// Returns `false` on a syntax error. An empty list is accepted here and
    /// rejected by the caller with a more specific message.
    fn list(&mut self, open: u8, close: u8, mut item: impl FnMut(&'a [u8]) -> bool) -> bool {
        self.skip_space();
        if !self.eat(open) {
            return false;
        }
        self.skip_space();
        if self.eat(close) {
            return true;
        }
        loop {
            self.skip_space();
            let token = self.token();
            if token.is_empty() || !item(token) {
                return false;
            }
            self.skip_space();
            if self.eat(close) {
                return true;
            }
            if !self.eat(b',') {
                return false;
            }
        }
    }
}

/// Parses the text form. Codes above `max_code` are rejected.
pub fn parse_text(input: &CStr, type_name: &str, max_code: u8) -> (Metadata, Vec<u8>) {
    let mut cursor = Cursor {
        bytes: input.to_bytes(),
        pos: 0,
    };
    let mut metadata = Vec::<f32>::with_capacity(4);
    if !cursor.list(b'(', b')', |token| {
        let Ok(token) = std::str::from_utf8(token) else {
            return false;
        };
        match token.parse::<f32>() {
            Ok(value) => {
                metadata.push(value);
                true
            }
            Err(_) => false,
        }
    }) {
        syntax_error(type_name, input);
    }
    let mut codes = Vec::<u8>::new();
    let mut out_of_range = None;
    if !cursor.list(b'[', b']', |token| {
        use std::num::IntErrorKind;
        let Ok(text) = std::str::from_utf8(token) else {
            return false;
        };
        match text.parse::<i64>() {
            Ok(value) if (0..=max_code as i64).contains(&value) => {
                codes.push(value as u8);
                true
            }
            Ok(_) => {
                out_of_range = Some(text.to_owned());
                false
            }
            Err(e)
                if matches!(
                    e.kind(),
                    IntErrorKind::PosOverflow | IntErrorKind::NegOverflow
                ) =>
            {
                out_of_range = Some(text.to_owned());
                false
            }
            Err(_) => false,
        }
    }) {
        if let Some(code) = out_of_range {
            pgrx::ereport!(
                ERROR,
                PgSqlErrorCode::ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
                format!("code \"{code}\" is out of range for type {type_name}"),
                format!("Codes must be between 0 and {max_code}.")
            );
        }
        syntax_error(type_name, input);
    }
    cursor.skip_space();
    if cursor.pos != cursor.bytes.len() {
        syntax_error(type_name, input);
    }
    let [sum_of_x2, norm_of_lattice, sum_of_code, sum_of_abs_x] = metadata[..] else {
        syntax_error(type_name, input);
    };
    let metadata = Metadata {
        sum_of_x2,
        norm_of_lattice,
        sum_of_code,
        sum_of_abs_x,
    };
    check_dim(type_name, codes.len().try_into().unwrap_or(u32::MAX));
    (metadata, codes)
}

/// Rejects dimensions that cannot be stored.
pub fn check_dim(type_name: &str, dim: u32) {
    if dim == 0 {
        pgrx::ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_DATA_EXCEPTION,
            format!("{type_name} must have at least 1 dimension")
        );
    }
    if dim > MAX_DIM {
        pgrx::ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
            format!("{type_name} cannot have more than {MAX_DIM} dimensions")
        );
    }
}

/// Raised by quantization when the metadata cannot be represented, for
/// example when the squared norm of the input overflows `real`.
pub fn check_quantized(type_name: &str, metadata: Metadata) {
    let values = [
        metadata.sum_of_x2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        metadata.sum_of_abs_x,
    ];
    if values.iter().any(|x| !x.is_finite()) {
        pgrx::ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
            "value out of range: overflow",
            format!("The vector is too large to be quantized to {type_name}.")
        );
    }
}

/// A bounds-checked reader over the receive buffer of a binary input function.
pub struct Reader<'a> {
    buf: &'a mut pgrx::pg_sys::StringInfoData,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a mut pgrx::pg_sys::StringInfoData) -> Self {
        Self { buf }
    }
    pub fn bytes(&mut self, n: usize) -> &'a [u8] {
        let cursor = usize::try_from(self.buf.cursor).unwrap_or(usize::MAX);
        let len = usize::try_from(self.buf.len).unwrap_or(0);
        if cursor > len || n > len - cursor {
            pgrx::ereport!(
                ERROR,
                PgSqlErrorCode::ERRCODE_PROTOCOL_VIOLATION,
                "insufficient data left in message"
            );
        }
        let result =
            unsafe { std::slice::from_raw_parts(self.buf.data.add(cursor).cast::<u8>(), n) };
        self.buf.cursor += n as i32;
        result
    }
    pub fn array<const N: usize>(&mut self) -> [u8; N] {
        self.bytes(N).try_into().expect("the length is checked")
    }
    pub fn u32(&mut self) -> u32 {
        u32::from_be_bytes(self.array())
    }
    pub fn f32(&mut self) -> f32 {
        f32::from_be_bytes(self.array())
    }
    pub fn metadata(&mut self) -> Metadata {
        Metadata {
            sum_of_x2: self.f32(),
            norm_of_lattice: self.f32(),
            sum_of_code: self.f32(),
            sum_of_abs_x: self.f32(),
        }
    }
}

/// pgvector's `vector` and `halfvec` accept at most this many dimensions.
pub const PGVECTOR_MAX_DIM: u32 = 16000;

/// Rejects dequantized values that pgvector would refuse to read back:
/// more than 16000 dimensions or non-finite coordinates.
pub fn check_dequantized(target: &str, dim: u32, finite: bool) {
    if dim > PGVECTOR_MAX_DIM {
        pgrx::ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
            format!("{target} cannot have more than {PGVECTOR_MAX_DIM} dimensions")
        );
    }
    if !finite {
        pgrx::ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
            format!("dequantized value is out of range for type {target}")
        );
    }
}
