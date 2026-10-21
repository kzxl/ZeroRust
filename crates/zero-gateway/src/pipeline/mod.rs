//! # Gravitational Pipeline Orchestration
//!
//! Sequentially executes registered middlewares across the request journey.

pub mod middleware;
pub mod ratelimit;

use middleware::{GatewayMiddleware, RequestContext};
use crate::error::GatewayError;

/// Ordered pipeline of middlewares.
pub struct Pipeline {
    middlewares: Vec<Box<dyn GatewayMiddleware>>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    pub fn use_middleware<M: GatewayMiddleware + 'static>(&mut self, mw: M) {
        self.middlewares.push(Box::new(mw));
    }

    /// Executes the pipeline sequentially over the given request context.
    pub fn run(&self, ctx: &mut RequestContext) -> Result<(), GatewayError> {
        for mw in &self.middlewares {
            mw.execute(ctx)?;
        }
        Ok(())
    }
}
