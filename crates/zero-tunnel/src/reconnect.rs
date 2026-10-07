//! Auto-reconnect state machine and session resume token governor.

/// Reconnection policy and timing thresholds.
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    /// Maximum reconnection attempts before giving up (default: 20).
    pub max_attempts: u32,
    /// Base initial retry backoff in milliseconds (default: 500 ms).
    pub initial_interval_ms: u64,
    /// Ceiling maximum retry interval in milliseconds (default: 10 000 ms).
    pub max_interval_ms: u64,
    /// Inactivity timeout in ms after which connection is considered dropped (default: 2000 ms).
    pub drop_timeout_ms: u64,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 20,
            initial_interval_ms: 500,
            max_interval_ms: 10_000,
            drop_timeout_ms: 2_000,
        }
    }
}

impl ReconnectPolicy {
    /// Calculates exponential backoff interval in ms for an attempt number.
    pub fn calculate_backoff(&self, attempt: u32) -> u64 {
        let shift = attempt.min(10);
        let exp = self.initial_interval_ms.saturating_mul(1 << shift);
        exp.min(self.max_interval_ms)
    }
}

/// Dynamic connection health and auto-reconnection state.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconnectState {
    /// Normal operational streaming.
    Connected,
    /// Network degraded (high latency or packet loss detected).
    Degraded {
        /// Round-trip time in milliseconds.
        rtt_ms: f32,
        /// Packet loss rate ($0.0 \dots 1.0$).
        loss_rate: f32,
    },
    /// Connection severed, attempting reconnection.
    Reconnecting {
        /// Current attempt count (1-based).
        attempt: u32,
        /// Monotonic timestamp in ms for next retry.
        next_retry_ms: u64,
        /// Secret session resume token.
        resume_token: u64,
    },
    /// Reconnection successful; awaiting keyframe refresh.
    Resumed,
    /// Reconnection exhausted or aborted permanently.
    Terminated,
}

/// Auto-reconnection state machine.
#[derive(Debug)]
pub struct ReconnectManager {
    policy: ReconnectPolicy,
    state: ReconnectState,
    last_packet_received_ms: u64,
    resume_token: u64,
    last_rendered_frame: u32,
}

impl ReconnectManager {
    /// Creates a new reconnect manager with a secret session token.
    pub fn new(policy: ReconnectPolicy, resume_token: u64) -> Self {
        Self {
            policy,
            state: ReconnectState::Connected,
            last_packet_received_ms: 0,
            resume_token,
            last_rendered_frame: 0,
        }
    }

    /// Records receipt of a valid packet and resets heartbeat timer.
    pub fn on_packet_received(&mut self, now_ms: u64, frame_id: u32) {
        self.last_packet_received_ms = now_ms;
        if frame_id > self.last_rendered_frame {
            self.last_rendered_frame = frame_id;
        }

        if let ReconnectState::Reconnecting { .. } = self.state {
            self.state = ReconnectState::Resumed;
        } else if let ReconnectState::Degraded { .. } = self.state {
            self.state = ReconnectState::Connected;
        }
    }

    /// Ticks the state machine and checks for heartbeat drop timeouts.
    pub fn tick(&mut self, now_ms: u64, rtt_ms: f32, loss_rate: f32) -> &ReconnectState {
        if self.last_packet_received_ms == 0 {
            self.last_packet_received_ms = now_ms;
            return &self.state;
        }

        let elapsed = now_ms.saturating_sub(self.last_packet_received_ms);

        match &mut self.state {
            ReconnectState::Connected
            | ReconnectState::Degraded { .. }
            | ReconnectState::Resumed => {
                if elapsed > self.policy.drop_timeout_ms {
                    // Trigger auto-reconnect
                    let backoff = self.policy.calculate_backoff(0);
                    self.state = ReconnectState::Reconnecting {
                        attempt: 1,
                        next_retry_ms: now_ms + backoff,
                        resume_token: self.resume_token,
                    };
                } else if loss_rate > 0.05 || rtt_ms > 120.0 {
                    self.state = ReconnectState::Degraded { rtt_ms, loss_rate };
                } else {
                    self.state = ReconnectState::Connected;
                }
            }
            ReconnectState::Reconnecting {
                attempt,
                next_retry_ms,
                ..
            } => {
                if now_ms >= *next_retry_ms {
                    if *attempt >= self.policy.max_attempts {
                        self.state = ReconnectState::Terminated;
                    } else {
                        *attempt += 1;
                        let backoff = self.policy.calculate_backoff(*attempt);
                        *next_retry_ms = now_ms + backoff;
                    }
                }
            }
            ReconnectState::Terminated => {}
        }

        &self.state
    }

    /// Current reconnection lifecycle state.
    pub fn state(&self) -> &ReconnectState {
        &self.state
    }

    /// Highest rendered frame identifier before disconnect.
    pub fn last_rendered_frame(&self) -> u32 {
        self.last_rendered_frame
    }

    /// Secret session resume token.
    pub fn resume_token(&self) -> u64 {
        self.resume_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_backoff_progression() {
        let policy = ReconnectPolicy::default();
        assert_eq!(policy.calculate_backoff(0), 500);
        assert_eq!(policy.calculate_backoff(1), 1000);
        assert_eq!(policy.calculate_backoff(2), 2000);
        assert_eq!(policy.calculate_backoff(3), 4000);
        assert_eq!(policy.calculate_backoff(4), 8000);
        assert_eq!(policy.calculate_backoff(5), 10_000); // Capped
    }

    #[test]
    fn test_reconnect_fsm_lifecycle() {
        let mut mgr = ReconnectManager::new(ReconnectPolicy::default(), 0x1234_5678);
        mgr.on_packet_received(1000, 10);
        assert_eq!(*mgr.state(), ReconnectState::Connected);

        // Network degrades
        mgr.tick(1500, 150.0, 0.08);
        assert!(matches!(mgr.state(), ReconnectState::Degraded { .. }));

        // Timeout triggers reconnecting
        mgr.tick(3500, 200.0, 0.1);
        assert!(matches!(mgr.state(), ReconnectState::Reconnecting { .. }));

        // Packet arrives -> Resumes
        mgr.on_packet_received(4000, 11);
        assert_eq!(*mgr.state(), ReconnectState::Resumed);
    }
}
