//! # cuda-resolve-agent
//!
//! Deliberative A2A agent built on cuda-equipment.
//! Implements Consider/Resolve/Forfeit protocol with Bayesian confidence.
//!
//! ```rust
//! use cuda_resolve_agent::{ResolveAgent, DeliberationConfig};
//! use cuda_equipment::{Confidence, FleetMessage, MessageType, VesselId, Fleet};
//!
//! let mut fleet = Fleet::new();
//! let agent = ResolveAgent::new(1, "architect", DeliberationConfig::default());
//! fleet.register(Box::new(agent));
//!
//! let responses = fleet.send(&FleetMessage::new(
//!     VesselId(0), VesselId(1),
//!     MessageType::Consider { proposal_id: 42 },
//! ));
//! ```

pub use cuda_equipment::{Confidence, FleetMessage, MessageType, VesselId, TileId,
    Agent, BaseAgent, EquipmentRegistry, SensorType, ActuatorType, Provenance};

/// Configuration for deliberation behavior.
#[derive(Debug, Clone)]
pub struct DeliberationConfig {
    pub confidence_threshold: f64,
    pub max_rounds: u32,
    pub decay_rate: f64,
    pub auto_accept_threshold: f64,
    pub auto_forfeit_threshold: f64,
}

impl Default for DeliberationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.85,
            max_rounds: 10,
            decay_rate: 0.95,
            auto_accept_threshold: 0.90,
            auto_forfeit_threshold: 0.20,
        }
    }
}

/// A proposal being deliberated.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub id: u64,
    pub from: VesselId,
    pub description: String,
    pub confidence: Confidence,
    pub supports: Vec<VesselId>,
    pub opposes: Vec<VesselId>,
    pub round: u32,
    pub resolved: bool,
    pub accepted: bool,
    pub created_at: u64,
}

impl Proposal {
    pub fn new(id: u64, from: VesselId, description: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self {
            id, from, description: description.to_string(),
            confidence: Confidence::HALF,
            supports: vec![], opposes: vec![],
            round: 0, resolved: false, accepted: false,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
        }
    }

    pub fn support(&mut self, voter: VesselId, voter_confidence: Confidence) {
        if !self.supports.contains(&voter) {
            self.supports.push(voter);
            self.confidence = self.confidence.combine(voter_confidence);
        }
    }

    pub fn oppose(&mut self, voter: VesselId, voter_confidence: Confidence) {
        if !self.opposes.contains(&voter) {
            self.opposes.push(voter);
            self.confidence = self.confidence.discount(voter_confidence.value());
        }
    }

    pub fn consensus_ratio(&self) -> f64 {
        let total = self.supports.len() + self.opposes.len();
        if total == 0 { return 0.5; }
        self.supports.len() as f64 / total as f64
    }

    pub fn should_resolve(&self, config: &DeliberationConfig) -> Option<bool> {
        if self.resolved { return Some(self.accepted); }
        if self.confidence.value() >= config.auto_accept_threshold { return Some(true); }
        if self.confidence.value() <= config.auto_forfeit_threshold { return Some(false); }
        if self.consensus_ratio() >= 0.8 && self.supports.len() >= 2 { return Some(true); }
        None
    }
}

/// A deliberative agent that participates in Consider/Resolve/Forfeit.
pub struct ResolveAgent {
    id: VesselId,
    name: String,
    confidence: Confidence,
    capabilities: Vec<String>,
    config: DeliberationConfig,
    proposals: std::collections::HashMap<u64, Proposal>,
    expertise: Vec<String>,
    messages_sent: u64,
    messages_received: u64,
}

impl ResolveAgent {
    pub fn new(id: u64, name: &str, config: DeliberationConfig) -> Self {
        Self {
            id: VesselId(id), name: name.to_string(), confidence: Confidence::HALF,
            capabilities: vec!["deliberation".to_string(), "consider".to_string(),
                "resolve".to_string(), "forfeit".to_string()],
            config, proposals: std::collections::HashMap::new(),
            expertise: vec![], messages_sent: 0, messages_received: 0,
        }
    }

    pub fn with_expertise(mut self, domains: &[&str]) -> Self {
        self.expertise = domains.iter().map(|s| s.to_string()).collect();
        self.capabilities = vec!["deliberation".to_string(), "consider".to_string(),
            "resolve".to_string(), "forfeit".to_string(),
            format!("expert:{}", domains.join(","))];
        self
    }

    pub fn proposals(&self) -> &std::collections::HashMap<u64, Proposal> { &self.proposals }
    pub fn proposal(&self, id: u64) -> Option<&Proposal> { self.proposals.get(&id) }

    fn evaluate_proposal(&self, proposal: &Proposal) -> Confidence {
        let mut conf = Confidence::HALF;
        // Expertise boost
        for domain in &self.expertise {
            if proposal.description.to_lowercase().contains(domain) {
                conf = Confidence::new(conf.value() + 0.15);
            }
        }
        // Confidence from existing support
        if !proposal.supports.is_empty() {
            conf = conf.combine(proposal.confidence);
        }
        conf
    }

    fn handle_consider(&mut self, proposal_id: u64, from: VesselId) -> Vec<FleetMessage> {
        let mut responses = vec![];
        
        // Create or look up proposal
        let entry = self.proposals.entry(proposal_id).or_insert_with(|| {
            Proposal::new(proposal_id, from, &format!("proposal_{}", proposal_id))
        });
        
        let eval = self.evaluate_proposal(entry);
        
        if eval.value() >= self.config.confidence_threshold {
            // Support
            entry.support(self.id, eval);
            responses.push(FleetMessage::new(self.id, from,
                MessageType::Resolve { proposal_id, accepted: true }));
            self.messages_sent += 1;
        } else if eval.value() <= self.config.auto_forfeit_threshold {
            // Forfeit
            responses.push(FleetMessage::new(self.id, from,
                MessageType::Forfeit { proposal_id, reason: "below_threshold".to_string() }));
            self.messages_sent += 1;
        } else {
            // Partial support — still considering
            entry.support(self.id, eval);
            // Broadcast confidence update
            responses.push(FleetMessage::new(self.id, from,
                MessageType::ConfidenceUpdate {
                    topic: format!("proposal_{}", proposal_id),
                    confidence: eval,
                }));
            self.messages_sent += 1;
        }
        
        responses
    }

    fn handle_resolve(&mut self, proposal_id: u64, accepted: bool, from: VesselId) -> Vec<FleetMessage> {
        if let Some(proposal) = self.proposals.get_mut(&proposal_id) {
            proposal.round += 1;
            if accepted {
                proposal.support(from, Confidence::SURE);
            } else {
                proposal.oppose(from, Confidence::LIKELY);
            }
            proposal.confidence = proposal.confidence.decay(1, self.config.decay_rate);
        }
        vec![]
    }

    fn handle_confidence_update(&mut self, topic: &str, confidence: Confidence) {
        // Extract proposal_id from topic
        if let Some(id_str) = topic.strip_prefix("proposal_") {
            if let Ok(pid) = id_str.parse::<u64>() {
                if let Some(proposal) = self.proposals.get_mut(&pid) {
                    proposal.confidence = proposal.confidence.combine(confidence);
                    self.confidence = self.confidence.combine(confidence);
                }
            }
        }
    }
}

impl Agent for ResolveAgent {
    fn id(&self) -> VesselId { self.id }
    fn name(&self) -> &str { &self.name }

    fn receive(&mut self, msg: &FleetMessage) -> Vec<FleetMessage> {
        self.messages_received += 1;
        match &msg.msg_type {
            MessageType::Consider { proposal_id } => {
                self.handle_consider(*proposal_id, msg.from)
            }
            MessageType::Resolve { proposal_id, accepted } => {
                self.handle_resolve(*proposal_id, *accepted, msg.from)
            }
            MessageType::Forfeit { proposal_id, reason } => {
                if let Some(proposal) = self.proposals.get_mut(proposal_id) {
                    proposal.oppose(msg.from, Confidence::UNLIKELY);
                }
                vec![]
            }
            MessageType::CapabilityQuery => {
                self.messages_sent += 1;
                vec![msg.reply(MessageType::CapabilityResponse {
                    capabilities: self.capabilities.join(","),
                })]
            }
            MessageType::Ping => {
                self.messages_sent += 1;
                vec![msg.reply(MessageType::Pong)]
            }
            MessageType::ConfidenceUpdate { topic, confidence } => {
                self.handle_confidence_update(topic, *confidence);
                vec![]
            }
            _ => vec![],
        }
    }

    fn capabilities(&self) -> Vec<String> { self.capabilities.clone() }
    fn self_confidence(&self) -> Confidence { self.confidence }
}

/// Orchestrator — coordinates multi-agent deliberation rounds.
pub struct Orchestrator {
    fleet: cuda_equipment::Fleet,
    config: DeliberationConfig,
    rounds_completed: u32,
}

impl Orchestrator {
    pub fn new(config: DeliberationConfig) -> Self {
        Self { fleet: cuda_equipment::Fleet::new(), config, rounds_completed: 0 }
    }

    pub fn register(&mut self, agent: Box<dyn Agent>) {
        self.fleet.register(agent);
    }

    /// Run one deliberation round for a proposal.
    pub fn deliberate(&mut self, proposal_id: u64, initiator: VesselId) -> DeliberationResult {
        let consider = FleetMessage::new(initiator, VesselId(0),
            MessageType::Consider { proposal_id });
        
        let responses = self.fleet.send(&consider);
        self.rounds_completed += 1;
        
        let supports = responses.iter()
            .filter(|r| matches!(&r.msg_type, MessageType::Resolve { accepted: true, .. }))
            .count();
        let opposes = responses.iter()
            .filter(|r| matches!(&r.msg_type, MessageType::Resolve { accepted: false, .. })
                || matches!(&r.msg_type, MessageType::Forfeit { .. }))
            .count();
        let pending = responses.len() - supports - opposes;
        
        DeliberationResult {
            round: self.rounds_completed,
            total_responses: responses.len(),
            supports, opposes, pending,
            converged: supports as f64 / (responses.len().max(1) as f64) >= 0.8,
            confidence: if responses.is_empty() { Confidence::ZERO }
                else { Confidence::new(supports as f64 / responses.len() as f64) },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeliberationResult {
    pub round: u32,
    pub total_responses: usize,
    pub supports: usize,
    pub opposes: usize,
    pub pending: usize,
    pub converged: bool,
    pub confidence: Confidence,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_equipment::Fleet;

    fn test_config() -> DeliberationConfig {
        DeliberationConfig { confidence_threshold: 0.5, auto_accept_threshold: 0.8,
            auto_forfeit_threshold: 0.15, ..Default::default() }
    }

    #[test]
    fn test_resolve_agent_creation() {
        let agent = ResolveAgent::new(1, "architect", test_config());
        assert_eq!(agent.name(), "architect");
        assert!(agent.capabilities().contains(&"deliberation".to_string()));
    }

    #[test]
    fn test_expertise_boost() {
        let agent = ResolveAgent::new(1, "expert", test_config())
            .with_expertise(&["rust", "cuda"]);
        assert!(agent.capabilities().iter().any(|c| c.contains("rust")));
    }

    #[test]
    fn test_consider_below_threshold() {
        let mut agent = ResolveAgent::new(1, "cautious", DeliberationConfig {
            confidence_threshold: 0.9, ..Default::default()
        });
        let msg = FleetMessage::new(VesselId(0), VesselId(1),
            MessageType::Consider { proposal_id: 1 });
        let responses = agent.receive(&msg);
        assert!(!responses.is_empty());
        // Should be a confidence update, not a resolve
        assert!(responses.iter().any(|r| matches!(r.msg_type, MessageType::ConfidenceUpdate { .. })));
    }

    #[test]
    fn test_consider_above_threshold() {
        let mut agent = ResolveAgent::new(1, "eager", DeliberationConfig {
            confidence_threshold: 0.3, auto_accept_threshold: 0.5,
            auto_forfeit_threshold: 0.1, ..Default::default()
        });
        let msg = FleetMessage::new(VesselId(0), VesselId(1),
            MessageType::Consider { proposal_id: 2 });
        let responses = agent.receive(&msg);
        assert!(responses.iter().any(|r| matches!(r.msg_type, MessageType::Resolve { accepted: true, .. })));
    }

    #[test]
    fn test_ping_pong() {
        let mut agent = ResolveAgent::new(1, "agent", test_config());
        let ping = FleetMessage::new(VesselId(0), VesselId(1), MessageType::Ping);
        let responses = agent.receive(&ping);
        assert_eq!(responses.len(), 1);
        assert!(matches!(responses[0].msg_type, MessageType::Pong));
    }

    #[test]
    fn test_fleet_deliberation() {
        let mut fleet = Fleet::new();
        fleet.register(Box::new(ResolveAgent::new(1, "arch", test_config())));
        fleet.register(Box::new(ResolveAgent::new(2, "validator", test_config())));
        fleet.register(Box::new(ResolveAgent::new(3, "optimizer", test_config())));
        
        let consider = FleetMessage::new(VesselId(0), VesselId(1),
            MessageType::Consider { proposal_id: 99 });
        let responses = fleet.send(&consider);
        assert!(responses.len() >= 1);
    }

    #[test]
    fn test_orchestrator() {
        let mut orch = Orchestrator::new(test_config());
        orch.register(Box::new(ResolveAgent::new(1, "a", test_config())));
        orch.register(Box::new(ResolveAgent::new(2, "b", test_config())));
        
        let result = orch.deliberate(1, VesselId(0));
        assert!(result.total_responses >= 1);
        assert_eq!(result.round, 1);
    }

    #[test]
    fn test_proposal_support_oppose() {
        let mut p = Proposal::new(1, VesselId(0), "test proposal");
        p.support(VesselId(1), Confidence::LIKELY);
        p.oppose(VesselId(2), Confidence::UNLIKELY);
        assert_eq!(p.supports.len(), 1);
        assert_eq!(p.opposes.len(), 1);
        assert!((p.consensus_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_proposal_should_resolve() {
        let mut p = Proposal::new(1, VesselId(0), "strong proposal");
        p.support(VesselId(1), Confidence::SURE);
        p.support(VesselId(2), Confidence::SURE);
        p.support(VesselId(3), Confidence::SURE);
        let config = DeliberationConfig::default();
        assert_eq!(p.should_resolve(&config), Some(true));
    }

    #[test]
    fn test_confidence_update_flow() {
        let mut agent = ResolveAgent::new(1, "agent", test_config());
        // First create proposal via consider
        let consider = FleetMessage::new(VesselId(0), VesselId(1),
            MessageType::Consider { proposal_id: 42 });
        agent.receive(&consider);
        
        // Then update confidence
        let update = FleetMessage::new(VesselId(2), VesselId(1),
            MessageType::ConfidenceUpdate {
                topic: "proposal_42".to_string(),
                confidence: Confidence::SURE,
            });
        agent.receive(&update);
        assert!(agent.proposal(42).is_some());
        assert!(agent.self_confidence().is_likely());
    }
}
