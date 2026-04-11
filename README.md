# cuda-resolve-agent

Deliberative A2A agent — Consider/Resolve/Forfeit with Bayesian confidence, built on cuda-equipment

Part of the Cocapn spatial layer — how agents perceive and navigate physical space.

## What It Does

### Key Types

- `DeliberationConfig` — core data structure
- `Proposal` — core data structure
- `ResolveAgent` — core data structure
- `Orchestrator` — core data structure
- `DeliberationResult` — core data structure

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-resolve-agent.git
cd cuda-resolve-agent

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_resolve_agent::*;

// See src/lib.rs for full API
// 10 unit tests included
```

### Available Implementations

- `Default for DeliberationConfig` — see source for methods
- `Proposal` — see source for methods
- `ResolveAgent` — see source for methods
- `Agent for ResolveAgent` — see source for methods
- `Orchestrator` — see source for methods

## Testing

```bash
cargo test
```

10 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: spatial
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates

- [cuda-sensor-agent](https://github.com/Lucineer/cuda-sensor-agent)
- [cuda-voxel-logic](https://github.com/Lucineer/cuda-voxel-logic)
- [cuda-world-model](https://github.com/Lucineer/cuda-world-model)
- [cuda-weather](https://github.com/Lucineer/cuda-weather)

## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
