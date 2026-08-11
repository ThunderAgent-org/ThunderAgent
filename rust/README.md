# ThunderAgent Dynamo policy

This crate keeps ThunderAgent's session and capacity state outside Dynamo. Dynamo owns request storage and calls the policy serially from its queue actor.

```rust
use std::sync::Arc;

use dynamo_kv_router::WorkerSelectionPolicyFactory;
use thunderagent_dynamo_policy::{ThunderAgentConfig, worker_selection_policy};

let config = ThunderAgentConfig::default();
let factory: WorkerSelectionPolicyFactory = Arc::new(move |router, worker_type, _partition| {
    worker_selection_policy(router.clone(), worker_type, config.clone())
        .expect("valid ThunderAgent configuration")
});
```

The returned `WorkerSelectionPolicy` contains all three external components:

- A queue admission policy that owns program, session, request, and capacity state.
- A scorer used only when no session assignment exists.
- A picker that honors the queue policy's session assignment when that worker remains eligible.

Requests without Dynamo session context bypass ThunderAgent admission and still use the scorer and picker.

The crate also exposes `register`, so it can be linked directly as Dynamo's `dynamo-worker-selection-policy-catalog` dependency. The registered policy type is `thunderagent`; its YAML parameters are the fields of `ThunderAgentConfig`.

[`worker-selection.yaml`](worker-selection.yaml) is a complete policy configuration using the default ThunderAgent values.
