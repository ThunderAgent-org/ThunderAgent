// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicyError,
};
use parking_lot::RwLock;

#[derive(Default)]
pub(crate) struct SessionAssignments {
    workers: RwLock<HashMap<String, WorkerWithDpRank>>,
}

impl SessionAssignments {
    pub(crate) fn set(&self, session_id: &str, worker: Option<WorkerWithDpRank>) {
        let mut workers = self.workers.write();
        match worker {
            Some(worker) => {
                workers.insert(session_id.to_owned(), worker);
            }
            None => {
                workers.remove(session_id);
            }
        }
    }

    fn get(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.workers.read().get(session_id).copied()
    }
}

/// Low-cost fallback for requests that do not yet have a session assignment.
pub(crate) struct ThunderAgentScorer;

impl WorkerScorer for ThunderAgentScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("worker load unavailable"))?;
        Ok(load.active_prefill_tokens() as f64
            + load.decode_cost_blocks() * f64::from(context.block_size()))
    }
}

/// Preserves the queue policy's session assignment, then falls back to minimum score.
pub(crate) struct ThunderAgentPicker {
    assignments: Arc<SessionAssignments>,
}

impl ThunderAgentPicker {
    pub(crate) fn new(assignments: Arc<SessionAssignments>) -> Self {
        Self { assignments }
    }
}

impl WorkerPicker for ThunderAgentPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if let Some(worker) = context
            .session_context()
            .and_then(|session| self.assignments.get(session.session_id()))
            && let Some(row) = candidates
                .iter()
                .position(|candidate| candidate.worker() == worker)
        {
            return Ok(row);
        }

        candidates
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                left.cost()
                    .total_cmp(&right.cost())
                    .then_with(|| left.worker().cmp(&right.worker()))
            })
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}
