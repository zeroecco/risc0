// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{cell::RefCell, collections::HashMap, sync::OnceLock};

use anyhow::{Context, Result, anyhow, bail, ensure};

use super::{ProverServer, keccak::prove_keccak};
use crate::{
    Assumption, AssumptionReceipt, CompositeReceipt, ExecutorEnv, InnerAssumptionReceipt,
    MaybePruned, Output, PreflightResults, ProverOpts, Receipt, ReceiptClaim, Segment, Session,
    UnionClaim, Unknown, VerifierContext, WorkClaim,
    claim::merge::Merge,
    host::{
        client::prove::opts::ReceiptKind,
        prove_info::ProveInfo,
        recursion::prove as recursion_prove,
        server::{exec::executor::ExecutorImpl, prove::union_peak::UnionPeak},
    },
    mmr::MerkleMountainAccumulator,
    receipt::{InnerReceipt, SegmentReceipt, SuccinctReceipt},
    sha::Digestible,
};

/// An implementation of a Prover that runs locally.
pub struct ProverImpl {
    opts: ProverOpts,
    segment_prover: RefCell<Option<Box<dyn risc0_circuit_rv32im::prove::SegmentProver>>>,
    recursion_provers:
        RefCell<HashMap<String, Box<dyn risc0_circuit_recursion::prove::RecursionProver>>>,
}

fn verify_internal_receipts_enabled() -> bool {
    static VERIFY_INTERNAL_RECEIPTS: OnceLock<bool> = OnceLock::new();
    *VERIFY_INTERNAL_RECEIPTS.get_or_init(|| {
        std::env::var("RISC0_VERIFY_INTERNAL_RECEIPTS")
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
            })
            .unwrap_or(true)
    })
}

impl ProverImpl {
    /// Construct a [ProverImpl].
    pub fn new(opts: ProverOpts) -> Self {
        let prover = Self {
            opts,
            segment_prover: RefCell::new(None),
            recursion_provers: RefCell::new(HashMap::new()),
        };

        if let Ok(spec) = std::env::var("RISC0_PREWARM_PO2S") {
            for po2 in spec.split(',').filter_map(|part| {
                let part = part.trim();
                if part.is_empty() {
                    None
                } else {
                    Some(part.parse::<usize>())
                }
            }) {
                match po2 {
                    Ok(po2) => {
                        if let Err(err) = prover
                            .with_segment_prover(|segment_prover| segment_prover.prewarm_po2(po2))
                        {
                            tracing::warn!("segment prover prewarm failed for po2={po2}: {err:#}");
                        }
                    }
                    Err(err) => {
                        tracing::warn!("ignoring invalid RISC0_PREWARM_PO2S entry: {err}");
                    }
                }
            }
        }

        prover
    }

    fn with_segment_prover<T>(
        &self,
        f: impl FnOnce(&dyn risc0_circuit_rv32im::prove::SegmentProver) -> Result<T>,
    ) -> Result<T> {
        if self.segment_prover.borrow().is_none() {
            *self.segment_prover.borrow_mut() =
                Some(risc0_circuit_rv32im::prove::segment_prover()?);
        }

        let borrow = self.segment_prover.borrow();
        f(borrow.as_deref().unwrap())
    }

    fn with_recursion_prover<T>(
        &self,
        hashfn: &str,
        f: impl FnOnce(&dyn risc0_circuit_recursion::prove::RecursionProver) -> Result<T>,
    ) -> Result<T> {
        let needs_init = { !self.recursion_provers.borrow().contains_key(hashfn) };
        if needs_init {
            self.recursion_provers.borrow_mut().insert(
                hashfn.to_string(),
                risc0_circuit_recursion::prove::recursion_prover(hashfn)?,
            );
        }

        let borrow = self.recursion_provers.borrow();
        f(borrow.get(hashfn).unwrap().as_ref())
    }
}

impl ProverServer for ProverImpl {
    fn prove(&self, env: ExecutorEnv<'_>, elf: &[u8]) -> Result<ProveInfo> {
        let ctx = VerifierContext::default().with_dev_mode(self.opts.dev_mode());
        self.prove_with_ctx(env, &ctx, elf)
    }

    fn prove_with_ctx(
        &self,
        env: ExecutorEnv<'_>,
        ctx: &VerifierContext,
        elf: &[u8],
    ) -> Result<ProveInfo> {
        let session = ExecutorImpl::from_elf(env, elf)?.run()?;
        self.prove_session(ctx, &session)
    }

    fn prove_session(&self, ctx: &VerifierContext, session: &Session) -> Result<ProveInfo> {
        tracing::debug!(
            "prove_session: exit_code = {:?}, journal = {:?}, segments: {}",
            session.exit_code,
            session.journal.as_ref().map(hex::encode),
            session.segments.len()
        );

        ensure!(
            self.opts.hashfn == "poseidon2",
            "provided `ProverOpts` has unsupported `hashfn` value of \"{}\"; \
            supported `hashfn` values are: \"poseidon2\".",
            &self.opts.hashfn
        );

        let mut segments = Vec::new();
        for segment_ref in session.segments.iter() {
            let segment = segment_ref.resolve()?;
            for hook in &session.hooks {
                hook.on_pre_prove_segment(&segment);
            }
            segments.push(self.prove_segment(ctx, &segment)?);
            for hook in &session.hooks {
                hook.on_post_prove_segment(&segment);
            }
        }

        let (assumptions, session_assumption_receipts): (Vec<_>, Vec<_>) =
            session.assumptions.iter().cloned().unzip();

        // Merge the output, including journal digest and assumptions, into the last segment.
        segments
            .last_mut()
            .ok_or_else(|| anyhow!("session is empty"))?
            .claim
            .output
            .merge_with(
                &session
                    .journal
                    .as_ref()
                    .map(|journal| Output {
                        journal: MaybePruned::Pruned(journal.digest()),
                        assumptions: assumptions.into(),
                    })
                    .into(),
            )
            .context("failed to merge output into final segment claim")?;

        let verifier_parameters = ctx
            .composite_verifier_parameters()
            .ok_or_else(|| anyhow!("composite receipt verifier parameters missing from context"))?
            .digest();

        let mut zkr_receipts = HashMap::new();
        let mut keccak_receipts: MerkleMountainAccumulator<UnionPeak> =
            MerkleMountainAccumulator::new();
        for proof_request in session.pending_keccaks.iter() {
            let receipt = prove_keccak(proof_request)?;
            tracing::debug!("adding keccak assumption: {}", receipt.claim.digest());
            keccak_receipts.insert(receipt)?;
        }

        // NOTE: Calling keccak_receipts.root() proves the union tree.
        if let Ok(root_receipt) = keccak_receipts.root() {
            let assumption = Assumption {
                claim: root_receipt.claim.digest(),
                control_root: root_receipt.control_root()?,
            };

            tracing::debug!("keccak root assumption: {:?}", assumption);
            zkr_receipts.insert(assumption, root_receipt.clone());
        }

        // TODO: add test case for when a single session refers to the same assumption multiple times
        let inner_assumption_receipts: Vec<_> = session_assumption_receipts
            .into_iter()
            .map(|assumption_receipt| match assumption_receipt {
                AssumptionReceipt::Proven(receipt) => Ok(receipt),
                AssumptionReceipt::Unresolved(assumption) => {
                    let receipt = zkr_receipts.get(&assumption).ok_or_else(|| {
                        anyhow!("no receipt available for unresolved assumption: {assumption:#?}")
                    })?;
                    Ok(InnerAssumptionReceipt::Succinct(receipt.clone()))
                }
            })
            .collect::<Result<_>>()?;

        let composite_receipt = CompositeReceipt {
            segments,
            assumption_receipts: inner_assumption_receipts,
            verifier_parameters,
        };

        let session_claim = session.claim()?;

        // Verify the receipt to catch if something is broken in the proving process.
        // NOTE: If the proof is very large, this could take > 1s, e.g. with 1000 segments.
        if verify_internal_receipts_enabled() {
            composite_receipt.verify_integrity_with_context(ctx)?;
        }
        check_claims(
            &session_claim,
            "composite",
            MaybePruned::Value(composite_receipt.claim()?),
        )?;

        if self.opts.receipt_kind == ReceiptKind::Composite {
            let receipt = Receipt::new(
                InnerReceipt::Composite(composite_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: None,
                stats: session.stats(),
            });
        }

        let (succinct_receipt, work_receipt) = match session.povw_job_id.is_some() {
            true => {
                let work_receipt = self.composite_to_succinct_povw(&composite_receipt)?;
                let unwrapped = self.unwrap_povw(&work_receipt)?;
                (unwrapped, Some(work_receipt))
            }
            false => (self.composite_to_succinct(&composite_receipt)?, None),
        };

        if self.opts.receipt_kind == ReceiptKind::Succinct {
            let receipt = Receipt::new(
                InnerReceipt::Succinct(succinct_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: work_receipt.map(Into::into),
                stats: session.stats(),
            });
        }

        let groth16_receipt = self.succinct_to_groth16(&succinct_receipt)?;

        if self.opts.receipt_kind == ReceiptKind::Groth16 {
            let receipt = Receipt::new(
                InnerReceipt::Groth16(groth16_receipt),
                session.journal.clone().unwrap_or_default().bytes,
            );
            return Ok(ProveInfo {
                receipt,
                work_receipt: work_receipt.map(Into::into),
                stats: session.stats(),
            });
        }

        // As long as the checks above are exhaustive, this code is unreachable. If this statement
        // is reached, this is an implementation error.
        unreachable!(
            "proving not implemented for receipt kind {:?}",
            self.opts.receipt_kind
        );
    }

    fn segment_preflight(&self, segment: &Segment) -> Result<PreflightResults> {
        tracing::debug!("segment_preflight");

        ensure!(
            segment.po2() <= self.opts.max_segment_po2,
            "segment po2 exceeds max on ProverOpts: {} > {}",
            segment.po2(),
            self.opts.max_segment_po2
        );
        let inner =
            self.with_segment_prover(|segment_prover| segment_prover.preflight(&segment.inner))?;

        Ok(PreflightResults {
            inner,
            terminate_state: segment.inner.claim.terminate_state,
            output: segment.output.clone(),
            segment_index: segment.index,
        })
    }

    fn prove_segment_core(
        &self,
        ctx: &VerifierContext,
        preflight_results: PreflightResults,
    ) -> Result<SegmentReceipt> {
        tracing::debug!("prove_segment_core");

        ensure!(
            self.opts.hashfn == "poseidon2",
            "provided `ProverOpts` has unsupported `hashfn` value of \"{}\"; \
            supported `hashfn` values are: \"poseidon2\".",
            &self.opts.hashfn
        );

        let po2 = preflight_results.inner.po2();
        let seal = self.with_segment_prover(|segment_prover| {
            segment_prover.prove_core(preflight_results.inner)
        })?;
        let mut claim = ReceiptClaim::decode_from_seal_v2(&seal, Some(po2))?;
        claim.output = preflight_results.output.into();

        let verifier_parameters = ctx
            .segment_verifier_parameters
            .as_ref()
            .ok_or_else(|| anyhow!("segment receipt verifier parameters missing from context"))?
            .digest();
        let receipt = SegmentReceipt {
            seal,
            index: preflight_results.segment_index,
            hashfn: self.opts.hashfn.clone(),
            claim,
            verifier_parameters,
        };
        if verify_internal_receipts_enabled() {
            receipt
                .verify_integrity_with_context(ctx)
                .context("verify segment")?;
        }

        Ok(receipt)
    }

    fn lift(&self, receipt: &SegmentReceipt) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::lift_with_recursion_prover(receipt, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify lift")?;
        }
        Ok(receipt)
    }

    fn lift_povw(
        &self,
        receipt: &SegmentReceipt,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::lift_povw_with_recursion_prover(receipt, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify lift_povw")?;
        }
        Ok(receipt)
    }

    fn join(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
        b: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::join_with_recursion_prover(a, b, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify join")?;
        }
        Ok(receipt)
    }

    fn join_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::join_povw_with_recursion_prover(a, b, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify join_povw")?;
        }
        Ok(receipt)
    }

    fn join_unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        b: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::join_unwrap_povw_with_recursion_prover(a, b, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt
                .verify_integrity()
                .context("verify join_unwrap_povw")?;
        }
        Ok(receipt)
    }

    fn resolve(
        &self,
        conditional: &SuccinctReceipt<ReceiptClaim>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::resolve_with_recursion_prover(
                conditional,
                assumption,
                Some(recursion_prover),
            )
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify resolve")?;
        }
        Ok(receipt)
    }

    fn resolve_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<WorkClaim<ReceiptClaim>>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::resolve_povw_with_recursion_prover(
                conditional,
                assumption,
                Some(recursion_prover),
            )
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify resolve_povw")?;
        }
        Ok(receipt)
    }

    fn resolve_unwrap_povw(
        &self,
        conditional: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
        assumption: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::resolve_unwrap_povw_with_recursion_prover(
                conditional,
                assumption,
                Some(recursion_prover),
            )
        })?;
        if verify_internal_receipts_enabled() {
            receipt
                .verify_integrity()
                .context("verify resolve_unwrap_povw")?;
        }
        Ok(receipt)
    }

    fn identity_p254(
        &self,
        a: &SuccinctReceipt<ReceiptClaim>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        // TODO: figure out how to verify this
        self.with_recursion_prover("poseidon_254", |recursion_prover| {
            recursion_prove::identity_p254_with_recursion_prover(a, Some(recursion_prover))
        })
    }

    fn prove_keccak(
        &self,
        request: &crate::ProveKeccakRequest,
    ) -> Result<SuccinctReceipt<Unknown>> {
        // TODO: figure out how to verify this
        prove_keccak(request)
    }

    fn union(
        &self,
        a: &SuccinctReceipt<Unknown>,
        b: &SuccinctReceipt<Unknown>,
    ) -> Result<SuccinctReceipt<UnionClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::union_with_recursion_prover(a, b, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify union")?;
        }
        Ok(receipt)
    }

    fn unwrap_povw(
        &self,
        a: &SuccinctReceipt<WorkClaim<ReceiptClaim>>,
    ) -> Result<SuccinctReceipt<ReceiptClaim>> {
        let receipt = self.with_recursion_prover(&self.opts.hashfn, |recursion_prover| {
            recursion_prove::unwrap_povw_with_recursion_prover(a, Some(recursion_prover))
        })?;
        if verify_internal_receipts_enabled() {
            receipt.verify_integrity().context("verify unwrap_povw")?;
        }
        Ok(receipt)
    }
}

fn check_claims(
    session_claim: &ReceiptClaim,
    other_name: &str,
    other_claim: MaybePruned<ReceiptClaim>,
) -> Result<()> {
    let session_claim_digest = session_claim.digest();
    let other_claim_digest = other_claim.digest();
    if session_claim_digest != other_claim_digest {
        tracing::debug!("session claim and {other_name} do not match");
        tracing::debug!("session claim: {session_claim:#?}");
        tracing::debug!("{other_name} claim: {other_claim:#?}");
        bail!(
            "session claim: {} != {other_name} claim: {}",
            hex::encode(session_claim_digest),
            hex::encode(other_claim_digest)
        );
    }
    Ok(())
}
