# ADR-0001: Record architecture decisions

- **Status:** Proposed
- **Date:** 2026-10-12

## Context

Decisions made in chat get lost, and this project has an unusual constraint: it must stay portable to wasifyai.

## Decision

Record significant decisions as short ADRs in `docs/decisions/` (Status, Date, Context, Decision, Consequences).
The first ADRs to write from `docs/DESIGN.md` are: providers and availability; model hub registry; generation jobs
and worker; LoRA training and trainer service; portability rules.

## Consequences

Intent can be checked before a design changes.
