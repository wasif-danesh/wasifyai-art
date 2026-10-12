import type { components } from "./schema.gen";

type S = components["schemas"];

export type Health = S["Health"];
export type ProviderInfo = S["ProviderInfo"];
export type ProviderCapabilities = S["ProviderCapabilities"];
export type ModelSummary = S["ModelSummary"];
export type ModelVersionSummary = S["ModelVersionSummary"];
export type ModelFileSummary = S["ModelFileSummary"];
export type ScanResult = S["ScanResult"];
export type VersionUpdate = S["VersionUpdate"];
export type FamilyInfo = S["FamilyInfo"];
export type GenerationRequest = S["GenerationRequest"];
export type JobSummary = S["JobSummary"];
export type CreationSummary = S["CreationSummary"];
export type CreationModelRef = S["CreationModelRef"];
export type LoraUse = S["LoraUse"];
export type UploadSummary = S["UploadSummary"];
