import type { ProviderInfo } from "@wasifyai-art/contracts";

/** What a provider offers, as short words for the dropdown hint. */
export function providerHint(p: ProviderInfo): string {
  const parts: string[] = [];
  if (p.capabilities.img2img) parts.push("image to image");
  if (p.capabilities.edit) parts.push("edit");
  if (p.capabilities.loras) parts.push(`up to ${p.capabilities.lora_stack_max} LoRAs`);
  if (p.capabilities.video) parts.push("video");
  return parts.join(", ");
}
