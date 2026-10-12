import { describe, expect, it } from "vitest";
import { providerHint } from "./labels";

const base = {
  id: "p",
  kind: "comfyui",
  title: "P",
  status: "configured" as const,
};

describe("providerHint", () => {
  it("lists only what the provider supports", () => {
    const text = providerHint({
      ...base,
      capabilities: {
        text2img: true,
        img2img: true,
        edit: false,
        video: false,
        loras: true,
        lora_stack_max: 5,
        negative_prompt: true,
      },
    });
    expect(text).toBe("image to image, up to 5 LoRAs");
  });
});
