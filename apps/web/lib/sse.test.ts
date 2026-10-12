import { describe, expect, it } from "vitest";
import { parseJobEvent } from "./sse";

describe("parseJobEvent", () => {
  it("parses known events", () => {
    expect(parseJobEvent("progress", '{"progress":0.5}')).toEqual({
      type: "progress",
      progress: 0.5,
    });
    expect(parseJobEvent("succeeded", '{"creation_id":"c1"}')).toEqual({
      type: "succeeded",
      creationId: "c1",
    });
    expect(parseJobEvent("failed", '{"message":"No"}')).toEqual({ type: "failed", message: "No" });
  });
  it("parses cancel", () => {
    expect(parseJobEvent("canceled", "{}")).toEqual({ type: "canceled" });
  });
  it("ignores unknown or malformed events", () => {
    expect(parseJobEvent("other", "{}")).toBeNull();
    expect(parseJobEvent("progress", "not json")).toBeNull();
    expect(parseJobEvent("progress", "{}")).toBeNull();
  });
});
