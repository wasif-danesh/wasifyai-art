import { describe, expect, it } from "vitest";
import { formatSize, parseWords } from "./format";

describe("format", () => {
  it("formats sizes", () => {
    expect(formatSize(7_800_000_000)).toBe("7.8 GB");
    expect(formatSize(120_000_000)).toBe("120 MB");
  });
  it("parses comma separated trigger words", () => {
    expect(parseWords(" kaia_x, , 1girl ")).toEqual(["kaia_x", "1girl"]);
  });
});
