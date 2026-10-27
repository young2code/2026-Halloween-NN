import { describe, expect, it } from "vitest";
import { parsePrompt } from "./prompt";

describe("parsePrompt", () => {
  it("turns a friendly prompt into one-hot conditions", () => {
    const result = parsePrompt("a tall green pumpkin with star eyes, fangs, and purple glow");
    expect(result.selections.shape).toBe(1);
    expect(result.selections.color).toBe(1);
    expect(result.selections.eyes).toBe(2);
    expect(result.selections.mouth).toBe(3);
    expect(result.selections.glow).toBe(3);
    expect(result.vector).toHaveLength(28);
    expect([...result.vector].reduce((a, b) => a + b, 0)).toBe(7);
    expect(result.unknown).toEqual([]);
  });

  it("reports words outside the tiny vocabulary", () => {
    expect(parsePrompt("pumpkin riding a bicycle").unknown).toEqual(["riding", "bicycle"]);
  });
});

