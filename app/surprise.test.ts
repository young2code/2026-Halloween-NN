import { describe, expect, it } from "vitest";
import { GROUPS, parsePrompt, VOCABULARY_BY_GROUP, type GroupName } from "./prompt";
import { surprisePrompt, timeSeed } from "./surprise";

describe("surprise prompts", () => {
  it("uses the low 32 bits of the supplied millisecond time", () => {
    expect(timeSeed(1_798_000_123_456)).toBe(1_798_000_123_456 >>> 0);
  });

  it("chooses one supported vocabulary term from every group", () => {
    const result = parsePrompt(surprisePrompt(1234));
    expect(result.unknown).toEqual([]);
    expect(Object.keys(result.matchedTerms)).toHaveLength(Object.keys(GROUPS).length);
  });

  it("can reach all 100 vocabulary choices", () => {
    const seen = Object.fromEntries(
      (Object.keys(GROUPS) as GroupName[]).map((group) => [group, new Set<string>()]),
    ) as Record<GroupName, Set<string>>;
    for (let seed = 0; seed < 10_000; seed += 1) {
      const { matchedTerms } = parsePrompt(surprisePrompt(seed));
      (Object.keys(GROUPS) as GroupName[]).forEach((group) => {
        const term = matchedTerms[group];
        if (term) seen[group].add(term);
      });
    }
    (Object.keys(GROUPS) as GroupName[]).forEach((group) => {
      expect(seen[group]).toEqual(new Set(VOCABULARY_BY_GROUP[group]));
    });
  });
});
