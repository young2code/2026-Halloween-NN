import { VOCABULARY_BY_GROUP, type GroupName } from "./prompt";
import { mulberry32 } from "./random";

export function timeSeed(now = Date.now()): number {
  return now >>> 0;
}

export function surprisePrompt(seed: number): string {
  const random = mulberry32(seed);
  const pick = (group: GroupName): string => {
    const terms = VOCABULARY_BY_GROUP[group];
    return terms[Math.floor(random() * terms.length)];
  };
  const shape = pick("shape");
  const expression = pick("expression");
  const eyes = pick("eyes");
  const mouth = pick("mouth");
  const color = pick("color");
  const glow = pick("glow");
  const accessory = pick("accessory");
  const templates = [
    `please carve me a ${expression} ${shape} ${color} pumpkin with ${eyes}, ${mouth}, ${glow}, and ${accessory}`,
    `make me a ${shape} ${color} jack o lantern that is ${expression} with ${eyes}, ${mouth}, ${glow}, and ${accessory}`,
    `a very ${expression} ${shape} ${color} pumpkin with ${eyes} and ${mouth}, plus ${glow} and ${accessory}`,
  ];
  return templates[Math.floor(random() * templates.length)];
}
