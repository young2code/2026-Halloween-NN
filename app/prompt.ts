export const GROUPS = {
  shape: ["round", "tall", "wide", "lumpy"],
  expression: ["happy", "silly", "surprised", "angry", "spooky"],
  eyes: ["triangle", "round eyes", "star eyes", "sleepy"],
  mouth: ["smile", "crooked", "toothy", "fangs"],
  color: ["orange", "green", "pale"],
  glow: ["yellow glow", "green glow", "blue glow", "purple glow"],
  accessory: ["no hat", "witch hat", "pirate hat", "leaf"],
} as const;

export type GroupName = keyof typeof GROUPS;
export type ParsedPrompt = {
  selections: Record<GroupName, number>;
  vector: Float32Array;
  recognized: string[];
  unknown: string[];
};

const defaults: Record<GroupName, number> = {
  shape: 0,
  expression: 0,
  eyes: 0,
  mouth: 0,
  color: 0,
  glow: 0,
  accessory: 0,
};

const aliases: Record<string, [GroupName, number]> = {
  round: ["shape", 0],
  circle: ["shape", 0],
  circular: ["shape", 0],
  plump: ["shape", 0],
  chubby: ["shape", 0],
  tall: ["shape", 1],
  towering: ["shape", 1],
  stretched: ["shape", 1],
  skinny: ["shape", 1],
  slender: ["shape", 1],
  wide: ["shape", 2],
  broad: ["shape", 2],
  huge: ["shape", 2],
  chunky: ["shape", 2],
  squat: ["shape", 2],
  lumpy: ["shape", 3],
  bumpy: ["shape", 3],
  knobby: ["shape", 3],
  uneven: ["shape", 3],
  happy: ["expression", 0],
  cheerful: ["expression", 0],
  joyful: ["expression", 0],
  friendly: ["expression", 0],
  merry: ["expression", 0],
  silly: ["expression", 1],
  goofy: ["expression", 1],
  funny: ["expression", 1],
  playful: ["expression", 1],
  surprised: ["expression", 2],
  shocked: ["expression", 2],
  amazed: ["expression", 2],
  startled: ["expression", 2],
  angry: ["expression", 3],
  mad: ["expression", 3],
  grumpy: ["expression", 3],
  fierce: ["expression", 3],
  furious: ["expression", 3],
  spooky: ["expression", 4],
  scary: ["expression", 4],
  eerie: ["expression", 4],
  haunted: ["expression", 4],
  creepy: ["expression", 4],
  triangle: ["eyes", 0],
  triangular: ["eyes", 0],
  "triangle eyes": ["eyes", 0],
  "pointy eyes": ["eyes", 0],
  "round eyes": ["eyes", 1],
  "circle eyes": ["eyes", 1],
  "button eyes": ["eyes", 1],
  "star eyes": ["eyes", 2],
  stars: ["eyes", 2],
  "starry eyes": ["eyes", 2],
  "sparkle eyes": ["eyes", 2],
  sleepy: ["eyes", 3],
  "sleepy eyes": ["eyes", 3],
  smile: ["mouth", 0],
  smiling: ["mouth", 0],
  grin: ["mouth", 0],
  grinning: ["mouth", 0],
  crooked: ["mouth", 1],
  zigzag: ["mouth", 1],
  jagged: ["mouth", 1],
  smirk: ["mouth", 1],
  toothy: ["mouth", 2],
  teeth: ["mouth", 2],
  "toothy grin": ["mouth", 2],
  "big teeth": ["mouth", 2],
  fangs: ["mouth", 3],
  vampire: ["mouth", 3],
  orange: ["color", 0],
  tangerine: ["color", 0],
  amber: ["color", 0],
  green: ["color", 1],
  emerald: ["color", 1],
  lime: ["color", 1],
  pale: ["color", 2],
  white: ["color", 2],
  ivory: ["color", 2],
  cream: ["color", 2],
  ghost: ["color", 2],
  ashen: ["color", 2],
  "yellow glow": ["glow", 0],
  "gold glow": ["glow", 0],
  yellow: ["glow", 0],
  "green glow": ["glow", 1],
  "toxic glow": ["glow", 1],
  "blue glow": ["glow", 2],
  "cyan glow": ["glow", 2],
  blue: ["glow", 2],
  "purple glow": ["glow", 3],
  "violet glow": ["glow", 3],
  purple: ["glow", 3],
  "no hat": ["accessory", 0],
  bareheaded: ["accessory", 0],
  "witch hat": ["accessory", 1],
  witch: ["accessory", 1],
  "wizard hat": ["accessory", 1],
  "pirate hat": ["accessory", 2],
  pirate: ["accessory", 2],
  leaf: ["accessory", 3],
};

export const VOCABULARY_BY_GROUP = (Object.keys(GROUPS) as GroupName[]).reduce(
  (vocabulary, group) => {
    vocabulary[group] = Object.entries(aliases)
      .filter(([, [aliasGroup]]) => aliasGroup === group)
      .map(([term]) => term);
    return vocabulary;
  },
  {} as Record<GroupName, string[]>,
);

export const VISUAL_VOCABULARY_SIZE = Object.values(VOCABULARY_BY_GROUP).flat().length;

const ignored = new Set([
  "a", "an", "the", "with", "and", "pumpkin", "jack", "o", "lantern", "jackolantern",
  "of", "in", "make", "me", "please", "carved", "face", "glowing", "glow", "eyes", "eye",
  "hat", "that", "has", "is", "very", "little", "big",
]);

export function parsePrompt(input: string): ParsedPrompt {
  const selections = { ...defaults };
  const normalized = input.toLowerCase().replace(/[’']/g, "").replace(/[^a-z\s-]/g, " ");
  const consumed = new Set<number>();
  const words = normalized.split(/\s+/).filter(Boolean);
  const recognized: string[] = [];

  for (let i = 0; i < words.length - 1; i += 1) {
    const phrase = `${words[i]} ${words[i + 1]}`;
    const match = aliases[phrase];
    if (match) {
      selections[match[0]] = match[1];
      recognized.push(phrase);
      consumed.add(i);
      consumed.add(i + 1);
    }
  }

  words.forEach((word, index) => {
    if (consumed.has(index)) return;
    const match = aliases[word];
    if (match) {
      selections[match[0]] = match[1];
      recognized.push(word);
    }
  });

  const unknown = [...new Set(words.filter((word, i) => !consumed.has(i) && !aliases[word] && !ignored.has(word)))];
  const vector = new Float32Array(Object.values(GROUPS).reduce((sum, values) => sum + values.length, 0));
  let offset = 0;
  (Object.keys(GROUPS) as GroupName[]).forEach((group) => {
    vector[offset + selections[group]] = 1;
    offset += GROUPS[group].length;
  });

  return { selections, vector, recognized, unknown };
}

export function labelFromSelection(group: GroupName, index: number): string {
  return GROUPS[group][index] ?? GROUPS[group][0];
}

