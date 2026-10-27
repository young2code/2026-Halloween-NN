export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function normalVector(length: number, seed: number): Float32Array {
  const random = mulberry32(seed);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u = Math.max(random(), Number.EPSILON);
    const v = random();
    const radius = Math.sqrt(-2 * Math.log(u));
    output[i] = radius * Math.cos(2 * Math.PI * v);
    if (i + 1 < length) output[i + 1] = radius * Math.sin(2 * Math.PI * v);
  }
  return output;
}

