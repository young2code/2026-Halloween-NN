const ramp = " .,:;irsXA253hMHGS#9B&@";

export type AsciiCell = { character: string; color: string };

export function imageToAsciiCells(imageData: ImageData, columns = 44): AsciiCell[][] {
  const scale = imageData.width / columns;
  const rows = Math.max(1, Math.round(imageData.height / (scale * 1.85)));
  const output: AsciiCell[][] = [];
  for (let y = 0; y < rows; y += 1) {
    const line: AsciiCell[] = [];
    for (let x = 0; x < columns; x += 1) {
      const px = Math.min(imageData.width - 1, Math.floor((x + 0.5) * scale));
      const py = Math.min(imageData.height - 1, Math.floor((y + 0.5) * scale * 1.85));
      const index = (py * imageData.width + px) * 4;
      const r = imageData.data[index];
      const g = imageData.data[index + 1];
      const b = imageData.data[index + 2];
      const brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
      line.push({
        character: ramp[Math.min(ramp.length - 1, Math.floor(brightness * ramp.length))],
        color: `rgb(${r} ${g} ${b})`,
      });
    }
    output.push(line);
  }
  return output;
}

export function imageToAscii(imageData: ImageData, columns = 44): string {
  return imageToAsciiCells(imageData, columns)
    .map((line) => line.map((cell) => cell.character).join("").replace(/\s+$/, ""))
    .join("\n");
}


