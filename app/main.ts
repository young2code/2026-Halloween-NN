import "./style.css";
import { imageToAscii, imageToAsciiCells } from "./ascii";
import { PumpkinGenerator } from "./inference";
import {
  GROUPS,
  labelFromSelection,
  parsePrompt,
  VISUAL_VOCABULARY_SIZE,
  VOCABULARY_BY_GROUP,
  type GroupName,
} from "./prompt";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("App root was not found.");

const examples = [
  "cheerful plump amber pumpkin with starry eyes and violet glow",
  "towering ivory haunted pumpkin with sleepy eyes and vampire fangs",
  "goofy emerald bumpy pumpkin with button eyes and a wizard hat",
  "broad furious pirate pumpkin with pointy eyes and cyan glow",
];

app.innerHTML = `
  <header class="hero">
    <nav aria-label="Primary navigation">
      <a class="brand" href="#top" aria-label="Tiny Pumpkin Machine home"><span>✦</span> TPM</a>
      <div><a href="#learn">How it learns</a><a href="https://github.com/young2code/2026-Halloween-NN">Source</a></div>
    </nav>
    <div class="hero-copy" id="top">
      <p class="eyebrow">A pocket-sized neural network</p>
      <h1>Describe it.<br /><em>Carve it.</em></h1>
      <p class="intro">A tiny conditional GAN dreams up jack-o’-lanterns on your device. No account. No cloud GPU. Your prompt never leaves the browser.</p>
      <a class="hero-cta" href="#studio">Enter the pumpkin patch <span aria-hidden="true">↓</span></a>
    </div>
    <div class="hero-mark" aria-hidden="true">
      <div class="moon"></div><div class="pumpkin-glyph">🎃</div>
      <span class="spark s1">✦</span><span class="spark s2">✧</span><span class="spark s3">✦</span>
    </div>
  </header>

  <main>
    <section class="studio-section" id="studio" aria-labelledby="studio-title">
      <div class="section-label"><span>01</span> The studio</div>
      <div class="studio-heading">
        <div><p class="eyebrow">Your imagination, 28 tiny switches</p><h2 id="studio-title">Carve a new character</h2></div>
        <div class="runtime-pill" id="runtime-pill" aria-live="polite"><span></span><b>Preparing model…</b></div>
      </div>

      <div class="studio-grid">
        <form class="prompt-panel" id="prompt-form">
          <label for="prompt">Tell the machine what you see</label>
          <textarea id="prompt" rows="4" maxlength="180">${examples[0]}</textarea>
          <p class="prompt-note" id="prompt-note">Try color, shape, mood, eyes, mouth, glow, and a hat.</p>
          <details class="vocabulary-guide" open>
            <summary><span><b>${VISUAL_VOCABULARY_SIZE}</b>-word prompt vocabulary</span><em>Click a word to try it</em></summary>
            <p>These playful synonyms map onto the 28 attributes the neural network learned.</p>
            <div class="vocabulary-list" id="vocabulary-list"></div>
          </details>
          <div class="chip-groups" id="chips"></div>
          <div class="actions">
            <button class="carve-button" id="carve" type="submit" disabled><span>Carve it</span><b aria-hidden="true">✦</b></button>
            <button class="dice-button" id="surprise" type="button" aria-label="Use a surprise prompt">⚄</button>
          </div>
          <details>
            <summary>Experiment controls</summary>
            <label class="seed-label" for="seed">Seed <input id="seed" type="number" min="0" max="4294967295" value="1031" /></label>
            <p>Keep the same seed to reproduce an image. Change it to explore another point in the model’s latent space.</p>
          </details>
        </form>

        <div class="result-panel">
          <div class="result-heading">
            <p class="eyebrow">The same pumpkin, translated</p>
            <h3 id="output-title">Colored ASCII portrait</h3>
          </div>
          <div class="canvas-frame" id="canvas-frame">
            <pre id="ascii-output" aria-label="Colored ASCII version of the generated pumpkin"></pre>
            <canvas id="pumpkin-canvas" width="512" height="512" aria-label="Generated jack-o'-lantern" hidden></canvas>
            <div class="canvas-placeholder" id="placeholder"><span>✦</span><b>Warming the lantern…</b><small>The tiny model loads once, then stays on your device.</small></div>
          </div>
          <div class="result-actions">
            <button type="button" id="again" disabled>Carve again</button>
            <button type="button" id="ascii-toggle" aria-pressed="true" disabled>Show image</button>
            <button type="button" id="copy-ascii" disabled>Copy plain text</button>
            <button type="button" id="download" disabled>Download PNG</button>
          </div>
          <p class="generation-status" id="generation-status" aria-live="polite"></p>
        </div>
      </div>
    </section>

    <section class="learn-section" id="learn" aria-labelledby="learn-title">
      <div class="section-label light"><span>02</span> Under the hood</div>
      <div class="learn-intro">
        <h2 id="learn-title">A counterfeiter,<br />a critic, and a truce.</h2>
        <p>A GAN learns through a contest between two neural networks. We train both; your browser only needs the artist.</p>
      </div>
      <ol class="learning-flow">
        <li><span class="flow-number">1</span><div><h3>The generator imagines</h3><p>Random numbers and seven chosen attributes enter a compact convolutional network. It turns them into a crisp 64×64 image that scales cleanly for pixel art.</p></div></li>
        <li><span class="flow-number">2</span><div><h3>The discriminator critiques</h3><p>A second network compares generated pumpkins with our labeled procedural artwork and learns to spot the fakes.</p></div></li>
        <li><span class="flow-number">3</span><div><h3>Both improve</h3><p>The artist learns to fool the critic; the critic gets sharper. After training, we discard the critic and export only the small generator.</p></div></li>
      </ol>
      <div class="model-facts">
        <div><strong>64²</strong><span>native pixels</span></div>
        <div><strong>28</strong><span>condition values</span></div>
        <div><strong>0</strong><span>server requests per image</span></div>
      </div>
    </section>

    <section class="privacy-section">
      <p class="eyebrow">Private by architecture</p>
      <h2>The pumpkin patch<br />lives in your browser.</h2>
      <p>The website downloads a small ONNX model. WebGPU runs it on compatible graphics hardware; WebAssembly provides a CPU fallback. Once loaded, generation needs no network connection.</p>
      <div class="privacy-diagram" aria-label="Prompt and seed flow through the local generator to an image, all inside the browser">
        <span>Prompt + seed</span><b>→</b><span>Tiny generator</span><b>→</b><span>Your image</span>
      </div>
    </section>
  </main>

  <footer>
    <p>Made as a small, inspectable AI learning project. Implemented with assistance from OpenAI Codex using <a href="https://developers.openai.com/api/docs/models/gpt-5.6-sol">GPT-5.6 Sol</a>.</p>
    <a href="#top">Back to the moon ↑</a>
  </footer>
`;

const promptForm = document.querySelector<HTMLFormElement>("#prompt-form")!;
const promptInput = document.querySelector<HTMLTextAreaElement>("#prompt")!;
const promptNote = document.querySelector<HTMLParagraphElement>("#prompt-note")!;
const seedInput = document.querySelector<HTMLInputElement>("#seed")!;
const carveButton = document.querySelector<HTMLButtonElement>("#carve")!;
const surpriseButton = document.querySelector<HTMLButtonElement>("#surprise")!;
const againButton = document.querySelector<HTMLButtonElement>("#again")!;
const downloadButton = document.querySelector<HTMLButtonElement>("#download")!;
const asciiToggle = document.querySelector<HTMLButtonElement>("#ascii-toggle")!;
const copyAsciiButton = document.querySelector<HTMLButtonElement>("#copy-ascii")!;
const asciiOutput = document.querySelector<HTMLElement>("#ascii-output")!;
const outputTitle = document.querySelector<HTMLElement>("#output-title")!;
const runtimePill = document.querySelector<HTMLElement>("#runtime-pill")!;
const placeholder = document.querySelector<HTMLElement>("#placeholder")!;
const status = document.querySelector<HTMLElement>("#generation-status")!;
const canvas = document.querySelector<HTMLCanvasElement>("#pumpkin-canvas")!;
const context = canvas.getContext("2d", { alpha: false })!;
const generator = new PumpkinGenerator();
let lastImage: ImageData | null = null;
let busy = false;

function renderChips(): void {
  const container = document.querySelector<HTMLDivElement>("#chips")!;
  (Object.keys(GROUPS) as GroupName[]).forEach((group) => {
    const row = document.createElement("div");
    row.className = "chip-row";
    const label = document.createElement("span");
    label.textContent = group;
    row.append(label);
    GROUPS[group].forEach((value, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip";
      button.textContent = value;
      button.dataset.group = group;
      button.dataset.index = String(index);
      button.addEventListener("click", () => selectChip(group, index));
      row.append(button);
    });
    container.append(row);
  });
  updateChips();
}

function renderVocabulary(): void {
  const container = document.querySelector<HTMLDivElement>("#vocabulary-list")!;
  (Object.keys(VOCABULARY_BY_GROUP) as GroupName[]).forEach((group) => {
    const section = document.createElement("section");
    section.className = "vocabulary-group";
    const label = document.createElement("span");
    label.textContent = group;
    const terms = document.createElement("div");
    VOCABULARY_BY_GROUP[group].forEach((term) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = term;
      button.addEventListener("click", () => {
        promptInput.value = `${promptInput.value.trim()} ${term}`;
        updatePromptFeedback();
        promptInput.focus();
      });
      terms.append(button);
    });
    section.append(label, terms);
    container.append(section);
  });
}

function selectChip(group: GroupName, index: number): void {
  const parsed = parsePrompt(promptInput.value);
  const current = labelFromSelection(group, parsed.selections[group]);
  const replacement = labelFromSelection(group, index);
  const pattern = new RegExp(current.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  promptInput.value = pattern.test(promptInput.value)
    ? promptInput.value.replace(pattern, replacement)
    : `${promptInput.value.trim()} ${replacement}`;
  updatePromptFeedback();
}

function updateChips(): void {
  const parsed = parsePrompt(promptInput.value);
  document.querySelectorAll<HTMLButtonElement>(".chip").forEach((chip) => {
    const group = chip.dataset.group as GroupName;
    chip.classList.toggle("selected", parsed.selections[group] === Number(chip.dataset.index));
  });
}

function updatePromptFeedback(): void {
  const parsed = parsePrompt(promptInput.value);
  promptNote.textContent = parsed.unknown.length
    ? `Tiny vocabulary note: I don’t know “${parsed.unknown.join("”, “")}” yet, so those words will be ignored.`
    : `All prompt words recognized from the ${VISUAL_VOCABULARY_SIZE}-word vocabulary. The chips show the 7 conditions sent to the network.`;
  promptNote.classList.toggle("warning", parsed.unknown.length > 0);
  updateChips();
}

function drawImage(image: ImageData): void {
  const scratch = document.createElement("canvas");
  scratch.width = image.width;
  scratch.height = image.height;
  scratch.getContext("2d")!.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(scratch, 0, 0, canvas.width, canvas.height);
  placeholder.hidden = true;
}

function renderAscii(image: ImageData): void {
  asciiOutput.replaceChildren();
  imageToAsciiCells(image).forEach((line) => {
    const row = document.createElement("span");
    row.className = "ascii-row";
    line.forEach((cell) => {
      const character = document.createElement("span");
      character.textContent = cell.character;
      character.style.color = cell.color;
      row.append(character);
    });
    asciiOutput.append(row);
  });
}

async function generate(): Promise<void> {
  if (busy || !generator.runtime) return;
  busy = true;
  carveButton.disabled = true;
  againButton.disabled = true;
  status.textContent = "The generator is imagining…";
  const started = performance.now();
  try {
    const parsed = parsePrompt(promptInput.value);
    const seed = Number(seedInput.value) >>> 0;
    lastImage = await generator.generate(parsed.vector, seed);
    drawImage(lastImage);
    renderAscii(lastImage);
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    status.textContent = `Carved locally in ${elapsed}s · seed ${seed}`;
    againButton.disabled = false;
    downloadButton.disabled = false;
    asciiToggle.disabled = false;
    copyAsciiButton.disabled = false;
  } catch (error) {
    status.textContent = `Generation failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    busy = false;
    carveButton.disabled = false;
  }
}

promptInput.addEventListener("input", updatePromptFeedback);
promptForm.addEventListener("submit", (event) => { event.preventDefault(); void generate(); });
surpriseButton.addEventListener("click", () => {
  promptInput.value = examples[Math.floor(Math.random() * examples.length)];
  seedInput.value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  updatePromptFeedback();
});
againButton.addEventListener("click", () => {
  seedInput.value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  void generate();
});
asciiToggle.addEventListener("click", () => {
  const showAscii = asciiOutput.hidden;
  asciiOutput.hidden = !showAscii;
  canvas.hidden = showAscii;
  asciiToggle.textContent = showAscii ? "Show image" : "Show ASCII";
  asciiToggle.setAttribute("aria-pressed", String(showAscii));
  outputTitle.textContent = showAscii ? "Colored ASCII portrait" : "Generated pumpkin";
});
downloadButton.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = `tiny-pumpkin-${Number(seedInput.value) >>> 0}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
});
copyAsciiButton.addEventListener("click", async () => {
  if (lastImage) await navigator.clipboard.writeText(imageToAscii(lastImage));
});

renderChips();
renderVocabulary();
updatePromptFeedback();
if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => void navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`));
}
void generator.initialize().then((runtime) => {
  runtimePill.innerHTML = `<span></span><b>${runtime === "webgpu" ? "WebGPU" : "CPU / WASM"}</b>`;
  runtimePill.classList.add("ready");
  carveButton.disabled = false;
  status.textContent = "The tiny generator is ready.";
  return generate();
}).catch((error: unknown) => {
  runtimePill.innerHTML = "<span></span><b>Could not load</b>";
  runtimePill.classList.add("error");
  placeholder.innerHTML = `<span>!</span><b>The model could not start.</b><small>${error instanceof Error ? error.message : String(error)}</small>`;
  status.textContent = "Try refreshing or use a current browser.";
});

