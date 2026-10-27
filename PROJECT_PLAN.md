# Tiny Pumpkin Machine — Project Plan

## 1. Product goal

Build an educational, family-friendly Halloween web app that generates small
jack-o'-lantern images entirely on the visitor's device. The released project
must be open on GitHub, cost nothing to host or operate, require no API key, and
send neither prompts nor generated images to a server.

The primary generator will be a small conditional GAN exported to ONNX. The web
app will prefer WebGPU and fall back to WebAssembly/CPU. A colored ASCII view
will turn the generated image into a second, playful representation.

## 2. Success criteria

- A visitor opens an HTTPS website, describes a pumpkin, and generates an image.
- No backend, cloud GPU, database, account, analytics, or paid service is used.
- The complete first-load payload has a target size under 15 MB and a hard
  release ceiling of 25 MB.
- WebGPU is used where supported; WASM inference works as the compatibility path.
- Generation works offline after the application assets have been cached.
- The same prompt and seed reproduce the same output.
- The website is usable with keyboard, mouse, and touch.
- The repository explains the ML concepts, training process, limitations, and
  browser deployment in beginner-friendly language.
- Tests, production build, ONNX validation, and a clean-browser smoke test pass
  before release.

## 3. User experience

### Main screen

- Project title and a one-sentence privacy statement.
- Prompt box with examples and clickable vocabulary chips.
- `Carve it!` button.
- Generated pumpkin with `Carve again`, `Download PNG`, `Copy prompt`, and
  `Show as ASCII` actions.
- A small runtime badge: `WebGPU`, `CPU/WASM`, or a helpful initialization error.
- Seed control hidden under an optional “experiment” panel.

### Prompt behavior

This is a deliberately small model, so the website will not pretend to
understand unrestricted English. A deterministic prompt compiler will recognize
a documented vocabulary and convert it to condition values.

Initial condition groups:

- shape: round, tall, wide, lumpy
- expression: happy, silly, surprised, angry, spooky
- eyes: triangle, round, star, sleepy
- mouth: smile, crooked, toothy, fangs
- pumpkin color: orange, green, pale
- glow: yellow, green, blue, purple
- accessory: none, witch hat, pirate hat, stem leaf

Unrecognized words remain visible and the interface gently suggests supported
alternatives. Controls and prompt text stay synchronized.

### Colored ASCII view

The generated image will be converted locally to a character grid. Brightness
selects characters while sampled pixel color supplies the foreground color.
CSS provides a dark background and optional glow. Users can copy plain text or
download a rendered PNG.

## 4. System architecture

```text
Training (developer machine; not part of the deployed app)

Procedural dataset generator + optional original artwork
                         |
                         v
               labeled 64x64 images
                         |
                         v
          PyTorch conditional GAN training
                         |
                         v
            generator-only checkpoint
                         |
                         v
         ONNX export + parity/size validation
                         |
                         v
              public/generator.onnx


Inference (visitor's browser)

Prompt -> prompt compiler -> condition vector --+
                                                 |
Seed ----------------------> latent vector -------+--> ONNX generator
                                                        |
                                    WebGPU preferred / WASM fallback
                                                        |
                                               64x64 RGB image
                                                        |
                                  display / upscale / PNG / ASCII
```

There is no deployed application server. GitHub Pages serves immutable static
files over HTTPS, and ONNX Runtime Web performs inference in the browser.

## 5. Model design

### Generator

- Input: random latent vector plus a multi-hot condition vector.
- Network: learned projection followed by four upsampling convolution blocks.
- Output: 3 x 128 x 128 RGB tensor.
- Target: approximately 1–5 million parameters and 3–12 MB after export.
- Export only the generator; the discriminator is training-only.

### Discriminator

- Convolutional image encoder.
- Condition projection or concatenated condition embedding.
- Spectral normalization and/or gradient penalty will be considered if basic
  adversarial training is unstable.

### Training safeguards

- Fixed validation seeds and condition sets saved at every checkpoint.
- Sample grids and loss curves for human inspection.
- Checkpoint resume support.
- Deterministic seed option.
- Dataset manifest and documented provenance.
- Tests that every condition combination has reasonable dataset coverage.

### Dataset strategy

Start with a reproducible procedural generator that draws pumpkin bodies,
carved features, glow, texture, and accessories. This gives us original,
automatically labeled training data without scraping copyrighted images. Any
hand-drawn additions must be owned by the contributor and recorded in the
dataset manifest.

An early baseline will compare the neural generator against held-out procedural
images. We will keep the procedural renderer in the repository so learners can
see both the source distribution and what the GAN learned from it.

## 6. Implementation stack

- Training: Python, PyTorch, Pillow, NumPy.
- Export and validation: PyTorch ONNX exporter and ONNX Runtime.
- Web application: TypeScript, Vite, lightweight HTML/CSS (framework only if it
  materially simplifies the interface).
- Browser inference: ONNX Runtime Web.
- Unit tests: Vitest for prompt/parser/UI logic and pytest for training utilities.
- Formatting/linting: Prettier, ESLint, Ruff.
- CI and deployment: GitHub Actions and GitHub Pages.

Exact dependency versions will be locked after the first working vertical slice.

## 7. Repository layout

```text
.
|-- app/                  # Browser application
|-- training/
|   |-- dataset/          # Procedural renderer and manifests
|   |-- models/           # Generator and discriminator definitions
|   |-- train.py
|   `-- evaluate.py
|-- export/               # ONNX export, optimization, and parity checks
|-- public/               # Released ONNX model and static assets
|-- tests/
|-- docs/                 # Architecture, training, and educational diagrams
|-- .github/workflows/    # CI and Pages deployment
|-- README.md
|-- MODEL_CARD.md
|-- DATASET_CARD.md
|-- CONTRIBUTING.md
|-- LICENSE
`-- THIRD_PARTY_NOTICES.md
```

## 8. Educational README outline

The README will include:

1. Live demo, screenshots, privacy promise, and quick start.
2. “What happens when I click Carve?” walkthrough.
3. Neural-network basics: parameters, tensors, convolutions, and latent vectors.
4. GAN explanation: generator and discriminator as an improving counterfeiter
   and critic, including the adversarial training loop.
5. Conditional generation: how labels steer the same generator.
6. Why the deployed site contains only the generator.
7. How a controlled prompt becomes a condition vector.
8. Training data creation and why provenance matters.
9. PyTorch-to-ONNX export and numerical parity testing.
10. WebGPU acceleration, WASM fallback, privacy, and zero server cost.
11. Failure modes such as mode collapse and how sample grids reveal them.
12. Reproduction instructions and suggested learner experiments.
13. Model and dataset limitations.
14. References and further reading.

Planned primary references:

- Goodfellow et al., “Generative Adversarial Nets” (2014).
- Radford, Metz, and Chintala, “Unsupervised Representation Learning with Deep
  Convolutional Generative Adversarial Networks” (2015).
- Mirza and Osindero, “Conditional Generative Adversarial Nets” (2014).
- PyTorch's maintained DCGAN tutorial.
- ONNX and ONNX Runtime Web documentation.
- W3C WebGPU specification and MDN WebGPU guide.

The prose will distinguish our implementation choices from ideas taken from
papers, tutorials, and libraries. Third-party code will retain its required
license notices; we will not copy tutorial code without attribution.

## 9. Phased implementation

### Phase 0 — Foundation

- Initialize Git and the application/training toolchains.
- Add licenses, formatting, tests, CI, and a minimal GitHub Pages deployment.
- Build a static interface with a temporary deterministic placeholder generator.

Exit: the public-shaped app builds and deploys without ML.

### Phase 1 — Data pipeline

- Implement procedural pumpkin rendering and condition labels.
- Generate dataset previews, manifests, and coverage reports.
- Add dataset documentation and deterministic regeneration.

Exit: a reproducible labeled dataset is ready for training.

### Phase 2 — Conditional GAN

- Implement generator/discriminator and training loop.
- Run a small overfit test, then baseline training.
- Track samples, stability, speed, and model size.
- Tune only as required to reach a fun, varied result.

Exit: a selected generator checkpoint passes visual review.

### Phase 3 — Browser inference

- Export the generator to ONNX.
- Verify PyTorch/ONNX output parity for fixed inputs.
- Integrate ONNX Runtime Web with WebGPU-first, WASM-second initialization.
- Add deterministic seed generation and browser performance measurements.

Exit: the trained model generates in a production browser with both execution
paths where supported.

### Phase 4 — Product polish

- Complete prompt compiler, suggestion chips, accessibility, responsive layout,
  PNG export, and colored ASCII mode.
- Add loading, unsupported-device, memory, and corrupted-cache recovery states.
- Add a small offline cache only after the base experience is stable.

Exit: a child can use the app without instructions.

### Phase 5 — Education and release

- Finish README, architecture guide, model card, dataset card, and references.
- Run unit, integration, build, model-size, parity, and clean-browser tests.
- Test desktop Chrome/Edge plus CPU fallback in Firefox/Safari where supported.
- Publish a tagged release and enable GitHub Pages.

Exit: source, model, documentation, and live website are public.

## 10. Verification and release gates

- `pytest` passes for data generation, model shapes, and export utilities.
- web unit tests, lint, type-check, and production build pass.
- ONNX output matches PyTorch within an agreed numerical tolerance.
- Model-size and total-payload budgets pass in CI.
- Fixed-seed output is reproducible.
- WebGPU failure automatically retries with WASM.
- No network request occurs when generating after assets have loaded.
- No secret, API key, telemetry, or user-data upload is present.
- Keyboard navigation, labels, focus states, contrast, and reduced-motion behavior
  receive a manual accessibility check.
- A clean browser profile completes first load and generation over HTTPS.
- All model, dataset, and third-party licenses are documented.

## 11. Zero-cost deployment plan

### Primary: GitHub Pages

GitHub Actions will build the static application on pushes to the default branch
and deploy the build artifact to GitHub Pages. The public repository, Pages site,
and Actions usage for this small project should fit the free offering. Because
inference is local, popularity creates no GPU bill.

The ONNX model will remain in the repository if the final file stays within the
project's size budget. This avoids depending on a second hosting provider.

### Optional: Vercel mirror

Vercel is not required. If desired later, its free Hobby plan can import the
GitHub repository and deploy the same static build. This project is personal and
non-commercial, which matches the current Hobby-plan restriction. We will avoid
serverless functions so there is no runtime service to configure or accidentally
consume.

## 12. Information and access needed from the project owner

Needed before GPU training:

- GPU model and VRAM, obtainable with `nvidia-smi` for an NVIDIA GPU.
- Approval of the initial visual direction: pixel-art, illustrated, or softly
  rendered. The default plan is illustrated pixel art because it tolerates a
  small network and dataset well.

Needed only when publishing:

- A free GitHub account.
- The desired public repository name and owner/account.
- Authentication/approval to create or push the repository and enable Pages.
- Optional custom domain ownership; otherwise GitHub supplies a free
  `github.io` address.

Not currently needed:

- Vercel account.
- Cloud provider or GPU account.
- Payment method.
- Domain purchase.
- Hugging Face account, unless the final model unexpectedly exceeds the chosen
  repository budget.

## 13. Release handoff

The release report will contain:

- product and visual design decisions;
- final architecture and data flow;
- model architecture, parameter count, file size, and measured performance;
- dataset provenance and training procedure;
- implementation and local-development instructions;
- test and browser-compatibility results;
- GitHub Pages deployment configuration and live URL;
- reproduction, retraining, and contribution instructions;
- limitations, licenses, references, and future experiments;
- any owner-controlled account settings required to maintain the release.

