import * as ort from "onnxruntime-web/webgpu";
import { normalVector } from "./random";

export type Runtime = "webgpu" | "wasm";

export class PumpkinGenerator {
  private session: ort.InferenceSession | null = null;
  runtime: Runtime | null = null;

  async initialize(): Promise<Runtime> {
    const modelUrl = `${import.meta.env.BASE_URL}generator.onnx`;
    if ("gpu" in navigator) {
      try {
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
        });
        this.runtime = "webgpu";
        return this.runtime;
      } catch (error) {
        console.warn("WebGPU initialization failed; retrying with WASM.", error);
      }
    }

    ort.env.wasm.numThreads = 1;
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    this.runtime = "wasm";
    return this.runtime;
  }

  async generate(condition: Float32Array, seed: number): Promise<ImageData> {
    if (!this.session) throw new Error("The pumpkin generator has not loaded yet.");
    const latent = normalVector(96, seed);
    const output = await this.session.run({
      latent: new ort.Tensor("float32", latent, [1, 96]),
      condition: new ort.Tensor("float32", condition, [1, 28]),
    });
    const tensor = output.image;
    if (!tensor || !(tensor.data instanceof Float32Array)) throw new Error("The model returned an unexpected image.");
    const height = tensor.dims[2];
    const width = tensor.dims[3];
    if (!height || !width) throw new Error("The model returned an image without dimensions.");
    const pixels = new Uint8ClampedArray(width * height * 4);
    const plane = width * height;
    for (let i = 0; i < plane; i += 1) {
      pixels[i * 4] = Math.round(Math.max(0, Math.min(1, (tensor.data[i] + 1) / 2)) * 255);
      pixels[i * 4 + 1] = Math.round(Math.max(0, Math.min(1, (tensor.data[plane + i] + 1) / 2)) * 255);
      pixels[i * 4 + 2] = Math.round(Math.max(0, Math.min(1, (tensor.data[plane * 2 + i] + 1) / 2)) * 255);
      pixels[i * 4 + 3] = 255;
    }
    return new ImageData(pixels, width, height);
  }
}

