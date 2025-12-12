import * as blazeface from "@tensorflow-models/blazeface";
import * as canvas from "canvas";
import * as tf from "@tensorflow/tfjs-node";
import { Landmark } from "../types";

let blazeModel: blazeface.BlazeFaceModel | null = null;
let modelsLoaded = false;

/**
 * Load TensorFlow.js models for face detection
 */
export async function loadModels(): Promise<void> {
  if (modelsLoaded) return;

  console.log("Loading BlazeFace model...");
  console.log("TF backend:", tf.getBackend());

  blazeModel = await blazeface.load();
  console.log("✓ BlazeFace loaded");

  modelsLoaded = true;
  console.log("✓ Model loaded successfully");
}

/**
 * Convert tensor coordinate to number array
 */
function toNumberArray(t: any): [number, number] {
  if (Array.isArray(t)) return [t[0], t[1]];
  if (typeof t.arraySync === "function") {
    return t.arraySync();
  }
  throw new Error("Invalid tensor coordinate format");
}

/**
 * Extract landmarks from prediction
 */
function extractLandmarks(
  prediction: any,
  startX: number,
  startY: number,
  width: number,
  height: number
): Landmark[] {
  const landmarks: Landmark[] = [];

  if (!prediction.landmarks) {
    console.log("⚠️  No landmarks in prediction");
    return landmarks;
  }

  console.log(`Landmarks type: ${typeof prediction.landmarks}`);

  if (Array.isArray(prediction.landmarks)) {
    console.log(
      `Processing ${prediction.landmarks.length} landmarks (array)...`
    );
    for (const lm of prediction.landmarks) {
      const [lx, ly] = toNumberArray(lm);
      landmarks.push({
        x: (lx - startX) / width,
        y: (ly - startY) / height,
      });
    }
  } else if (typeof prediction.landmarks.arraySync === "function") {
    console.log("Processing landmarks (tensor)...");
    const lmArray = prediction.landmarks.arraySync();
    for (let i = 0; i < lmArray.length; i++) {
      const lm = lmArray[i];
      landmarks.push({
        x: (lm[0] - startX) / width,
        y: (lm[1] - startY) / height,
      });
    }
  }

  console.log(`✓ Extracted ${landmarks.length} landmarks`);
  return landmarks;
}

/**
 * Extract face region and landmarks from image
 */
export async function detectFace(imageBuffer: Buffer): Promise<{
  faceTensor: tf.Tensor3D;
  landmarks: Landmark[];
} | null> {
  try {
    console.log("=== Starting face detection ===");
    console.log(`Buffer size: ${imageBuffer.length} bytes`);

    await loadModels();
    if (!blazeModel) {
      throw new Error("BlazeFace model not loaded");
    }

    console.log("Loading image...");
    const img = await canvas.loadImage(imageBuffer);
    console.log(`✓ Image loaded: ${img.width}x${img.height}`);

    console.log("Creating canvas...");
    const canvasEl = canvas.createCanvas(img.width, img.height);
    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(img, 0, 0);
    console.log("✓ Canvas created");

    console.log("Converting to tensor...");
    const inputTensor = tf.browser.fromPixels(canvasEl as any);
    console.log(`✓ Tensor shape: ${inputTensor.shape}`);

    console.log("Running BlazeFace detection...");
    let predictions = await blazeModel.estimateFaces(inputTensor as any, false);

    if (!predictions || predictions.length === 0) {
      console.log("❌ No faces detected, trying with returnTensors=true...");
      predictions = await blazeModel.estimateFaces(inputTensor as any, true);

      if (!predictions || predictions.length === 0) {
        inputTensor.dispose();
        return null;
      }
    }

    console.log(`✓ Found ${predictions.length} face(s)`);
    const prediction = predictions[0];

    // Extract bounding box
    const [x1, y1] = toNumberArray(prediction.topLeft);
    const [x2, y2] = toNumberArray(prediction.bottomRight);
    console.log(`Raw bbox: (${x1}, ${y1}) to (${x2}, ${y2})`);

    // Add padding
    const padding = Math.round(Math.max(x2 - x1, y2 - y1) * 0.12);
    let startX = Math.max(0, Math.floor(x1) - padding);
    let startY = Math.max(0, Math.floor(y1) - padding);
    let endX = Math.min(img.width, Math.ceil(x2) + padding);
    let endY = Math.min(img.height, Math.ceil(y2) + padding);

    const width = endX - startX;
    const height = endY - startY;
    console.log(
      `Final bbox: x=${startX}, y=${startY}, w=${width}, h=${height}`
    );

    if (width <= 0 || height <= 0) {
      console.log("❌ Invalid bounding box dimensions");
      inputTensor.dispose();
      return null;
    }

    // Extract face region
    const faceCanvas = canvas.createCanvas(width, height);
    const faceCtx = faceCanvas.getContext("2d");
    faceCtx.drawImage(
      canvasEl,
      startX,
      startY,
      width,
      height,
      0,
      0,
      width,
      height
    );
    console.log("✓ Face region extracted");

    const faceTensor = tf.browser.fromPixels(faceCanvas as any) as tf.Tensor3D;
    const landmarks = extractLandmarks(
      prediction,
      startX,
      startY,
      width,
      height
    );

    inputTensor.dispose();
    console.log("=== Face detection complete ===");

    return { faceTensor, landmarks };
  } catch (err) {
    console.error("Error in detectFace:", err);
    console.error("Stack:", (err as Error).stack);
    return null;
  }
}

export function isModelsLoaded(): boolean {
  return modelsLoaded;
}
