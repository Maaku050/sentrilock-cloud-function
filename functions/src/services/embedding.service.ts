import * as tf from "@tensorflow/tfjs-node";
import { Landmark } from "../types";
import { detectFace } from "./faceDetection.service";

/**
 * Create face embedding from tensor and landmarks
 */
async function createEmbedding(
  faceTensor: tf.Tensor3D,
  landmarks: Landmark[]
): Promise<Float32Array> {
  const lmFeatures: number[] = [];

  // Flatten landmark coordinates
  for (const landmark of landmarks) {
    lmFeatures.push(landmark.x);
    lmFeatures.push(landmark.y);
  }

  // Compute color statistics from resized face
  const resized = tf.image
    .resizeBilinear(faceTensor, [64, 64])
    .toFloat()
    .div(tf.scalar(255.0));
  const data = await resized.data();

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  for (let i = 0; i < data.length; i += 3) {
    rSum += data[i];
    gSum += data[i + 1];
    bSum += data[i + 2];
  }
  const pxCount = data.length / 3;
  const avgR = rSum / pxCount;
  const avgG = gSum / pxCount;
  const avgB = bSum / pxCount;

  // Sample texture features (64 values from 64x64 image)
  const sampled: number[] = [];
  const sampleCount = 64;
  const step = Math.max(1, Math.floor(data.length / 3 / sampleCount));
  for (let i = 0; i < sampleCount; i++) {
    const idx = i * step * 3;
    sampled.push(data[idx] ?? 0);
    sampled.push(data[idx + 1] ?? 0);
    sampled.push(data[idx + 2] ?? 0);
  }

  resized.dispose();

  // Combine all features
  const combined: number[] = [...lmFeatures, avgR, avgG, avgB, ...sampled];

  // L2-normalize the embedding
  const arr = new Float32Array(combined);
  let norm = 0;
  for (let i = 0; i < arr.length; i++) {
    norm += arr[i] * arr[i];
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < arr.length; i++) {
    arr[i] = arr[i] / norm;
  }

  return arr;
}

/**
 * Extract face embedding from image buffer
 */
export async function extractFaceEmbedding(
  imageBuffer: Buffer
): Promise<Float32Array | null> {
  try {
    const detection = await detectFace(imageBuffer);

    if (!detection) {
      console.log("No face detected");
      return null;
    }

    const { faceTensor, landmarks } = detection;
    console.log("Creating embedding...");

    const embedding = await createEmbedding(faceTensor, landmarks);

    faceTensor.dispose();
    console.log(`✓ Embedding created: ${embedding.length} dimensions`);

    return embedding;
  } catch (err) {
    console.error("Error in extractFaceEmbedding:", err);
    return null;
  }
}
