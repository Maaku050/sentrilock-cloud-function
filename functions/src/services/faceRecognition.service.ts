import FormData from "form-data";
import fetch from "node-fetch";

const FACE_SERVICE_URL =
  process.env.FACE_SERVICE_URL || "http://localhost:5000";

/**
 * Extract face embedding using InsightFace Python service
 */
export async function extractFaceEmbedding(
  imageBuffer: Buffer
): Promise<Float32Array | null> {
  try {
    console.log("=== Calling InsightFace service ===");
    console.log(`Buffer size: ${imageBuffer.length} bytes`);

    // Create form data
    const formData = new FormData();
    formData.append("image", imageBuffer, {
      filename: "image.jpg",
      contentType: "image/jpeg",
    });

    // Call InsightFace service
    const response = await fetch(`${FACE_SERVICE_URL}/extract-embedding`, {
      method: "POST",
      body: formData,
      headers: formData.getHeaders(),
    });

    const result = await response.json();

    if (!result.success) {
      console.log(`❌ ${result.message || result.error}`);
      return null;
    }

    console.log("✓ Face detected");
    console.log(
      `✓ Embedding size: ${result.embedding_size} dimensions (InsightFace)`
    );
    console.log(`✓ Number of faces: ${result.num_faces_detected}`);
    if (result.face?.quality_score) {
      console.log(
        `✓ Face quality score: ${(result.face.quality_score * 100).toFixed(2)}%`
      );
    }
    if (result.face?.gender) {
      console.log(`✓ Gender: ${result.face.gender}, Age: ${result.face.age}`);
    }
    console.log("=== Face extraction complete ===");

    return new Float32Array(result.embedding);
  } catch (error) {
    console.error("Error calling InsightFace service:", error);
    return null;
  }
}

/**
 * Compare two embeddings using Python service
 */
export async function compareEmbeddings(
  embedding1: Float32Array,
  embedding2: number[]
): Promise<{
  distance: number;
  similarity: number;
  isMatch: boolean;
} | null> {
  try {
    const response = await fetch(`${FACE_SERVICE_URL}/compare-embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        embedding1: Array.from(embedding1),
        embedding2: embedding2,
      }),
    });

    const result = await response.json();

    if (!result.success) {
      console.error("Comparison failed:", result.error);
      return null;
    }

    return {
      distance: result.distance,
      similarity: result.similarity,
      isMatch: result.is_match,
    };
  } catch (error) {
    console.error("Error comparing embeddings:", error);
    return null;
  }
}

/**
 * Check if face recognition service is available
 */
export async function checkServiceHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${FACE_SERVICE_URL}/health`);
    const result = await response.json();
    return result.status === "healthy";
  } catch (error) {
    console.error("Face recognition service unavailable:", error);
    return false;
  }
}
