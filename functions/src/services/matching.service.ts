import { db } from "../config/firebase";
import { MatchResult, PersonDocument } from "../types";
import { compareEmbeddings } from "./faceRecognition.service";

/**
 * Find matching person from stored embeddings using InsightFace
 * InsightFace uses different thresholds than face-recognition
 */
export async function findMatchingPerson(
  embedding: Float32Array,
  similarityThreshold = 70 // 70% similarity required (InsightFace scale)
): Promise<MatchResult | null> {
  try {
    const personsSnapshot = await db.collection("known_persons").get();

    if (personsSnapshot.empty) {
      console.log("No known persons in database");
      return null;
    }

    let bestMatch: MatchResult | null = null;

    for (const doc of personsSnapshot.docs) {
      const person = doc.data() as PersonDocument;
      const storedEmbedding = person.embedding;

      if (!storedEmbedding || !Array.isArray(storedEmbedding)) {
        continue;
      }

      // Use Python service to compare embeddings
      const comparison = await compareEmbeddings(embedding, storedEmbedding);

      if (!comparison) {
        console.log(`⚠️  Failed to compare with ${person.name}`);
        continue;
      }

      console.log(
        `Person: ${person.name} | ` +
          `Similarity: ${comparison.similarity.toFixed(2)}% | ` +
          `Distance: ${comparison.distance.toFixed(4)} | ` +
          `Match: ${comparison.isMatch ? "YES" : "NO"}`
      );

      // Check if similarity meets threshold
      if (comparison.similarity >= similarityThreshold) {
        if (
          bestMatch === null ||
          comparison.similarity > bestMatch.confidence * 100
        ) {
          bestMatch = {
            personId: doc.id,
            name: person.name,
            distance: comparison.distance,
            confidence: comparison.similarity / 100, // Convert back to 0-1 range
          };
        }
      }
    }

    if (bestMatch !== null) {
      console.log(
        `✓ Match found: ${bestMatch.name} ` +
          `(${(bestMatch.confidence * 100).toFixed(2)}% confidence)`
      );
      return bestMatch;
    }

    console.log("No matching person found (unknown face)");
    return null;
  } catch (error) {
    console.error("Error finding matching person:", error);
    return null;
  }
}
