import { Request, Response } from "express";
import { extractFaceEmbedding } from "../services/embedding.service";
import { findMatchingPerson } from "../services/matching.service";
import { getPersonById } from "../services/storage.service";

/**
 * Handle face recognition authorization
 */
export async function recognizeFace(
  req: Request,
  res: Response
): Promise<void> {
  console.log("=== Face Recognition Authorization ===");
  console.log("Time:", new Date().toISOString());

  try {
    const imageBuffer = req.body as Buffer;

    const embedding = await extractFaceEmbedding(imageBuffer);

    if (!embedding) {
      res.status(200).json({
        status: "success",
        message: "No face detected in image",
        isAuthorized: false,
      });
      return;
    }

    const matchedPerson = await findMatchingPerson(embedding);

    if (matchedPerson) {
      console.log(`✓ AUTHORIZED: ${matchedPerson.name}`);

      const userData = await getPersonById(matchedPerson.personId);

      res.status(200).json({
        status: "success",
        message: "Face recognized - Access granted",
        isAuthorized: true,
        user: {
          id: matchedPerson.personId,
          name: matchedPerson.name,
          confidence: parseFloat((matchedPerson.confidence * 100).toFixed(2)),
          registeredDate: userData?.registeredDate || null,
        },
      });
    } else {
      console.log("❌ UNAUTHORIZED: Unknown face");
      res.status(200).json({
        status: "success",
        message: "Face not recognized - Access denied",
        isAuthorized: false,
      });
    }
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({
      status: "error",
      message: "Failed to process face recognition",
      isAuthorized: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}
