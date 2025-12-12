import { Request, Response } from "express";
import { extractFaceEmbedding } from "../services/embedding.service";
import {
  uploadImage,
  savePerson,
  getAllPersons,
  deletePerson,
} from "../services/storage.service";

/**
 * Register a new person
 */
export async function registerPerson(
  req: Request,
  res: Response
): Promise<void> {
  console.log("=== Register New Person ===");

  try {
    const imageBuffer = req.body as Buffer;
    const name = req.query.name as string;
    const email = (req.query.email as string) || null;
    const phone = (req.query.phone as string) || null;

    if (!name) {
      res.status(400).json({
        status: "error",
        message: "Name parameter is required (?name=John)",
      });
      return;
    }

    const embedding = await extractFaceEmbedding(imageBuffer);

    if (!embedding) {
      res.status(400).json({
        status: "error",
        message: "No face detected. Please provide a clear photo.",
      });
      return;
    }

    const { imageUrl, filename } = await uploadImage(
      imageBuffer,
      name,
      email,
      phone
    );

    const personId = await savePerson(
      name,
      email,
      phone,
      embedding,
      imageUrl,
      filename
    );

    res.status(200).json({
      status: "success",
      message: `Person '${name}' registered successfully`,
      data: {
        personId,
        name,
        email,
        phone,
        imageUrl,
        embeddingSize: embedding.length,
        registeredDate: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({
      status: "error",
      message: "Failed to register person",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * Get all registered persons
 */
export async function listPersons(req: Request, res: Response): Promise<void> {
  try {
    const persons = await getAllPersons();

    res.json({
      status: "success",
      count: persons.length,
      persons,
    });
  } catch (error) {
    res.status(500).json({
      status: "error",
      message: "Failed to fetch persons",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * Delete a person
 */
export async function removePersonById(
  req: Request,
  res: Response
): Promise<void> {
  try {
    const { id } = req.params;
    await deletePerson(id);

    res.json({
      status: "success",
      message: "Person deleted successfully",
    });
  } catch (error) {
    res.status(500).json({
      status: "error",
      message: "Failed to delete person",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}
