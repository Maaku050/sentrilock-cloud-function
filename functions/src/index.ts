import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import express, {Request, Response, raw} from "express";
import cors from "cors";
import * as faceapi from "face-api.js";
import * as canvas from "canvas";
import sharp from "sharp";

// Patch face-api.js to use node-canvas
const {Canvas, Image, ImageData} = canvas;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(faceapi.env as any).monkeyPatch({Canvas, Image, ImageData});

// Initialize Firebase Admin
admin.initializeApp();

const app = express();
const bucket = admin.storage().bucket();
const db = admin.firestore();

// Enable CORS for all routes
app.use(cors({origin: true}));
app.use(express.json());

// Load face-api models (runs once when function starts)
let modelsLoaded = false;

/**
 * Load face-api.js models for face detection and recognition
 */
async function loadModels() {
  if (modelsLoaded) return;

  try {
    console.log("Loading face-api.js models...");
    // You'll need to include models in your deployment
    const modelPath = "./models";

    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);

    modelsLoaded = true;
    console.log("✓ Face-api models loaded successfully");
  } catch (error) {
    console.error("Failed to load face-api models:", error);
    throw error;
  }
}

/**
 * Extract face descriptor (embedding) from image buffer
 * @param {Buffer} imageBuffer - Image buffer to process
 * @return {Promise<Float32Array | null>} Face embedding or null
 */
async function extractFaceEmbedding(
  imageBuffer: Buffer
): Promise<Float32Array | null> {
  try {
    await loadModels();

    const fixedImage = await sharp(imageBuffer).rotate().toBuffer();
    const img = await canvas.loadImage(fixedImage);

    // face-api.js types don't match node-canvas types
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const detection = await faceapi
      .detectSingleFace(img as any)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      console.log("No face detected in image");
      return null;
    }

    console.log(
      `✓ Face detected with confidence: ${detection.detection.score}`
    );
    return detection.descriptor;
  } catch (error) {
    console.error("Error extracting face embedding:", error);
    return null;
  }
}

/**
 * Calculate Euclidean distance between two embeddings
 * @param {Float32Array} embedding1 - First embedding
 * @param {number[]} embedding2 - Second embedding
 * @return {number} Distance between embeddings
 */
function calculateDistance(
  embedding1: Float32Array,
  embedding2: number[]
): number {
  let sum = 0;
  for (let i = 0; i < embedding1.length; i++) {
    const diff = embedding1[i] - embedding2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// Interface for match result
interface MatchResult {
  personId: string;
  name: string;
  distance: number;
  confidence: number;
}

/**
 * Find matching person from stored embeddings
 * @param {Float32Array} embedding - Face embedding to match
 * @return {Promise<MatchResult | null>} Matched person or null
 */
async function findMatchingPerson(
  embedding: Float32Array
): Promise<MatchResult | null> {
  try {
    // Get all known persons from Firestore
    const personsSnapshot = await db.collection("known_persons").get();

    if (personsSnapshot.empty) {
      console.log("No known persons in database");
      return null;
    }

    let bestMatch: MatchResult | null = null;
    const threshold = 0.6; // Distance threshold for matching

    personsSnapshot.forEach((doc) => {
      const person = doc.data();
      const storedEmbedding = person.embedding;

      if (!storedEmbedding || !Array.isArray(storedEmbedding)) {
        return;
      }

      const distance = calculateDistance(embedding, storedEmbedding);
      const confidence = Math.max(0, 1 - distance);

      console.log(
        `Distance to ${person.name}: ${distance.toFixed(4)} ` +
          `(confidence: ${(confidence * 100).toFixed(2)}%)`
      );

      if (distance < threshold) {
        if (bestMatch === null || distance < bestMatch.distance) {
          const newMatch: MatchResult = {
            personId: doc.id,
            name: person.name,
            distance: distance,
            confidence: confidence,
          };
          bestMatch = newMatch;
        }
      }
    });

    if (bestMatch !== null) {
      const match: MatchResult = bestMatch;
      console.log(
        `✓ Match found: ${match.name} ` +
          `(${(match.confidence * 100).toFixed(2)}% confidence)`
      );
      return match;
    }

    console.log("No matching person found (unknown face)");
    return null;
  } catch (error) {
    console.error("Error finding matching person:", error);
    return null;
  }
}

// Root endpoint - API info
app.get("/", (req: Request, res: Response) => {
  res.json({
    status: "online",
    message: "ESP32-CAM Facial Recognition API",
    timestamp: new Date().toISOString(),
    endpoints: {
      "GET /": "API information",
      "GET /ping": "Simple ping test",
      "POST /upload": "Image upload with facial recognition",
      "POST /register": "Register a new person with their face",
      "GET /persons": "List all registered persons",
      "GET /images": "List all uploaded images",
      "GET /images/latest": "Get latest uploaded image URL",
      "DELETE /persons/:id": "Delete a registered person",
    },
  });
});

// Simple ping endpoint
app.get("/ping", (req: Request, res: Response) => {
  console.log("Ping received from:", req.ip);

  res.json({
    status: "success",
    message: "pong",
    timestamp: new Date().toISOString(),
    clientIP: req.ip,
  });
});

// Face recognition authorization endpoint
app.post(
  "/upload",
  raw({type: "image/jpeg", limit: "10mb"}),
  async (req: Request, res: Response): Promise<void> => {
    console.log("=== Face Recognition Authorization ===");
    console.log("Time:", new Date().toISOString());
    console.log("Client IP:", req.ip);

    try {
      const imageBuffer = req.body as Buffer;

      console.log(`✓ Received ${imageBuffer.length} bytes`);

      // Validate JPEG format
      if (imageBuffer[0] !== 0xff || imageBuffer[1] !== 0xd8) {
        console.log("❌ Invalid JPEG format");
        res.status(400).json({
          status: "error",
          message: "Invalid JPEG format",
          isAuthorized: false,
        });
        return;
      }

      console.log("✓ Valid JPEG format confirmed");

      // Extract face embedding
      console.log("Extracting face embedding...");
      const embedding = await extractFaceEmbedding(imageBuffer);

      if (!embedding) {
        console.log("❌ No face detected");
        res.status(200).json({
          status: "success",
          message: "No face detected in image",
          isAuthorized: false,
        });
        return;
      }

      console.log("✓ Face detected, finding match...");

      // Find matching person
      const matchedPerson = await findMatchingPerson(embedding);

      if (matchedPerson) {
        // Authorized - Face matched
        console.log(`✓ AUTHORIZED: ${matchedPerson.name}`);

        // Get full user data from Firestore
        const personDoc = await db
          .collection("known_persons")
          .doc(matchedPerson.personId)
          .get();

        const userData = personDoc.data();

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
        // Not authorized - No match found
        console.log("❌ UNAUTHORIZED: Unknown face");
        res.status(200).json({
          status: "success",
          message: "Face not recognized - Access denied",
          isAuthorized: false,
        });
      }

      console.log("=== Authorization Complete ===\n");
    } catch (error) {
      console.error("❌ Error processing authorization:", error);
      res.status(500).json({
        status: "error",
        message: "Failed to process face recognition",
        isAuthorized: false,
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }
);

// Register a new person with their face
app.post(
  "/register",
  raw({type: "image/jpeg", limit: "10mb"}),
  async (req: Request, res: Response): Promise<void> => {
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

      console.log(`Registering person: ${name}`);

      // Validate JPEG format
      if (imageBuffer[0] !== 0xff || imageBuffer[1] !== 0xd8) {
        res.status(400).json({
          status: "error",
          message: "Invalid JPEG format",
        });
        return;
      }

      // Extract face embedding
      const embedding = await extractFaceEmbedding(imageBuffer);

      if (!embedding) {
        res.status(400).json({
          status: "error",
          message:
            "No face detected in image. Please provide a clear face photo.",
        });
        return;
      }

      console.log("✓ Face embedding extracted");

      // Generate filename for storage
      const timestamp = Date.now();
      const date = new Date(timestamp);
      const sanitizedName = name.replace(/\s+/g, "_").toLowerCase();
      const filename = `registered_faces/${sanitizedName}_${timestamp}.jpg`;

      console.log(`Uploading image to Storage: ${filename}`);

      // Upload image to Firebase Storage
      const file = bucket.file(filename);

      await file.save(imageBuffer, {
        metadata: {
          contentType: "image/jpeg",
          metadata: {
            name: name,
            email: email || "",
            phone: phone || "",
            registeredAt: date.toISOString(),
            source: "registration",
          },
        },
      });

      console.log("✓ Image uploaded to Storage");

      // Make the file publicly accessible
      await file.makePublic();

      // Get public URL
      const imageUrl = `https://storage.googleapis.com/${bucket.name}/${filename}`;

      console.log(`✓ Public URL: ${imageUrl}`);

      // Save to Firestore with image reference
      const docRef = await db.collection("known_persons").add({
        name: name,
        email: email,
        phone: phone,
        embedding: Array.from(embedding),
        imageUrl: imageUrl,
        imagePath: filename,
        registeredAt: admin.firestore.FieldValue.serverTimestamp(),
        registeredDate: date.toISOString(),
      });

      console.log(`✓ Person registered: ${name} (ID: ${docRef.id})`);

      res.status(200).json({
        status: "success",
        message: `Person '${name}' registered successfully`,
        data: {
          personId: docRef.id,
          name: name,
          email: email,
          phone: phone,
          imageUrl: imageUrl,
          embeddingSize: embedding.length,
          registeredDate: date.toISOString(),
        },
      });
    } catch (error) {
      console.error("❌ Error registering person:", error);
      res.status(500).json({
        status: "error",
        message: "Failed to register person",
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }
);

// Get list of registered persons
app.get("/persons", async (req: Request, res: Response): Promise<void> => {
  try {
    const snapshot = await db.collection("known_persons").get();

    const persons = snapshot.docs.map((doc) => ({
      id: doc.id,
      name: doc.data().name,
      registeredDate: doc.data().registeredDate,
    }));

    res.json({
      status: "success",
      count: persons.length,
      persons: persons,
    });
  } catch (error) {
    console.error("Error fetching persons:", error);
    res.status(500).json({
      status: "error",
      message: "Failed to fetch persons",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Delete a registered person
app.delete(
  "/persons/:id",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const personId = req.params.id;
      await db.collection("known_persons").doc(personId).delete();

      res.json({
        status: "success",
        message: `Person ${personId} deleted successfully`,
      });
    } catch (error) {
      console.error("Error deleting person:", error);
      res.status(500).json({
        status: "error",
        message: "Failed to delete person",
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }
);

// Get list of all uploaded images
app.get("/images", async (req: Request, res: Response): Promise<void> => {
  try {
    const limit = parseInt(req.query.limit as string) || 20;

    const snapshot = await db
      .collection("captures")
      .orderBy("timestamp", "desc")
      .limit(limit)
      .get();

    const images = snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));

    res.json({
      status: "success",
      count: images.length,
      images: images,
    });
  } catch (error) {
    console.error("Error fetching images:", error);
    res.status(500).json({
      status: "error",
      message: "Failed to fetch images",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Get latest uploaded image
app.get(
  "/images/latest",
  async (req: Request, res: Response): Promise<void> => {
    try {
      const snapshot = await db
        .collection("captures")
        .orderBy("timestamp", "desc")
        .limit(1)
        .get();

      if (snapshot.empty) {
        res.status(404).json({
          status: "error",
          message: "No images found",
        });
        return;
      }

      const latestImage = {
        id: snapshot.docs[0].id,
        ...snapshot.docs[0].data(),
      };

      res.json({
        status: "success",
        image: latestImage,
      });
    } catch (error) {
      console.error("Error fetching latest image:", error);
      res.status(500).json({
        status: "error",
        message: "Failed to fetch latest image",
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }
);

// Health check endpoint
app.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    modelsLoaded: modelsLoaded,
  });
});

// Export the Express app as a Firebase Function
export const esp32cam = functions.https.onRequest(
  {
    timeoutSeconds: 300,
    memory: "1GiB",
  },
  app
);
