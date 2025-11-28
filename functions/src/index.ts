import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import express, { Request, Response, raw } from 'express';
import cors from 'cors';
import * as blazeface from '@tensorflow-models/blazeface';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node';

// Initialize Firebase Admin
admin.initializeApp();

const app = express();
const bucket = admin.storage().bucket();
const db = admin.firestore();

// Enable CORS for all routes
app.use(cors({ origin: true }));
app.use(express.json());

let blazeModel: blazeface.BlazeFaceModel | null = null;
let modelsLoaded = false;

/**
 * Load TensorFlow.js models for face detection
 */
async function loadModelsNode() {
  if (modelsLoaded) return;
  console.log('Loading BlazeFace model...');

  console.log('TF backend:', tf.getBackend());

  // Load BlazeFace detector only
  blazeModel = await blazeface.load();
  console.log('✓ BlazeFace loaded');

  modelsLoaded = true;
  console.log('✓ Model loaded successfully');
}

/**
 * Convert tensor coordinate to number array
 * @param {any} t - Tensor coordinate
 * @return {[number, number]} Number array
 */
function toNumberArray(t: any): [number, number] {
  if (Array.isArray(t)) return [t[0], t[1]];
  if (typeof t.arraySync === 'function') {
    return t.arraySync();
  }
  throw new Error('Invalid tensor coordinate format');
}

/**
 * Extract face embedding using TensorFlow.js
 * @param {Buffer} imageBuffer - Image buffer to process
 * @return {Promise<Float32Array | null>} Face embedding or null
 */
async function extractFaceEmbeddingNode(
  imageBuffer: Buffer
): Promise<Float32Array | null> {
  try {
    console.log('=== Starting face extraction ===');
    console.log(`Buffer size: ${imageBuffer.length} bytes`);

    await loadModelsNode();
    if (!blazeModel) {
      console.error('BlazeFace model not loaded!');
      throw new Error('Models not initialized');
    }

    console.log('Loading image...');
    const img = await canvas.loadImage(imageBuffer);
    console.log(`✓ Image loaded: ${img.width}x${img.height}`);

    console.log('Creating canvas...');
    const canvasEl = canvas.createCanvas(img.width, img.height);
    const ctx = canvasEl.getContext('2d');
    ctx.drawImage(img, 0, 0);
    console.log('✓ Canvas created');

    console.log('Converting to tensor...');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const inputTensor = tf.browser.fromPixels(canvasEl as any);
    console.log(`✓ Tensor shape: ${inputTensor.shape}`);

    console.log('Running BlazeFace detection...');
    console.log('Detection parameters: returnTensors=false');

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const predictions = await blazeModel.estimateFaces(
      inputTensor as any,
      false // Try with returnTensors=false first
    );

    console.log(`BlazeFace returned: ${predictions?.length || 0} predictions`);

    if (!predictions || predictions.length === 0) {
      console.log('❌ No faces detected by BlazeFace');
      console.log('Trying with returnTensors=true...');

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const predictions2 = await blazeModel.estimateFaces(
        inputTensor as any,
        true
      );

      console.log(
        `BlazeFace (returnTensors=true): ${predictions2?.length || 0}`
      );

      if (!predictions2 || predictions2.length === 0) {
        inputTensor.dispose();
        return null;
      }

      // Use predictions2 if found
      return await processFaceDetection(
        predictions2[0],
        img,
        canvasEl,
        inputTensor
      );
    }

    console.log(`✓ Found ${predictions.length} face(s)`);

    const landmarksInfo = predictions[0].landmarks
      ? Array.isArray(predictions[0].landmarks)
        ? predictions[0].landmarks.length
        : 'tensor'
      : 'none';

    console.log(
      `Prediction details:`,
      JSON.stringify({
        topLeft: predictions[0].topLeft,
        bottomRight: predictions[0].bottomRight,
        probability: predictions[0].probability,
        landmarks: landmarksInfo,
      })
    );

    const result = await processFaceDetection(
      predictions[0],
      img,
      canvasEl,
      inputTensor
    );

    return result;
  } catch (err) {
    console.error('Error in extractFaceEmbeddingNode:', err);
    console.error('Stack:', (err as Error).stack);
    return null;
  }
}

/**
 * Process detected face and create embedding
 * @param {any} prediction - BlazeFace prediction
 * @param {any} img - Original image
 * @param {any} canvasEl - Canvas element
 * @param {tf.Tensor} inputTensor - Input tensor
 * @return {Promise<Float32Array | null>} Face embedding
 */
async function processFaceDetection(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  prediction: any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  img: any,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  canvasEl: any,
  inputTensor: tf.Tensor
): Promise<Float32Array | null> {
  try {
    const p = prediction;
    const [x1, y1] = toNumberArray(p.topLeft);
    const [x2, y2] = toNumberArray(p.bottomRight);

    console.log(`Raw bbox: (${x1}, ${y1}) to (${x2}, ${y2})`);

    let startX = Math.max(0, Math.floor(x1));
    let startY = Math.max(0, Math.floor(y1));
    let endX = Math.min(img.width, Math.ceil(x2));
    let endY = Math.min(img.height, Math.ceil(y2));

    // Add padding
    const padding = Math.round(Math.max(endX - startX, endY - startY) * 0.12);
    startX = Math.max(0, startX - padding);
    startY = Math.max(0, startY - padding);
    endX = Math.min(img.width, endX + padding);
    endY = Math.min(img.height, endY + padding);

    const width = endX - startX;
    const height = endY - startY;

    console.log(
      `Final bbox: x=${startX}, y=${startY}, w=${width}, h=${height}`
    );

    if (width <= 0 || height <= 0) {
      console.log('❌ Invalid bounding box dimensions');
      inputTensor.dispose();
      return null;
    }

    // Extract face region
    const faceCanvas = canvas.createCanvas(width, height);
    const faceCtx = faceCanvas.getContext('2d');
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

    console.log('✓ Face region extracted');

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const faceTensor = tf.browser.fromPixels(faceCanvas as any) as tf.Tensor3D;

    // Extract landmarks from BlazeFace prediction
    const landmarks: Array<{ x: number; y: number }> = [];

    if (p.landmarks) {
      console.log(`Landmarks type: ${typeof p.landmarks}`);
      console.log(`Landmarks:`, p.landmarks);

      // Check if landmarks is an array or tensor
      if (Array.isArray(p.landmarks)) {
        console.log(`Processing ${p.landmarks.length} landmarks (array)...`);
        for (const lm of p.landmarks) {
          const [lx, ly] = toNumberArray(lm);
          landmarks.push({
            x: (lx - startX) / width,
            y: (ly - startY) / height,
          });
        }
      } else if (typeof p.landmarks.arraySync === 'function') {
        // landmarks is a tensor
        console.log('Processing landmarks (tensor)...');
        const lmArray = p.landmarks.arraySync();
        console.log(`Landmarks array shape:`, lmArray.length);
        for (let i = 0; i < lmArray.length; i++) {
          const lm = lmArray[i];
          landmarks.push({
            x: (lm[0] - startX) / width,
            y: (lm[1] - startY) / height,
          });
        }
      } else {
        console.log('⚠️  Unknown landmarks format');
        console.log('Landmarks keys:', Object.keys(p.landmarks));
      }

      console.log(`✓ Extracted ${landmarks.length} landmarks`);
    } else {
      console.log('⚠️  No landmarks in prediction, using face region only');
    }

    console.log('Creating embedding...');
    const embedding = await createEmbeddingFromFaceTensor(
      faceTensor,
      landmarks
    );

    faceTensor.dispose();
    inputTensor.dispose();

    console.log(`✓ Embedding created: ${embedding.length} dimensions`);
    console.log('=== Face extraction complete ===');

    return embedding;
  } catch (err) {
    console.error('Error in processFaceDetection:', err);
    inputTensor.dispose();
    return null;
  }
}

/**
 * Create face embedding from tensor and landmarks
 * @param {tf.Tensor3D} faceTensor - Face image tensor
 * @param {Array} landmarks - Face landmarks from BlazeFace
 * @return {Promise<Float32Array>} Face embedding
 */
async function createEmbeddingFromFaceTensor(
  faceTensor: tf.Tensor3D,
  landmarks: Array<{ x: number; y: number }>
): Promise<Float32Array> {
  const lmFeatures: number[] = [];

  // Flatten landmark coordinates
  for (let i = 0; i < landmarks.length; i++) {
    lmFeatures.push(landmarks[i].x);
    lmFeatures.push(landmarks[i].y);
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
  const minLength = Math.min(embedding1.length, embedding2.length);
  for (let i = 0; i < minLength; i++) {
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
    const personsSnapshot = await db.collection('known_persons').get();

    if (personsSnapshot.empty) {
      console.log('No known persons in database');
      return null;
    }

    let bestMatch: MatchResult | null = null;
    const threshold = 0.8;

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

    console.log('No matching person found (unknown face)');
    return null;
  } catch (error) {
    console.error('Error finding matching person:', error);
    return null;
  }
}

// Root endpoint
app.get('/', (req: Request, res: Response) => {
  res.json({
    status: 'online',
    message: 'ESP32-CAM Facial Recognition API (TensorFlow.js)',
    timestamp: new Date().toISOString(),
    endpoints: {
      'GET /': 'API information',
      'GET /ping': 'Simple ping test',
      'POST /upload': 'Face recognition authorization',
      'POST /register': 'Register a new person',
      'GET /persons': 'List all registered persons',
      'DELETE /persons/:id': 'Delete a registered person',
    },
  });
});

// Ping endpoint
app.get('/ping', (req: Request, res: Response) => {
  console.log('Ping received from:', req.ip);
  res.json({
    status: 'success',
    message: 'pong',
    timestamp: new Date().toISOString(),
    clientIP: req.ip,
  });
});

// Face recognition authorization endpoint
app.post(
  '/upload',
  raw({ type: 'image/jpeg', limit: '10mb' }),
  async (req: Request, res: Response): Promise<void> => {
    console.log('=== Face Recognition Authorization ===');
    console.log('Time:', new Date().toISOString());

    try {
      const imageBuffer = req.body as Buffer;
      console.log(`✓ Received ${imageBuffer.length} bytes`);

      // Validate JPEG
      if (imageBuffer[0] !== 0xff || imageBuffer[1] !== 0xd8) {
        res.status(400).json({
          status: 'error',
          message: 'Invalid JPEG format',
          isAuthorized: false,
        });
        return;
      }

      const embedding = await extractFaceEmbeddingNode(imageBuffer);

      if (!embedding) {
        res.status(200).json({
          status: 'success',
          message: 'No face detected in image',
          isAuthorized: false,
        });
        return;
      }

      const matchedPerson = await findMatchingPerson(embedding);

      if (matchedPerson) {
        console.log(`✓ AUTHORIZED: ${matchedPerson.name}`);

        const personDoc = await db
          .collection('known_persons')
          .doc(matchedPerson.personId)
          .get();

        const userData = personDoc.data();

        res.status(200).json({
          status: 'success',
          message: 'Face recognized - Access granted',
          isAuthorized: true,
          user: {
            id: matchedPerson.personId,
            name: matchedPerson.name,
            confidence: parseFloat((matchedPerson.confidence * 100).toFixed(2)),
            registeredDate: userData?.registeredDate || null,
          },
        });
      } else {
        console.log('❌ UNAUTHORIZED: Unknown face');
        res.status(200).json({
          status: 'success',
          message: 'Face not recognized - Access denied',
          isAuthorized: false,
        });
      }
    } catch (error) {
      console.error('❌ Error:', error);
      res.status(500).json({
        status: 'error',
        message: 'Failed to process face recognition',
        isAuthorized: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// Register endpoint
app.post(
  '/register',
  raw({ type: 'image/jpeg', limit: '10mb' }),
  async (req: Request, res: Response): Promise<void> => {
    console.log('=== Register New Person ===');

    try {
      const imageBuffer = req.body as Buffer;
      const name = req.query.name as string;
      const email = (req.query.email as string) || null;
      const phone = (req.query.phone as string) || null;

      if (!name) {
        res.status(400).json({
          status: 'error',
          message: 'Name parameter is required (?name=John)',
        });
        return;
      }

      if (imageBuffer[0] !== 0xff || imageBuffer[1] !== 0xd8) {
        res.status(400).json({
          status: 'error',
          message: 'Invalid JPEG format',
        });
        return;
      }

      const embedding = await extractFaceEmbeddingNode(imageBuffer);

      if (!embedding) {
        res.status(400).json({
          status: 'error',
          message: 'No face detected. Please provide a clear photo.',
        });
        return;
      }

      const timestamp = Date.now();
      const date = new Date(timestamp);
      const sanitizedName = name.replace(/\s+/g, '_').toLowerCase();
      const filename = `registered_faces/${sanitizedName}_${timestamp}.jpg`;

      const file = bucket.file(filename);
      await file.save(imageBuffer, {
        metadata: {
          contentType: 'image/jpeg',
          metadata: {
            name: name,
            email: email || '',
            phone: phone || '',
            registeredAt: date.toISOString(),
          },
        },
      });

      await file.makePublic();
      const imageUrl = `https://storage.googleapis.com/${bucket.name}/${filename}`;

      const docRef = await db.collection('known_persons').add({
        name: name,
        email: email,
        phone: phone,
        embedding: Array.from(embedding),
        imageUrl: imageUrl,
        imagePath: filename,
        registeredAt: admin.firestore.FieldValue.serverTimestamp(),
        registeredDate: date.toISOString(),
      });

      console.log(`✓ Person registered: ${name}`);

      res.status(200).json({
        status: 'success',
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
      console.error('❌ Error:', error);
      res.status(500).json({
        status: 'error',
        message: 'Failed to register person',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
);

// Get persons
app.get('/persons', async (req: Request, res: Response): Promise<void> => {
  try {
    const snapshot = await db.collection('known_persons').get();
    const persons = snapshot.docs.map((doc) => ({
      id: doc.id,
      name: doc.data().name,
      email: doc.data().email,
      registeredDate: doc.data().registeredDate,
    }));

    res.json({
      status: 'success',
      count: persons.length,
      persons: persons,
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: 'Failed to fetch persons',
    });
  }
});

// Delete person
app.delete(
  '/persons/:id',
  async (req: Request, res: Response): Promise<void> => {
    try {
      await db.collection('known_persons').doc(req.params.id).delete();
      res.json({
        status: 'success',
        message: 'Person deleted successfully',
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: 'Failed to delete person',
      });
    }
  }
);

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    modelsLoaded: modelsLoaded,
    model: 'TensorFlow.js (BlazeFace + FaceMesh)',
  });
});

export const esp32cam = functions.https.onRequest(
  {
    timeoutSeconds: 300,
    memory: '1GiB',
  },
  app
);
