import { bucket, db } from "../config/firebase";
import { FieldValue } from "firebase-admin/firestore";
import { PersonData } from "../types";

/**
 * Upload image to Firebase Storage
 */
export async function uploadImage(
  imageBuffer: Buffer,
  name: string,
  email: string | null,
  phone: string | null
): Promise<{ imageUrl: string; filename: string }> {
  const timestamp = Date.now();
  const date = new Date(timestamp);
  const sanitizedName = name.replace(/\s+/g, "_").toLowerCase();
  const filename = `registered_faces/${sanitizedName}_${timestamp}.jpg`;

  const file = bucket.file(filename);
  await file.save(imageBuffer, {
    metadata: {
      contentType: "image/jpeg",
      metadata: {
        name: name,
        email: email || "",
        phone: phone || "",
        registeredAt: date.toISOString(),
      },
    },
  });

  await file.makePublic();
  const imageUrl = `https://storage.googleapis.com/${bucket.name}/${filename}`;

  return { imageUrl, filename };
}

/**
 * Save person data to Firestore
 */
export async function savePerson(
  name: string,
  email: string | null,
  phone: string | null,
  embedding: Float32Array,
  imageUrl: string,
  imagePath: string
): Promise<string> {
  const date = new Date();

  const personData: PersonData = {
    name,
    email,
    phone,
    embedding: Array.from(embedding),
    imageUrl,
    imagePath,
    registeredAt: FieldValue.serverTimestamp(),
    registeredDate: date.toISOString(),
  };

  const docRef = await db.collection("known_persons").add(personData);
  console.log(`✓ Person registered: ${name} (ID: ${docRef.id})`);

  return docRef.id;
}

/**
 * Get all persons from database
 */
export async function getAllPersons() {
  const snapshot = await db.collection("known_persons").get();
  return snapshot.docs.map((doc) => ({
    id: doc.id,
    name: doc.data().name,
    email: doc.data().email,
    phone: doc.data().phone,
    registeredDate: doc.data().registeredDate,
  }));
}

/**
 * Get person by ID
 */
export async function getPersonById(personId: string) {
  const doc = await db.collection("known_persons").doc(personId).get();
  return doc.exists ? doc.data() : null;
}

/**
 * Delete person from database
 */
export async function deletePerson(personId: string): Promise<void> {
  await db.collection("known_persons").doc(personId).delete();
  console.log(`✓ Person deleted: ${personId}`);
}
