import * as admin from "firebase-admin";
import { getFirestore } from "firebase-admin/firestore";

// Initialize Firebase Admin
admin.initializeApp();

export const db = getFirestore();
export const bucket = admin.storage().bucket();

export default admin;
