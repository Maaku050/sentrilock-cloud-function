export interface MatchResult {
  personId: string;
  name: string;
  distance: number;
  confidence: number;
}

export interface PersonDocument {
  name: string;
  email?: string | null;
  phone?: string | null;
  embedding: number[];
  imageUrl?: string;
  imagePath?: string;
  registeredAt?: FirebaseFirestore.Timestamp;
  registeredDate?: string;
}

export interface Landmark {
  x: number;
  y: number;
}

export interface PersonData {
  name: string;
  email: string | null;
  phone: string | null;
  embedding: number[];
  imageUrl: string;
  imagePath: string;
  registeredAt: FirebaseFirestore.FieldValue;
  registeredDate: string;
}

export interface RegisterRequest {
  name: string;
  email?: string;
  phone?: string;
}
