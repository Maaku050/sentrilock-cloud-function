import { Request, Response } from "express";
import { isModelsLoaded } from "../services/faceDetection.service";

/**
 * Root endpoint
 */
export function getApiInfo(req: Request, res: Response): void {
  res.json({
    status: "online",
    message: "ESP32-CAM Facial Recognition API (TensorFlow.js)",
    timestamp: new Date().toISOString(),
    endpoints: {
      "GET /": "API information",
      "GET /ping": "Simple ping test",
      "GET /health": "Health check",
      "POST /upload": "Face recognition authorization",
      "POST /register": "Register a new person",
      "GET /persons": "List all registered persons",
      "DELETE /persons/:id": "Delete a registered person",
    },
  });
}

/**
 * Ping endpoint
 */
export function ping(req: Request, res: Response): void {
  console.log("Ping received from:", req.ip);
  res.json({
    status: "success",
    message: "pong",
    timestamp: new Date().toISOString(),
    clientIP: req.ip,
  });
}

/**
 * Health check endpoint
 */
export function healthCheck(req: Request, res: Response): void {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    modelsLoaded: isModelsLoaded(),
    model: "TensorFlow.js (BlazeFace)",
  });
}
