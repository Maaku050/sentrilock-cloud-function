import * as functions from "firebase-functions";
import express from "express";
import cors from "cors";
import routes from "./routes";
import { errorHandler } from "./middleware/errorHandler";

const app = express();

// Middleware
app.use(cors({ origin: true }));
app.use(express.json());

// Routes
app.use("/", routes);

// Error handling middleware (must be last)
app.use(errorHandler);

// Export Cloud Function
export const esp32cam = functions.https.onRequest(
  {
    timeoutSeconds: 300,
    memory: "1GiB",
  },
  app
);
