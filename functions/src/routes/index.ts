import { Router } from "express";
import healthRoutes from "./health.routes";
import faceRecognitionRoutes from "./faceRecognition.routes";
import personRoutes from "./person.routes";

const router = Router();

// Mount all routes
router.use("/", healthRoutes);
router.use("/", faceRecognitionRoutes);
router.use("/", personRoutes);

export default router;
