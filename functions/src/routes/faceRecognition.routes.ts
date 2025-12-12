import { Router, raw } from "express";
import { recognizeFace } from "../controllers/faceRecognition.controller";
import { validateJpegImage } from "../middleware/validateImage";

const router = Router();

router.post(
  "/upload",
  raw({ type: "image/jpeg", limit: "10mb" }),
  validateJpegImage,
  recognizeFace
);

export default router;
