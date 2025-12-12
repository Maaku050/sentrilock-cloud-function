import { Router } from "express";
import {
  getApiInfo,
  ping,
  healthCheck,
} from "../controllers/health.controller";

const router = Router();

router.get("/", getApiInfo);
router.get("/ping", ping);
router.get("/health", healthCheck);

export default router;
