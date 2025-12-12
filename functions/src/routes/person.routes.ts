import { raw, Router } from "express";
import {
  registerPerson,
  listPersons,
  removePersonById,
} from "../controllers/person.controller";
import { validateJpegImage } from "../middleware/validateImage";

const router = Router();

router.post(
  "/register",
  raw({ type: "image/jpeg", limit: "10mb" }),
  validateJpegImage,
  registerPerson
);

router.get("/persons", listPersons);

router.delete("/persons/:id", removePersonById);

export default router;
