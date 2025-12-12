import { Request, Response, NextFunction } from "express";

export function validateJpegImage(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const imageBuffer = req.body as Buffer;

  if (!imageBuffer || imageBuffer.length === 0) {
    res.status(400).json({
      status: "error",
      message: "No image data provided",
    });
    return;
  }

  // Validate JPEG signature (0xFFD8)
  if (imageBuffer[0] !== 0xff || imageBuffer[1] !== 0xd8) {
    res.status(400).json({
      status: "error",
      message: "Invalid JPEG format",
    });
    return;
  }

  console.log(`✓ Received ${imageBuffer.length} bytes`);
  next();
}
