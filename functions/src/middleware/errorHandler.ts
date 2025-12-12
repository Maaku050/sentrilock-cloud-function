import { Request, Response, NextFunction } from "express";

export function errorHandler(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  console.error("❌ Error:", err);
  console.error("Stack:", err.stack);

  res.status(500).json({
    status: "error",
    message: "Internal server error",
    error: err.message,
  });
}
