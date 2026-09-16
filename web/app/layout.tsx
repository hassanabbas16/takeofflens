import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TakeoffLens",
  description: "Blueprint analysis: OCR and LLM takeoff extraction from floor plans.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-50 text-neutral-900 antialiased">
        {children}
      </body>
    </html>
  );
}
