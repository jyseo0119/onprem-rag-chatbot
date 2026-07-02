import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "On-Prem RAG Chatbot",
  description: "Ask questions about your internal documents — answers cite their sources.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
