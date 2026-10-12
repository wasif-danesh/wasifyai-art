import Link from "next/link";
import type { ReactNode } from "react";
import "./tailwind.css";

export const metadata = { title: "Art Hub", description: "Model hub and art generator" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="flex items-center justify-between border-b border-[var(--border)] px-6 py-3">
          <span className="font-medium">Art Hub</span>
          <nav className="flex gap-4 text-sm text-[var(--muted)]">
            <Link href="/">Home</Link>
            <Link href="/create">Create</Link>
            <Link href="/jobs">Jobs</Link>
            <Link href="/models">Models</Link>
            <span>Training</span>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
