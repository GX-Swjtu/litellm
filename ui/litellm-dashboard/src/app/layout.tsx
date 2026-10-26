import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

import { NuqsAdapter } from "nuqs/adapters/next/app";
import { ThemeProvider } from "next-themes";

import { AuthProvider } from "@/contexts/AuthContext";
import ReactQueryProvider from "@/contexts/ReactQueryProvider";
import { Toaster } from "@/components/ui/sonner";

const inter = localFont({
  src: "../../public/fonts/inter/InterVariable.woff2",
  weight: "100 900",
  display: "swap",
});

export const metadata: Metadata = {
  title: "LiteLLM Dashboard",
  description: "LiteLLM Proxy Admin UI",
  icons: { icon: "/get_favicon" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // next-themes stamps the theme class on <html> before paint, which the exported markup
    // cannot predict; suppressHydrationWarning confines that mismatch to this element.
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
          <NuqsAdapter>
            <ReactQueryProvider>
              <AuthProvider>{children}</AuthProvider>
              <Toaster />
            </ReactQueryProvider>
          </NuqsAdapter>
        </ThemeProvider>
      </body>
    </html>
  );
}
