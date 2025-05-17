// app/layout.tsx
import React from 'react';
import type { Metadata } from 'next';
import { ClusterProvider } from '@/context/ClusterContext';
import './globals.css';

export const metadata: Metadata = {
  title: 'Entity Resolution Review Tool',
  description: 'A tool for reviewing and validating entity resolution results',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ClusterProvider>
          {children}
        </ClusterProvider>
      </body>
    </html>
  );
}