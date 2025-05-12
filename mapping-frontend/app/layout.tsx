// app/layout.tsx
import './globals.css';
import { ClusterProvider } from '../context/ClusterContext';

export const metadata = {
  title: 'HITL Entity Resolution Review',
  description: 'Human-in-the-Loop interface for entity resolution review',
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