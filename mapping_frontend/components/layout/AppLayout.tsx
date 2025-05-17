// components/layout/AppLayout.tsx
import React, { ReactNode } from 'react';
import AppHeader from './AppHeader';

interface AppLayoutProps {
  children: ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <AppHeader />
      <main className="flex-1">
        {children}
      </main>
      <footer className="py-3 px-4 text-center text-sm text-gray-500 border-t">
        Entity Resolution Review Tool &copy; {new Date().getFullYear()}
      </footer>
    </div>
  );
};

export default AppLayout;