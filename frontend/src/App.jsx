import React, { useState } from 'react';
import Header from './components/Header';
import StairPage from './pages/StairPage';

export default function App() {
  const [resetKey, setResetKey] = useState(0);

  const handleReset = () => {
    setResetKey((prev) => prev + 1);
  };

  return (
    <div className="app-container">
      {/* Aniai Minimalist Top Header */}
      <Header onReset={handleReset} />

      {/* Main Single-Focus Content */}
      <main className="main-content">
        <StairPage key={resetKey} />
      </main>
    </div>
  );
}
