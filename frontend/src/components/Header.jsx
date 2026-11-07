import React from 'react';
import { ShieldCheck, RotateCcw } from 'lucide-react';

export default function Header({ onReset }) {
  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid #e5e7eb',
      }}
    >
      <div
        style={{
          maxWidth: '1280px',
          margin: '0 auto',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Brand Logo - Aniai Minimalist Tech Style */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              fontSize: '18px',
              fontWeight: 900,
              letterSpacing: '-0.5px',
              color: '#0f172a',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            SAFESTEP
            <span
              style={{
                fontSize: '10px',
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: '4px',
                background: '#0f172a',
                color: '#ffffff',
                letterSpacing: '0.5px',
              }}
            >
              AI
            </span>
          </div>
          <span style={{ color: '#d1d5db', fontSize: '13px' }}>|</span>
          <span style={{ fontSize: '12px', color: '#6b7280', fontWeight: 500 }}>
            Night Stair Vision Guard
          </span>
        </div>

        {/* Right Info & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '9999px',
              background: '#f0fdf4',
              border: '1px solid #bbf7d0',
              fontSize: '11px',
              fontWeight: 600,
              color: '#166534',
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: '#16a34a',
                display: 'inline-block',
              }}
            />
            Engine Ready
          </div>

          {onReset && (
            <button
              onClick={onReset}
              className="btn-outline"
              style={{ padding: '6px 12px', fontSize: '12px' }}
              title="처음 상태로 초기화"
            >
              <RotateCcw size={13} />
              초기화
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
