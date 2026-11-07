import React, { useRef } from 'react';
import { UploadCloud, X, Image as ImageIcon } from 'lucide-react';

export default function FileUploader({
  imagePreview,
  setImagePreview,
  onFileSelect,
  label = '계단 사진을 끌어다 놓거나 클릭하여 선택하세요',
  height = '190px',
}) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      previewAndCallback(file);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      previewAndCallback(file);
    }
  };

  const previewAndCallback = (file) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target.result;
      const img = new Image();
      img.onload = () => {
        const maxDim = 1600;
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          if (w > h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          const compressed = canvas.toDataURL('image/jpeg', 0.88);
          setImagePreview(compressed);
          if (onFileSelect) onFileSelect(file, compressed);
        } else {
          setImagePreview(dataUrl);
          if (onFileSelect) onFileSelect(file, dataUrl);
        }
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setImagePreview(null);
    if (onFileSelect) onFileSelect(null, null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      style={{
        width: '100%',
        minHeight: height,
        borderRadius: '12px',
        border: '1.5px dashed #cbd5e1',
        background: '#ffffff',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.15s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#0f172a';
        e.currentTarget.style.background = '#f8fafc';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#cbd5e1';
        e.currentTarget.style.background = '#ffffff';
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleChange}
      />

      {imagePreview ? (
        <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: height, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '12px' }}>
          <img
            src={imagePreview}
            alt="Preview"
            style={{
              maxHeight: height,
              maxWidth: '100%',
              objectFit: 'contain',
              borderRadius: '8px',
            }}
          />
          <button
            onClick={handleClear}
            style={{
              position: 'absolute',
              top: '12px',
              right: '12px',
              background: '#0f172a',
              border: 'none',
              borderRadius: '50%',
              width: '26px',
              height: '26px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              cursor: 'pointer',
              boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
            }}
            title="이미지 제거"
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '24px 16px' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              background: '#f1f5f9',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '10px',
              color: '#334155',
            }}
          >
            <UploadCloud size={20} />
          </div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: '#111827' }}>
            {label}
          </div>
          <div style={{ fontSize: '11px', color: '#6b7280', marginTop: '4px' }}>
            JPG, PNG 지원 · 업로드 시 자동 탐지 실행
          </div>
        </div>
      )}
    </div>
  );
}
