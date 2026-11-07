import React, { useState, useEffect, useRef } from 'react';
import {
  UploadCloud,
  Camera,
  Video,
  Square,
  RefreshCw,
  Eye,
  ShieldAlert,
  Smartphone,
  ImageIcon,
} from 'lucide-react';
import FileUploader from '../components/FileUploader';
import { detectStairs } from '../utils/api';

export default function StairPage() {
  // Input Mode: 'upload' (default) or 'camera'
  const [inputMode, setInputMode] = useState('upload');

  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeView, setActiveView] = useState('annotated'); // 'annotated' | 'original' | 'depth'

  // Camera State
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Enumerate cameras
  useEffect(() => {
    const getCameras = async () => {
      try {
        if (navigator.mediaDevices?.enumerateDevices) {
          const all = await navigator.mediaDevices.enumerateDevices();
          const videoInputs = all.filter((d) => d.kind === 'videoinput');
          setDevices(videoInputs);
          if (videoInputs.length > 0 && !selectedDeviceId) {
            const backCam = videoInputs.find(
              (d) =>
                d.label.toLowerCase().includes('back') ||
                d.label.toLowerCase().includes('rear') ||
                d.label.toLowerCase().includes('droidcam') ||
                d.label.toLowerCase().includes('iriun') ||
                d.label.toLowerCase().includes('camo')
            );
            setSelectedDeviceId(backCam ? backCam.deviceId : videoInputs[0].deviceId);
          }
        }
      } catch (err) {
        console.warn('카메라 목록 조회 실패:', err);
      }
    };

    getCameras();

    return () => {
      stopCamera();
    };
  }, []);

    // Start Camera
    const startCamera = async (devId = selectedDeviceId) => {
      try {
        stopCamera();
        const constraints = {
          video: devId
            ? { deviceId: { exact: devId } }
            : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setIsCameraActive(true);
      } catch (err) {
        alert('카메라 시작 실패: ' + err.message + '\n카메라 접근 권한을 확인해주세요.');
      }
    };

    // Stop Camera
    const stopCamera = () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
    };

    // Switch Input Mode
    const handleModeChange = (newMode) => {
      setInputMode(newMode);
      if (newMode === 'camera') {
        startCamera();
      } else {
        stopCamera();
      }
    };

    // Capture Frame from Video
    const handleCaptureFrame = () => {
      if (!videoRef.current) return;
      const video = videoRef.current;
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        alert('카메라 영상이 아직 준비되지 않았습니다.');
        return;
      }
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.88);
      setImagePreview(dataUrl);
      runDetection(dataUrl);
    };

    // Run Backend Detection API
    const runDetection = async (base64Img) => {
      if (!base64Img) return;
      setLoading(true);
      try {
        const options = {
          gamma: 1.6,
          preferred_mode: 'descending',
          enhancement_mode: 'auto',
        };
        const res = await detectStairs(base64Img, options);
        setResult(res);
        setActiveView('annotated');
      } catch (err) {
        alert('계단 분석 중 오류가 발생했습니다: ' + (err.message || '서버 연결 실패'));
      } finally {
        setLoading(false);
      }
    };

    const metrics = result?.metrics;
    const hazardLevel = metrics?.hazard_level || 'SAFE';

    const hazardBadge = {
      DANGER: { bg: '#fef2f2', border: '#fecaca', text: '#dc2626', label: '위험 (DANGER)' },
      CAUTION: { bg: '#fffbeb', border: '#fde68a', text: '#d97706', label: '주의 (CAUTION)' },
      SAFE: { bg: '#f0fdf4', border: '#bbf7d0', text: '#16a34a', label: '안전 (SAFE)' },
    }[hazardLevel] || { bg: '#f0fdf4', border: '#bbf7d0', text: '#16a34a', label: '안전 (SAFE)' };

    return (
      <div>
          {/* Hero Header */}
          <section style={{ marginBottom: '32px', textAlign: 'left' }}>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '3px 10px',
                borderRadius: '9999px',
                background: '#f1f5f9',
                color: '#334155',
                fontSize: '11px',
                fontWeight: 700,
                letterSpacing: '0.5px',
                marginBottom: '12px',
              }}
            >
              VISION AI FOR PEDESTRIAN SAFETY
            </div>
            <h1
              style={{
                fontSize: '30px',
                fontWeight: 800,
                letterSpacing: '-0.8px',
                color: '#0f172a',
                lineHeight: 1.25,
                marginBottom: '10px',
              }}
            >
              야간 계단 탐지 & 보행 안전 가이드
            </h1>
              <p style={{ fontSize: '15px', color: '#64748b', lineHeight: 1.6, maxWidth: '720px' }}>
                사진을 업로드하거나 스마트폰/웹캠 실시간 영상을 연결하여
                계단 첫 단차(Drop-Off)와 디딤판 엣지, 보행 안전 위험도를 즉시 분석합니다.
              </p>

              {/* Input Mode Selector (Upload vs Live Camera) */}
              <div style={{ marginTop: '20px', display: 'inline-flex', background: '#f1f5f9', padding: '4px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                <button
                  onClick={() => handleModeChange('upload')}
                  style={{
                    padding: '8px 18px',
                    borderRadius: '7px',
                    border: 'none',
                    background: inputMode === 'upload' ? '#ffffff' : 'transparent',
                    color: inputMode === 'upload' ? '#0f172a' : '#64748b',
                    fontWeight: 700,
                    fontSize: '13px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    boxShadow: inputMode === 'upload' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <UploadCloud size={16} /> 사진 업로드
                </button>

                <button
                  onClick={() => handleModeChange('camera')}
                  style={{
                    padding: '8px 18px',
                    borderRadius: '7px',
                    border: 'none',
                    background: inputMode === 'camera' ? '#ffffff' : 'transparent',
                    color: inputMode === 'camera' ? '#0f172a' : '#64748b',
                    fontWeight: 700,
                    fontSize: '13px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    boxShadow: inputMode === 'camera' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <Camera size={16} /> 실시간 영상 (카메라)
                </button>
              </div>
          </section>

            {/* Main Studio Grid (Left Controls : Right Viewport) */}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 4.2fr) minmax(420px, 7.8fr)', gap: '28px' }}>

              {/* Left Column */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

                      {/* Mode 1: Photo Upload Box */}
                      {inputMode === 'upload' && (
                        <div className="aniai-card" style={{ padding: '24px' }}>
                          <div style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a', marginBottom: '14px' }}>
                            계단 사진 등록
                          </div>

                          <FileUploader
                            imagePreview={imagePreview}
                            setImagePreview={setImagePreview}
                            label="분석할 계단 사진 끌어다 놓기 또는 클릭"
                            height="220px"
                            onFileSelect={(file, b64) => {
                              if (b64) runDetection(b64);
                            }}
                          />

                          {imagePreview && (
                            <button
                              className="btn-dark"
                              onClick={() => runDetection(imagePreview)}
                              disabled={loading}
                              style={{ width: '100%', marginTop: '16px', padding: '12px' }}
                            >
                              <RefreshCw size={14} className={loading ? 'spin' : ''} />
                              {loading ? 'AI 분석 중...' : '계단 탐지 실행'}
                            </button>
                          )}
                        </div>
                      )}

                      {/* Mode 2: Real-time Camera Controls */}
                      {inputMode === 'camera' && (
                        <div className="aniai-card" style={{ padding: '24px' }}>
                          <div style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a', marginBottom: '14px' }}>
                            카메라 장치 제어
                          </div>

              {/* Camera Device Select */}
                      <div style={{ marginBottom: '16px' }}>
                        <label style={{ fontSize: '12px', color: '#64748b', display: 'block', marginBottom: '6px' }}>
                          연결된 카메라 장치:
                        </label>
                        <select
                          value={selectedDeviceId}
                          onChange={(e) => {
                            setSelectedDeviceId(e.target.value);
                            if (isCameraActive) {
                              startCamera(e.target.value);
                            }
                          }}
                          style={{
                            width: '100%',
                            background: '#ffffff',
                            border: '1px solid #cbd5e1',
                            color: '#0f172a',
                            padding: '8px 12px',
                            borderRadius: '8px',
                            fontSize: '13px',
                            outline: 'none',
                            cursor: 'pointer',
                          }}
                        >
                          <option value="">기본 카메라 (자동 감지)</option>
                          {devices.map((d, idx) => (
                            <option key={d.deviceId || idx} value={d.deviceId}>
                              {d.label || `카메라 장치 ${idx + 1}`}
                            </option>
                          ))}
                        </select>
                      </div>

                          {/* Camera Start / Stop / Capture Actions */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            {!isCameraActive ? (
                              <button
                                className="btn-dark"
                                onClick={() => startCamera()}
                                style={{ width: '100%', padding: '12px' }}
                              >
                                <Camera size={15} /> 카메라 스트림 켜기
                              </button>
                            ) : (
                              <>
                                <button
                                  className="btn-dark"
                                  onClick={handleCaptureFrame}
                                  disabled={loading}
                                  style={{ width: '100%', padding: '12px' }}
                                >
                                  <Camera size={15} /> 📸 현재 화면 정밀 분석
                                </button>
                                <button
                                  className="btn-outline"
                                  onClick={stopCamera}
                                  style={{ width: '100%', padding: '10px', color: '#dc2626', borderColor: '#fecaca' }}
                                >
                                  <Square size={13} /> 카메라 정지
                                </button>
                              </>
                            )}
                          </div>
                        </div>
          )}

                      {/* Safety Assessment Guidance Card (Shown when result exists) */}
                      {metrics && (
                        <div className="aniai-card" style={{ padding: '20px 24px', background: hazardBadge.bg, borderColor: hazardBadge.border }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                            <ShieldAlert size={18} color={hazardBadge.text} />
                            <span style={{ fontSize: '14px', fontWeight: 800, color: hazardBadge.text }}>
                              {hazardBadge.label}
                            </span>
                          </div>
                          <p style={{ fontSize: '13px', color: '#1e293b', lineHeight: 1.5, fontWeight: 500 }}>
                            {metrics.guidance_message}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Right Column: Viewport & Metrics */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

                      {/* Main Visualizer Card */}
                      <div className="aniai-card" style={{ padding: '24px' }}>
                        {/* Viewport Header */}
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            paddingBottom: '16px',
                            borderBottom: '1px solid #f1f5f9',
                            marginBottom: '16px',
                          }}
                        >
                          <div style={{ fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>
                            {inputMode === 'camera' && isCameraActive && !result ? '실시간 카메라 화면' : '탐지 결과 뷰포트'}
                          </div>

                          {/* View Switcher (When photo or result available) */}
                          {result && (
                            <div style={{ display: 'flex', gap: '6px' }}>
                              <button
                                onClick={() => setActiveView('annotated')}
                                style={{
                                  padding: '6px 12px',
                                  borderRadius: '6px',
                                  border: '1px solid',
                                  borderColor: activeView === 'annotated' ? '#0f172a' : '#e2e8f0',
                                  background: activeView === 'annotated' ? '#0f172a' : '#ffffff',
                                  color: activeView === 'annotated' ? '#ffffff' : '#64748b',
                                  fontSize: '12px',
                                  fontWeight: 600,
                                  cursor: 'pointer',
                                }}
                              >
                                🎯 분석 결과 (HUD)
                              </button>
                              <button
                                onClick={() => setActiveView('original')}
                                style={{
                                  padding: '6px 12px',
                                  borderRadius: '6px',
                                  border: '1px solid',
                                  borderColor: activeView === 'original' ? '#0f172a' : '#e2e8f0',
                                  background: activeView === 'original' ? '#0f172a' : '#ffffff',
                                  color: activeView === 'original' ? '#ffffff' : '#64748b',
                                  fontSize: '12px',
                                  fontWeight: 600,
                                  cursor: 'pointer',
                                }}
                              >
                                📷 원본 사진
                              </button>
                              <button
                                onClick={() => setActiveView('depth')}
                                style={{
                                  padding: '6px 12px',
                                  borderRadius: '6px',
                                  border: '1px solid',
                                  borderColor: activeView === 'depth' ? '#0f172a' : '#e2e8f0',
                                  background: activeView === 'depth' ? '#0f172a' : '#ffffff',
                                  color: activeView === 'depth' ? '#ffffff' : '#64748b',
                                  fontSize: '12px',
                                  fontWeight: 600,
                                  cursor: 'pointer',
                                }}
                              >
                                🌈 깊이 맵
                              </button>
                            </div>
                          )}
                        </div>

                        {/* Viewport Frame */}
                        <div
                          style={{
                            width: '100%',
                            minHeight: '440px',
                            background: '#090d16',
                            borderRadius: '10px',
                            overflow: 'hidden',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            position: 'relative',
                          }}
                        >
                          {/* 1. Live Camera Stream View */}
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            style={{
                              display: inputMode === 'camera' && isCameraActive && !result ? 'block' : 'none',
                              width: '100%',
                              maxHeight: '520px',
                              objectFit: 'contain',
                            }}
                          />

                          {loading && (
                            <div style={{ color: '#94a3b8', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <RefreshCw size={16} className="spin" /> AI 계단 분석 중...
                            </div>
                          )}

                          {/* 2. Photo / Result Views */}
                          {!loading && (inputMode === 'upload' || result) && (
                            <>
                              {activeView === 'annotated' && result?.annotated_image && (
                                <img
                                  src={result.annotated_image}
                                  alt="Detected Stair HUD"
                                  style={{ width: '100%', maxHeight: '520px', objectFit: 'contain', display: 'block' }}
                                />
                              )}

                              {activeView === 'original' && result?.original_image && (
                                <img
                                  src={result.original_image}
                                  alt="Original Stair"
                                  style={{ width: '100%', maxHeight: '520px', objectFit: 'contain', display: 'block' }}
                                />
                              )}

                  {activeView === 'depth' && result?.depth_image && (
                                <img
                                  src={result.depth_image}
                                  alt="Depth Map"
                                  style={{ width: '100%', maxHeight: '520px', objectFit: 'contain', display: 'block' }}
                                />
                              )}

                              {!result && (
                                <div style={{ color: '#64748b', fontSize: '13px', textAlign: 'center', padding: '40px' }}>
                                  {inputMode === 'camera'
                                    ? '카메라를 켜고 [현재 화면 정밀 분석]을 누르면 계단 분석이 실행됩니다.'
                                    : '왼쪽에서 계단 사진을 등록하시면 계단 엣지와 안전 분석 결과가 표시됩니다.'}
                                </div>
                              )}
                            </>
                          )}
                        </div>

                        {/* Key Metrics 4-Box Grid */}
                        {metrics && (
                          <div
                            style={{
                              display: 'grid',
                              gridTemplateColumns: 'repeat(4, 1fr)',
                              gap: '12px',
                              marginTop: '20px',
                            }}
                          >
                            <div style={{ background: '#f8fafc', padding: '14px 16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                              <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>위험도 판정</div>
                              <div style={{ fontSize: '16px', fontWeight: 800, color: hazardBadge.text }}>
                                {hazardLevel}
                              </div>
                            </div>

                            <div style={{ background: '#f8fafc', padding: '14px 16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                              <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>첫 단차 거리</div>
                              <div style={{ fontSize: '16px', fontWeight: 800, color: '#0f172a' }}>
                                {metrics.distance_to_first_step_m} <span style={{ fontSize: '12px', fontWeight: 500, color: '#64748b' }}>m</span>
                              </div>
                            </div>

                            <div style={{ background: '#f8fafc', padding: '14px 16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                              <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>감지된 계단 수</div>
                              <div style={{ fontSize: '16px', fontWeight: 800, color: '#0f172a' }}>
                                {metrics.step_count} <span style={{ fontSize: '12px', fontWeight: 500, color: '#64748b' }}>단</span>
                              </div>
                            </div>

                            <div style={{ background: '#f8fafc', padding: '14px 16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
                              <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>분석 신뢰도</div>
                              <div style={{ fontSize: '16px', fontWeight: 800, color: '#0f172a' }}>
                                {metrics.confidence_pct} <span style={{ fontSize: '12px', fontWeight: 500, color: '#64748b' }}>%</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                );
}
