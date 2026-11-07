import React from 'react';
import {
  Sparkles,
  Layers,
  Footprints,
  Cpu,
  Eye,
  CheckCircle2,
  ShieldAlert,
  Sliders,
  Maximize2,
  Activity,
  Zap,
} from 'lucide-react';

export default function TechSpecsPage({ setActiveTab }) {
  return (
    <div style={{ padding: '36px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
      {/* Hero Banner */}
      <div
        className="glass-card"
        style={{
          background: 'linear-gradient(135deg, #ffffff 0%, #f0fdf4 50%, #eff6ff 100%)',
          border: '1px solid #cbd5e1',
          padding: '36px 40px',
          marginBottom: '32px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div style={{ position: 'relative', zIndex: 2, maxWidth: '880px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 12px',
              borderRadius: '20px',
              background: '#e0f2fe',
              border: '1px solid #bae6fd',
              color: '#0284c7',
              fontSize: '12px',
              fontWeight: 700,
              marginBottom: '16px',
            }}
          >
            <Zap size={14} /> NIGHT VISION CV PIPELINE
          </div>
          <h2 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.5px', marginBottom: '12px' }}>
            야간 저조도 계단 탐지 및 단안 깊이 추정 아키텍처
          </h2>
          <p style={{ fontSize: '15px', color: '#475569', lineHeight: 1.7 }}>
            야간 조도 부족, 가로등 그림자, 반사광 등 열악한 환경에서도 보행자의 낙상을 방지하기 위해
            <b> 저조도 암부 복원(Retinexformer/CLAHE)</b>과 <b>YOLO 인스턴스 세그멘테이션</b>,
            그리고 <b>단안 깊이 추정(Depth Anything V2)</b>을 융합한 실시간 하이브리드 비전 파이프라인입니다.
          </p>

          <div style={{ marginTop: '20px' }}>
            <button
              className="btn-primary"
              onClick={() => setActiveTab('stair')}
              style={{ fontSize: '13px', padding: '10px 20px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}
            >
              <Eye size={16} /> 실시간 야간 계단 탐지 스튜디오 시작하기
            </button>
          </div>
        </div>
      </div>

      {/* 4-Stage Core Pipeline Cards */}
      <h3 style={{ fontSize: '19px', fontWeight: 700, color: '#0f172a', marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Layers size={20} color="#0284c7" /> 4단계 종단간(End-to-End) 안전 추론 파이프라인
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        {/* Step 1 */}
        <div className="glass-card" style={{ borderColor: '#e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '11px', fontWeight: 800, color: '#7c3aed', background: '#f5f3ff', border: '1px solid #ddd6fe', padding: '3px 8px', borderRadius: '6px' }}>
              STAGE 01
            </span>
            <Sparkles size={18} color="#7c3aed" />
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            저조도 암부 지능 복원
          </h4>
          <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6, marginBottom: '12px' }}>
            야간의 0.1~5 Lux 극저조도 환경에서 계단 모서리의 수평 에지가 손실되는 것을 방지합니다.
          </p>
          <div style={{ fontSize: '12px', color: '#334155', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '10px 12px', borderRadius: '8px', lineHeight: 1.6 }}>
            • <b>Retinexformer AI</b>: 원-스테이지 트랜스포머 기반 조도/반사율 맵 분해<br />
            • <b>Adaptive CLAHE</b>: 타일 기반 명암 국소 정규화 (15ms 초고속 처리)
          </div>
        </div>

        {/* Step 2 */}
        <div className="glass-card" style={{ borderColor: '#e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '11px', fontWeight: 800, color: '#0284c7', background: '#f0f9ff', border: '1px solid #bae6fd', padding: '3px 8px', borderRadius: '6px' }}>
              STAGE 02
            </span>
            <Cpu size={18} color="#0284c7" />
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            YOLO 다중 계단 세그멘테이션
          </h4>
          <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6, marginBottom: '12px' }}>
            각 계단 디딤판(Tread)의 다각형 마스크를 분할 검출하고 경계 좌표를 정밀 추출합니다.
          </p>
          <div style={{ fontSize: '12px', color: '#334155', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '10px 12px', borderRadius: '8px', lineHeight: 1.6 }}>
            • 야간 증강 데이터셋(Night-Augmented) 기반 특화 학습 가중치 탑재<br />
            • 각 디딤판의 경계선(Boundary Contour)과 중심점 고속 추적
          </div>
        </div>

        {/* Step 3 */}
        <div className="glass-card" style={{ borderColor: '#e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '11px', fontWeight: 800, color: '#e11d48', background: '#fff1f2', border: '1px solid #fecdd3', padding: '3px 8px', borderRadius: '6px' }}>
              STAGE 03
            </span>
            <Footprints size={18} color="#e11d48" />
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            1st Step Quad 기하학 추출
          </h4>
          <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6, marginBottom: '12px' }}>
            보행자가 가장 먼저 발을 디뎌야 하는 '첫 번째 단차(Drop-Off)'를 4점 다각형으로 특정합니다.
          </p>
          <div style={{ fontSize: '12px', color: '#334155', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '10px 12px', borderRadius: '8px', lineHeight: 1.6 }}>
            • 하향 1인칭 시점에서 위험도가 가장 높은 바닥 근접 계단 자동 선별<br />
            • 좌상/우상/우하/좌하 4개 정밀 핀 포인트 계산 및 AR 오버레이
          </div>
        </div>

        {/* Step 4 */}
        <div className="glass-card" style={{ borderColor: '#e2e8f0' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ fontSize: '11px', fontWeight: 800, color: '#059669', background: '#ecfdf5', border: '1px solid #a7f3d0', padding: '3px 8px', borderRadius: '6px' }}>
              STAGE 04
            </span>
            <ShieldAlert size={18} color="#059669" />
          </div>
          <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            단안 깊이 기반 실시간 경보
          </h4>
          <p style={{ fontSize: '13px', color: '#64748b', lineHeight: 1.6, marginBottom: '12px' }}>
            단일 카메라로 단차까지의 물리적 거리(m)와 수직 낙차(cm)를 추정하여 위험도를 판정합니다.
          </p>
          <div style={{ fontSize: '12px', color: '#334155', background: '#f8fafc', border: '1px solid #e2e8f0', padding: '10px 12px', borderRadius: '8px', lineHeight: 1.6 }}>
            • <b>Depth Anything V2</b>: 단일 RGB 프레임에서 절대 깊이 맵 복원<br />
            • 전방 0.9m 이내 진입 시 DANGER 알림 및 초록 발 디딤 타겟 투영
          </div>
        </div>
      </div>

      {/* Model Benchmark Specs Table */}
      <div className="glass-card" style={{ padding: '28px' }}>
        <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a', marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} color="var(--cyan-accent)" /> 핵심 AI 모델 벤치마크 및 구동 사양
        </h4>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc', color: '#475569' }}>
                <th style={{ padding: '12px 14px' }}>모듈명</th>
                <th style={{ padding: '12px 14px' }}>탑재 신경망 아키텍처</th>
                <th style={{ padding: '12px 14px' }}>평균 추론 속도</th>
                <th style={{ padding: '12px 14px' }}>주요 역할</th>
                <th style={{ padding: '12px 14px' }}>상태</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>저조도 복원 엔진</td>
                <td style={{ padding: '12px 14px', color: '#7c3aed', fontFamily: 'var(--font-mono)' }}>Retinexformer / CLAHE</td>
                <td style={{ padding: '12px 14px', color: '#0284c7', fontWeight: 600 }}>15 ~ 45 ms</td>
                <td style={{ padding: '12px 14px', color: '#475569' }}>야간 암부 밝기 및 대비 동적 복원</td>
                <td style={{ padding: '12px 14px' }}>
                  <span className="badge badge-success" style={{ fontSize: '11px' }}>Active</span>
                </td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>계단 세그멘테이션</td>
                <td style={{ padding: '12px 14px', color: '#0284c7', fontFamily: 'var(--font-mono)' }}>YOLO26-Seg (Fine-tuned)</td>
                <td style={{ padding: '12px 14px', color: '#0284c7', fontWeight: 600 }}>22 ~ 35 ms</td>
                <td style={{ padding: '12px 14px', color: '#475569' }}>다중 디딤판 폴리곤 및 챌판 분할</td>
                <td style={{ padding: '12px 14px' }}>
                  <span className="badge badge-success" style={{ fontSize: '11px' }}>Active</span>
                </td>
              </tr>
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>단안 깊이 추정</td>
                <td style={{ padding: '12px 14px', color: '#e11d48', fontFamily: 'var(--font-mono)' }}>Depth Anything V2 Small</td>
                <td style={{ padding: '12px 14px', color: '#0284c7', fontWeight: 600 }}>30 ~ 65 ms</td>
                <td style={{ padding: '12px 14px', color: '#475569' }}>카메라-단차 간 메트릭 상대 거리 맵 도출</td>
                <td style={{ padding: '12px 14px' }}>
                  <span className="badge badge-success" style={{ fontSize: '11px' }}>Active</span>
                </td>
              </tr>
              <tr>
                <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>실시간 AR HUD 렌더러</td>
                <td style={{ padding: '12px 14px', color: '#059669', fontFamily: 'var(--font-mono)' }}>HTML5 Canvas 2D Matrix</td>
                <td style={{ padding: '12px 14px', color: '#059669', fontWeight: 600 }}>60 FPS (Zero Latency)</td>
                <td style={{ padding: '12px 14px', color: '#475569' }}>스마트폰 비디오 스트림 위에 투시 오버레이</td>
                <td style={{ padding: '12px 14px' }}>
                  <span className="badge badge-success" style={{ fontSize: '11px' }}>Active</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
