import React from 'react';
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Footprints,
  ArrowDownRight,
  Eye,
  Sliders,
  HelpCircle,
  Smartphone,
  Compass,
} from 'lucide-react';

export default function SafetyGuidePage({ setActiveTab }) {
  return (
    <div style={{ padding: '36px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
      {/* Hero Banner */}
      <div
        className="glass-card"
        style={{
          background: 'linear-gradient(135deg, #ffffff 0%, #fff1f2 50%, #fff7ed 100%)',
          border: '1px solid #fecdd3',
          padding: '36px 40px',
          marginBottom: '32px',
        }}
      >
        <div style={{ maxWidth: '880px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 12px',
              borderRadius: '20px',
              background: '#ffe4e6',
              border: '1px solid #fecdd3',
              color: '#e11d48',
              fontSize: '12px',
              fontWeight: 700,
              marginBottom: '16px',
            }}
          >
            <ShieldAlert size={14} /> PEDESTRIAN SAFETY CRITERIA
          </div>
          <h2 style={{ fontSize: '28px', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.5px', marginBottom: '12px' }}>
            야간 계단 보행 안전 기준 및 낙상 방지 가이드
          </h2>
          <p style={{ fontSize: '15px', color: '#475569', lineHeight: 1.7 }}>
            국토교통부 표준 건축 기준(단차 높이 16~18cm, 디딤판 깊이 26~28cm)을 기준으로,
            어두운 환경에서 인지하기 어려운 '첫 번째 낭떠러지(1st Drop-Off)'를 컴퓨터 비전으로 탐지하여
            보행자에게 최적의 착지 궤적(Foot Target)과 조기 경보를 제공합니다.
          </p>

          <div style={{ marginTop: '20px' }}>
            <button
              className="btn-primary"
              onClick={() => setActiveTab('stair')}
              style={{
                fontSize: '13px',
                padding: '10px 20px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                background: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)',
                boxShadow: '0 4px 14px rgba(225, 29, 72, 0.35)',
              }}
            >
              <Eye size={16} /> 계단 탐지 스튜디오에서 직접 테스트
            </button>
          </div>
        </div>
      </div>

      {/* Safety Matrix */}
      <h3 style={{ fontSize: '19px', fontWeight: 700, color: '#0f172a', marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Compass size={20} color="#e11d48" /> 건축 표준 대비 AI 위험도 3단계 판정 기준
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        {/* Safe Level */}
        <div className="glass-card" style={{ borderColor: '#bbf7d0', background: '#f0fdf4' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <span style={{ fontSize: '12px', fontWeight: 800, color: '#166534', background: '#dcfce7', border: '1px solid #bbf7d0', padding: '4px 10px', borderRadius: '6px' }}>
              LEVEL 1: 안전 (SAFE)
            </span>
            <CheckCircle2 size={20} color="#16a34a" />
          </div>
          <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            전방 1.5m 이상 원거리 또는 완만한 경사
          </h4>
          <p style={{ fontSize: '13px', color: '#334155', lineHeight: 1.6, marginBottom: '14px' }}>
            보행자의 발밑에 급격한 단차가 없으며, 계단 디딤판 폭(28cm 이상)이 충분히 확보되어 실족 위험이 매우 낮은 상태입니다.
          </p>
          <div style={{ fontSize: '12px', color: '#166534', borderTop: '1px solid #bbf7d0', paddingTop: '10px', fontWeight: 500 }}>
            <b>HUD 피드백:</b> 녹색 테두리 오버레이, "정상 보행 가능"
          </div>
        </div>

        {/* Caution Level */}
        <div className="glass-card" style={{ borderColor: '#fde68a', background: '#fffbeb' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <span style={{ fontSize: '12px', fontWeight: 800, color: '#92400e', background: '#fef3c7', border: '1px solid #fde68a', padding: '4px 10px', borderRadius: '6px' }}>
              LEVEL 2: 주의 (CAUTION)
            </span>
            <AlertTriangle size={20} color="#d97706" />
          </div>
          <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            전방 1.0 ~ 1.5m 단차 접근 구간
          </h4>
          <p style={{ fontSize: '13px', color: '#334155', lineHeight: 1.6, marginBottom: '14px' }}>
            곧 계단이 시작되는 구간으로, 어두운 조명이나 스마트폰 주시로 인해 발을 헛디딜 가능성이 있는 전환 지점입니다.
          </p>
          <div style={{ fontSize: '12px', color: '#92400e', borderTop: '1px solid #fde68a', paddingTop: '10px', fontWeight: 500 }}>
            <b>HUD 피드백:</b> 황색 하이라이트 펄스, "계단 시작 주의 / 발끝 확인"
          </div>
        </div>

        {/* Danger Level */}
        <div className="glass-card" style={{ borderColor: '#fecaca', background: '#fef2f2' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <span style={{ fontSize: '12px', fontWeight: 800, color: '#991b1b', background: '#fee2e2', border: '1px solid #fecaca', padding: '4px 10px', borderRadius: '6px' }}>
              LEVEL 3: 위험 (DANGER)
            </span>
            <ShieldAlert size={20} color="#dc2626" />
          </div>
          <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
            전방 0.9m 이내 즉각 낙하(Drop-Off) 구간
          </h4>
          <p style={{ fontSize: '13px', color: '#334155', lineHeight: 1.6, marginBottom: '14px' }}>
            다음 한 걸음 내에 계단 내리막 단차(수직 16~20cm)가 위치하여 즉각적인 발 디딤 안전 확보가 필요한 초긴급 상태입니다.
          </p>
          <div style={{ fontSize: '12px', color: '#991b1b', borderTop: '1px solid #fecaca', paddingTop: '10px', fontWeight: 500 }}>
            <b>HUD 피드백:</b> 적색 스트로브 경보, "🚨 첫 번째 디딤판 착지 유도선 투영"
          </div>
        </div>
      </div>

      {/* Practical Usage Tips for Smartphones */}
      <div className="glass-card" style={{ padding: '28px' }}>
        <h4 style={{ fontSize: '17px', fontWeight: 700, color: '#0f172a', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Smartphone size={20} color="var(--cyan-accent)" /> 스마트폰을 활용한 실전 보행 가이드 활용 팁
        </h4>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>
              1. 카메라 각도 (하향 45도)
            </div>
            <p style={{ fontSize: '12px', color: '#475569', lineHeight: 1.6 }}>
              스마트폰을 정면이 아닌 전방 하향 45도 각도로 기울여 본인의 발끝 1~2걸음 앞 바닥과 계단 첫 모서리가 앵글 중앙에 오도록 유지해주세요.
            </p>
          </div>

          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>
              2. 저조도 복원 모드 선택
            </div>
            <p style={{ fontSize: '12px', color: '#475569', lineHeight: 1.6 }}>
              가로등이 거의 없는 극암 환경에서는 <b>Retinexformer AI</b> 모드를, 실시간 빠른 이동 중에는 <b>CLAHE 고속 모드</b>를 사용하시면 최적의 반응 속도를 체감할 수 있습니다.
            </p>
          </div>

          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#0f172a', marginBottom: '6px' }}>
              3. AR 발 디딤 타겟(Foot Target) 확인
            </div>
            <p style={{ fontSize: '12px', color: '#475569', lineHeight: 1.6 }}>
              내리막 계단을 마주쳤을 때 화면에 투영되는 4점 사각 디딤판 중앙의 초록색 마커를 향해 발바닥 중앙을 착지하시면 헛디딤과 미끄러짐을 완벽히 방지합니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
