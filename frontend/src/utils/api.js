// API client utility for SafeStep Night Stair Vision & Safety Guide

export async function fetchHealth() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('Healthcheck failed');
  return res.json();
}

/**
 * 야간 계단 탐지 & 보행 안전 가이드 API 호출
 * @param {File|string} fileOrBase64 - 이미지 파일 객체 또는 base64 문자열
 * @param {object} params - { gamma, stair_type_sample, preferred_mode, enhancement_mode }
 */
export async function detectStairs(fileOrBase64, params = {}) {
  const formData = new FormData();
  if (fileOrBase64) {
    if (typeof fileOrBase64 === 'string') {
      formData.append('image_base64', fileOrBase64);
    } else {
      formData.append('file', fileOrBase64);
    }
  }
  formData.append('gamma', params.gamma ?? 1.6);
  formData.append('stair_type_sample', params.stair_type_sample || 'descending');
  formData.append('preferred_mode', params.preferred_mode || 'descending');
  formData.append('enhancement_mode', params.enhancement_mode || 'auto');

  const res = await fetch('/api/stair/detect', { method: 'POST', body: formData });
  if (!res.ok) throw new Error('Stair detection failed');
  return res.json();
}
