# BandMatch 🎵

**BandMatch**는 두 개의 레퍼런스 음원을 기준으로 타깃 음원의 주파수 대역별 에너지 분포를 비교 분석하는 macOS용 오디오 분석 도구입니다.

## 주요 기능

- 🎚️ **주파수 대역 분석**: 저음/중저음/중음/중고음/고음 5개 대역 에너지 비교
- 📊 **정량적 비교**: 대역별 dB 차이를 수치로 제공
- 🎯 **LUFS 정규화**: ITU-R BS.1770-4 표준 기반 라우드니스 매칭
- 📈 **시각화**: 바 차트, 레이더 차트로 직관적인 결과 표시
- 💾 **다양한 출력**: JSON, CSV, PDF 리포트 생성
- 🖥️ **듀얼 인터페이스**: GUI와 CLI 모두 지원

## 설치 방법

### 필수 요구사항

- macOS 13.0 이상
- Python 3.11 이상
- FFmpeg (MP3/M4A 지원용)

### 1. FFmpeg 설치

```bash
# Homebrew를 통한 설치
brew install ffmpeg
```

### 2. 프로젝트 클론 및 설정

```bash
# 프로젝트 클론
git clone https://github.com/yourusername/bandmatch.git
cd bandmatch

# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### GUI 모드

```bash
# GUI 실행
python bandmatch.py gui

# 또는 기본 실행 (GUI가 기본값)
python bandmatch.py
```

GUI에서:
1. Reference A, Reference B, Target 오디오 파일을 드래그&드롭 또는 Browse로 선택
2. 설정 조정 (LUFS, 대역 프리셋, 집계 방식)
3. "Analyze" 클릭
4. 결과 확인 및 내보내기

### CLI 모드

```bash
# 기본 분석
python bandmatch.py cli \
  --ref-a reference1.wav \
  --ref-b reference2.wav \
  --target mix.wav \
  --json report.json

# 상세 옵션
python bandmatch.py cli \
  --ref-a ref_a.wav \
  --ref-b ref_b.wav \
  --target target.wav \
  --sr 48000 \
  --lufs -14 \
  --preset mastering \
  --aggregate median \
  --weights "1.5,1" \
  --json output.json \
  --csv output.csv \
  --pdf output.pdf \
  --verbose
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--ref-a` | Reference A 오디오 파일 | 필수 |
| `--ref-b` | Reference B 오디오 파일 | 필수 |
| `--target` | Target 오디오 파일 | 필수 |
| `--sr` | 샘플레이트 | 48000 |
| `--lufs` | 타깃 LUFS | -14.0 |
| `--bands` | 커스텀 대역 정의 | - |
| `--preset` | 대역 프리셋 (default/mastering/podcast/edm/voice) | default |
| `--aggregate` | 시간 집계 방식 (median/mean/percentile_95) | median |
| `--weights` | 레퍼런스 가중치 | "1,1" |
| `--n-fft` | FFT 크기 | 4096 |
| `--json` | JSON 출력 파일 | - |
| `--csv` | CSV 출력 파일 | - |
| `--pdf` | PDF 출력 파일 | - |
| `--charts/--no-charts` | 차트 생성 여부 | true |
| `--verbose` | 상세 출력 | false |

## 대역 정의

### 기본 대역 (Default)
- **저음 (Low)**: 20-80 Hz
- **중저음 (Low-Mid)**: 80-250 Hz
- **중음 (Mid)**: 250-2,000 Hz
- **중고음 (High-Mid)**: 2,000-6,000 Hz
- **고음 (High)**: 6,000-20,000 Hz

### 프리셋
- **mastering**: 7개 대역 (Sub/Bass/Low-Mid/Mid/Upper-Mid/Presence/Air)
- **podcast**: 음성 최적화 5개 대역
- **edm**: EDM 장르 최적화 6개 대역
- **voice**: 보컬 분석용 5개 대역

### 커스텀 대역
```bash
# 커스텀 대역 정의 예시
--bands "20-60,60-200,200-800,800-4000,4000-20000"
```

## 판정 기준

| Delta (dB) | 판정 | 설명 |
|------------|------|------|
| < ±1.0 | 적정 | 레퍼런스와 거의 일치 |
| ±1.0-3.0 | 약간 부족/과다 | 미세 조정 권장 |
| ±3.0-6.0 | 부족/과다 | 조정 권장 |
| > ±6.0 | 크게 부족/과다 | 강력한 조정 권장 |

## 출력 형식

### JSON
```json
{
  "bands": ["Low", "Low-Mid", "Mid", "High-Mid", "High"],
  "baseline_db": [-23.1, -20.4, -18.6, -19.2, -21.0],
  "target_db": [-21.5, -18.0, -19.1, -22.4, -24.5],
  "delta_db": [1.6, 2.4, -0.5, -3.2, -3.5],
  "judgement": ["약간 과다", "과다", "적정", "부족", "부족"],
  "warnings": []
}
```

### CSV
```csv
Band,Baseline_dB,Target_dB,Delta_dB,Judgement
Low,-23.1,-21.5,1.6,약간 과다
Low-Mid,-20.4,-18.0,2.4,과다
...
```

## 테스트 실행

```bash
# 기본 테스트 실행
python tests/test_basic.py

# pytest 사용 (설치 필요)
pytest tests/
```

## 프로젝트 구조

```
bandmatch/
├── bandmatch.py         # 메인 엔트리 포인트
├── cli.py              # CLI 인터페이스
├── audio_io.py         # 오디오 파일 I/O
├── loudness.py         # LUFS 측정/정규화
├── bands.py            # 주파수 대역 정의
├── spectrum.py         # 스펙트럼 분석
├── reference.py        # 레퍼런스 결합
├── comparison.py       # 대역 비교 및 판정
├── report.py           # 리포트 생성
├── ui/
│   └── main_window.py  # PyQt6 GUI
├── tests/
│   └── test_basic.py   # 기본 테스트
├── requirements.txt    # 의존성 목록
└── README.md          # 문서
```

## 알려진 제한사항

- 최소 10초 이상의 오디오 권장 (짧은 오디오는 신뢰도 낮음)
- MP3 저비트레이트 파일은 고주파 왜곡 가능
- 레퍼런스 간 극단적 차이(>3dB)시 경고 표시

## 문제 해결

### FFmpeg를 찾을 수 없음
```bash
# FFmpeg 설치 확인
ffmpeg -version

# 설치되지 않은 경우
brew install ffmpeg
```

### PyQt6 import 오류
```bash
# PyQt6 재설치
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip
pip install PyQt6
```

### 권한 오류
```bash
# 실행 권한 부여
chmod +x bandmatch.py cli.py
```

## 라이선스

MIT License

## 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 감사의 글

- librosa - 오디오 분석 라이브러리
- pyloudnorm - LUFS 측정 구현
- PyQt6 - GUI 프레임워크
- FFmpeg - 멀티미디어 처리
