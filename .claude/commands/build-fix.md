# /build-fix - Fix Build Errors

빌드 오류를 진단하고 수정합니다.

## Usage

```
/build-fix              # 전체 프로젝트 빌드 검사
```

## Workflow

1. **빌드 실행**: 빌드 명령 실행하여 오류 수집
2. **오류 분석**: 오류 메시지 파싱 및 원인 분석
3. **수정 제안**: 구체적인 수정 방안 제시
4. **검증**: 수정 후 재빌드하여 확인

## Build Commands

```bash
# sdy/rag/ 디렉토리에서

# 린팅
uv run ruff check src tests

# 포맷 검사
uv run ruff format --check src tests

# 테스트
uv run pytest
```

## Common Errors

### Python Errors
```
ModuleNotFoundError: No module named 'x'
→ 의존성 설치: uv add x

ImportError: cannot import name 'X' from 'Y'
→ 임포트 경로 확인

TypeError: X() got an unexpected keyword argument 'y'
→ 함수 시그니처 확인
```

### Ruff Errors
```
E501: Line too long
→ 줄 길이 88자 제한

F401: Module imported but unused
→ 사용하지 않는 임포트 제거

I001: Import block is un-sorted or un-formatted
→ uv run ruff check --fix
```

## Output Format

```markdown
## Build Fix Report

### Errors Found
1. [파일:라인] [오류 메시지]
   - 원인: [원인 분석]
   - 수정: [수정 방안]

### Applied Fixes
- [수정 1]
- [수정 2]

### Verification
✅ Build successful / ❌ Build failed (remaining issues)
```
