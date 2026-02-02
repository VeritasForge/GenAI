# /review - Code Review

변경된 코드를 검토합니다.

## Usage

```
/review              # 현재 브랜치의 모든 변경사항 검토
/review [파일경로]    # 특정 파일만 검토
/review --staged     # 스테이징된 변경사항만 검토
```

## Review Checklist

### Security (Critical)
- [ ] 하드코딩된 비밀 (API 키, 비밀번호) 없음
- [ ] Command injection 방지
- [ ] 입력 검증

### Quality (High)
- [ ] 함수 길이 50줄 이하
- [ ] 파일 길이 800줄 이하
- [ ] 중첩 깊이 4단계 이하
- [ ] 적절한 에러 핸들링
- [ ] 테스트 커버리지

### Maintainability (Medium)
- [ ] 명확한 변수/함수 이름
- [ ] 매직 넘버 상수화
- [ ] 중복 코드 제거
- [ ] 타입 힌트 완전성

## Review Commands

```bash
# 변경사항 확인
git diff

# 스테이징된 변경사항
git diff --staged

# 특정 커밋 이후 변경사항
git diff HEAD~3
```

## Output Format

```markdown
## Code Review Summary

### Files Reviewed
- [파일1]
- [파일2]

### Findings

#### Critical
- [파일:라인] [문제] → [수정 제안]

#### Suggestions
- [개선 제안]

### Verdict
✅ Approve / ⚠️ Approve with comments / ❌ Request changes
```
