# Git Workflow Rules

Git 사용에 관한 규칙입니다.

## Branch Naming

```
feature/[issue-number]-short-description
bugfix/[issue-number]-short-description
hotfix/[issue-number]-short-description
refactor/short-description
docs/short-description
```

Examples:
- `feature/add-pdf-loader`
- `bugfix/fix-chunk-overlap`
- `refactor/cleanup-pipeline`

## Commit Message Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 포맷팅 등 (코드 변경 없음)
- `refactor`: 리팩토링 (기능 변경 없음)
- `test`: 테스트 추가/수정
- `chore`: 빌드, 의존성 등

### Scopes
- `loader`: 문서 로딩
- `splitter`: 청킹
- `embedder`: 임베딩
- `store`: 벡터 저장소
- `retriever`: 검색
- `generator`: 답변 생성
- `pipeline`: 파이프라인 통합
- `cli`: CLI 인터페이스
- `deps`: 의존성

### Examples
```
feat(loader): add PDF file support

- Add pdf_loader.py for PDF parsing
- Include page number in metadata
- Unit tests added

Closes #12
```

```
fix(splitter): correct overlap calculation at document boundary

Chunks at the end of documents were missing overlap content
due to off-by-one error in stride calculation.

Fixes #34
```

## Commit Rules

### Do
- 작은 단위로 자주 커밋
- 하나의 커밋에 하나의 논리적 변경
- 테스트와 구현을 함께 커밋
- 의미 있는 커밋 메시지 작성

### Don't
- 여러 기능을 하나의 커밋에 포함
- "WIP", "fix", "update" 같은 모호한 메시지
- 빌드가 깨진 상태로 커밋
- 민감한 정보 커밋

## Dangerous Commands (Use with Caution)

```bash
# ⚠️ 강제 푸시 (개인 브랜치만)
git push --force-with-lease

# ❌ main에 절대 사용 금지
git push --force origin main
git reset --hard origin/main
```

## Pre-commit Hooks

자동으로 실행되는 검사:

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
uv run pytest
```
