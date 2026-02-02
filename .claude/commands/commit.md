# /commit - Complete and Push Changes

작업을 마무리하고 변경사항을 커밋 및 푸시합니다.

## Usage

```
/commit              # 현재 세션의 모든 변경사항을 커밋 및 푸시
/commit [message]    # 커스텀 메시지로 커밋 및 푸시
```

## Workflow

작업 마무리 시 다음 단계를 자동으로 수행합니다:

1. **Pre-Commit Check**: 테스트 및 린트 확인
2. **Status Check**: 변경된 파일 확인 (`git status`, `git diff HEAD`)
3. **Commit Message**: Conventional Commits 규칙에 따라 메시지 생성
4. **Commit**: 변경사항 스테이징 및 커밋
5. **Push**: 원격 저장소로 푸시
6. **Report**: 결과 보고

## Steps

### Step 1: Pre-Commit Check

커밋 전 테스트와 린트를 확인합니다.

```bash
# sdy/rag/ 디렉토리에서
cd sdy/rag && uv run pytest -v
cd sdy/rag && uv run ruff check src tests
```

테스트 실패 시 커밋을 중단하고 사용자에게 보고합니다.

### Step 2: Check Status

변경된 파일을 확인합니다.

```bash
git status
git diff HEAD
```

### Step 3: Generate Commit Message

Conventional Commits 규칙에 따라 커밋 메시지를 생성합니다.

**Format:**
```
<type>(<scope>): <subject>

[optional body]

Co-Authored-By: Claude <model> <noreply@anthropic.com>
```

**Types:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드, 의존성 등

**Scopes (sdy/rag):**
- `loader`: 문서 로딩
- `splitter`: 청킹
- `embedder`: 임베딩
- `store`: 벡터 저장소
- `retriever`: 검색
- `generator`: 답변 생성
- `pipeline`: 파이프라인 통합
- `cli`: CLI 인터페이스
- `deps`: 의존성
- `docs`: 문서

**Example Messages:**
```bash
# Feature
feat(loader): add PDF file support

- Add pdf parsing logic
- Include page number in metadata
- Unit tests added

Co-Authored-By: Claude <model> <noreply@anthropic.com>

# Fix
fix(splitter): correct overlap calculation at document boundary

Chunks at the end of documents were missing overlap content
due to off-by-one error in stride calculation.

Co-Authored-By: Claude <model> <noreply@anthropic.com>

# Docs
docs(claude): update architecture section in CLAUDE.md

Co-Authored-By: Claude <model> <noreply@anthropic.com>
```

### Step 4: Commit

변경사항을 스테이징하고 커밋합니다.

```bash
# 관련 파일만 스테이징 (git add . 대신 개별 파일 지정 권장)
git add <files>

# HEREDOC을 사용한 커밋 (포맷팅 보장)
git commit -m "$(cat <<'EOF'
feat(loader): add PDF file support

- Add pdf parsing logic
- Include page number in metadata

Co-Authored-By: Claude <model> <noreply@anthropic.com>
EOF
)"
```

### Step 5: Push

원격 저장소로 푸시합니다.

```bash
# 현재 브랜치를 원격으로 푸시
git push

# 새 브랜치의 경우 upstream 설정
git push -u origin <branch-name>
```

### Step 6: Report

사용자에게 결과를 보고합니다.

```markdown
Changes successfully committed and pushed!

**Commit:** feat(loader): add PDF file support
**Branch:** feature/add-pdf-loader
**Files Changed:** 3 files (+80, -10)

**Summary:**
- 3 files committed
- Pushed to origin/feature/add-pdf-loader
```

## Edge Cases

### No Changes to Commit

```
No changes to commit. Working tree is clean.
```

### Push Conflict

```
Push rejected. Remote has changes.

Suggested action:
git pull --rebase
git push
```

### Test Failure

```
Tests failed. Commit aborted.

Failed tests:
- test_should_load_pdf_file_with_metadata

Fix the failing tests before committing.
```

## Pre-Commit Checks

커밋 전 다음 항목을 자동으로 확인합니다:

- [ ] 테스트 통과 (`uv run pytest` in `sdy/rag/`)
- [ ] Linter 통과 (`uv run ruff check`)
- [ ] 비밀 정보 없음 (하드코딩된 API 키 등)

## References

- `.claude/rules/git-workflow.md` - Git 워크플로우 규칙
- `.claude/rules/testing.md` - 테스트 규칙
- `.claude/rules/security.md` - 보안 규칙
