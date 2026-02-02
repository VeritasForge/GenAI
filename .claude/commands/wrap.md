# /wrap - Update Project Documentation

작업 완료 후 프로젝트 문서(`CLAUDE.md`)를 현재 코드베이스 상태에 맞게 업데이트합니다.

## Usage

```
/wrap              # 현재 코드베이스 분석 후 문서 업데이트
/wrap --check      # 문서와 코드의 일치 여부만 확인 (업데이트 없음)
```

## Document Structure

이 프로젝트는 다음과 같은 문서 구조를 따릅니다:

- **`CLAUDE.md`**: AI용 컨텍스트 + 프로젝트 가이드 (아키텍처, 규약, 명령어, 변경 이력)
- **`.claude/rules/`**: 코딩 스타일, 테스트, 보안, Git 워크플로우 규칙

## Workflow

1. **Analyze Codebase**: 현재 디렉토리 구조 및 파일 분석
2. **Read Documents**: `CLAUDE.md` 읽기
3. **Detect Changes**: 문서와 코드 간 불일치 탐지
4. **Update Documents**: 변경사항 반영
5. **Report**: 업데이트 내역 보고

## Steps

### Step 1: Analyze Codebase

현재 프로젝트의 구조를 분석합니다.

```bash
# 프로젝트 구조 확인
ls -la sdy/rag/src/rag/
ls -la sdy/rag/tests/

# 의존성 확인
cat sdy/rag/pyproject.toml
```

**분석 대상:**
- 모듈 변경 (`sdy/rag/src/rag/` 내 새 모듈, 삭제된 모듈)
- 외부 의존성 변경 (`pyproject.toml`)
- CLI 명령어 변경 (`cli.py`)
- 환경변수 변경 (`.env.example`)
- 데이터 파일 변경 (`sdy/rag/data/`)

### Step 2: Read Current Documents

기존 문서를 읽고 내용을 파악합니다.

```bash
cat CLAUDE.md
```

### Step 3: Detect Changes

문서와 코드 간 불일치를 탐지합니다.

**체크리스트:**

#### CLAUDE.md
- [ ] Section 1 (Repository Structure): 새 서브프로젝트 추가되었는가?
- [ ] Section 2 (Commands): CLI 명령어가 최신인가?
- [ ] Section 3 (Architecture): 모듈 구조 변경사항 반영되었는가?
- [ ] Section 4 (Conventions): 의존성/설정 변경되었는가?
- [ ] Section 8 (Claude Code Configuration): 명령어/에이전트 변경사항 반영되었는가?

### Step 4: Update Documents

탐지된 변경사항에 따라 문서를 업데이트합니다.

#### CLAUDE.md Update Rules

**수정 대상:**
```markdown
## 1. Repository Structure (새 서브프로젝트 추가 시)
- `sdy/new_project/` — 설명

## 2. Commands (CLI 명령어 변경 시)
# 새 명령어 추가/변경

## 3. Architecture (모듈 구조 변경 시)
- **`new_module.py`** — 설명

## 4. Conventions (의존성/설정 변경 시)
- 새 의존성, 환경변수 등

## 8. Claude Code Configuration (명령어/에이전트 변경 시)
- Available Commands 테이블 업데이트
```

**수정하지 않음:**
- Section 5 (Core Principles) — 원칙 변경 시만
- Section 6 (AI 사고 프로세스) — 프로세스 변경 시만
- Section 7 (Test Protection Protocol) — 프로세스 변경 시만
- `.claude/rules/` 파일들 — 규칙 변경 시만

### Step 5: Report Changes

업데이트 결과를 보고합니다.

```markdown
## Documentation Update Report

### CLAUDE.md
Updated:
- Section 3 Architecture: Added new_module.py
- Section 2 Commands: Added new CLI command

No changes needed:
- Section 5 Core Principles
- Section 7 Test Protection Protocol

### Summary
- 1 file updated
- 2 sections modified
- 0 inconsistencies remaining
```

## Update Guidelines

### Language
- **CLAUDE.md**: 한국어 기본, 기술 용어는 영어 유지

### Format
- 기존 문서의 마크다운 형식 유지
- 섹션 번호 체계 유지
- 코드 블록 스타일 일관성 유지

### Anti-Patterns

**Don't:**
- 기존 섹션 구조를 크게 변경
- 문서에 없던 새 섹션을 임의로 추가
- Core Principles를 사용자 승인 없이 변경
- 불필요한 내용 추가 (간결하게 유지)

**Do:**
- 기존 형식과 톤 유지
- 변경사항만 업데이트 (불필요한 수정 지양)
- CLAUDE.md는 간결하게 유지 (300줄 이하 목표)

## Check Mode

`--check` 플래그를 사용하면 업데이트 없이 일치 여부만 확인합니다.

```bash
/wrap --check
```

**Output:**
```markdown
## Documentation Check Report

### CLAUDE.md
Inconsistencies found:
- Section 3: New module not documented
- Section 2: New CLI command not documented

Consistent sections:
- Section 5 Core Principles
- Section 7 Test Protection Protocol

### Action Required
Run `/wrap` to update documents automatically.
```

## Integration with Other Commands

`/wrap`은 다음 명령어들과 함께 사용됩니다:

```bash
# TDD 개발 → 문서 업데이트 → 커밋
/tdd [feature]
/wrap
/commit

# 리뷰 → 문서 업데이트 → 커밋
/review
/wrap
/commit
```

## References

- `CLAUDE.md` - 프로젝트 컨텍스트
- `.claude/rules/coding-style.md` - 코딩 스타일 규칙
- `.claude/rules/git-workflow.md` - Git 워크플로우 규칙

## Notes

- 문서 업데이트는 코드 변경 후 수행 권장
- 문서는 항상 현재 코드베이스 상태를 반영해야 합니다
- Core Principles 수정이 필요한 경우 반드시 사용자에게 확인
