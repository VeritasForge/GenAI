# Security Rules

모든 코드 작성 및 커밋 전 반드시 준수해야 하는 보안 규칙입니다.

## Mandatory Checks

### 1. No Hardcoded Secrets
```python
# ❌ NEVER do this
API_KEY = "sk-proj-abc123..."
DATABASE_URL = "postgresql://user:password@host/db"

# ✅ Always use environment variables
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
```

### 2. Input Validation
```python
# ✅ Validate all user inputs
def load_file(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix != ".txt":
        raise ValueError(f"Only .txt files are supported, got '{path.suffix}'")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"File is empty: {path}")
    return Document(content=content, metadata={...})
```

### 3. Command Injection Prevention
```python
# ❌ NEVER do this
import subprocess
subprocess.run(f"claude -p {user_input}", shell=True)

# ✅ Use list arguments (no shell=True)
subprocess.run(["claude", "-p", user_input], capture_output=True, text=True)
```

### 4. Path Traversal Prevention
```python
# ❌ Vulnerable
def load_file(filename: str) -> Document:
    return Path(f"data/{filename}").read_text()

# ✅ Validate path stays within expected directory
def load_file(filename: str, base_dir: Path) -> Document:
    path = (base_dir / filename).resolve()
    if not str(path).startswith(str(base_dir.resolve())):
        raise ValueError("Path traversal detected")
    return path.read_text()
```

### 5. Error Messages
```python
# ❌ Exposes sensitive information
except Exception as e:
    return {"error": str(e)}  # May reveal stack traces

# ✅ Generic error messages
except Exception as e:
    logger.error(f"Error: {e}")  # Log details for debugging
    raise RuntimeError("An error occurred")  # Generic to user
```

## Files to Never Commit

```gitignore
# .gitignore - MUST include
.env
.env.local
.env.*.local
*.pem
*.key
credentials.json
secrets.yaml
chroma_db/
```

## Response Protocol

보안 이슈 발견 시:
1. **즉시 작업 중단**
2. **이슈 심각도 평가** (Critical/High/Medium/Low)
3. **수정 계획 수립**
4. **노출된 비밀은 즉시 로테이션**
5. **전체 코드베이스 유사 이슈 스캔**
