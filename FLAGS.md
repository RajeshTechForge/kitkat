# Project Flags

Track code that needs attention: bugs, improvements, future changes, and technical debt.

---

## 🔴 Critical (Fix Before Release)

| File | Line | Priority | Description | Status | Issue |
|------|------|----------|-------------|--------|-------|
|  |  |  |  |  |  |


## 🟡 Important (Fix in Next Release)

| File | Line | Priority | Description | Target Version | Issue |
|------|------|----------|-------------|----------------|-------|
|  |  |  |  |  |  |


## 🔵 Technical Debt (Schedule Later)

| File | Line | Priority | Description | Target Version | Issue |
|------|------|----------|-------------|----------------|-------|
| `agents/adapters/managed.py` | func `_to_llm_request()` | P2 | This is a text-based fallback. The ideal long-term fix is to extend `LLMRequest` / `Message` to carry structured tool-call content natively, so the provider layer can format it correctly (Anthropic `tool_use` blocks, OpenAI `function_call`, etc.). That's a larger change to the core models. | - | - |


## 🟢 Future Improvements

| File | Line | Priority | Description | Target Version | Issue |
|------|------|----------|-------------|----------------|-------|
|  |  |  |  |  |  |


## ⚠️ Deprecated (Remove Later)

| File | Line | Description | Removed In | Issue |
|------|------|-------------|------------|-------|
| |  |  |  |  |


## 🏴 Hacks / Workarounds (Temporary)

| File | Line | Description | Reason | Remove By |
|------|------|-------------|--------|-----------|
| |  |  |  |  |

---

## Legend

- **Priority**: P0 (Critical) > P1 (Important) > P2 (Debt) > P3 (Nice-to-have)
- **Status**: ⏳ Not Started | 🔄 In Progress | ✅ Done | ❌ Won't Fix
- **Issue**: Link to GitHub/GitLab issue for details and discussion

---

## Quick Search

Run these commands to find flags in code:

```bash
# Find all TODO/FIXME/XXX/HACK with issue numbers
grep -rn "TODO\|FIXME\|XXX\|HACK\|DEPRECATED" --include="*.py" .

# With ripgrep (faster)
rg "TODO|FIXME|XXX|HACK" -t py .

# Count by type
rg "TODO" -t py . | wc -l
rg "FIXME" -t py . | wc -l
```

---

## Adding New Flags

When adding a flag in code, follow this format:

```python
# TODO(#issue): Description of what needs to be done
# FIXME(#issue): Description of the bug or issue
```

See the [Inline Markers](#inline-markers) section for examples.

---

## Inline Markers Reference

See below for how to properly format inline markers in code:

---

<!-- @flag:inline-markers-start -->

## Inline Markers

Use these formats in your code files. **Always include an issue number.**

### TODO
```python
# TODO(#123): Add validation for email format
def validate_email(email: str) -> bool:
    pass
```

### FIXME
```python
# FIXME(#456): Handle None case — currently raises AttributeError
def get_config(key: str) -> dict:
    return self.config[key]  # ← breaks if key missing
```

### HACK
```python
# HACK(#789): Temp workaround for API returning wrong Content-Type
# Remove after API v3.2 is released
response.headers['Content-Type'] = 'application/json'
```

### DEPRECATED
```python
# DEPRECATED(#321): Use parse_json() instead. Removed in v3.0.
def parse_response(response):
    pass
```

### XXX
```python
# XXX(#654): Brittle — depends on external API response order
# Add integration test before v2.1 release
def extract_user(data):
    return data['name'], data['email'], data['id']  # assumes order
```

### NOTE
```python
# NOTE(#777): This algorithm is O(n²) — consider optimization
# For 10k+ items, switch to heap-based approach
def find_duplicates(items: list) -> list:
    pass
```

### REVIEW
```python
# REVIEW(#888): Thread safety not verified — needs review
# Related to issue #555 about concurrent access
class ConnectionPool:
    pass
```

### OPTIMIZE
```python
# OPTIMIZE(#999): Cache results to avoid repeated DB queries
def get_user_stats(user_id: int) -> dict:
    pass
```

<!-- @flag:inline-markers-end -->

---

## Maintenance

- Review this file before each release
- Remove entries when resolved (move to CHANGELOG)
- Update "Target Version" when priorities change
- Run `rg "TODO|FIXME" -t py .` monthly to sync with code

---

*Last updated: 2026-07-27*