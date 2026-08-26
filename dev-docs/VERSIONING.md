# Versioning Policy

This document defines how versions are numbered, what counts as our public
API, how we handle deprecations, and how releases are cut. It applies from
`v1.0.0` onward and is binding for all maintainers and contributors.

If you're proposing a change and unsure whether it's a MAJOR, MINOR, or
PATCH bump, this file is the source of truth.


## 1. We follow Semantic Versioning (SemVer 2.0.0)

Version numbers take the form `MAJOR.MINOR.PATCH`, e.g. `1.4.2`.

| Segment | Bumped when...                                                        |
|---------|------------------------------------------------------------------------|
| MAJOR   | A backward-incompatible change is made to the **public API**           |
| MINOR   | Backward-compatible functionality is added                             |
| PATCH   | Backward-compatible bug fixes are made, with no API changes            |

Full spec: https://semver.org

We additionally follow **PEP 440** for how pre-release identifiers are
formatted, since that's what pip and PyPI actually parse. See §5.


## 2. What counts as the public API

SemVer only means something if "the API" is clearly defined. The following
is our contract:

### Public (covered by SemVer guarantees)
- Anything exported via `__all__` in a package's `__init__.py`
- Anything documented on the official documentation site
- Public function/method signatures, including parameter names for
  keyword arguments
- Behavior explicitly documented (return types, exceptions raised,
  side effects)

### NOT public (may change in any release, including PATCH)
- Anything under a module or path prefixed with `_` (e.g. `mylib._internal`)
- Anything not listed in `__all__`, even if importable
- Private attributes/methods prefixed with `_`
- Undocumented behavior, error message text/wording, and internal
  exception hierarchies not explicitly documented
- CLI output formatting, unless explicitly documented as stable
- Anything marked `Experimental` or `Provisional` in the docs (see §6)

**Rule of thumb:** if it's not in `__all__` and not in the docs, it isn't
part of the contract — treat it as an implementation detail, and don't
file a breaking-change bump for changing it.


## 3. What triggers each version bump

### MAJOR (`X.0.0`)
- Removing or renaming a public function, class, method, or parameter
- Changing a public function's behavior in a way existing correct code
  would notice (e.g. changed return type, changed default value,
  stricter validation that now raises where it didn't before)
- Dropping support for a previously-supported Python version
- Changing required dependencies in a way that breaks existing installs

### MINOR (`x.Y.0`)
- Adding a new public function, class, method, or module
- Adding a new optional parameter with a backward-compatible default
- Marking something as deprecated (the deprecation itself is not breaking)
- Performance improvements with no behavior change
- Expanding support (e.g. adding a new supported Python version)

### PATCH (`x.y.Z`)
- Bug fixes that restore documented/intended behavior
- Documentation fixes
- Internal refactors with zero public-facing effect
- Dependency version bumps that don't change our public behavior

**Note:** a bug fix that happens to change behavior someone was
(unintentionally) relying on is still a PATCH if the old behavior was
never the documented/intended behavior. Judgment call, when in doubt,
open a maintainer discussion before releasing.


## 4. Before `v1.0.0`

Versions `0.y.z` are considered **unstable**: any release, including a
`0.y` bump, may contain breaking changes. This is standard SemVer
behavior for the `0.x` line and is why we're able to move fast pre-1.0.

`v1.0.0` is a public commitment that the API is stable. Before we tag it:

- [ ] The public API surface (§2) is finalized and documented
- [ ] `__all__` is set correctly in every public module
- [ ] A release-candidate cycle has been run (see §5) with no
      API-breaking issues found
- [ ] `CHANGELOG.md` and `MIGRATING.md` (0.x → 1.0 guide) are complete
- [ ] `python_requires` in `pyproject.toml` reflects our actual
      minimum supported version


## 5. Pre-releases (PEP 440)

We use PEP 440 pre-release identifiers, which pip and PyPI understand
natively (`pip install` skips these by default unless `--pre` is passed):

```
1.0.0a1     # alpha — early, unstable, API may still shift
1.0.0b1     # beta — API frozen, testing for bugs
1.0.0rc1    # release candidate — believed ready, final verification
1.0.0       # final release
```

Typical release flow for a MAJOR or MINOR version:

```
1.0.0rc1 → (fix reported issues) → 1.0.0rc2 → 1.0.0
```

Pre-releases are published to PyPI (not just TestPyPI) so real users
can opt in with `pip install mylib --pre` and give feedback before the
version is final.


## 6. Experimental / Provisional API

Sometimes we want to ship something before we're ready to commit to it
forever. Anything explicitly marked **`Experimental`** or
**`Provisional`** in its docstring and documentation page:

- Is exported and usable like normal public API
- Is **not** covered by SemVer guarantees — it may change or be removed
  in a MINOR or even PATCH release
- Must include a docstring note, e.g.:

  ```python
  def new_thing():
      """
      .. warning::
          Experimental. This API may change without notice in a future
          MINOR or PATCH release. Not covered by SemVer guarantees.
      """
  ```

Once an experimental API is considered stable, it graduates to full
public API status via a MINOR release, with a changelog entry noting
the graduation.


## 7. Deprecation policy

We don't remove public API without warning. The process:

1. **Deprecate** — mark the old API with a runtime warning:

   ```python
   import warnings

   def old_method(self):
       warnings.warn(
           "old_method() is deprecated and will be removed in v2.0.0; "
           "use process() instead.",
           DeprecationWarning,
           stacklevel=2,
       )
       return self.process()
   ```

   This ships in a MINOR release. Document the replacement and the
   version it will be removed in, in both the docstring and the
   changelog.

2. **Coexist** — the deprecated API must keep working for at least
   one full MINOR release cycle, and will not be removed before the
   next MAJOR version.

3. **Remove** — the deprecated API is removed only in a MAJOR release,
   and the removal is called out explicitly in `CHANGELOG.md` and
   `MIGRATING.md`.

Minimum deprecation window: **one MAJOR version**, or **6 months**,
whichever is longer.


## 8. Python version support

We follow a rolling support window aligned with upstream Python's
support lifecycle:

- We support all Python minor versions that are not yet End-of-Life,
  down to a minimum we state in `pyproject.toml` (`python_requires`)
- Dropping a Python version is a **MAJOR** bump
- Adding support for a new Python version is a **MINOR** bump


## 9. Dependencies

- Changing a dependency's minimum required version in a way that could
  break existing environments is treated as a **MINOR** bump at
  minimum (**MAJOR** if it changes our own public behavior)
- New required (non-optional) dependencies are a **MINOR** bump at
  minimum, and should be avoided in PATCH releases entirely
- Optional/extras dependencies can be added in a MINOR release


## 10. Single source of truth for the version number

The version is derived automatically from the git tag via
`setuptools_scm` — it is **not** hand-edited in multiple files. Do not
manually set `__version__` in source files; it's generated at build
time.

To cut a release, the version is set by creating an annotated git tag:

```bash
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

- Tags are always prefixed with `v`
- Tags are annotated (`-a`), never lightweight
- Tags are immutable once pushed — a mistake is fixed with a new PATCH
  release, never by moving or force-pushing a tag
- A version, once published to PyPI, can never be re-uploaded or
  reused, even if yanked


## 11. Where releases are documented

- **`CHANGELOG.md`** — every release, following the
  [Keep a Changelog](https://keepachangelog.com/) format, grouped into
  `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`
- **`MIGRATING.md`** — step-by-step upgrade guide, written whenever a
  MAJOR version ships breaking changes
- Git tag + GitHub Release notes — auto-generated or summarized from
  `CHANGELOG.md` at tag time


## 12. Quick reference

| Change                                             | Bump  |
|-----------------------------------------------------|-------|
| Remove/rename public function or parameter          | MAJOR |
| Change default return value/behavior of public API  | MAJOR |
| Drop support for a Python version                    | MAJOR |
| Remove a deprecated API (after deprecation window)   | MAJOR |
| Add new public function/class/module                | MINOR |
| Add new optional parameter with safe default         | MINOR |
| Deprecate an existing API (mark only, not remove)    | MINOR |
| Add support for a new Python version                 | MINOR |
| Bug fix restoring documented behavior                | PATCH |
| Internal refactor, no public effect                  | PATCH |
| Documentation-only change                            | PATCH |

When genuinely unsure, bump up, not down — treat ambiguous cases as the
larger of the two candidate bumps and discuss before release.
