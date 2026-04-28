# Agent Workflow for `students-AI_math_essentials`

This repository is notebook-heavy and teaching-oriented.

## Core Rules
- Use `bd` for task tracking.
- Prefer additive changes over broad rewrites.
- Do not renumber existing `TODO` blocks in lab notebooks.
- Keep student notebooks and solution notebooks structurally aligned.
- Preserve the teaching flow inside notebooks: contract -> theory -> manual example -> TODO -> checks -> diagnostics.

## Repo Shape
- `themes/00-Foundations/` contains shared prerequisite material, warm-up examples, and optional real-data showcases.
- `themes/01-RNN/` contains the core RNN theory and labs.
- `themes/02-Attention/` continues the same sequence-learning line with attention.

## Notebook Expectations
- New prerequisite notebooks should run top-to-bottom on CPU.
- Use `python3` in shell-facing instructions.
- Keep examples small enough for self-study and classroom use.
- When adding new links, verify relative paths from the file that references them.

## Beads Minimum Workflow
- `bd ready` to see unblocked work.
- `bd create "Title" -p 1` to open a new task.
- `bd update <id> --claim` to claim work.
- `bd show <id>` to inspect task state and history.

## Notebook Quality Checks
После правок учебных notebook-ов обязательно прогонять структурные и quality-контракты:

```bash
python3 scripts/validate_notebooks.py
python3 scripts/check_notebook_contracts.py
python3 scripts/check_lab_quality_contracts.py
python3 scripts/check_runtime_gpu_contracts.py
git diff --check
git status --short --branch
```

Правило: тяжелое обучение не запускать в обычном quality-маршруте, только
структурные и контрактные проверки.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
