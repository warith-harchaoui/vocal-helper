#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# nomoreclaude.sh — purge every trace of Claude / Anthropic / Claude Code
# from a git repository : commit history (author, committer, subject, body,
# trailers) AND tracked working-tree files (headers, doc credits, etc.).
#
# Default mode is *audit* — it scans and reports without mutating anything.
# Pass ``--apply`` to actually rewrite history and patch files. Pass
# ``--push`` (only with ``--apply``) to force-push the rewritten branches
# afterwards.
#
# Usage
# -----
#
#   ./nomoreclaude.sh                  # dry-run — exit 1 if findings
#   ./nomoreclaude.sh --apply          # rewrite history + files (local only)
#   ./nomoreclaude.sh --apply --push   # ... then force-push to ``origin``
#
# Dependencies
# ------------
#
# * ``git`` (always required).
# * ``git-filter-repo`` (preferred for ``--apply`` ; ``brew install
#   git-filter-repo`` on macOS). Falls back to ``git filter-branch`` with a
#   loud warning if ``git-filter-repo`` is absent.
#
# Safety
# ------
#
# * History rewriting is destructive. ``--apply`` operates on a clone-safe
#   *local* branch only ; ``--push`` is required to publish.
# * Before pushing, the script prints the new commit graph and waits for
#   ``y`` on stdin. CI / non-interactive runs should pass ``--yes`` to
#   skip the prompt (use deliberately).
# * Co-authors / pre-existing tags signed with the old commits will lose
#   their signatures — that's the cost of a history rewrite. Re-sign as
#   needed after the push.
#
# Author
# ------
# Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui
# ---------------------------------------------------------------------------

set -euo pipefail

APPLY=0
PUSH=0
ASSUME_YES=0
REPO_PATH="."

for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --push)  PUSH=1 ;;
        --yes)   ASSUME_YES=1 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            if [[ -d "$arg" ]]; then
                REPO_PATH="$arg"
            else
                echo "unknown arg: $arg" >&2
                exit 2
            fi
            ;;
    esac
done

cd "$REPO_PATH"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: $(pwd) is not a git repository." >&2
    exit 2
fi

REPO_NAME="$(basename "$(git rev-parse --show-toplevel)")"
echo "## nomoreclaude.sh — repo: $REPO_NAME"
echo "## mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo audit)"
echo

# ---------------------------------------------------------------------------
# Patterns. Anchor on word boundaries where useful to limit false positives
# (``claude`` is also a French given name, etc.), but cast wide enough to
# catch the standard Claude Code attribution block.
# ---------------------------------------------------------------------------
# Word-bounded so neutral occurrences (e.g. ``nomoreclaude`` in this
# script's own filename) don't count as hits. The standard Claude Code
# attribution patterns still match because they sit on word boundaries.
PATTERN='(\bclaude\b|\banthropic\b|\bclaude[- ]code\b|co-authored-by:[[:space:]]*claude|generated with[[:space:]]+claude)'

# ---------------------------------------------------------------------------
# Audit step — always runs.
# ---------------------------------------------------------------------------
echo "=== git history scan (all refs) ==="
HISTORY_HITS=0
# Author + committer + subject + body, one block per commit, case-insensitive.
HISTORY_OUT="$(git log --all \
    --format='--- %H%n%an <%ae>%n%cn <%ce>%n%s%n%b' \
    -i --grep="$PATTERN" \
    --author="$PATTERN" \
    --committer="$PATTERN" || true)"
if [[ -n "$HISTORY_OUT" ]]; then
    echo "$HISTORY_OUT" | head -200
    HISTORY_HITS=$(grep -cE '^---' <<<"$HISTORY_OUT" || true)
    echo "history matches: $HISTORY_HITS commit(s)"
else
    echo "history matches: 0 commit(s)"
fi
echo

echo "=== tracked-file scan ==="
FILE_HITS=0
FILE_OUT="$(git grep -inE "$PATTERN" -- ':!*.lock' ':!*.svg' ':!nomoreclaude.sh' 2>/dev/null || true)"
if [[ -n "$FILE_OUT" ]]; then
    echo "$FILE_OUT"
    FILE_HITS=$(wc -l <<<"$FILE_OUT" | tr -d ' ')
    echo "file matches: $FILE_HITS line(s)"
else
    echo "file matches: 0 line(s)"
fi
echo

TOTAL=$((HISTORY_HITS + FILE_HITS))
if [[ $TOTAL -eq 0 ]]; then
    echo "✅ $REPO_NAME is clean — no Claude / Anthropic mentions found."
    exit 0
fi

if [[ $APPLY -eq 0 ]]; then
    echo "⚠  $REPO_NAME has $TOTAL match(es). Re-run with --apply to scrub."
    exit 1
fi

# ---------------------------------------------------------------------------
# Apply step — destructive.
# ---------------------------------------------------------------------------
echo "=== applying scrub ==="

# Refuse to operate on a dirty tree — file rewrites would be lost.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: working tree is dirty. Commit or stash before --apply." >&2
    exit 2
fi

USE_FILTER_REPO=1
if ! command -v git-filter-repo >/dev/null 2>&1; then
    USE_FILTER_REPO=0
    echo "⚠  git-filter-repo not found. Falling back to git filter-branch — slower"
    echo "   and noisier. Install with 'brew install git-filter-repo' on macOS."
fi

# Patch tracked files first (in-tree) — this becomes one normal commit
# *before* the history rewrite, so the rewrite picks it up as the current
# tip. The history rewrite then removes Claude lines from *every* commit
# (including this one), but we still get a clean tip.
if [[ $FILE_HITS -gt 0 ]]; then
    echo "patching tracked files…"
    # Drop entire lines that mention Claude/Anthropic from text files. The
    # ``-iE`` makes the match case-insensitive ; ``-v`` keeps non-matching
    # lines. Binary files are skipped by ``git grep --files-with-matches``.
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        tmp="$(mktemp)"
        grep -ivE "$PATTERN" -- "$f" >"$tmp" || true
        mv "$tmp" "$f"
        echo "  patched $f"
    done < <(git grep -lE "$PATTERN" -- ':!*.lock' ':!*.svg' ':!nomoreclaude.sh' 2>/dev/null || true)
    git add -A
    if ! git diff --cached --quiet; then
        git commit -m "chore: drop Claude/Anthropic mentions from tracked files"
    fi
fi

if [[ $USE_FILTER_REPO -eq 1 ]]; then
    # git-filter-repo refuses to run on a non-fresh clone by default ;
    # ``--force`` is explicit acknowledgement that we know.
    git filter-repo --force \
        --message-callback "
import re
pat = re.compile(r'(?i)$PATTERN')
new = b'\n'.join(line for line in message.split(b'\n') if not pat.search(line.decode('utf-8', 'replace')))
return new.strip(b'\n') + b'\n'
" \
        --commit-callback "
import re
pat = re.compile(r'(?i)$PATTERN')
def scrub(name, email):
    name_s = name.decode('utf-8', 'replace')
    email_s = email.decode('utf-8', 'replace')
    if pat.search(name_s) or pat.search(email_s):
        return b'Warith HARCHAOUI', b'warith@deraison.ai'
    return name, email
commit.author_name, commit.author_email = scrub(commit.author_name, commit.author_email)
commit.committer_name, commit.committer_email = scrub(commit.committer_name, commit.committer_email)
"
else
    # Fallback path — only scrubs commit messages, NOT author/committer
    # identities. ``git filter-branch --env-filter`` could do the latter
    # but is gnarly and slow ; if you hit this path, install
    # git-filter-repo and re-run.
    git filter-branch -f --msg-filter "
        sed -E '/(claude|anthropic|claude[- ]code|co-authored-by:.*claude|generated with.*claude)/Id'
    " -- --all
fi

echo
echo "=== post-scrub audit ==="
"$0" "$REPO_PATH"
POST_RC=$?

if [[ $POST_RC -ne 0 ]]; then
    echo "⚠  scrub left residual matches — review the output above before pushing."
    exit "$POST_RC"
fi

if [[ $PUSH -eq 0 ]]; then
    echo
    echo "Local history rewritten ; remote is untouched."
    echo "Inspect with: git log --oneline --decorate"
    echo "When ready: re-run with --apply --push  (force-pushes every branch)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Push step — needs explicit confirmation.
# ---------------------------------------------------------------------------
echo
echo "=== force-pushing to origin ==="
if [[ $ASSUME_YES -eq 0 ]]; then
    read -rp "Type 'yes' to force-push every branch + tag to origin: " ans
    if [[ "$ans" != "yes" ]]; then
        echo "aborted."
        exit 1
    fi
fi
git push --force --all origin
git push --force --tags origin
echo "✅ origin force-pushed. Collaborators must re-clone or rebase."
