#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_PATH="${REPO_ROOT}/docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE_RESULT_TEMPLATE.md"

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "Template not found: ${TEMPLATE_PATH}" >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
Usage: new_deferred_carry_smoke_result.sh [--blocked] [output-path]
       new_deferred_carry_smoke_result.sh [--blocked] --output <output-path> [--edited-file <path> ...]
       new_deferred_carry_smoke_result.sh [--blocked] [result flags]
       new_deferred_carry_smoke_result.sh --blocked --reuse-latest-blocked [--max-age-hours <hours>]

Notes:
- Backward-compatible form keeps [output-path] as the first positional argument.
- Use --edited-file for exact path evidence when the repo has unrelated pre-existing churn.
- Result flags:
  --key-release pass|fail|blocked
  --click-jump pass|fail|blocked
  --mode-switch pass|fail|blocked
  --direction-reversal pass|fail|blocked
  --ambiguous yes|no|unknown
- Reuse flags:
  --reuse-latest-blocked
  --max-age-hours <hours> (default: 24; only used with --reuse-latest-blocked)
USAGE
}

is_valid_result_value() {
  local value="$1"
  [[ "${value}" == "pass" || "${value}" == "fail" || "${value}" == "blocked" ]]
}

is_valid_ambiguous_value() {
  local value="$1"
  [[ "${value}" == "yes" || "${value}" == "no" || "${value}" == "unknown" ]]
}

blocked_mode=0
output_path=""
edited_files=()
positional=()
key_release_result=""
click_jump_result=""
mode_switch_result=""
direction_reversal_result=""
ambiguous_value=""
reuse_latest_blocked=0
max_age_hours=24

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blocked)
      blocked_mode=1
      shift
      ;;
    -o|--output)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      output_path="$2"
      shift 2
      ;;
    -e|--edited-file)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      edited_files+=("$2")
      shift 2
      ;;
    --key-release)
      if [[ $# -lt 2 ]] || ! is_valid_result_value "$2"; then
        echo "Invalid value for $1 (expected pass|fail|blocked)" >&2
        usage >&2
        exit 1
      fi
      key_release_result="$2"
      shift 2
      ;;
    --click-jump)
      if [[ $# -lt 2 ]] || ! is_valid_result_value "$2"; then
        echo "Invalid value for $1 (expected pass|fail|blocked)" >&2
        usage >&2
        exit 1
      fi
      click_jump_result="$2"
      shift 2
      ;;
    --mode-switch)
      if [[ $# -lt 2 ]] || ! is_valid_result_value "$2"; then
        echo "Invalid value for $1 (expected pass|fail|blocked)" >&2
        usage >&2
        exit 1
      fi
      mode_switch_result="$2"
      shift 2
      ;;
    --direction-reversal)
      if [[ $# -lt 2 ]] || ! is_valid_result_value "$2"; then
        echo "Invalid value for $1 (expected pass|fail|blocked)" >&2
        usage >&2
        exit 1
      fi
      direction_reversal_result="$2"
      shift 2
      ;;
    --ambiguous)
      if [[ $# -lt 2 ]] || ! is_valid_ambiguous_value "$2"; then
        echo "Invalid value for $1 (expected yes|no|unknown)" >&2
        usage >&2
        exit 1
      fi
      ambiguous_value="$2"
      shift 2
      ;;
    --reuse-latest-blocked)
      reuse_latest_blocked=1
      shift
      ;;
    --max-age-hours)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[0-9]+$ ]] || [[ "$2" -le 0 ]]; then
        echo "Invalid value for $1 (expected positive integer hours)" >&2
        usage >&2
        exit 1
      fi
      max_age_hours="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        positional+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
default_output="${REPO_ROOT}/docs/evidence/deferred-carry/deferred-carry-smoke-${timestamp//[: ]/-}.md"

# Keep one-arg positional compatibility for output path.
if [[ -z "${output_path}" && ${#positional[@]} -gt 0 ]]; then
  output_path="${positional[0]}"
  positional=("${positional[@]:1}")
fi
output_path="${output_path:-${default_output}}"

if [[ ${#positional[@]} -gt 0 ]]; then
  echo "Unexpected argument(s): ${positional[*]}" >&2
  usage >&2
  exit 1
fi

if [[ "${reuse_latest_blocked}" -eq 1 && "${blocked_mode}" -ne 1 ]]; then
  echo "--reuse-latest-blocked requires --blocked" >&2
  usage >&2
  exit 1
fi

if [[ "${reuse_latest_blocked}" -eq 1 ]]; then
  evidence_dir="${REPO_ROOT}/docs/evidence/deferred-carry"
  max_age_seconds=$((max_age_hours * 3600))
  latest_blocked_file=""
  if [[ -d "${evidence_dir}" ]]; then
    latest_blocked_file="$(find "${evidence_dir}" -type f -name 'deferred-carry-smoke-*.md' -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n 1 || true)"
  fi

  if [[ -n "${latest_blocked_file}" ]] && rg -q "^- Result: blocked$" "${latest_blocked_file}" 2>/dev/null; then
    now_epoch="$(date +%s)"
    if mtime_epoch="$(stat -f %m "${latest_blocked_file}" 2>/dev/null)"; then
      :
    elif mtime_epoch="$(stat -c %Y "${latest_blocked_file}" 2>/dev/null)"; then
      :
    else
      echo "Could not read mtime for ${latest_blocked_file}" >&2
      exit 1
    fi

    age_seconds=$((now_epoch - mtime_epoch))
    if [[ "${age_seconds}" -le "${max_age_seconds}" ]]; then
      echo "REUSED_BLOCKED_ARTIFACT:${latest_blocked_file}"
      echo "age_seconds=${age_seconds}"
      echo "max_age_seconds=${max_age_seconds}"
      exit 0
    fi
  fi
fi

branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<unknown-branch>')"
commit="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '<unknown-commit>')"

if [[ ${#edited_files[@]} -eq 0 ]]; then
  while IFS= read -r path; do
    [[ -n "${path}" ]] && edited_files+=("${path}")
  done < <(
    git -C "${REPO_ROOT}" diff --name-only -- \
      Lostless-Mask-Editor/nodes/mask_editor.py \
      docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md \
      docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE_RESULT_TEMPLATE.md \
      docs/MASK_EDITOR_DEFERRED_CARRY_RESULTS_TEMPLATE.md \
      docs/DEFERRED_CARRY_STATE_MACHINE_SMOKE.md \
      scripts/new_deferred_carry_smoke_result.sh 2>/dev/null || true
  )

  while IFS= read -r path; do
    [[ -n "${path}" ]] && edited_files+=("${path}")
  done < <(
    git -C "${REPO_ROOT}" ls-files --others --exclude-standard -- \
      Lostless-Mask-Editor/nodes/mask_editor.py \
      docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md \
      docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE_RESULT_TEMPLATE.md \
      docs/MASK_EDITOR_DEFERRED_CARRY_RESULTS_TEMPLATE.md \
      docs/DEFERRED_CARRY_STATE_MACHINE_SMOKE.md \
      scripts/new_deferred_carry_smoke_result.sh 2>/dev/null || true
  )
fi

if [[ ${#edited_files[@]} -eq 0 ]]; then
  edited_files=("Lostless-Mask-Editor/nodes/mask_editor.py")
fi

# Deduplicate while preserving order.
deduped_edited_files=()
seen_paths="|"
for path in "${edited_files[@]}"; do
  if [[ "${seen_paths}" != *"|${path}|"* ]]; then
    deduped_edited_files+=("${path}")
    seen_paths="${seen_paths}${path}|"
  fi
done

edited_paths_csv=""
for path in "${deduped_edited_files[@]}"; do
  if [[ -z "${edited_paths_csv}" ]]; then
    edited_paths_csv="${path}"
  else
    edited_paths_csv="${edited_paths_csv}, ${path}"
  fi
done

mkdir -p "$(dirname "${output_path}")"

awk \
  -v now="${timestamp}" \
  -v branch_commit="${branch} @ ${commit}" \
  -v edited_paths="${edited_paths_csv}" \
  -v blocked_mode="${blocked_mode}" \
  -v key_release_result="${key_release_result}" \
  -v click_jump_result="${click_jump_result}" \
  -v mode_switch_result="${mode_switch_result}" \
  -v direction_reversal_result="${direction_reversal_result}" \
  -v ambiguous_value="${ambiguous_value}" \
  '
  /^1\. Key release flush$/ {
    current_case = "key_release"
    print
    next
  }
  /^2\. Click-jump boundary$/ {
    current_case = "click_jump"
    print
    next
  }
  /^3\. Mode\/tool switch boundary$/ {
    current_case = "mode_switch"
    print
    next
  }
  /^4\. Direction reversal boundary$/ {
    current_case = "direction_reversal"
    print
    next
  }
  /^- Date\/time:/ {
    print "- Date/time: " now
    next
  }
  /^- Branch\/commit:/ {
    print "- Branch/commit: " branch_commit
    next
  }
  /^- Edited file paths:/ {
    print "- Edited file paths: " edited_paths
    next
  }
  /^- Result: pass \| fail \| blocked$/ {
    if (blocked_mode == 1) {
      print "- Result: blocked"
    } else if (current_case == "key_release" && key_release_result != "") {
      print "- Result: " key_release_result
    } else if (current_case == "click_jump" && click_jump_result != "") {
      print "- Result: " click_jump_result
    } else if (current_case == "mode_switch" && mode_switch_result != "") {
      print "- Result: " mode_switch_result
    } else if (current_case == "direction_reversal" && direction_reversal_result != "") {
      print "- Result: " direction_reversal_result
    } else {
      print
    }
    next
  }
  /^- Notes:$/ {
    if (blocked_mode == 1) {
      print "- Notes: Manual ComfyUI UI/device validation unavailable on this machine; run this checklist on the target host and update this artifact with pass/fail outcomes."
    } else {
      print
    }
    next
  }
  /^- Ambiguous behavior observed: yes \| no \| unknown$/ {
    if (blocked_mode == 1) {
      print "- Ambiguous behavior observed: unknown"
    } else if (ambiguous_value != "") {
      print "- Ambiguous behavior observed: " ambiguous_value
    } else {
      print
    }
    next
  }
  /^- Follow-up required:$/ {
    if (blocked_mode == 1) {
      print "- Follow-up required: Execute docs/MASK_EDITOR_DEFERRED_CARRY_SMOKE.md on the ComfyUI host and replace each blocked result with pass/fail plus notes."
    } else {
      print
    }
    next
  }
  { print }
  ' "${TEMPLATE_PATH}" > "${output_path}"

echo "Created deferred-carry smoke result artifact:"
echo "${output_path}"
