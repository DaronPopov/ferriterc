# diag.sh — shared installer diagnostics contract
# Sourced by scripts/install/install.sh and related scripts.

diag_escape_json() {
  local s="${1:-}"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

diag_emit() {
  local component="${1:-installer}"
  local status="${2:-PASS}"
  local code="${3:-GEN-0000}"
  local summary="${4:-}"
  local remediation="${5:-none}"
  local stream="${6:-stderr}"

  local fmt="${FERRITE_DIAG_FORMAT:-plain}"
  local out_fd=2
  [[ "$stream" == "stdout" ]] && out_fd=1

  if [[ "$fmt" == "json" ]]; then
    printf '{"component":"%s","status":"%s","code":"%s","summary":"%s","remediation":"%s"}\n' \
      "$(diag_escape_json "$component")" \
      "$(diag_escape_json "$status")" \
      "$(diag_escape_json "$code")" \
      "$(diag_escape_json "$summary")" \
      "$(diag_escape_json "$remediation")" >&${out_fd}
  else
    printf '[diag] component=%s status=%s code=%s summary=%s remediation=%s\n' \
      "$component" "$status" "$code" "$summary" "$remediation" >&${out_fd}
  fi
}
