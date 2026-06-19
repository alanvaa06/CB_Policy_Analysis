# context-size-check.ps1 — UserPromptSubmit hook
# Stats docs/context/*.md every turn; warns when any file is over its hard cap.
# Stdout on exit 0 is injected into context, so Claude sees the warning and can
# react by running /compact-context. Caps: approx tokens = bytes / 4.
$ErrorActionPreference = 'SilentlyContinue'

$root = $env:CLAUDE_PROJECT_DIR
if (-not $root) { $root = (Get-Location).Path }
$ctx = Join-Path $root 'docs/context'

# caps in bytes (tokens * 4)
$caps = [ordered]@{
  'memory.md'     = 44000   # 11k tokens
  'lessons.md'    = 28000   #  7k tokens
  'todo.md'       = 10000   #  2.5k tokens
  'results.md'    = 24000   #  6k tokens
  'sesion-log.md' = 16000   #  4k tokens
}

$over = @()
foreach ($name in $caps.Keys) {
  $path = Join-Path $ctx $name
  if (Test-Path $path) {
    $bytes = (Get-Item $path).Length
    if ($bytes -gt $caps[$name]) {
      $tok = [math]::Round($bytes / 4)
      $cap = [math]::Round($caps[$name] / 4)
      $over += "$name ~${tok}t > ${cap}t cap"
    }
  }
}

if ($over.Count -gt 0) {
  Write-Output "[context-size] OVER CAP: $($over -join '; '). Run /compact-context."
}
exit 0
