param(
    [string]$Network = "5-rce_default",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"

if (-not $Command -or $Command.Count -eq 0) {
    Write-Error "Usage: .\scripts\stage2_tool.ps1 [-Network 5-rce_default] <tool> [args...]"
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

docker run --rm `
    --network $Network `
    -v "${ProjectRoot}:/work" `
    -w /work `
    webattacksim-stage2-tools `
    @Command
