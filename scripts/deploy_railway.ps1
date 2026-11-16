# Run after Railway trial/plan is active and a Groq API key is available.
param(
  [Parameter(Mandatory = $true)]
    [string]$GroqApiKey,
    [string]$Model = "openai/gpt-oss-20b",
  [string]$ProjectName = "greenko-rag-assistant"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

railway whoami
railway init -n $ProjectName
railway up
railway variables set "GROQ_API_KEY=$GroqApiKey"
railway variables set "GROQ_MODEL=$Model"
railway domain
Write-Host "Paste the public URL into README.md under Live URL."
