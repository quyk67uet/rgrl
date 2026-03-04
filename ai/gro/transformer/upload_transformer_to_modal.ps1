# upload_transformer_to_modal.ps1
# Upload Transformer Policy files to Modal volume

# Enter this command to run the script:
# .\upload_transformer_to_modal.ps1

Write-Host "================================================================================"
Write-Host "UPLOADING TRANSFORMER FILES TO MODAL" -ForegroundColor Cyan
Write-Host "================================================================================"

$files = @(
    @{Local="policy_model.py"; Remote="/data/gro/transformer/policy_model.py"},
    @{Local="env.py"; Remote="/data/gro/transformer/env.py"},
    @{Local="train_orchestrator_sb3.py"; Remote="/data/gro/transformer/train_orchestrator_sb3.py"},
    @{Local="..\scripts\guidance.py"; Remote="/data/gro/transformer/guidance.py"}
)

foreach ($file in $files) {
    Write-Host ""
    Write-Host "📄 Uploading: $($file.Local)" -ForegroundColor Yellow
    Write-Host "   -> $($file.Remote)"
    
    if (-Not (Test-Path $file.Local)) {
        Write-Host "   ❌ File not found!" -ForegroundColor Red
        continue
    }
    
    # Upload using Modal CLI
    $cmd = "modal volume put --force vhas-training-data $($file.Local) $($file.Remote)"
    Write-Host "   ⚠️  Force overwrite enabled" -ForegroundColor DarkYellow

    Write-Host "   Running: $cmd" -ForegroundColor Gray
    
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Success!" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Failed!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================================================"
Write-Host "UPLOAD COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================"
Write-Host ""
Write-Host "Files uploaded to Modal volume: vhas-training-data"
Write-Host "Remote path: /data/gro/transformer/"
Write-Host ""
Write-Host "🧪 Test pipeline with 2048 steps:"
Write-Host "   modal run --detach train_transformer_deployed.py::start_training --total-timesteps 2048" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 Full training with 1M steps:"
Write-Host "   modal run --detach train_transformer_deployed.py::start_training --total-timesteps 1000000" -ForegroundColor Cyan
Write-Host "================================================================================"
