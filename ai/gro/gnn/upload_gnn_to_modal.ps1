# upload_gnn_to_modal.ps1
# Upload GNN Policy files + RSL-RL source to Modal volume
#
# Chạy script:
#   .\upload_gnn_to_modal.ps1
#
# YÊU CẦU:
#   - Đang đứng trong thư mục: D:\VHAS\gro\gnn
#   - Đã cài Modal CLI và đăng nhập

Write-Host "================================================================================"
Write-Host "UPLOADING GNN FILES + RSL-RL TO MODAL" -ForegroundColor Cyan
Write-Host "================================================================================"

# 1) Upload các file GNN core (env, wrapper, policy, vec_env, train)
$files = @(
    @{Local="env.py";               Remote="/data/gro/gnn/env.py"},
    @{Local="wrappers.py";          Remote="/data/gro/gnn/wrappers.py"},
    @{Local="actor_critic_gnn.py";  Remote="/data/gro/gnn/actor_critic_gnn.py"},
    @{Local="vec_env_wrapper.py";   Remote="/data/gro/gnn/vec_env_wrapper.py"},
    @{Local="train.py";             Remote="/data/gro/gnn/train.py"}
)

foreach ($file in $files) {
    Write-Host ""
    Write-Host "Uploading: $($file.Local)" -ForegroundColor Yellow
    Write-Host "   -> $($file.Remote)"
    
    if (-Not (Test-Path $file.Local)) {
        Write-Host "   File not found!" -ForegroundColor Red
        continue
    }
    
    $cmd = "modal volume put --force vhas-training-data $($file.Local) $($file.Remote)"
    Write-Host "   Force overwrite enabled" -ForegroundColor DarkYellow
    Write-Host "   Running: $cmd" -ForegroundColor Gray
    
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Success!" -ForegroundColor Green
    } else {
        Write-Host "   Failed!" -ForegroundColor Red
    }
}

# 2) Upload full thư mục rsl_rl (source code framework) lên Modal
Write-Host ""
Write-Host "Uploading RSL-RL framework directory..." -ForegroundColor Yellow

# Đường dẫn local tới repo rsl_rl (từ gro/gnn đi lên 2 cấp)
$rslLocal = "..\..\rsl_rl"
$rslRemote = "/data/rsl_rl"

if (-Not (Test-Path $rslLocal)) {
    Write-Host "   RSL-RL folder not found at $rslLocal" -ForegroundColor Red
} else {
    $cmdRsl = "modal volume put --force vhas-training-data $rslLocal $rslRemote"
    Write-Host "   This may take a while (uploading entire framework)" -ForegroundColor DarkYellow
    Write-Host "   Running: $cmdRsl" -ForegroundColor Gray
    
    Invoke-Expression $cmdRsl
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   RSL-RL uploaded successfully!" -ForegroundColor Green
    } else {
        Write-Host "   Failed to upload RSL-RL!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================================================"
Write-Host "UPLOAD COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================"
Write-Host ""
Write-Host "Files uploaded to Modal volume: vhas-training-data"
Write-Host "Remote code paths:"
Write-Host "   - /data/gro/gnn/*.py      (GNN env, wrappers, policy, train)"
Write-Host "   - /data/rsl_rl/           (RSL-RL framework source)"
Write-Host ""
Write-Host "Test pipeline with ~200k timesteps:"
Write-Host "   modal run --detach train_gnn_deployed.py::start_training --total-timesteps 200000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Full training with 1M timesteps:"
Write-Host "   modal run --detach train_gnn_deployed.py::start_training --total-timesteps 1000000" -ForegroundColor Cyan
Write-Host "================================================================================"

