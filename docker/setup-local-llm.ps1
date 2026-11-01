param(
    [string]$RobotContainer = 'cargo_robot_project',
    [string]$Model = 'qwen2.5:3b'
)
$ErrorActionPreference = 'Stop'
if ($Model -match 'cloud') { throw 'Specify a local model, not a cloud model.' }

function Invoke-Docker {
    param([string[]]$DockerArgs)
    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed: $($DockerArgs[0])" }
}

& docker inspect $RobotContainer --format '{{.Name}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Robot container does not exist: $RobotContainer" }
& docker network inspect cargo_ai_local --format '{{.Name}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Invoke-Docker -DockerArgs @('network','create','cargo_ai_local') }
$networks = & docker inspect $RobotContainer --format '{{json .NetworkSettings.Networks}}' | ConvertFrom-Json
if (-not $networks.PSObject.Properties['cargo_ai_local']) {
    Invoke-Docker -DockerArgs @('network','connect','cargo_ai_local',$RobotContainer)
}
& docker inspect cargo_ollama --format '{{.Name}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Invoke-Docker -DockerArgs @('run','-d','--name','cargo_ollama','--network','cargo_ai_local',
        '--network-alias','cargo-ollama','--memory','6g','-e','OLLAMA_NO_CLOUD=1',
        '-e','OLLAMA_NUM_PARALLEL=1','-e','OLLAMA_MAX_LOADED_MODELS=1',
        '-v','cargo_ollama_models:/root/.ollama','ollama/ollama@sha256:dea72d1e22270fe095b68fdf756d2b489d615643c5a2eaebbc5f569181e322d8')
} else {
    $details = (& docker inspect cargo_ollama | ConvertFrom-Json)[0]
    $ownedVolume = @($details.Mounts | Where-Object { $_.Destination -eq '/root/.ollama' -and $_.Name -eq 'cargo_ollama_models' })
    if ($ownedVolume.Count -ne 1 -or $details.Config.Env -notcontains 'OLLAMA_NO_CLOUD=1') {
        throw 'cargo_ollama belongs to another setup. Keep it unchanged and choose another setup.'
    }
    Invoke-Docker -DockerArgs @('update','--memory','6g','cargo_ollama')
    Invoke-Docker -DockerArgs @('start','cargo_ollama')
}
Invoke-Docker -DockerArgs @('exec','cargo_ollama','ollama','pull',$Model)
Write-Output "Local LLM ready. Run inside the robot container:"
Write-Output "python3 scripts/run_language_mission.py --backend ollama --model $Model --gui"
