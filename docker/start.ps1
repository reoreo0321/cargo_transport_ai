[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$ContainerName = 'cargo_robot_project',
    [ValidateRange(1024, 65535)]
    [int]$Port = 6081
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$image = 'airobotbook/ros2-desktop-ai-robot-book-humble@sha256:b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63'

& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Start Docker Desktop in Linux-container mode and retry.'
}

$existing = & docker container ls --all --format '{{.Names}}' --filter "name=^/$ContainerName$"
if ($LASTEXITCODE -ne 0) { throw 'Could not list Docker containers.' }

if ($existing -contains $ContainerName) {
    $details = (& docker inspect $ContainerName | ConvertFrom-Json)[0]
    $mount = @($details.Mounts | Where-Object { $_.Destination -eq '/workspace/cargo_ws' })
    if ($mount.Count -ne 1) {
        throw 'This container belongs to another setup. Choose another -ContainerName.'
    }
    $actualSource = [System.IO.Path]::GetFullPath($mount[0].Source).TrimEnd([char[]]'\/')
    if (-not $actualSource.Equals($repoRoot.TrimEnd([char[]]'\/'), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The existing container uses a different project folder. Choose another -ContainerName.'
    }
    if (-not $details.State.Running) {
        & docker start $ContainerName | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not start the container.' }
    }
} else {
    & docker run --detach --name $ContainerName `
        --publish "127.0.0.1:${Port}:80" --shm-size=512m `
        --mount "type=bind,source=$repoRoot,target=/workspace/cargo_ws" $image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project container.' }
}

# Only this mounted project is changed. Existing lecture containers are untouched.
& docker exec --user root $ContainerName chown -R ubuntu:ubuntu /workspace/cargo_ws
if ($LASTEXITCODE -ne 0) { throw 'Could not set project ownership for ubuntu.' }

$published = & docker port $ContainerName '80/tcp'
Write-Host "Project: $repoRoot"
Write-Host "Container: $ContainerName"
Write-Host "Desktop port: $published"
Write-Host 'Open the browser URL using the port above (normally http://localhost:6081/).'
Write-Host "Terminal: docker exec -it --user ubuntu --workdir /workspace/cargo_ws $ContainerName bash"
