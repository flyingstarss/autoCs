param(
    [switch]$Stop,
    [int]$Port = 8000
)

# Restart the cs-agent console on Windows:
#   1) kill stale uvicorn (the old process keeps holding port 8000 -> WinError 10013)
#   2) pick a free port (8000 is also sometimes inside a Windows reserved range)
#   3) start uvicorn, wait for /healthz with the new build marker, open the console
# Usage:  restart_console.bat            (restart + open browser)
#         restart_console.bat -Stop      (only stop the running server)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONPATH = Join-Path $root 'src'
$env:PYTHONIOENCODING = 'utf-8'

if ($env:PYTHON) {
    $python = $env:PYTHON
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = 'python'
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $python = 'py'
} else {
    throw 'Python not found: install Python 3.11+ or set the PYTHON environment variable to the interpreter path'
}

function Test-PortFree([int]$p) {
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $p)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

Write-Host ('==== [1/4] stop stale uvicorn (target port ' + $Port + ') ====')

# uvicorn --reload = supervisor (cmdline contains "uvicorn") + child that binds the port.
# Kill the supervisor first, otherwise it respawns the child and the port stays busy.
$supers = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'uvicorn' })
if ($supers.Count -gt 0) {
    foreach ($p in $supers) {
        Write-Host ('   stop supervisor PID ' + $p.ProcessId)
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 800
}

$owners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { $_.OwningProcess } | Select-Object -Unique)
foreach ($procId in $owners) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
    if ($proc -and $proc.Name -eq 'python.exe') {
        Write-Host ('   stop listener PID ' + $procId)
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    } elseif ($proc) {
        Write-Host ('   [WARN] port ' + $Port + ' is held by ' + $proc.Name + ' (PID ' + $procId + '), not python')
    }
}
Start-Sleep -Milliseconds 700

if ($Stop) {
    Write-Host 'Stopped. Port should be free now.'
    exit 0
}

$chosen = $null
foreach ($candidate in @($Port, $Port + 10, $Port + 20, $Port + 30, $Port + 40)) {
    if (Test-PortFree $candidate) { $chosen = $candidate; break }
}
if (-not $chosen) {
    Write-Host ('[FAIL] ports ' + $Port + '..' + ($Port + 40) + ' are all unavailable.')
    Write-Host '       Check Windows reserved ranges: netsh int ipv4 show excludedportrange protocol=tcp'
    exit 1
}
if ($chosen -ne $Port) {
    Write-Host ('[WARN] port ' + $Port + ' is blocked (other program or Windows reserved range), using ' + $chosen + ' instead')
}

Write-Host ('==== [2/4] start uvicorn on 127.0.0.1:' + $chosen + ' ====')
$uvArgs = @('-m', 'uvicorn', 'cs_agent_system.api:app', '--host', '127.0.0.1', '--port', "$chosen", '--reload')
$uv = Start-Process -FilePath $python -ArgumentList $uvArgs -WorkingDirectory $root -PassThru
Write-Host ('   launcher PID ' + $uv.Id + ' (logs in the new uvicorn window)')

Write-Host '==== [3/4] wait for /healthz with build marker ===='
$ok = $false
$body = ''
for ($i = 1; $i -le 45; $i++) {
    if (Test-PortFree $chosen) { Start-Sleep -Milliseconds 700; continue }
    try {
        $r = Invoke-WebRequest -UseBasicParsing ('http://127.0.0.1:' + $chosen + '/healthz') -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            $body = $r.Content
            if ($body -match 'per-turn-budget') {
                $ok = $true
                break
            }
            Write-Host '   [WARN] an OLD process answers on this port (no build marker). Stop it and retry.'
            break
        }
    } catch { }
    Start-Sleep -Milliseconds 700
}

if (-not $ok) {
    Write-Host '[FAIL] no valid response. Read the uvicorn window, or start manually:'
    Write-Host ('       ' + $python + ' -m uvicorn cs_agent_system.api:app --port ' + $chosen)
    exit 1
}

Write-Host '==== [4/4] healthz ===='
Write-Host $body
Write-Host ''
Write-Host ('Console: http://127.0.0.1:' + $chosen + '/')
Write-Host 'Expect: build = per-turn-budget+deepseek, gateway = LangChainLLMGateway'
Start-Process ('http://127.0.0.1:' + $chosen + '/')
