# UDP arrival probe for the MAP65 export.
# Run on the RECEIVING PC (PC-B) with MAP65 CLOSED so port 50002 is free.
# Prints the size + source of each datagram that arrives; says so if none do.
#
#   powershell -ExecutionPolicy Bypass -File udp_listen.ps1            # port 50002
#   powershell -ExecutionPolicy Bypass -File udp_listen.ps1 -Port 50002 -Count 5
param([int]$Port = 50002, [int]$Count = 5)

$u = New-Object System.Net.Sockets.UdpClient $Port
$u.Client.ReceiveTimeout = 10000          # 10 s, so it won't hang forever
$e = New-Object System.Net.IPEndPoint([Net.IPAddress]::Any, 0)
Write-Host "listening on UDP $Port (10 s timeout per packet)..."
for ($i = 0; $i -lt $Count; $i++) {
    try {
        $d = $u.Receive([ref]$e)
        Write-Host ("#{0}: {1} bytes from {2}" -f ($i + 1), $d.Length, $e.Address)
    } catch {
        Write-Host "no packet within timeout - nothing is arriving on UDP $Port"
        break
    }
}
$u.Close()
